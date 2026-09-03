"""Tests for Runner calibration integration — v0.4-B.

Regression coverage for the owner-confirmed correctness blockers:

    A. production plan provenance — ``runner.run(reference_set=...)`` must
       populate all five calibration-context fields on the plan AND the
       manifest, and they must agree with ``CapabilityProfile.calibration``
    B. partial candidate measurement (31 graded + 1 failure) → no calibration
    C. ungradable candidate measurement (31 graded + 1 ungradable) → no calibration
    D. different ReferenceSet content → different plan_id (same target config)

Per §19, the runner integration path uses the *real* trust chain built by
``build_trusted_reference_root`` (production builders + repositories), never
``model_construct`` with fake SHA-256s.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.runner import PreflightError, UnifiedAuditRunner
from llmtrace.scoring.calibration import ReferenceCalibrationPolicy
from tests.execution.conftest import TrustedFakeBackend
from tests.execution.test_runner import _completion_json, _mock_openai, _models_json
from tests.reference.helpers import build_trusted_reference_root


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The protocol config-precheck probe reads the key from the environment."""
    monkeypatch.setenv("TEST_KEY", "test-key")


def _make_config():
    from llmtrace.config import AuditConfig, Protocol

    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com/v1",
        model="demo-model",
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
    )


def _runner(tmp_path: Path, **kwargs) -> UnifiedAuditRunner:
    """A runner whose artifact repository is the *same* store that produced
    the reference runs — the validator must be able to read the member source
    runs from ``self._repository``."""
    repo = RunArtifactRepository(tmp_path / "ref")
    return UnifiedAuditRunner(
        _make_config(),
        api_key="test-key",
        target_id="test-target",
        repository=repo,
        code_backend=TrustedFakeBackend(),
        **kwargs,
    )


def _runner_at(artifact_root: Path, set_path: Path) -> UnifiedAuditRunner:
    """A runner bound to an explicit artifact root (for plan-id tests)."""
    return UnifiedAuditRunner(
        _make_config(),
        api_key="test-key",
        target_id="test-target",
        repository=RunArtifactRepository(artifact_root),
        code_backend=TrustedFakeBackend(),
        reference_set_path=set_path,
    )


def _calibration_plan_fields(result) -> dict[str, str]:
    plan = result.plan
    return {
        "reference_set_id": plan.reference_set_id,
        "reference_set_version": plan.reference_set_version,
        "reference_set_content_sha256": plan.reference_set_content_sha256,
        "calibration_policy_id": plan.calibration_policy_id,
        "calibration_policy_version": plan.calibration_policy_version,
    }


class TestRunnerPreflightReferenceSet:
    def test_invalid_reference_set_file(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad_ref_set.json"
        bad_path.write_text("not valid json")
        runner = _runner(tmp_path, reference_set_path=bad_path)
        with pytest.raises(PreflightError, match="reference set"):
            runner._preflight()

    def test_content_hash_mismatch_rejected(self, tmp_path: Path) -> None:
        """A set whose declared self-hash no longer recomputes fails preflight."""
        _ref_root, set_path = build_trusted_reference_root(tmp_path / "ref")
        from llmtrace.reference.reference_set import ReferenceSet

        reference_set = ReferenceSet.model_validate_json(set_path.read_text(encoding="utf-8"))
        bad_path = tmp_path / "ref" / "references" / "sets" / "bad.json"
        bad = reference_set.model_copy(update={"content_sha256": "0" * 64})
        bad_path.write_text(bad.model_dump_json())
        runner = _runner(tmp_path, reference_set_path=bad_path)
        with pytest.raises(PreflightError, match="rejected for formal calibration"):
            runner._preflight()

    def test_trusted_reference_set_sets_state(self, tmp_path: Path) -> None:
        """Preflight on a real trust chain resolves CalibrationContext."""
        _ref_root, set_path = build_trusted_reference_root(tmp_path / "ref")
        runner = _runner(tmp_path, reference_set_path=set_path)
        runner._preflight()
        assert runner._calibration_context is not None
        assert runner._calibration_context.reference_set.reference_set_id == "calib-set"
        assert runner._calibration_context.calibration_policy is not None
        assert (
            runner._calibration_context.calibration_policy.policy_id
            == ReferenceCalibrationPolicy.create_v1().policy_id
        )
        # Verified profiles must be pinned in context
        assert len(runner._calibration_context.verified_profiles) == 5

    def test_forged_member_snapshot_sha_rejected(self, tmp_path: Path) -> None:
        """D at the runner level: forged self-hash + wrong member SHA fails preflight."""
        _ref_root, set_path = build_trusted_reference_root(tmp_path / "ref")
        forged = set_path.read_text(encoding="utf-8")
        import json as _json

        data = _json.loads(forged)
        data["members"][0]["snapshot_sha256"] = "1" * 64
        from llmtrace.reference.reference_set import ReferenceSet

        reference_set = ReferenceSet.model_validate(data)
        reference_set = reference_set.model_copy(update={"content_sha256": reference_set.compute_content_sha256()})
        forged_path = tmp_path / "ref" / "references" / "sets" / "forged.json"
        forged_path.write_text(reference_set.model_dump_json())
        runner = _runner(tmp_path, reference_set_path=forged_path)
        with pytest.raises(PreflightError, match="rejected for formal calibration"):
            runner._preflight()


class TestProductionPlanProvenance:
    @pytest.mark.asyncio
    async def test_full_run_populates_calibration_provenance(self, tmp_path: Path, api_key_env: None) -> None:
        """A: a complete run with a trusted, compatible ReferenceSet must
        carry all five calibration-context fields on the plan AND the
        manifest, consistent with the calibrated profile."""
        _ref_root, set_path = build_trusted_reference_root(tmp_path / "ref")
        runner = _runner(tmp_path, reference_set_path=set_path)

        with respx.mock as mock:
            _mock_openai(mock)
            result = await runner.run()

        assert result.status.value.startswith("COMPLETED")
        # The plan carries the bundle — never None.
        plan_fields = _calibration_plan_fields(result)
        assert all(v is not None for v in plan_fields.values())
        assert plan_fields["reference_set_id"] == "calib-set"
        assert plan_fields["calibration_policy_id"] == ReferenceCalibrationPolicy.create_v1().policy_id

        # The candidate measurement was complete, so calibration really ran.
        assert result.measurement_summary is not None
        assert result.measurement_summary.graded_item_count == 32
        assert result.capability_profile is not None
        assert result.capability_profile.calibration is not None
        assert result.capability_profile.calibrated_total_score is not None

        # The manifest copies the plan context verbatim.
        manifest = runner._repository.load_manifest(result.execution_id)
        for field, value in plan_fields.items():
            assert getattr(manifest, field) == value

        # Manifest + plan + CapabilityProfile.calibration all agree.
        calibration = result.capability_profile.calibration
        assert calibration.reference_set_id == manifest.reference_set_id
        assert calibration.reference_set_version == manifest.reference_set_version
        assert calibration.reference_set_content_sha256 == manifest.reference_set_content_sha256
        assert calibration.policy_id == manifest.calibration_policy_id
        assert calibration.policy_version == manifest.calibration_policy_version

    @pytest.mark.asyncio
    async def test_partial_measurement_never_calibrates(self, tmp_path: Path, api_key_env: None) -> None:
        """B: 31 graded + 1 failure — raw profile survives, formal calibration
        does not happen, and a warning explains why."""
        _ref_root, set_path = build_trusted_reference_root(tmp_path / "ref")
        runner = _runner(tmp_path, reference_set_path=set_path)
        post_calls = {"n": 0}

        async def _responder(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json=_models_json())
            post_calls["n"] += 1
            if post_calls["n"] == 4:  # first benchmark item fails
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json=_completion_json("The answer is (A). The answer is 42."))

        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").mock(side_effect=_responder)
            mock.post("http://test.example.com/v1/chat/completions").mock(side_effect=_responder)
            result = await runner.run()

        assert result.measurement_summary is not None
        assert result.measurement_summary.failure_item_count == 1
        assert result.measurement_summary.graded_item_count == 31
        assert result.capability_profile is not None
        assert result.capability_profile.calibration is None
        assert result.capability_profile.calibrated_total_score is None
        assert all(d.calibrated_score is None for d in result.capability_profile.dimensions)
        assert any("candidate measurement incomplete" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_ungradable_measurement_never_calibrates(self, tmp_path: Path, api_key_env: None) -> None:
        """C: 31 graded + 1 ungradable — same refusal."""
        _ref_root, set_path = build_trusted_reference_root(tmp_path / "ref")
        runner = _runner(tmp_path, reference_set_path=set_path)
        post_calls = {"n": 0}

        async def _responder(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json=_models_json())
            post_calls["n"] += 1
            if post_calls["n"] == 4:  # first benchmark item cannot be graded
                return httpx.Response(200, json=_completion_json("I do not know."))
            return httpx.Response(200, json=_completion_json("The answer is (A). The answer is 42."))

        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").mock(side_effect=_responder)
            mock.post("http://test.example.com/v1/chat/completions").mock(side_effect=_responder)
            result = await runner.run()

        assert result.measurement_summary is not None
        assert result.measurement_summary.ungradable_item_count == 1
        assert result.measurement_summary.graded_item_count == 31
        assert result.capability_profile is not None
        assert result.capability_profile.calibration is None
        assert result.capability_profile.calibrated_total_score is None
        assert any("candidate measurement incomplete" in w for w in result.warnings)

    def test_plan_id_differs_across_reference_sets(self, tmp_path: Path) -> None:
        """D: same target config + different ReferenceSet content → different
        plan identity (the plan binds the calibration context)."""
        _root_a, set_a = build_trusted_reference_root(tmp_path / "a", description="reference set A")
        _root_b, set_b = build_trusted_reference_root(tmp_path / "b", description="reference set B")
        runner_a = _runner_at(tmp_path / "a", set_a)
        runner_b = _runner_at(tmp_path / "b", set_b)
        runner_a._preflight()
        runner_b._preflight()

        plan_a = runner_a._plan()
        plan_b = runner_b._plan()
        assert plan_a.reference_set_content_sha256 != plan_b.reference_set_content_sha256
        assert plan_a.plan_id != plan_b.plan_id
