"""Runner ↔ Reference Calibration integration tests (v0.4-B §22).

Full chain: fixture ReferenceSet (trust chain intact) → UnifiedAuditRunner →
Quick Suite → raw profile → calibration → calibrated profile → claimed-model
gap → JSON / HTML reports.  Plus backward compatibility and fail-closed paths.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx

from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.artifacts import RunArtifactRepository, sha256_of
from llmtrace.execution.models import UnifiedRunStatus
from llmtrace.execution.runner import PreflightError, UnifiedAuditRunner
from llmtrace.reference.builder import ReferenceSnapshotBuilder
from llmtrace.reference.reference_set import ReferenceSetBuilder
from llmtrace.scoring.reference import ReferenceRepository

from ..reference.helpers import commit_run, make_capability_profile, make_manifest
from .conftest import TrustedFakeBackend
from .test_runner import API_KEY, TARGET_ID, _mock_openai

# Five reference configurations spanning low → flagship capability levels
# (§12).  Scores per dimension are uniform per model; what matters for the
# fixture is the spread across identities.
_REFERENCE_MODELS: tuple[tuple[str, float], ...] = (
    ("ref-model-low", 0.30),
    ("ref-model-midlow", 0.45),
    ("ref-model-mid", 0.60),
    ("ref-model-high", 0.75),
    ("ref-model-flagship", 0.90),
)

_COVERAGE_WEIGHT = 0.75


def _build_reference_fixture(
    artifact_root: Path,
    reference_root: Path,
    *,
    models: tuple[tuple[str, float], ...] = _REFERENCE_MODELS,
    set_id: str = "llmtrace-reference-v1",
    set_version: str = "0.1.0",
) -> Path:
    """Build a complete trusted ReferenceSet fixture and return its set path.

    Each member carries the full v0.4-A trust chain: qualified run artifact →
    trusted snapshot (with integrity sidecar) → ReferenceSet member.
    """
    repository = RunArtifactRepository(artifact_root)
    snapshot_repository = ReferenceRepository(directory=reference_root / "snapshots")
    snapshot_builder = ReferenceSnapshotBuilder()

    snapshots = []
    for index, (model_id, score) in enumerate(models):
        execution_id = f"aaaaaaaa-0000-0000-0000-{index:012d}"
        manifest = make_manifest(execution_id=execution_id).model_copy(
            update={"candidate_model_id": model_id}
        )
        profile = make_capability_profile(score).model_copy(
            update={"provisional_raw_index": round(score * _COVERAGE_WEIGHT, 6)}
        )
        # Rewrite the artifact with the model-specific profile.
        commit_run(
            artifact_root,
            execution_id=execution_id,
            manifest=manifest,
            profile=profile,
        )
        snapshot = snapshot_builder.build(
            execution_id=execution_id,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id=f"snap-ref-{index}",
            created_by="operator",
        )
        snapshots.append(snapshot)

    reference_set = ReferenceSetBuilder().build(
        reference_set_id=set_id,
        reference_set_version=set_version,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        snapshots=snapshots,
        snapshot_sha256s={
            s.snapshot_id: sha256_of(snapshot_repository.read_raw(s.snapshot_id)) for s in snapshots
        },
    )

    sets_dir = reference_root / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    set_path = sets_dir / f"{set_id}_{set_version}.json"
    set_path.write_text(reference_set.model_dump_json(indent=2), encoding="utf-8")
    return set_path


async def _run_with_reference_set(
    config: AuditConfig,
    tmp_path: Path,
    set_path: Path | None,
) -> object:
    repo = RunArtifactRepository(tmp_path)
    with respx.mock as mock:
        _mock_openai(mock)
        runner = UnifiedAuditRunner(
            config,
            api_key=API_KEY,
            target_id=TARGET_ID,
            repository=repo,
            code_backend=TrustedFakeBackend(),
            reference_set_path=set_path,
        )
        return await runner.run()


@pytest.fixture
def config() -> AuditConfig:
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com/v1",
        model="my-real-model",
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
        output_dir="reports",
    )


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", API_KEY)


class TestRunnerCalibrationIntegration:
    @pytest.mark.asyncio
    async def test_full_calibration_pipeline_with_claimed_gap(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        """§22 Runner Integration: the flagship model is the claimed model."""
        # The claimed model exists among the reference identities.
        models = (
            ("ref-model-low", 0.30),
            ("ref-model-midlow", 0.45),
            ("ref-model-mid", 0.60),
            ("my-real-model", 0.75),
            ("ref-model-flagship", 0.90),
        )
        set_path = _build_reference_fixture(tmp_path, tmp_path / "references", models=models)
        result = await _run_with_reference_set(config, tmp_path, set_path)

        assert result.status == UnifiedRunStatus.COMPLETED
        assert result.capability_profile is not None

        # Formal 0–100 scores exist.
        profile = result.capability_profile
        assert profile.calibration is not None
        assert profile.calibration.reference_set_id == "llmtrace-reference-v1"
        assert profile.calibration.reference_identity_count == 5
        assert profile.calibrated_total_score is not None
        assert 0.0 <= profile.calibrated_total_score <= 100.0
        for dim in profile.dimensions:
            assert dim.calibrated_score is not None
            assert 0.0 <= dim.calibrated_score <= 100.0
            # Raw measurement retained (§18: raw scores are kept).
            assert dim.raw_normalized_score >= 0.0

        # Claimed model gap against the compatible trusted reference.
        gap = result.claimed_model_gap
        assert gap is not None
        assert gap.claimed_model_id == "my-real-model"
        assert gap.reference_model_id == "my-real-model"
        assert gap.total_delta == pytest.approx(gap.candidate_total_score - gap.reference_total_score)
        assert len(gap.dimension_gaps) == 4

        # Artifacts carry the calibration provenance (§10).
        repo = RunArtifactRepository(tmp_path)
        manifest = repo.load_manifest(result.execution_id)
        assert manifest.calibration_policy_id == "llmtrace-reference-calibration-v1"
        assert manifest.reference_set_id == "llmtrace-reference-v1"
        assert manifest.reference_set_content_sha256 is not None

        # JSON report: formal scores + provenance + claimed_model_gap (§18).
        run_dir = tmp_path / "runs" / result.execution_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["capability_profile"]["calibration_status"] == "CALIBRATED"
        assert report["capability_profile"]["calibrated_total_score"] is not None
        assert report["capability_profile"]["calibration"]["reference_set_id"] == "llmtrace-reference-v1"
        assert report["claimed_model_gap"]["available"] is True
        assert report["claimed_model_gap"]["total_delta"] == pytest.approx(gap.total_delta)

        # HTML report: Capability Score + Claimed Model Comparison (§19).
        html = (run_dir / "report.html").read_text(encoding="utf-8")
        assert "Capability Score" in html
        assert "Claimed Model Comparison" in html
        assert "my-real-model" in html

    @pytest.mark.asyncio
    async def test_backward_compatible_without_reference_set(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        """§16/§22: no calibration → run still works, stays UNCALIBRATED."""
        result = await _run_with_reference_set(config, tmp_path, None)

        assert result.status == UnifiedRunStatus.COMPLETED
        assert result.capability_profile is not None
        assert result.capability_profile.calibration is None
        assert result.capability_profile.calibrated_total_score is None
        assert result.claimed_model_gap is None

        run_dir = tmp_path / "runs" / result.execution_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["capability_profile"]["calibration_status"] == "UNCALIBRATED"
        assert "claimed_model_gap" not in report
        assert "UNCALIBRATED" in (run_dir / "report.html").read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_insufficient_identities_skip_calibration_fail_closed(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        """§22 Fail Closed: < 5 distinct identities → no fake scores."""
        models = (
            ("ref-model-a", 0.30),
            ("ref-model-b", 0.60),
            ("ref-model-c", 0.90),
        )
        set_path = _build_reference_fixture(tmp_path, tmp_path / "references", models=models)
        result = await _run_with_reference_set(config, tmp_path, set_path)

        # Fail-closed is a skip with a warning — the run itself is healthy.
        assert result.status == UnifiedRunStatus.COMPLETED_WITH_WARNINGS
        assert result.capability_profile is not None
        assert result.capability_profile.calibration is None
        assert result.capability_profile.calibrated_total_score is None
        assert result.claimed_model_gap is None
        assert any("reference calibration skipped" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_claimed_model_not_in_reference_set(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        """§13: no compatible reference → gap unavailable, no fuzzy match."""
        set_path = _build_reference_fixture(tmp_path, tmp_path / "references")
        result = await _run_with_reference_set(config, tmp_path, set_path)

        # Gap unavailability is a warning-carrying skip, not a run failure.
        assert result.status == UnifiedRunStatus.COMPLETED_WITH_WARNINGS
        assert result.capability_profile is not None
        assert result.capability_profile.calibration is not None
        # Calibration succeeded, but the claimed model has no trusted reference.
        assert result.claimed_model_gap is None
        assert any("claimed-model gap unavailable" in w for w in result.warnings)

        run_dir = tmp_path / "runs" / result.execution_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["capability_profile"]["calibration_status"] == "CALIBRATED"
        assert "claimed_model_gap" not in report


class TestRunnerCalibrationPreflight:
    @pytest.mark.asyncio
    async def test_corrupt_reference_set_rejected_preflight(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        """A tampered set file fails closed before any HTTP request."""
        set_path = _build_reference_fixture(tmp_path, tmp_path / "references")
        payload = json.loads(set_path.read_text(encoding="utf-8"))
        payload["description"] = "tampered after signing"
        set_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(PreflightError) as excinfo:
            await _run_with_reference_set(config, tmp_path, set_path)
        assert "reference set" in str(excinfo.value).lower()
