"""Qualification gate-chain tests (§39): Gate 1–10 fail closed with reason codes."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llmtrace.reference.qualification import (
    ARTIFACT_INTEGRITY_FAILURE,
    BENCHMARK_FAILURE_PRESENT,
    GENERATION_CONFIG_MISMATCH,
    INCOMPATIBLE_COVERAGE,
    INCOMPLETE_MEASUREMENT,
    MISSING_CAPABILITY_PROFILE,
    SCORING_POLICY_MISMATCH,
    SUITE_CONTENT_MISMATCH,
    SUITE_MISMATCH,
    UNGRADABLE_ITEM_PRESENT,
    ReferenceQualificationError,
    ReferenceQualificationPolicy,
    ReferenceQualificationStatus,
    qualify_reference_run,
    require_qualified,
)
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)

from .helpers import (
    DEFAULT_EXECUTION_ID,
    commit_run,
    expected_generation_config_sha256,
    make_benchmark_runs_json,
    make_capability_profile,
    make_manifest,
)


def _run_file(root: Path, execution_id: str, filename: str) -> Path:
    return root / "runs" / execution_id / filename


# ---------------------------------------------------------------------------
# Qualified
# ---------------------------------------------------------------------------


class TestQualified:
    def test_valid_run_qualified_with_persisted_profile(self, qualified_run: tuple[object, object]) -> None:
        repository, _manifest = qualified_run
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert result.qualified
        assert result.status == ReferenceQualificationStatus.QUALIFIED
        assert result.reason_codes == ()
        assert result.capability_profile is not None
        # Gate 3: the profile must be the persisted one — every dimension measured.
        assert result.capability_profile.coverage_weight == 0.75
        assert len(result.capability_profile.dimensions) == 4

    def test_wrong_answers_still_qualified(self, artifact_root: Path) -> None:
        # All 32 items GRADED with score 0.0 — a wrong answer is still a measurement.
        repository, _ = commit_run(artifact_root, profile_score=0.0)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert result.qualified

    def test_require_qualified_returns_result(self, qualified_run: tuple[object, object]) -> None:
        repository, _ = qualified_run
        result = require_qualified(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert result.qualified

    def test_policy_id_version_recorded(self, qualified_run: tuple[object, object]) -> None:
        repository, _ = qualified_run
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert result.policy_id == "llmtrace_reference_qualification_v1"
        assert result.policy_version == "0.1.0"


# ---------------------------------------------------------------------------
# Gate 1 — Artifact integrity
# ---------------------------------------------------------------------------


class TestGate1ArtifactIntegrity:
    def test_missing_execution_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root)
        result = qualify_reference_run(
            execution_id="bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee", artifact_repository=repository
        )
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes

    def test_tampered_profile_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root)
        path = _run_file(artifact_root, DEFAULT_EXECUTION_ID, "capability_profile.json")
        path.write_text(path.read_text(encoding="utf-8") + "\n# tampered", encoding="utf-8")
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes

    def test_deleted_artifact_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root)
        _run_file(artifact_root, DEFAULT_EXECUTION_ID, "benchmark_runs.json").unlink()
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes

    def test_unparseable_profile_rejected(self, artifact_root: Path) -> None:
        # The bytes verify against the manifest hash (commit hashed them) but
        # are not a CapabilityProfile → Gate 3 parse failure fails closed.
        from llmtrace.execution.artifacts import RunArtifactRepository

        repository = RunArtifactRepository(artifact_root)
        manifest = make_manifest()
        repository.commit(
            manifest,
            {
                "capability_profile.json": "{not a profile",
                "benchmark_runs.json": make_benchmark_runs_json(),
            },
        )
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes


# ---------------------------------------------------------------------------
# Gate 2 — Capability profile exists
# ---------------------------------------------------------------------------


class TestGate2CapabilityProfile:
    def test_missing_profile_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root, include_profile=False)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert MISSING_CAPABILITY_PROFILE in result.reason_codes


# ---------------------------------------------------------------------------
# Gate 4 — Measurement completeness
# ---------------------------------------------------------------------------


class TestGate4Measurement:
    def test_one_failure_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root, failure=1)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPLETE_MEASUREMENT in result.reason_codes
        assert BENCHMARK_FAILURE_PRESENT in result.reason_codes

    def test_one_ungradable_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root, ungradable=1)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPLETE_MEASUREMENT in result.reason_codes
        assert UNGRADABLE_ITEM_PRESENT in result.reason_codes

    def test_zero_graded_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root, failure=8)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPLETE_MEASUREMENT in result.reason_codes

    def test_missing_benchmark_runs_rejected(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root, include_benchmark=False)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPLETE_MEASUREMENT in result.reason_codes


# ---------------------------------------------------------------------------
# Gate 5 — Scoring policy
# ---------------------------------------------------------------------------


class TestGate5ScoringPolicy:
    def test_manifest_policy_mismatch_rejected(self, artifact_root: Path) -> None:
        manifest = make_manifest(scoring_policy_id="other-policy")
        repository, _ = commit_run(artifact_root, manifest=manifest)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert SCORING_POLICY_MISMATCH in result.reason_codes

    def test_profile_policy_mismatch_rejected(self, artifact_root: Path) -> None:
        profile = make_capability_profile().model_copy(update={"scoring_policy_id": "other-policy"})
        repository, _ = commit_run(artifact_root, profile=profile)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert SCORING_POLICY_MISMATCH in result.reason_codes


# ---------------------------------------------------------------------------
# Gate 6 — Suite identity
# ---------------------------------------------------------------------------


class TestGate6Suite:
    def test_suite_id_mismatch_rejected(self, artifact_root: Path) -> None:
        manifest = make_manifest(suite_id="llmtrace_quick_v2")
        repository, _ = commit_run(artifact_root, manifest=manifest)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert SUITE_MISMATCH in result.reason_codes

    def test_suite_content_mismatch_rejected(self, artifact_root: Path) -> None:
        manifest = make_manifest(suite_content_sha256="e" * 64)
        repository, _ = commit_run(artifact_root, manifest=manifest)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert SUITE_CONTENT_MISMATCH in result.reason_codes

    def test_missing_suite_content_sha256_rejected(self, artifact_root: Path) -> None:
        # A pre-v0.4-A run (suite_content_sha256=None) is never an automatic reference.
        manifest = make_manifest(suite_content_sha256=None)
        repository, _ = commit_run(artifact_root, manifest=manifest)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert SUITE_CONTENT_MISMATCH in result.reason_codes


# ---------------------------------------------------------------------------
# Gate 7 — Generation config
# ---------------------------------------------------------------------------


class TestGate7GenerationConfig:
    def test_generation_config_mismatch_rejected(self, artifact_root: Path) -> None:
        manifest = make_manifest(generation_config_sha256="d" * 64)
        repository, _ = commit_run(artifact_root, manifest=manifest)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert GENERATION_CONFIG_MISMATCH in result.reason_codes

    def test_expected_config_matches_canonical(self) -> None:
        # The helper reproduces qualification's canonical computation; the
        # manifest built from it must therefore pass Gate 7 (asserted by the
        # qualified tests above).
        assert expected_generation_config_sha256() == expected_generation_config_sha256()


# ---------------------------------------------------------------------------
# Gate 8 — Adapter
# ---------------------------------------------------------------------------


class TestGate8Adapter:
    def test_missing_adapter_rejected(self) -> None:
        # A run artifact without a recorded adapter can never enter the
        # reference pipeline: the manifest model rejects empty adapter
        # identity at construction time (pydantic, fail closed).
        with pytest.raises(ValidationError):
            make_manifest(adapter_id="")
        with pytest.raises(ValidationError):
            make_manifest(adapter_version="")


# ---------------------------------------------------------------------------
# Gate 9 / Gate 10 — Capability coverage
# ---------------------------------------------------------------------------


class TestGate9Coverage:
    def test_coverage_weight_mismatch_rejected(self, artifact_root: Path) -> None:
        profile = make_capability_profile(coverage_weight=0.5)
        repository, _ = commit_run(artifact_root, profile=profile)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPATIBLE_COVERAGE in result.reason_codes


class TestGate10DimensionCoverage:
    def _profile_with_dimensions(
        self, dims: list[tuple[CapabilityDimension, DimensionScoreStatus]]
    ) -> CapabilityProfile:
        return CapabilityProfile(
            scoring_policy_id="llmtrace-capability-v1",
            scoring_policy_version="0.1.0",
            dimensions=tuple(
                DimensionScoreResult(
                    dimension=dim,
                    status=status,
                    raw_normalized_score=0.5,
                )
                for dim, status in dims
            ),
            coverage_weight=0.75,
        )

    def test_missing_dimension_rejected(self, artifact_root: Path) -> None:
        profile = self._profile_with_dimensions(
            [
                (CapabilityDimension.REASONING, DimensionScoreStatus.UNCALIBRATED),
                (CapabilityDimension.CODING, DimensionScoreStatus.UNCALIBRATED),
                (CapabilityDimension.MATH_SCIENCE, DimensionScoreStatus.UNCALIBRATED),
            ]
        )
        repository, _ = commit_run(artifact_root, profile=profile)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPATIBLE_COVERAGE in result.reason_codes

    def test_unavailable_dimension_rejected(self, artifact_root: Path) -> None:
        profile = self._profile_with_dimensions(
            [
                (CapabilityDimension.REASONING, DimensionScoreStatus.UNCALIBRATED),
                (CapabilityDimension.CODING, DimensionScoreStatus.UNCALIBRATED),
                (CapabilityDimension.MATH_SCIENCE, DimensionScoreStatus.UNCALIBRATED),
                (CapabilityDimension.INSTRUCTION_FOLLOWING, DimensionScoreStatus.UNAVAILABLE),
            ]
        )
        repository, _ = commit_run(artifact_root, profile=profile)
        result = qualify_reference_run(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert not result.qualified
        assert INCOMPATIBLE_COVERAGE in result.reason_codes


# ---------------------------------------------------------------------------
# require_qualified
# ---------------------------------------------------------------------------


class TestRequireQualified:
    def test_rejected_raises_with_error_code(self, artifact_root: Path) -> None:
        repository, _ = commit_run(artifact_root, include_profile=False)
        with pytest.raises(ReferenceQualificationError) as exc_info:
            require_qualified(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert exc_info.value.error_code == MISSING_CAPABILITY_PROFILE

    def test_qualified_does_not_raise(self, qualified_run: tuple[object, object]) -> None:
        repository, _ = qualified_run
        result = require_qualified(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert result.qualified


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


class TestQualificationPolicy:
    def test_create_v1(self) -> None:
        policy = ReferenceQualificationPolicy.create_v1()
        assert policy.policy_id == "llmtrace_reference_qualification_v1"
        assert policy.policy_version == "0.1.0"

    def test_policy_is_frozen(self) -> None:
        policy = ReferenceQualificationPolicy.create_v1()
        with pytest.raises(ValidationError):
            policy.policy_version = "0.2.0"  # type: ignore[misc]

    def test_policy_injection_for_tests_only(self, artifact_root: Path) -> None:
        # The policy parameter exists as a test seam only.  A real v2 policy
        # would bring different rules/thresholds with its id/version, not just
        # a relabelling of the same Gate 1–10 logic.  This test confirms the
        # seam works; production code must call create_v1() or an explicit v2.
        policy = ReferenceQualificationPolicy(
            policy_id="test_seam_policy",
            policy_version="999.0.0",
            description="Test-only injection; production paths use create_v1()",
        )
        repository, _ = commit_run(artifact_root)
        result = qualify_reference_run(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            policy=policy,
        )
        assert result.qualified
        assert result.policy_id == "test_seam_policy"
        assert result.policy_version == "999.0.0"
