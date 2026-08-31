"""Tests for CapabilityComparator, ComparisonResult, and DimensionDiff."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmtrace.scoring.comparison import (
    CapabilityComparator,
    ComparisonResult,
    DimensionDiff,
)
from llmtrace.scoring.errors import (
    IncompatibleCoverageError,
    SuiteMismatchError,
    SuiteVersionMismatchError,
)
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.reference import (
    ReferenceProvenance,
    ReferenceSnapshot,
)

_FULL_SCORES: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: 1.0,
    CapabilityDimension.CODING: 1.0,
    CapabilityDimension.MATH_SCIENCE: 1.0,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 1.0,
}

_CANDIDATE_SCORES: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: 0.8,
    CapabilityDimension.CODING: 0.9,
    CapabilityDimension.MATH_SCIENCE: 0.7,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 1.0,
}

_EXPECTED_DELTAS: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: -0.2,
    CapabilityDimension.CODING: -0.1,
    CapabilityDimension.MATH_SCIENCE: -0.3,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 0.0,
}


def _dim(dimension: CapabilityDimension, score: float) -> DimensionScoreResult:
    return DimensionScoreResult(
        dimension=dimension,
        status=DimensionScoreStatus.UNCALIBRATED,
        raw_normalized_score=score,
    )


def _profile(
    scores: dict[CapabilityDimension, float],
    *,
    coverage_weight: float = 0.75,
) -> CapabilityProfile:
    return CapabilityProfile(
        scoring_policy_id="llmtrace-capability-v1",
        scoring_policy_version="0.1.0",
        dimensions=tuple(_dim(d, s) for d, s in scores.items()),
        coverage_weight=coverage_weight,
    )


def _provenance() -> ReferenceProvenance:
    return ReferenceProvenance(
        source_type="benchmark_run",
        created_by="llmtrace",
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
        suite_sha256="abc123",
        benchmark_revision="quick-v1-rev",
        runner_version="0.3.0",
    )


def _snapshot(
    *,
    suite_id: str = "llmtrace_quick_v1",
    suite_version: str = "0.1.0",
    profile: CapabilityProfile | None = None,
) -> ReferenceSnapshot:
    return ReferenceSnapshot(
        snapshot_id="openai-gpt-x-quick-v1",
        model_id="gpt-x",
        provider_id="openai",
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
        suite_id=suite_id,
        suite_version=suite_version,
        capability_profile=profile if profile is not None else _profile(_FULL_SCORES),
        provenance=_provenance(),
    )


def _compare(
    reference: ReferenceSnapshot | None = None,
    candidate: CapabilityProfile | None = None,
    *,
    candidate_suite_id: str = "llmtrace_quick_v1",
    candidate_suite_version: str = "0.1.0",
) -> ComparisonResult:
    ref = reference if reference is not None else _snapshot()
    cand = candidate if candidate is not None else _profile(_CANDIDATE_SCORES)
    return CapabilityComparator().compare(
        ref,
        cand,
        candidate_suite_id=candidate_suite_id,
        candidate_suite_version=candidate_suite_version,
        candidate_model_id="candidate-api",
    )


class TestDimensionDiff:
    def test_delta_equals_candidate_minus_reference(self) -> None:
        diff = DimensionDiff(
            dimension=CapabilityDimension.REASONING,
            reference_score=1.0,
            candidate_score=0.8,
            delta=-0.2,
        )
        assert diff.delta == pytest.approx(-0.2)

    def test_inconsistent_delta_rejected(self) -> None:
        with pytest.raises(ValueError):
            DimensionDiff(
                dimension=CapabilityDimension.REASONING,
                reference_score=1.0,
                candidate_score=0.8,
                delta=0.5,  # wrong sign
            )

    def test_diff_is_frozen(self) -> None:
        diff = DimensionDiff(
            dimension=CapabilityDimension.REASONING,
            reference_score=1.0,
            candidate_score=0.8,
            delta=-0.2,
        )
        with pytest.raises((TypeError, ValueError)):
            diff.delta = 0.0  # type: ignore[misc]


class TestComparisonResult:
    def test_comparison_deltas(self) -> None:
        result = _compare()
        assert result.model_a == "gpt-x"
        assert result.model_b == "candidate-api"
        assert result.reference_snapshot_id == "openai-gpt-x-quick-v1"
        assert result.suite_id == "llmtrace_quick_v1"
        assert result.suite_version == "0.1.0"

        by_dim = {d.dimension: d for d in result.dimension_diffs}
        assert set(by_dim) == set(_EXPECTED_DELTAS)
        for dim, expected_delta in _EXPECTED_DELTAS.items():
            diff = by_dim[dim]
            assert diff.delta == pytest.approx(expected_delta)
            assert diff.candidate_score == pytest.approx(_CANDIDATE_SCORES[dim])
            assert diff.reference_score == pytest.approx(_FULL_SCORES[dim])

    def test_dimension_delta_dict(self) -> None:
        result = _compare()
        dd = result.dimension_delta_dict()
        assert set(dd) == {d.value for d in _CANDIDATE_SCORES}
        assert dd["reasoning"]["delta"] == pytest.approx(-0.2)
        assert dd["reasoning"]["reference"] == pytest.approx(1.0)
        assert dd["reasoning"]["candidate"] == pytest.approx(0.8)

    def test_coverage_diff_computed(self) -> None:
        ref = _snapshot(profile=_profile(_FULL_SCORES, coverage_weight=0.75))
        cand = _profile(_CANDIDATE_SCORES, coverage_weight=0.65)
        result = _compare(reference=ref, candidate=cand)
        assert result.coverage_diff == pytest.approx(-0.10)

    def test_no_identity_conclusion_fields(self) -> None:
        """ComparisonResult must not carry identity/detection conclusions."""
        result = _compare()
        assert "identity" not in result.model_dump()
        assert "detected" not in result.model_dump()
        assert "is_model" not in result.model_dump()


class TestCoverageMismatch:
    def test_missing_dimension_raises_incompatible_coverage(self) -> None:
        ref = _snapshot(profile=_profile(_FULL_SCORES))  # 4 dimensions
        candidate_3 = _profile(
            {
                CapabilityDimension.REASONING: 0.8,
                CapabilityDimension.CODING: 0.9,
                CapabilityDimension.MATH_SCIENCE: 0.7,
            }
        )  # 3 dimensions
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(reference=ref, candidate=candidate_3)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"
        assert "INCOMPATIBLE_COVERAGE" in str(excinfo.value)

    def test_extra_dimension_raises_incompatible_coverage(self) -> None:
        ref = _snapshot(profile=_profile(_FULL_SCORES))
        candidate_5 = _profile(
            {
                **_CANDIDATE_SCORES,
                CapabilityDimension.DATA_ANALYSIS: 0.5,
            }
        )
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(reference=ref, candidate=candidate_5)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"


class TestSuiteMismatch:
    def test_suite_id_mismatch_rejected(self) -> None:
        with pytest.raises(SuiteMismatchError):
            _compare(candidate_suite_id="llmtrace_quick_v2")

    def test_suite_version_mismatch_rejected(self) -> None:
        with pytest.raises(SuiteVersionMismatchError):
            _compare(candidate_suite_version="0.2.0")
