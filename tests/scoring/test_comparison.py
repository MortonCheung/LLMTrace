"""Tests for CapabilityComparator, ComparisonResult, and DimensionDiff."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmtrace.scoring.comparison import (
    COMPARABLE_STATUSES,
    CapabilityComparator,
    ComparisonResult,
    DimensionDiff,
    _comparable_dimension_score_map,
    comparable_dimensions,
)
from llmtrace.scoring.errors import (
    IncompatibleCoverageError,
    ScoringPolicyMismatchError,
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

_TEST_SUITE_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

_REFERENCE_SCORES: dict[CapabilityDimension, float] = {
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

_ALL_DIMENSIONS = tuple(_REFERENCE_SCORES)


def _dim(
    dimension: CapabilityDimension,
    score: float,
    *,
    status: DimensionScoreStatus = DimensionScoreStatus.UNCALIBRATED,
) -> DimensionScoreResult:
    return DimensionScoreResult(
        dimension=dimension,
        status=status,
        raw_normalized_score=score,
    )


def _profile(
    scores: dict[CapabilityDimension, float],
    *,
    coverage_weight: float = 0.75,
    statuses: dict[CapabilityDimension, DimensionScoreStatus] | None = None,
    policy_id: str = "llmtrace-capability-v1",
    policy_version: str = "0.1.0",
) -> CapabilityProfile:
    return CapabilityProfile(
        scoring_policy_id=policy_id,
        scoring_policy_version=policy_version,
        dimensions=tuple(
            _dim(d, s, status=(statuses or {}).get(d, DimensionScoreStatus.UNCALIBRATED)) for d, s in scores.items()
        ),
        coverage_weight=coverage_weight,
    )


def _provenance() -> ReferenceProvenance:
    return ReferenceProvenance(
        source_type="benchmark_run",
        created_by="llmtrace",
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
        suite_sha256=_TEST_SUITE_SHA256,
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
        capability_profile=profile if profile is not None else _profile(_REFERENCE_SCORES),
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


# ===========================================================================
# Comparability rules
# ===========================================================================


class TestComparableStatuses:
    def test_scored_and_uncalibrated_are_comparable(self) -> None:
        assert DimensionScoreStatus.SCORED in COMPARABLE_STATUSES
        assert DimensionScoreStatus.UNCALIBRATED in COMPARABLE_STATUSES

    def test_unavailable_and_insufficient_data_are_not_comparable(self) -> None:
        assert DimensionScoreStatus.UNAVAILABLE not in COMPARABLE_STATUSES
        assert DimensionScoreStatus.INSUFFICIENT_DATA not in COMPARABLE_STATUSES

    def test_score_map_excludes_unmeasured_dimensions(self) -> None:
        profile = _profile(
            _REFERENCE_SCORES,
            statuses={CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.UNAVAILABLE},
        )
        scores = _comparable_dimension_score_map(profile)
        assert CapabilityDimension.MATH_SCIENCE not in scores
        assert len(scores) == 3

    def test_score_map_includes_scored_status(self) -> None:
        profile = _profile(
            _REFERENCE_SCORES,
            statuses={CapabilityDimension.CODING: DimensionScoreStatus.SCORED},
        )
        assert CapabilityDimension.CODING in _comparable_dimension_score_map(profile)

    def test_comparable_dimensions_helper(self) -> None:
        profile = _profile(
            _REFERENCE_SCORES,
            statuses={
                CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.UNAVAILABLE,
                CapabilityDimension.CODING: DimensionScoreStatus.INSUFFICIENT_DATA,
            },
        )
        assert comparable_dimensions(profile) == (
            CapabilityDimension.REASONING,
            CapabilityDimension.INSTRUCTION_FOLLOWING,
        )


# ===========================================================================
# DimensionDiff
# ===========================================================================


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


# ===========================================================================
# ComparisonResult — happy path
# ===========================================================================


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
            assert diff.reference_score == pytest.approx(_REFERENCE_SCORES[dim])

    def test_reference_1_vs_candidate_08_delta(self) -> None:
        """Explicit 1.0 vs 0.8 delta check from the v0.3-C spec."""
        result = _compare()
        by_dim = {d.dimension: d for d in result.dimension_diffs}
        assert by_dim[CapabilityDimension.REASONING].reference_score == pytest.approx(1.0)
        assert by_dim[CapabilityDimension.REASONING].candidate_score == pytest.approx(0.8)
        assert by_dim[CapabilityDimension.REASONING].delta == pytest.approx(-0.2)

    def test_dimension_delta_dict(self) -> None:
        result = _compare()
        dd = result.dimension_delta_dict()
        assert set(dd) == {d.value for d in _CANDIDATE_SCORES}
        assert dd["reasoning"]["delta"] == pytest.approx(-0.2)
        assert dd["reasoning"]["reference"] == pytest.approx(1.0)
        assert dd["reasoning"]["candidate"] == pytest.approx(0.8)

    def test_coverage_diff_is_zero_when_coverage_matches(self) -> None:
        """coverage_diff is informational only — it is not a compatibility substitute."""
        result = _compare()
        assert result.coverage_diff == pytest.approx(0.0)

    def test_scored_status_is_comparable(self) -> None:
        ref = _snapshot(
            profile=_profile(
                _REFERENCE_SCORES,
                statuses=dict.fromkeys(_ALL_DIMENSIONS, DimensionScoreStatus.SCORED),
            )
        )
        cand = _profile(
            _CANDIDATE_SCORES,
            statuses=dict.fromkeys(_ALL_DIMENSIONS, DimensionScoreStatus.SCORED),
        )
        result = _compare(reference=ref, candidate=cand)
        by_dim = {d.dimension: d for d in result.dimension_diffs}
        assert by_dim[CapabilityDimension.MATH_SCIENCE].delta == pytest.approx(-0.3)

    def test_no_identity_conclusion_fields(self) -> None:
        """ComparisonResult must not carry identity/detection conclusions."""
        result = _compare()
        dumped = result.model_dump()
        assert "identity" not in dumped
        assert "detected" not in dumped
        assert "is_model" not in dumped
        assert "similarity" not in dumped
        assert "calibrated" not in dumped


# ===========================================================================
# Gate step 3–4: scoring policy compatibility
# ===========================================================================


class TestScoringPolicyMismatch:
    def test_policy_id_mismatch_rejected(self) -> None:
        cand = _profile(_CANDIDATE_SCORES, policy_id="another-policy")
        with pytest.raises(ScoringPolicyMismatchError) as excinfo:
            _compare(candidate=cand)
        assert excinfo.value.error_code == "SCORING_POLICY_MISMATCH"

    def test_policy_version_mismatch_rejected(self) -> None:
        cand = _profile(_CANDIDATE_SCORES, policy_version="0.2.0")
        with pytest.raises(ScoringPolicyMismatchError) as excinfo:
            _compare(candidate=cand)
        assert excinfo.value.error_code == "SCORING_POLICY_MISMATCH"

    def test_policy_mismatch_is_not_a_suite_mismatch(self) -> None:
        """Suite and scoring policy are different concepts — different error types."""
        cand = _profile(_CANDIDATE_SCORES, policy_id="another-policy")
        with pytest.raises(ScoringPolicyMismatchError) as excinfo:
            _compare(candidate=cand)
        assert not isinstance(excinfo.value, SuiteMismatchError)
        assert not isinstance(excinfo.value, SuiteVersionMismatchError)

    def test_policy_mismatch_rejected_even_with_identical_scores(self) -> None:
        """Same suite, same dimensions, same numbers — still incomparable under a new policy."""
        cand = _profile(_REFERENCE_SCORES, policy_version="9.9.9")
        with pytest.raises(ScoringPolicyMismatchError):
            _compare(candidate=cand)

    def test_policy_mismatch_is_checked_before_coverage(self) -> None:
        """Gate order: policy is evaluated before any coverage/delta work."""
        cand = _profile(
            {CapabilityDimension.REASONING: 0.8},  # also coverage-mismatched
            policy_id="another-policy",
            coverage_weight=0.25,
        )
        with pytest.raises(ScoringPolicyMismatchError):
            _compare(candidate=cand)


# ===========================================================================
# Gate step 5–6: comparable coverage compatibility
# ===========================================================================


class TestCoverageMismatch:
    def test_missing_dimension_raises_incompatible_coverage(self) -> None:
        ref = _snapshot(profile=_profile(_REFERENCE_SCORES))  # 4 dimensions
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
        ref = _snapshot(profile=_profile(_REFERENCE_SCORES))
        candidate_5 = _profile(
            {
                **_CANDIDATE_SCORES,
                CapabilityDimension.DATA_ANALYSIS: 0.5,
            }
        )
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(reference=ref, candidate=candidate_5)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"

    def test_candidate_unavailable_dimension_rejected(self) -> None:
        """Same keys, but candidate has no valid math data — must not become a -1.0 delta."""
        cand = _profile(
            _CANDIDATE_SCORES,
            statuses={CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.UNAVAILABLE},
            coverage_weight=0.60,  # production shape: unavailable dimension drops out
        )
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(candidate=cand)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"

    def test_candidate_unavailable_never_produces_fake_delta(self) -> None:
        """The exact regression: unavailable must not read as 'candidate is much worse'."""
        cand = _profile(
            _CANDIDATE_SCORES,
            statuses={CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.UNAVAILABLE},
        )
        with pytest.raises(IncompatibleCoverageError):
            result = _compare(candidate=cand)
            # if a result had been produced, math would have shown a bogus -1.0
            raise AssertionError(f"comparison should have failed closed, got {result}")

    def test_candidate_insufficient_data_rejected(self) -> None:
        cand = _profile(
            _CANDIDATE_SCORES,
            statuses={CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.INSUFFICIENT_DATA},
            coverage_weight=0.60,
        )
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(candidate=cand)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"

    def test_reference_unavailable_dimension_rejected(self) -> None:
        """Symmetric case: reference invalid, candidate valid → also rejected."""
        ref = _snapshot(
            profile=_profile(
                _REFERENCE_SCORES,
                statuses={CapabilityDimension.CODING: DimensionScoreStatus.UNAVAILABLE},
                coverage_weight=0.55,
            )
        )
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(reference=ref)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"

    def test_reference_insufficient_data_rejected(self) -> None:
        ref = _snapshot(
            profile=_profile(
                _REFERENCE_SCORES,
                statuses={CapabilityDimension.CODING: DimensionScoreStatus.INSUFFICIENT_DATA},
                coverage_weight=0.55,
            )
        )
        with pytest.raises(IncompatibleCoverageError):
            _compare(reference=ref)

    def test_coverage_weight_mismatch_rejected(self) -> None:
        """Comparable keys can match while coverage still differs — gate must catch it."""
        cand = _profile(_CANDIDATE_SCORES, coverage_weight=0.65)
        with pytest.raises(IncompatibleCoverageError) as excinfo:
            _compare(candidate=cand)
        assert excinfo.value.error_code == "INCOMPATIBLE_COVERAGE"
        assert "coverage_weight" in str(excinfo.value)

    def test_matching_coverage_both_sides_unavailable_is_comparable(self) -> None:
        """Both sides missing the same dimension is symmetric — deltas stay meaningful."""
        statuses = {CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.UNAVAILABLE}
        ref = _snapshot(profile=_profile(_REFERENCE_SCORES, statuses=statuses, coverage_weight=0.60))
        cand = _profile(_CANDIDATE_SCORES, statuses=statuses, coverage_weight=0.60)
        result = _compare(reference=ref, candidate=cand)
        by_dim = {d.dimension: d for d in result.dimension_diffs}
        assert CapabilityDimension.MATH_SCIENCE not in by_dim
        assert by_dim[CapabilityDimension.REASONING].delta == pytest.approx(-0.2)


# ===========================================================================
# Gate step 1–2: suite compatibility
# ===========================================================================


class TestSuiteMismatch:
    def test_suite_id_mismatch_rejected(self) -> None:
        with pytest.raises(SuiteMismatchError):
            _compare(candidate_suite_id="llmtrace_quick_v2")

    def test_suite_version_mismatch_rejected(self) -> None:
        with pytest.raises(SuiteVersionMismatchError):
            _compare(candidate_suite_version="0.2.0")
