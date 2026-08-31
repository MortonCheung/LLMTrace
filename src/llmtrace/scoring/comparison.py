"""Capability comparison engine — ReferenceSnapshot vs CapabilityProfile.

Compares a reference snapshot against a candidate capability profile, producing
per-dimension deltas (candidate − reference).  The comparison MUST NOT emit an
identity conclusion ("candidate is GPT-X") — benchmark similarity is not proof
of identity.

Compatibility gate
------------------
Every comparison runs through a fail-closed gate *before* any delta is
computed, in this exact order:

    1. suite_id
    2. suite_version
    3. scoring_policy_id
    4. scoring_policy_version
    5. comparable dimension coverage (by DimensionScoreStatus)
    6. coverage_weight
    7. dimension delta

Anything that fails the gate raises instead of producing a
``ComparisonResult``.  Partial coverage is never dressed up as a capability
drop, and profiles measured under different scoring policies are never treated
as directly comparable.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreStatus,
)

from .errors import (
    IncompatibleCoverageError,
    ScoringPolicyMismatchError,
    SuiteMismatchError,
    SuiteVersionMismatchError,
)
from .reference import ReferenceSnapshot

# ---------------------------------------------------------------------------
# Comparability rules
# ---------------------------------------------------------------------------

#: Statuses that carry a real measurement and may therefore be compared.
#: Everything else means "we do not have a valid number for this dimension".
COMPARABLE_STATUSES: frozenset[DimensionScoreStatus] = frozenset(
    {
        DimensionScoreStatus.SCORED,
        DimensionScoreStatus.UNCALIBRATED,
    }
)

#: Tolerance for treating two coverage_weight values as the same coverage.
_COVERAGE_ABS_TOL = 1e-9
_COVERAGE_REL_TOL = 1e-9


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class DimensionDiff(BaseModel):
    """Per-dimension difference between a reference and a candidate score."""

    dimension: CapabilityDimension = Field(..., description="Capability dimension")
    reference_score: float = Field(..., ge=0.0, le=1.0, description="Reference raw_normalized_score")
    candidate_score: float = Field(..., ge=0.0, le=1.0, description="Candidate raw_normalized_score")
    delta: float = Field(..., ge=-1.0, le=1.0, description="candidate_score - reference_score")

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _check_delta_consistency(self) -> DimensionDiff:
        """delta must equal candidate_score - reference_score."""
        expected = self.candidate_score - self.reference_score
        if not math.isclose(self.delta, expected, abs_tol=1e-9):
            raise ValueError(
                f"delta {self.delta} must equal candidate_score - reference_score "
                f"({self.candidate_score} - {self.reference_score} = {expected})"
            )
        return self


class ComparisonResult(BaseModel):
    """Result of comparing a reference snapshot against a candidate profile.

    Contains only relative statements (higher/lower per dimension); it carries
    no identity conclusion and no overall 0–100 score.
    """

    model_a: str = Field(..., min_length=1, description="Reference model label")
    model_b: str = Field(..., min_length=1, description="Candidate label")
    reference_snapshot_id: str = Field(..., min_length=1, description="Reference snapshot identifier")
    suite_id: str = Field(..., min_length=1, description="Suite both profiles were measured on")
    suite_version: str = Field(..., min_length=1, description="Suite version both profiles were measured on")
    dimension_diffs: tuple[DimensionDiff, ...] = Field(default_factory=tuple, description="Per-dimension deltas")
    coverage_diff: float = Field(
        default=0.0,
        description="candidate.coverage_weight - reference.coverage_weight (≈0 after the gate passes)",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    def dimension_delta_dict(self) -> dict[str, dict[str, float]]:
        """Return ``{dimension: {reference, candidate, delta}}`` for reporting."""
        return {
            d.dimension.value: {
                "reference": d.reference_score,
                "candidate": d.candidate_score,
                "delta": d.delta,
            }
            for d in self.dimension_diffs
        }


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def comparable_dimensions(profile: CapabilityProfile) -> tuple[CapabilityDimension, ...]:
    """Return the dimensions of *profile* that carry a real measurement.

    A ``DimensionScoreResult`` exists for every enabled dimension even when
    nothing could be measured, so presence alone says nothing about coverage.
    Only SCORED / UNCALIBRATED count; UNAVAILABLE and INSUFFICIENT_DATA mean
    "no valid number" and must never be turned into a delta.
    """
    return tuple(d.dimension for d in profile.dimensions if d.status in COMPARABLE_STATUSES)


def _comparable_dimension_score_map(profile: CapabilityProfile) -> dict[CapabilityDimension, float]:
    """Map each *comparable* dimension to its raw_normalized_score, preserving profile order."""
    return {d.dimension: d.raw_normalized_score for d in profile.dimensions if d.status in COMPARABLE_STATUSES}


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


class CapabilityComparator:
    """Compares a ``ReferenceSnapshot`` against a candidate ``CapabilityProfile``.

    Enforces that both profiles were measured on the same suite, suite version
    and scoring policy, and that they cover the same set of *comparable*
    dimensions with the same coverage weight — a silent partial comparison is
    forbidden.
    """

    def compare(
        self,
        reference: ReferenceSnapshot,
        candidate: CapabilityProfile,
        *,
        candidate_suite_id: str,
        candidate_suite_version: str,
        candidate_model_id: str = "candidate",
    ) -> ComparisonResult:
        """Compare *reference* with *candidate* and return a ``ComparisonResult``.

        Args:
            reference: The reference snapshot to compare against.
            candidate: The candidate capability profile to compare.
            candidate_suite_id: Suite the candidate was measured on.
            candidate_suite_version: Suite version the candidate was measured on.
            candidate_model_id: Label for the candidate (defaults to ``candidate``).

        Raises:
            SuiteMismatchError: If suite ids differ.
            SuiteVersionMismatchError: If suite versions differ.
            ScoringPolicyMismatchError: If scoring policy id or version differs.
            IncompatibleCoverageError: If comparable dimension coverage or
                coverage_weight differs.
        """
        self._check_suite(reference, candidate_suite_id, candidate_suite_version)
        self._check_scoring_policy(reference, candidate)

        reference_scores = _comparable_dimension_score_map(reference.capability_profile)
        candidate_scores = _comparable_dimension_score_map(candidate)

        self._check_coverage(reference, candidate, reference_scores, candidate_scores)

        diffs = tuple(
            DimensionDiff(
                dimension=dim,
                reference_score=reference_scores[dim],
                candidate_score=candidate_scores[dim],
                delta=candidate_scores[dim] - reference_scores[dim],
            )
            for dim in reference_scores
        )

        coverage_diff = candidate.coverage_weight - reference.capability_profile.coverage_weight

        return ComparisonResult(
            model_a=reference.model_id,
            model_b=candidate_model_id,
            reference_snapshot_id=reference.snapshot_id,
            suite_id=reference.suite_id,
            suite_version=reference.suite_version,
            dimension_diffs=diffs,
            coverage_diff=coverage_diff,
        )

    # -- Gate steps --------------------------------------------------------

    @staticmethod
    def _check_suite(
        reference: ReferenceSnapshot,
        candidate_suite_id: str,
        candidate_suite_version: str,
    ) -> None:
        """Gate steps 1–2: suite identity must match exactly."""
        if candidate_suite_id != reference.suite_id:
            raise SuiteMismatchError(
                f"Suite mismatch: reference suite_id '{reference.suite_id}' != candidate suite_id "
                f"'{candidate_suite_id}'"
            )
        if candidate_suite_version != reference.suite_version:
            raise SuiteVersionMismatchError(
                f"Suite version mismatch: reference '{reference.suite_version}' != candidate "
                f"'{candidate_suite_version}'"
            )

    @staticmethod
    def _check_scoring_policy(reference: ReferenceSnapshot, candidate: CapabilityProfile) -> None:
        """Gate steps 3–4: the same dimension score must mean the same thing.

        Two profiles can share a suite and still be incomparable: if the
        scoring policy (or its version) differs, the per-dimension numbers were
        produced by different rules and subtracting them is meaningless.
        """
        reference_policy_id = reference.capability_profile.scoring_policy_id
        reference_policy_version = reference.capability_profile.scoring_policy_version

        if candidate.scoring_policy_id != reference_policy_id:
            raise ScoringPolicyMismatchError(
                f"Scoring policy mismatch: reference scoring_policy_id '{reference_policy_id}' != "
                f"candidate '{candidate.scoring_policy_id}'"
            )
        if candidate.scoring_policy_version != reference_policy_version:
            raise ScoringPolicyMismatchError(
                f"Scoring policy version mismatch: reference '{reference_policy_version}' != "
                f"candidate '{candidate.scoring_policy_version}'"
            )

    @staticmethod
    def _check_coverage(
        reference: ReferenceSnapshot,
        candidate: CapabilityProfile,
        reference_scores: dict[CapabilityDimension, float],
        candidate_scores: dict[CapabilityDimension, float],
    ) -> None:
        """Gate steps 5–6: comparable dimensions and coverage weight must match.

        A dimension that is UNAVAILABLE or INSUFFICIENT_DATA is *not* covered.
        Treating it as a zero score would report "no data" as "much worse",
        which is exactly the failure this gate exists to prevent.
        """
        if set(reference_scores) != set(candidate_scores):
            ref_names = sorted(d.value for d in reference_scores)
            cand_names = sorted(d.value for d in candidate_scores)
            raise IncompatibleCoverageError(
                f"INCOMPATIBLE_COVERAGE: comparable reference dimensions {ref_names} != "
                f"comparable candidate dimensions {cand_names}"
            )

        reference_coverage_weight = reference.capability_profile.coverage_weight
        if not math.isclose(
            candidate.coverage_weight,
            reference_coverage_weight,
            rel_tol=_COVERAGE_REL_TOL,
            abs_tol=_COVERAGE_ABS_TOL,
        ):
            raise IncompatibleCoverageError(
                f"INCOMPATIBLE_COVERAGE: coverage_weight reference {reference_coverage_weight} != "
                f"candidate {candidate.coverage_weight}"
            )
