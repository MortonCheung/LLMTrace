"""Capability comparison engine — ReferenceSnapshot vs CapabilityProfile.

Compares a reference snapshot against a candidate capability profile, producing
per-dimension deltas (candidate − reference).  The comparison MUST NOT emit an
identity conclusion ("candidate is GPT-X") — benchmark similarity is not proof
of identity.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from llmtrace.scoring.models import CapabilityDimension, CapabilityProfile

from .errors import (
    IncompatibleCoverageError,
    SuiteMismatchError,
    SuiteVersionMismatchError,
)
from .reference import ReferenceSnapshot

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
        description="candidate.coverage_weight - reference.coverage_weight",
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
# Comparator
# ---------------------------------------------------------------------------


def _dimension_score_map(profile: CapabilityProfile) -> dict[CapabilityDimension, float]:
    """Map each dimension to its raw_normalized_score, preserving profile order."""
    return {d.dimension: d.raw_normalized_score for d in profile.dimensions}


class CapabilityComparator:
    """Compares a ``ReferenceSnapshot`` against a candidate ``CapabilityProfile``.

    Enforces that both profiles were measured on the same suite and suite
    version, and that they cover the same set of dimensions — a silent
    partial comparison is forbidden.
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
            IncompatibleCoverageError: If dimension coverage differs.
        """
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

        reference_scores = _dimension_score_map(reference.capability_profile)
        candidate_scores = _dimension_score_map(candidate)

        if set(reference_scores) != set(candidate_scores):
            ref_names = sorted(d.value for d in reference_scores)
            cand_names = sorted(d.value for d in candidate_scores)
            raise IncompatibleCoverageError(
                f"INCOMPATIBLE_COVERAGE: reference dimensions {ref_names} != candidate dimensions {cand_names}"
            )

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
