"""Scoring data models: dimensions, task specs, results, and profiles."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Capability dimensions
# ---------------------------------------------------------------------------


class CapabilityDimension(StrEnum):
    """Capability dimensions for model evaluation.

    Seven long-term dimensions.  v0.3 activates the first four;
    the remaining three are reserved for future expansion without
    changing historical data structures.
    """

    REASONING = "reasoning"
    CODING = "coding"
    MATH_SCIENCE = "math_science"
    INSTRUCTION_FOLLOWING = "instruction_following"
    DATA_ANALYSIS = "data_analysis"
    LONG_CONTEXT = "long_context"
    TOOL_USE = "tool_use"


# Long-term global weights delivered by the policy (Section 六).
_LONG_TERM_WEIGHTS: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: 0.25,
    CapabilityDimension.CODING: 0.20,
    CapabilityDimension.MATH_SCIENCE: 0.15,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 0.15,
    CapabilityDimension.DATA_ANALYSIS: 0.10,
    CapabilityDimension.LONG_CONTEXT: 0.10,
    CapabilityDimension.TOOL_USE: 0.05,
}


def _validate_weights_sum(weights: dict[CapabilityDimension, float]) -> dict[CapabilityDimension, float]:
    """Validate that dimension weights sum to exactly 1.0."""
    total = sum(weights.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise ValueError(f"Global dimension weights must sum to 1.0, got {total}")
    for dim, w in weights.items():
        if not math.isfinite(w):
            raise ValueError(f"Weight for {dim.value} is not finite: {w}")
    return weights


# ---------------------------------------------------------------------------
# Scoring status
# ---------------------------------------------------------------------------


class DimensionScoreStatus(StrEnum):
    """Status of a dimension score.

    - SCORED: sufficient graded tasks with valid calibration.
    - UNCALIBRATED: sufficient graded tasks, but no reference calibration (v0.3 default).
    - INSUFFICIENT_DATA: tasks exist but effective graded coverage below minimum.
    - UNAVAILABLE: no eligible tasks at all.
    """

    SCORED = "scored"
    UNCALIBRATED = "uncalibrated"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Task → dimension mapping
# ---------------------------------------------------------------------------


class TaskScoringSpec(BaseModel):
    """Explicit mapping from a benchmark task to a capability dimension.

    Must NOT use heuristics (e.g. ``"math" in task_id``) for dimension
    assignment — every task MUST have an explicit TaskScoringSpec.

    ``task_weight`` is the intra-dimension weight.  Weights are normalised
    within each dimension at aggregation time.
    """

    task_id: str = Field(..., min_length=1, description="Benchmark task identifier")
    dimension: CapabilityDimension = Field(..., description="Capability dimension this task belongs to")
    task_weight: float = Field(..., gt=0.0, description="Intra-dimension weight (> 0)")
    capability_score_eligible: bool = Field(
        default=True,
        description="False for smoke / diagnostic tasks that must not contribute to scores",
    )
    source_id: str = Field(default="", description="Benchmark source identifier")
    suite_id: str = Field(default="", description="Benchmark suite identifier")
    scoring_notes: str = Field(default="", description="Optional scoring notes")

    model_config = {"frozen": True, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Evidence UUID validation
# ---------------------------------------------------------------------------


def _validate_evidence_uuids(refs: tuple[str, ...]) -> tuple[str, ...]:
    """Validate that every entry is a valid UUID and return normalized forms."""
    result: list[str] = []
    for r in refs:
        try:
            result.append(str(UUID(r)))
        except ValueError as exc:
            raise ValueError(f"Invalid evidence UUID: '{r}'") from exc
    return tuple(result)


# ---------------------------------------------------------------------------
# Per-dimension score result
# ---------------------------------------------------------------------------


class DimensionScoreResult(BaseModel):
    """Aggregate score for a single capability dimension."""

    dimension: CapabilityDimension = Field(..., description="Capability dimension")
    status: DimensionScoreStatus = Field(..., description="Scoring status for this dimension")
    raw_normalized_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted aggregate of eligible normalized_scores (0.0–1.0)",
    )
    calibrated_score: float | None = Field(
        default=None,
        description="Calibrated 0–100 score. None until Reference Calibration is available.",
    )
    task_count: int = Field(default=0, ge=0, description="Total eligible tasks in this dimension")
    graded_task_count: int = Field(default=0, ge=0, description="Eligible tasks that were successfully graded")
    task_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Coverage ratio: graded_task_weight / planned_task_weight",
    )
    global_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="This dimension's share of the full 1.0 global weight",
    )
    weighted_contribution: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="raw_normalized_score × global_weight",
    )
    evidence_refs: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evidence UUIDs from contributing GradeResults and TaskAttempts",
    )
    source_task_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Task IDs that contributed to this dimension score",
    )
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Non-fatal warnings")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("calibrated_score")
    @classmethod
    def _validate_calibrated_score_range(cls, v: float | None) -> float | None:
        """calibrated_score must be None or in [0, 100]."""
        if v is not None and (v < 0.0 or v > 100.0):
            raise ValueError(f"calibrated_score must be in [0, 100], got {v}")
        return v


# ---------------------------------------------------------------------------
# Capability profile (top-level aggregate)
# ---------------------------------------------------------------------------


class CapabilityProfile(BaseModel):
    """Immutable capability profile for an evaluated model.

    Contains per-dimension scores, evidence references, and a provisional
    raw index.  calibrated_total_score remains None until Reference
    Calibration is implemented.
    """

    profile_version: str = Field(
        default="0.1.0",
        min_length=1,
        description="CapabilityProfile schema version",
    )
    scoring_policy_id: str = Field(..., min_length=1, description="Scoring policy identifier")
    scoring_policy_version: str = Field(..., min_length=1, description="Scoring policy version")
    dimensions: tuple[DimensionScoreResult, ...] = Field(
        default_factory=tuple,
        description="Per-dimension score results",
    )
    coverage_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sum of global weights for dimensions that have valid results",
    )
    calibrated_total_score: float | None = Field(
        default=None,
        description="Calibrated 0–100 total score. None until Reference Calibration is available.",
    )
    provisional_raw_index: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Sum of (raw_normalized_score × global_weight) across evaluated dimensions. "
            "NOT divided by coverage_weight.  This is a raw weighted sum, not a final "
            "0–100 capability score."
        ),
    )
    evidence_refs: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Union of all dimension evidence_refs (deduplicated, first-seen order)",
    )
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Non-fatal warnings")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("calibrated_total_score")
    @classmethod
    def _validate_calibrated_total_score_range(cls, v: float | None) -> float | None:
        """calibrated_total_score must be None or in [0, 100]."""
        if v is not None and (v < 0.0 or v > 100.0):
            raise ValueError(f"calibrated_total_score must be in [0, 100], got {v}")
        return v

    @field_validator("evidence_refs")
    @classmethod
    def _validate_evidence_uuids(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Validate that evidence_refs entries are valid UUIDs."""
        return _validate_evidence_uuids(v)
