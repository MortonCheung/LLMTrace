"""Benchmark domain models.

Defines the unified data model for capability evaluation across external
benchmark harnesses (lm-eval, LiveBench, EvalPlus, Inspect AI, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Status of a single task attempt."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class GradeStatus(StrEnum):
    """Status of a grading result."""

    GRADED = "graded"
    UNGRADABLE = "ungradable"
    ERROR = "error"


class FailureCategory(StrEnum):
    """Category of an adapter / execution failure."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    ADAPTER = "adapter"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Provenance – shared origin tracking
# ---------------------------------------------------------------------------


class BenchmarkProvenance(BaseModel):
    """Provenance fields shared by TaskAttempt, GradeResult, BenchmarkRunResult.

    Captures exactly which source, suite, and adapter were used for a
    particular evaluation artefact.
    """

    source_id: NonEmptyStr = Field(..., description="Benchmark source identifier")
    source_revision: NonEmptyStr = Field(..., description="Source data revision (hash or tag)")
    suite_id: NonEmptyStr = Field(..., description="Suite identifier")
    suite_version: NonEmptyStr = Field(..., description="Suite version string")
    adapter_id: NonEmptyStr = Field(..., description="Adapter identifier")
    adapter_version: NonEmptyStr = Field(..., description="Adapter version string")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _parse_uuid(value: str) -> str:
    """Parse a string as a UUID and return its normalized string form."""
    return str(UUID(value))


def _normalize_evidence_refs(refs: list[str]) -> list[str]:
    """Deduplicate and normalise evidence UUIDs, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        norm = _parse_uuid(r)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# Shared evidence-reference field type: UUID validation + normalisation + dedup
# applied consistently to every model that carries evidence references.
EvidenceRefs = Annotated[list[str], AfterValidator(_normalize_evidence_refs)]


def validate_evidence_refs(
    refs: Sequence[str],
    available_evidence_ids: set[UUID] | frozenset[UUID],
) -> None:
    """Validate that every evidence reference exists in *available_evidence_ids*.

    Args:
        refs: Evidence reference strings (each must be a valid UUID).
        available_evidence_ids: Set of known/valid evidence UUIDs.

    Raises:
        ValueError: If any reference is not a valid UUID or does not exist
                    in the available set.
    """
    for ref in refs:
        try:
            uid = UUID(ref)
        except ValueError as exc:
            raise ValueError(f"Invalid evidence reference: '{ref}' is not a valid UUID") from exc
        if uid not in available_evidence_ids:
            raise ValueError(f"Evidence reference '{ref}' not found in available evidence IDs")


# ---------------------------------------------------------------------------
# Benchmark Source & Suite
# ---------------------------------------------------------------------------


class BenchmarkSource(BaseModel):
    """Identifies a benchmark source (e.g., MMLU, LiveBench, EvalPlus)."""

    source_id: NonEmptyStr = Field(..., description="Unique source identifier, e.g. 'mmlu', 'livebench'")
    name: NonEmptyStr = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Brief description of the benchmark source")
    url: str = Field(default="", description="Reference URL for the benchmark")

    model_config = {"extra": "forbid"}


class SuiteVersion(BaseModel):
    """Immutable version identifier for a benchmark suite."""

    version: NonEmptyStr = Field(..., description="Semantic version string, e.g. '1.0.0'")
    released_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Release timestamp in ISO-8601 (UTC)",
    )
    notes: str = Field(default="", description="Release notes for this version")

    model_config = {"frozen": True, "extra": "forbid"}


class BenchmarkSuite(BaseModel):
    """A benchmark suite consisting of multiple tasks."""

    suite_id: NonEmptyStr = Field(..., description="Unique suite identifier, e.g. 'mmlu'")
    name: NonEmptyStr = Field(..., description="Human-readable suite name")
    version: SuiteVersion = Field(..., description="Suite version")
    source_id: NonEmptyStr = Field(..., description="Source this suite belongs to")
    source_revision: NonEmptyStr = Field(..., description="Revision hash or tag of the source data")
    description: str = Field(default="", description="Suite description")
    tasks: list[TaskSpec] = Field(default_factory=list, description="Tasks included in this suite")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """Specification of a single benchmark task."""

    task_id: NonEmptyStr = Field(..., description="Unique task identifier within the suite")
    name: NonEmptyStr = Field(..., description="Human-readable task name")
    description: str = Field(default="", description="Task description")
    category: str = Field(default="", description="Task category, e.g. 'math', 'reasoning'")
    num_samples: int = Field(default=0, ge=0, description="Number of samples/examples in this task")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Structured failure
# ---------------------------------------------------------------------------


class AdapterFailure(BaseModel):
    """Structured failure information for adapter / execution errors."""

    error_code: NonEmptyStr = Field(..., description="Machine-readable error code")
    category: FailureCategory = Field(default=FailureCategory.UNKNOWN, description="Failure category")
    message: str = Field(..., description="Human-readable error message")
    retryable: bool = Field(default=False, description="Whether this failure can be retried")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional diagnostic details")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Run Plan (generated by Planner)
# ---------------------------------------------------------------------------


class BudgetEstimate(BaseModel):
    """Estimated resource budget for a benchmark run."""

    planned_requests: int = Field(..., ge=0, description="Planned number of API requests")
    maximum_requests: int = Field(..., ge=0, description="Maximum requests including all retries")
    maximum_retries: int = Field(default=0, ge=0, description="Maximum retry attempts per request")
    estimated_input_tokens: int | None = Field(default=None, ge=0, description="Estimated total input tokens")
    estimated_output_tokens: int | None = Field(default=None, ge=0, description="Estimated total output tokens")
    estimated_duration_seconds: float | None = Field(
        default=None, ge=0, description="Estimated wall-clock duration in seconds"
    )
    estimated_cost: float | None = Field(
        default=None, ge=0, description="Estimated cost, or None if pricing unavailable"
    )
    currency: str = Field(default="USD", min_length=1, description="Currency code for cost estimate")
    assumptions: list[str] = Field(default_factory=list, description="Key assumptions behind the estimate")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_consistency(self) -> BudgetEstimate:
        """Validate that maximum_requests == planned_requests * (1 + maximum_retries)."""
        expected = self.planned_requests * (1 + self.maximum_retries)
        if self.maximum_requests != expected:
            raise ValueError(
                f"maximum_requests ({self.maximum_requests}) must equal "
                f"planned_requests ({self.planned_requests}) * "
                f"(1 + maximum_retries ({self.maximum_retries})) = {expected}"
            )
        return self


class RunPlan(BaseModel):
    """Deterministic execution plan generated from a suite and task spec.

    plan_id is a deterministic SHA-256 digest of the canonicalised inputs
    so that identical inputs always produce the same plan_id.
    """

    plan_id: NonEmptyStr = Field(
        ...,
        description="Deterministic SHA-256 plan identifier (set by planner)",
    )
    suite_id: NonEmptyStr = Field(..., description="Suite this plan targets")
    suite_version: NonEmptyStr = Field(..., description="Suite version this plan targets")
    source_id: NonEmptyStr = Field(..., description="Source this plan targets")
    source_revision: NonEmptyStr = Field(..., description="Source revision this plan targets")
    adapter_id: NonEmptyStr = Field(..., description="Adapter to use for execution")
    adapter_version: NonEmptyStr = Field(..., description="Adapter version")
    task_ids: list[NonEmptyStr] = Field(..., description="Task IDs to execute (deterministic order)")
    total_samples: int = Field(default=0, ge=0, description="Total number of samples across all tasks")
    budget: BudgetEstimate = Field(..., description="Resource budget estimate")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional plan metadata")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Task Attempt (single execution attempt for one task sample)
# ---------------------------------------------------------------------------


class TaskAttempt(BenchmarkProvenance):
    """Records one attempt at a benchmark task, referencing evidence."""

    attempt_id: NonEmptyStr = Field(..., description="Unique attempt identifier")
    task_id: NonEmptyStr = Field(..., description="Task identifier")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Attempt status")
    evidence_refs: EvidenceRefs = Field(
        default_factory=list,
        description="Evidence UUIDs (as strings) collected during this attempt",
    )
    failure: AdapterFailure | None = Field(default=None, description="Structured failure information on failure")
    started_at: datetime | None = Field(default=None, description="Attempt start time")
    finished_at: datetime | None = Field(default=None, description="Attempt finish time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional attempt metadata")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_failure_consistency(self) -> TaskAttempt:
        """failure must be present IFF status is FAILURE."""
        if self.status == TaskStatus.FAILURE:
            if self.failure is None:
                raise ValueError("failure must be set when status is FAILURE")
        elif self.failure is not None:
            raise ValueError(f"failure must be None when status is {self.status.value}")
        return self


# ---------------------------------------------------------------------------
# Grading Results
# ---------------------------------------------------------------------------


class GradeResult(BenchmarkProvenance):
    """Grading result for a single task attempt."""

    grade_id: NonEmptyStr = Field(..., description="Unique grade identifier")
    attempt_id: NonEmptyStr = Field(..., description="Reference to the TaskAttempt")
    task_id: NonEmptyStr = Field(..., description="Task identifier")
    grader_id: NonEmptyStr = Field(..., description="Grader identifier")
    status: GradeStatus = Field(default=GradeStatus.GRADED, description="Grading status")
    raw_score: float = Field(..., description="Raw score from the grader")
    normalized_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized score in [0, 1]",
    )
    evidence_refs: EvidenceRefs = Field(
        default_factory=list,
        description="Evidence UUIDs referenced for grading",
    )
    error_message: str | None = Field(default=None, description="Error message if grading failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional grading metadata")

    model_config = {"extra": "forbid"}


class DimensionResult(BaseModel):
    """Result for one evaluation dimension within a task.

    Dimensions are always nested under a parent GradeResult or
    BenchmarkRunResult; the parent provides the full provenance.
    """

    dimension_id: NonEmptyStr = Field(..., description="Dimension identifier, e.g. 'accuracy', 'f1'")
    name: NonEmptyStr = Field(..., description="Human-readable dimension name")
    value: float = Field(..., description="Dimension value")
    normalized_value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized value in [0, 1]",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Aggregate Run Result
# ---------------------------------------------------------------------------


class BenchmarkRunResult(BenchmarkProvenance):
    """Aggregate result from a complete benchmark run."""

    run_id: NonEmptyStr = Field(..., description="Unique run identifier")
    task_attempts: list[TaskAttempt] = Field(default_factory=list, description="All task attempts")
    grade_results: list[GradeResult] = Field(default_factory=list, description="All grading results")
    dimensions: list[DimensionResult] = Field(default_factory=list, description="Per-dimension aggregate results")
    started_at: datetime | None = Field(default=None, description="Run start time")
    finished_at: datetime | None = Field(default=None, description="Run finish time")
    evidence_refs: EvidenceRefs = Field(
        default_factory=list,
        description="All evidence UUIDs referenced across this run",
    )
    error_count: int = Field(default=0, ge=0, description="Number of failed task attempts")
    skip_count: int = Field(default=0, ge=0, description="Number of skipped tasks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional run metadata")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _sync_counts(self) -> BenchmarkRunResult:
        self.error_count = sum(1 for a in self.task_attempts if a.status == TaskStatus.FAILURE)
        self.skip_count = sum(1 for a in self.task_attempts if a.status == TaskStatus.SKIPPED)
        return self
