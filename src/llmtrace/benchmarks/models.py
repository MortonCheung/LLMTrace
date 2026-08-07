"""Benchmark domain models.

Defines the unified data model for capability evaluation across external
benchmark harnesses (lm-eval, LiveBench, EvalPlus, Inspect AI, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Protocol, runtime_checkable
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)

if TYPE_CHECKING:
    from llmtrace.models.evidence import HTTPEvidence

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

    _PROVENANCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "source_id",
        "source_revision",
        "suite_id",
        "suite_version",
        "adapter_id",
        "adapter_version",
    )

    def provenance_matches(self, other: BenchmarkProvenance) -> bool:
        """Check whether this provenance matches *other* on all fields."""
        return all(getattr(self, f) == getattr(other, f) for f in self._PROVENANCE_FIELDS)


def validate_provenance_consistency(
    parent: BenchmarkProvenance,
    child: BenchmarkProvenance,
    child_label: str,
    parent_label: str = "run_result",
) -> None:
    """Validate that *child* provenance matches *parent*.

    Shared by reporting (benchmark_mapper) and scoring (aggregator) layers.

    Args:
        parent: The enclosing provenance record (e.g. BenchmarkRunResult).
        child: The child record (e.g. TaskAttempt or GradeResult).
        child_label: Label for the child in error messages.
        parent_label: Label for the parent in error messages.

    Raises:
        ValueError: If any provenance field differs between parent and child.
    """
    for field in BenchmarkProvenance._PROVENANCE_FIELDS:
        parent_val = getattr(parent, field)
        child_val = getattr(child, field)
        if child_val != parent_val:
            raise ValueError(
                f"Provenance mismatch on '{field}': {child_label} has '{child_val}', {parent_label} has '{parent_val}'"
            )


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _parse_uuid(value: str) -> str:
    """Parse a string as a UUID and return its normalized string form."""
    return str(UUID(value))


def normalize_evidence_tuple(refs: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate and normalise evidence UUIDs, preserving first-seen order.

    Shared by benchmarks and scoring layers — every model that carries
    evidence references must use this (or the EvidenceRefs annotated type
    for list fields).
    """
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        norm = _parse_uuid(r)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return tuple(out)


def _normalize_evidence_refs(refs: list[str]) -> list[str]:
    """Deduplicate and normalise evidence UUIDs (list variant)."""
    return list(normalize_evidence_tuple(tuple(refs)))


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


# ---------------------------------------------------------------------------
# CompletionOptions — generation kwargs contract for Provider.complete()
# ---------------------------------------------------------------------------

# Recognised generation kwargs from lm-eval; anything else must fail explicitly.
_RECOGNISED_GEN_KWARGS = frozenset({"until", "stop", "temperature", "max_gen_toks", "max_tokens", "do_sample"})


class CompletionOptions(BaseModel):
    """Typed generation options passed to Provider.complete().

    Supports both the lm-eval ``generation_kwargs`` namespace and
    the OpenAI-/Anthropic-compatible key sets:

    ================== ================ ===================
    lm-eval key         OpenAI key       Anthropic key
    ================== ================ ===================
    until / stop         stop             stop_sequences
    temperature          temperature      temperature
    max_gen_toks /       max_tokens       max_tokens
    max_tokens
    do_sample            (ignored)        (ignored)
    ================== ================ ===================
    """

    until: list[str] | None = None
    stop: list[str] | None = None
    temperature: float | None = None
    max_gen_toks: int | None = None
    max_tokens: int | None = None
    do_sample: bool | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def from_lm_eval_kwargs(cls, gen_kwargs: dict[str, object]) -> CompletionOptions:
        """Build from lm-eval generation_kwargs, failing on unknown keys."""
        unknown = set(gen_kwargs.keys()) - _RECOGNISED_GEN_KWARGS
        if unknown:
            raise ValueError(
                f"Unsupported generation kwargs: {sorted(unknown)}. Supported keys: {sorted(_RECOGNISED_GEN_KWARGS)}"
            )

        return cls(
            until=cls._as_str_list(gen_kwargs.get("until")),
            stop=cls._as_str_list(gen_kwargs.get("stop")),
            temperature=cls._as_float_or_none(gen_kwargs.get("temperature")),
            max_gen_toks=cls._as_int_or_none(gen_kwargs.get("max_gen_toks")),
            max_tokens=cls._as_int_or_none(gen_kwargs.get("max_tokens")),
            do_sample=cls._as_bool_or_none(gen_kwargs.get("do_sample")),
        )

    @staticmethod
    def _as_str_list(value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v) for v in value]
        raise TypeError(f"Expected list[str] or None, got {type(value).__name__}")

    @staticmethod
    def _as_float_or_none(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"Expected float or None, got {type(value).__name__}")

    @staticmethod
    def _as_int_or_none(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        raise TypeError(f"Expected int or None, got {type(value).__name__}")

    @staticmethod
    def _as_bool_or_none(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        raise TypeError(f"Expected bool or None, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# CompletionProvider Protocol — contract for any Provider used by the bridge
# ---------------------------------------------------------------------------


@runtime_checkable
class CompletionProvider(Protocol):
    """Protocol that any Provider must satisfy when used with the lm-eval bridge.

    At minimum, the provider must implement an async ``complete()`` method
    that accepts an optional ``CompletionOptions`` and returns an HTTPEvidence.
    """

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: CompletionOptions | None = None,
    ) -> HTTPEvidence:  # noqa: F821  (forward ref via TYPE_CHECKING)
        ...


# ---------------------------------------------------------------------------
# SmokeTaskManifest — fixed identity for the smoke task
# ---------------------------------------------------------------------------


class SmokeTaskManifest(BaseModel):
    """Fixed provenance and identity for the LLMTrace smoke task.

    This manifest is the single source of truth for smoke task identity
    and is shared by RunPlan, TaskAttempt, and GradeResult.
    """

    task_id: NonEmptyStr = Field(default="llmtrace_smoke")
    suite_id: NonEmptyStr = Field(default="llmtrace_smoke")
    suite_version: NonEmptyStr = Field(default="1.0.0")
    source_id: NonEmptyStr = Field(default="lm-eval")
    source_revision: NonEmptyStr = Field(default="0000000-smoke")
    metric: NonEmptyStr = Field(default="exact_match")
    filter: NonEmptyStr = Field(default="none")
    capability_score_eligible: bool = Field(default=False)

    model_config = {"frozen": True, "extra": "forbid"}


# ---------------------------------------------------------------------------
# LmEvalMetricResult — controlled metric schema for metadata
# ---------------------------------------------------------------------------


class LmEvalMetricResult(BaseModel):
    """Controlled, schema-validated metric extracted from an lm-eval run.

    Only a whitelisted set of fields is stored; the full raw per-task
    results dict MUST NOT be placed in TaskAttempt.metadata.
    """

    task_name: NonEmptyStr = Field(..., description="lm-eval task name")
    metric_name: NonEmptyStr = Field(..., description="Metric name, e.g. 'exact_match'")
    filter_name: str = Field(default="none", description="Filter name from lm-eval metric key")
    value: float = Field(..., description="Metric value")
    fewshot: int = Field(default=0, description="Number of few-shot examples")
    lm_eval_version: str = Field(default="unknown", description="lm-eval package version")
    task_revision: str = Field(default="unknown", description="lm-eval task revision")
    generation_options: CompletionOptions | None = Field(
        default=None, description="Generation options used for this run"
    )

    model_config = {"extra": "forbid"}
