"""Benchmark report models — JSON-safe data layer for benchmark run reports.

These models map benchmark execution artefacts (RunPlan, TaskAttempt,
GradeResult, BenchmarkRunResult) into a structured, serializable report
section.  They are intentionally shallow: no direct references to
Provider objects, lm-eval internals, exception objects, or Pydantic
parents outside this module.

This is the *first layer* of the benchmark report pipeline:
  - data models (this module)
  - mapper (benchmark_mapper.py)
  - JSON serialization (future)
  - HTML sections (future)

No total_score or capability_score is computed here — that work belongs
in a later aggregation layer.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# BenchmarkRunSummary — top-level numeric summary
# ---------------------------------------------------------------------------


class BenchmarkRunSummary(BaseModel):
    """Numeric summary of a single benchmark run section.

    All unknown / unavailable values must be None, never 0.
    """

    planned_requests: int = Field(..., ge=0, description="Planned request count from RunPlan.budget")
    maximum_requests: int = Field(..., ge=0, description="Maximum request count including retries")
    actual_requests: int = Field(..., ge=0, description="Actual request count (from Evidence counting)")
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    skip_count: int = Field(default=0, ge=0)
    ungraded_count: int = Field(default=0, ge=0, description="SUCCESS attempts with no matching GradeResult")
    ungradable_count: int = Field(default=0, ge=0, description="GradeResult.status == UNGRADABLE")
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    estimated_output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# FailureReportItem — structured, safe failure representation
# ---------------------------------------------------------------------------


class FailureReportItem(BaseModel):
    """A JSON-safe representation of an AdapterFailure.

    No exception objects, raw dicts, or foreign Pydantic objects.
    """

    error_code: str = Field(..., description="Machine-readable error code")
    category: str = Field(..., description="Failure category enum value as string")
    message: str = Field(..., description="Human-readable error message")
    retryable: bool = Field(default=False)
    safe_details: dict[str, str] = Field(
        default_factory=dict,
        description="Diagnostic details limited to str→str (no objects or nested dicts)",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# TaskReportItem — per-task row in the report
# ---------------------------------------------------------------------------


class TaskReportItem(BaseModel):
    """Report view of a single TaskAttempt + optional GradeResult.

    One TaskReportItem per TaskAttempt.  GradeResult is merged by
    attempt_id in the mapper.  smoke tasks are excluded from capability
    scoring.
    """

    task_id: str = Field(..., description="Unique task identifier")
    attempt_id: str = Field(..., description="Matching TaskAttempt.attempt_id")
    status: str = Field(..., description="TaskAttempt.status as string")
    grader_id: str | None = Field(default=None, description="Grader identifier from GradeResult")
    grade_status: str | None = Field(default=None, description="GradeStatus as string")
    raw_score: float | None = Field(default=None, description="None when ungraded or failed")
    normalized_score: float | None = Field(default=None, description="None when ungraded or failed")
    evidence_refs: list[str] = Field(default_factory=list, description="Evidence UUIDs as strings")
    failure: FailureReportItem | None = Field(default=None, description="Present only on FAILURE")
    capability_score_eligible: bool = Field(
        default=True,
        description="False for smoke/internal tasks; true for real benchmarks",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Safe metadata dict (no Pydantic objects, exceptions, or external objects)",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# BenchmarkReportSection — top-level report section for one benchmark run
# ---------------------------------------------------------------------------


class BenchmarkReportSection(BaseModel):
    """Report section representing exactly one BenchmarkRunResult.

    This is a pure data container.  No HTML, CLI, or aggregation logic
    lives here.  JSON output is ISO-8601 for datetimes and string for
    UUID references.
    """

    run_id: UUID = Field(..., description="BenchmarkRunResult.run_id as UUID")
    plan_id: str = Field(..., description="RunPlan.plan_id")
    suite_id: str = Field(..., description="Suite identifier from provenance")
    suite_version: str = Field(..., description="Suite version from provenance")
    source_id: str = Field(..., description="Source identifier from provenance")
    source_revision: str = Field(..., description="Source revision from provenance")
    adapter_id: str = Field(..., description="Adapter identifier from provenance")
    adapter_version: str = Field(..., description="Adapter version from provenance")
    status: str = Field(..., description="Overall run status: 'success', 'partial_failure', 'failure'")
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    planned_requests: int = Field(..., ge=0)
    maximum_requests: int = Field(..., ge=0)
    actual_requests: int = Field(..., ge=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    estimated_output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    tasks: list[TaskReportItem] = Field(default_factory=list)
    summary: BenchmarkRunSummary = Field(...)
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
