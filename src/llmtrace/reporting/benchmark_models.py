"""Benchmark report models — JSON-safe data layer for benchmark run reports.

These models map benchmark execution artefacts (RunPlan, TaskAttempt,
GradeResult, BenchmarkRunResult) into a structured, serializable report
section.  They are intentionally shallow: no direct references to
Provider objects, lm-eval internals, exception objects, or Pydantic
parents outside this module.

This is the *first layer* of the benchmark report pipeline:
  - data models (this module)
  - mapper (benchmark_mapper.py)
  - JSON serialization (json_report.py)
  - HTML sections (future)

No total_score or capability_score is computed here — that work belongs
in a later aggregation layer.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Report-specific enums
# ---------------------------------------------------------------------------


class BenchmarkReportStatus(StrEnum):
    """Overall status of a benchmark run section."""

    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    FAILURE = "failure"
    INCOMPLETE = "incomplete"
    SKIPPED = "skipped"


class TaskReportStatus(StrEnum):
    """Status of a single task in a report."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ReportGradeStatus(StrEnum):
    """Grading status in report output."""

    GRADED = "graded"
    UNGRADABLE = "ungradable"
    ERROR = "error"


class ReportFailureCategory(StrEnum):
    """Failure category in report output."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    ADAPTER = "adapter"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# JSON-safe value type
# ---------------------------------------------------------------------------

# Allowed types for recursive metadata sanitization (used at runtime, not in Pydantic schema)
JsonSafeScalar = bool | int | float | str | None


# ---------------------------------------------------------------------------
# BenchmarkRunSummary — top-level numeric summary
# ---------------------------------------------------------------------------


class BenchmarkRunSummary(BaseModel):
    """Numeric summary of a single benchmark run section.

    All unknown / unavailable values must be None, never forged as 0.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    planned_requests: int = Field(..., ge=0)
    maximum_requests: int = Field(..., ge=0)
    actual_requests: int = Field(..., ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    skip_count: int = Field(default=0, ge=0)
    ungraded_count: int = Field(default=0, ge=0, description="SUCCESS attempts with no matching GradeResult")
    ungradable_count: int = Field(default=0, ge=0, description="GradeResult.status == UNGRADABLE")
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    estimated_output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FailureReportItem — structured, safe failure representation
# ---------------------------------------------------------------------------


# Sensitive keys that must be rejected or redacted from safe_details
_SENSITIVE_KEY_PATTERNS = frozenset(
    {
        "api_key",
        "authorization",
        "token",
        "secret",
        "password",
        "cookie",
    }
)


class FailureReportItem(BaseModel):
    """A JSON-safe representation of an AdapterFailure.

    No exception objects, raw dicts, or foreign Pydantic objects.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    error_code: str = Field(..., min_length=1, description="Machine-readable error code")
    category: ReportFailureCategory
    message: str
    retryable: bool = Field(default=False)
    safe_details: dict[str, str] = Field(
        default_factory=dict,
        description="Diagnostic details limited to str→str, sensitive keys redacted",
    )


# ---------------------------------------------------------------------------
# TaskReportItem — per-task row in the report
# ---------------------------------------------------------------------------


class ItemReportItem(BaseModel):
    """Report view of a single BenchmarkItemResult."""

    model_config = ConfigDict(extra="forbid", strict=True)

    item_id: str = Field(..., min_length=1)
    status: str = Field(..., description="graded / ungradable / failure")
    raw_score: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0)
    grader_id: str | None = Field(default=None)
    evidence_refs: list[str] = Field(default_factory=list)
    error_message: str | None = Field(default=None)
    failure_message: str | None = Field(default=None, description="Failure message for display (from AdapterFailure)")
    failure_category: str | None = Field(default=None, description="Failure category for display")
    failure_error_code: str | None = Field(default=None, description="Machine-readable error code")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskReportItem(BaseModel):
    """Report view of a single TaskAttempt + optional GradeResult.

    One TaskReportItem per TaskAttempt.  GradeResult is merged by
    attempt_id in the mapper.  Smoke tasks are excluded from capability
    scoring via explicit metadata, not name guessing.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    task_id: str = Field(..., min_length=1)
    attempt_id: str = Field(..., min_length=1)
    status: TaskReportStatus
    grader_id: str | None = Field(default=None)
    grade_status: ReportGradeStatus | None = Field(default=None)
    raw_score: float | None = Field(default=None, description="None when ungraded, failed, or ungradable")
    normalized_score: float | None = Field(default=None, description="None when ungraded, failed, or ungradable")
    evidence_refs: list[str] = Field(default_factory=list)
    items: list[ItemReportItem] = Field(default_factory=list, description="Per-item results for this task")
    failure: FailureReportItem | None = Field(default=None)
    capability_score_eligible: bool = Field(default=True, description="False for smoke/infrastructure tasks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="JSON-safe metadata (validated at runtime)")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata_json_safe(cls, v: object) -> object:
        """Validate metadata values through the shared json_safety module.

        Accepts any Mapping (e.g. dict, MappingProxyType) and returns a plain dict.
        """
        from collections.abc import Mapping

        from llmtrace.reporting.json_safety import validate_json_mapping

        if not isinstance(v, Mapping):
            raise ValueError(f"metadata must be a Mapping, got {type(v).__name__}")
        validate_json_mapping(v)
        return dict(v)


# ---------------------------------------------------------------------------
# BenchmarkReportSection — top-level report section for one benchmark run
# ---------------------------------------------------------------------------


class BenchmarkReportSection(BaseModel):
    """Report section representing exactly one BenchmarkRunResult.

    This is a pure data container.  No HTML, CLI, or aggregation logic
    lives here.  JSON output is ISO-8601 for datetimes and string for
    UUID references.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: UUID
    plan_id: str = Field(..., min_length=1)
    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    source_revision: str = Field(..., min_length=1)
    adapter_id: str = Field(..., min_length=1)
    adapter_version: str = Field(..., min_length=1)
    status: BenchmarkReportStatus
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
