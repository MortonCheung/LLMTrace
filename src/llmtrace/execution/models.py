"""Unified execution domain models — plan, result, and artifact manifest."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from llmtrace.analysis.behavior_drift import BehaviorDriftResult
from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.benchmarks.models import BenchmarkRunResult, RunPlan
from llmtrace.models.audit import AuditResult
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.reporting.benchmark_models import BenchmarkReportSection
from llmtrace.scoring.comparison import ComparisonResult
from llmtrace.scoring.models import CapabilityProfile


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime {value.isoformat()}")
    return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class UnifiedRunStatus(StrEnum):
    """Overall status of one unified execution."""

    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------


class UnifiedExecutionPlan(BaseModel):
    """Full execution plan, built before any real API request is sent."""

    plan_id: str = Field(..., min_length=1, description="Deterministic plan identifier")
    target_id: str = Field(..., min_length=1, description="Stable target label")
    candidate_model_id: str = Field(..., min_length=1, description="Declared model label")

    protocol_probe_requests: int = Field(..., ge=0, description="Planned protocol probe requests")
    benchmark_requests: int = Field(..., ge=0, description="Planned benchmark requests (Quick Suite = 32)")
    planned_requests: int = Field(..., ge=0, description="protocol + benchmark")
    maximum_requests: int = Field(..., ge=0, description="Hard ceiling enforced by RequestBudget")

    maximum_output_token_ceiling: int = Field(..., ge=0, description="Hard output-token ceiling, not a prediction")
    estimated_cost: float | None = Field(default=None, description="None — no trusted pricing source yet")

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    scoring_policy_id: str = Field(..., min_length=1)
    scoring_policy_version: str = Field(..., min_length=1)
    generation_config_sha256: str = Field(..., min_length=1, description="SHA-256 of the canonical generation config")
    requires_secure_code_sandbox: bool = Field(..., description="HumanEval requires a secure sandbox")

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _check_consistency(self) -> UnifiedExecutionPlan:
        if self.planned_requests != self.protocol_probe_requests + self.benchmark_requests:
            raise ValueError(
                f"planned_requests ({self.planned_requests}) must equal "
                f"protocol ({self.protocol_probe_requests}) + benchmark ({self.benchmark_requests})"
            )
        if self.maximum_requests < self.planned_requests:
            raise ValueError(
                f"maximum_requests ({self.maximum_requests}) must be >= planned_requests ({self.planned_requests})"
            )
        return self


# ---------------------------------------------------------------------------
# Artifact manifest
# ---------------------------------------------------------------------------

MANIFEST_VERSION = "0.1.0"


class RunArtifactManifest(BaseModel):
    """Append-only manifest for one execution's artifact directory.

    Contains no API key, no Authorization header, no raw request headers —
    only redacted target metadata and artifact hashes.
    """

    manifest_version: str = Field(default=MANIFEST_VERSION, min_length=1)
    execution_id: str = Field(..., min_length=1, description="UUID directory name — never user input")
    report_id: str = Field(..., min_length=1)

    target_id: str = Field(..., min_length=1)
    candidate_model_id: str = Field(..., min_length=1)
    base_url_redacted: str = Field(..., min_length=1, description="Redacted base URL (no credentials/query secrets)")
    protocol: str = Field(..., min_length=1)

    created_at: datetime = Field(...)
    completed_at: datetime | None = Field(default=None)
    status: UnifiedRunStatus = Field(...)

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    adapter_id: str = Field(..., min_length=1)
    adapter_version: str = Field(..., min_length=1)
    scoring_policy_id: str = Field(..., min_length=1)
    scoring_policy_version: str = Field(..., min_length=1)
    generation_config_sha256: str = Field(..., min_length=1)

    planned_requests: int = Field(..., ge=0)
    actual_requests: int = Field(..., ge=0)

    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="artifact filename → SHA-256 of its content",
    )

    baseline_execution_id: str | None = Field(default=None, description="History baseline this run was compared to")
    baseline_behavior_snapshot_sha256: str | None = Field(default=None)
    reference_snapshot_id: str | None = Field(default=None)

    warnings: tuple[str, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("created_at", "completed_at")
    @classmethod
    def _validate_manifest_utc(cls, v: datetime | None, info: ValidationInfo) -> datetime | None:
        if v is None:
            return None
        return _require_utc(v, str(info.field_name))


# ---------------------------------------------------------------------------
# Unified run result
# ---------------------------------------------------------------------------


class UnifiedRunResult(BaseModel):
    """Everything one unified execution produced."""

    execution_id: str = Field(..., min_length=1, description="UUID — safe directory name")
    status: UnifiedRunStatus = Field(...)
    plan: UnifiedExecutionPlan = Field(...)

    protocol_audit: AuditResult | None = Field(default=None)
    benchmark_plans: tuple[RunPlan, ...] = Field(default_factory=tuple)
    benchmark_runs: tuple[BenchmarkRunResult, ...] = Field(default_factory=tuple)
    benchmark_sections: tuple[BenchmarkReportSection, ...] = Field(default_factory=tuple)

    capability_profile: CapabilityProfile | None = Field(default=None)
    behavior_snapshot: BehaviorRunSnapshot | None = Field(default=None)
    behavior_drift: BehaviorDriftResult | None = Field(default=None)
    reference_comparison: ComparisonResult | None = Field(default=None)

    evidence: tuple[HTTPEvidence, ...] = Field(default_factory=tuple, description="Central evidence, arrival order")
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    started_at: datetime = Field(...)
    finished_at: datetime | None = Field(default=None)

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("started_at", "finished_at")
    @classmethod
    def _validate_run_utc(cls, v: datetime | None, info: ValidationInfo) -> datetime | None:
        if v is None:
            return None
        return _require_utc(v, str(info.field_name))
