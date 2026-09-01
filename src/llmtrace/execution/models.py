"""Unified execution domain models — plan, result, and artifact manifest."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

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


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _normalize_sha256(value: str, field_name: str) -> str:
    """Require a real 64-char hex SHA-256 digest, normalised to lowercase."""
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{field_name} must be exactly 64 hexadecimal characters (SHA-256), got {len(value)} chars: {value!r}"
        )
    return value.lower()


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
    suite_content_sha256: str = Field(
        ...,
        description="SHA-256 (64 lowercase hex) of the canonical suite content identity",
    )
    scoring_policy_id: str = Field(..., min_length=1)
    scoring_policy_version: str = Field(..., min_length=1)
    generation_config_sha256: str = Field(..., min_length=1, description="SHA-256 of the canonical generation config")
    requires_secure_code_sandbox: bool = Field(..., description="HumanEval requires a secure sandbox")

    calibration_policy_id: str | None = Field(
        default=None,
        description="Calibration policy id; None when calibration is not requested",
    )
    calibration_policy_version: str | None = Field(
        default=None,
        description="Calibration policy version; None when calibration is not requested",
    )
    reference_set_id: str | None = Field(default=None, description="ReferenceSet id; None when no calibration")
    reference_set_version: str | None = Field(
        default=None, description="ReferenceSet version; None when no calibration"
    )
    reference_set_content_sha256: str | None = Field(
        default=None,
        description="ReferenceSet content hash; None when no calibration",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("suite_content_sha256")
    @classmethod
    def _validate_suite_content_sha256(cls, v: str) -> str:
        return _normalize_sha256(v, "suite_content_sha256")

    @field_validator("reference_set_content_sha256")
    @classmethod
    def _validate_ref_set_content_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _normalize_sha256(v, "reference_set_content_sha256")

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

MANIFEST_VERSION = "0.3.0"


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
    suite_content_sha256: str | None = Field(
        default=None,
        description="Canonical suite content identity; None for pre-v0.4-A runs (never accepted as reference input)",
    )
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

    calibration_policy_id: str | None = Field(
        default=None,
        description="Calibration policy id; None when calibration was not requested or failed",
    )
    calibration_policy_version: str | None = Field(default=None)
    reference_set_id: str | None = Field(default=None)
    reference_set_version: str | None = Field(default=None)
    reference_set_content_sha256: str | None = Field(default=None)

    warnings: tuple[str, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("created_at", "completed_at")
    @classmethod
    def _validate_manifest_utc(cls, v: datetime | None, info: ValidationInfo) -> datetime | None:
        if v is None:
            return None
        return _require_utc(v, str(info.field_name))

    @field_validator("suite_content_sha256")
    @classmethod
    def _validate_manifest_suite_content_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _normalize_sha256(v, "suite_content_sha256")


# ---------------------------------------------------------------------------
# Benchmark measurement summary
# ---------------------------------------------------------------------------


class BenchmarkMeasurementSummary(BaseModel):
    """Deterministic measurement health of the benchmark stage.

    Computed from the canonical ``BenchmarkRunResult → TaskAttempt.item_results``
    chain — it never changes scoring, it only surfaces how much of the Quick
    Suite was actually measured.  A GRADED item with score 0.0 is a *valid*
    measurement (a wrong answer is still a measured answer); only FAILURE /
    UNGRADABLE items represent lost or degraded measurement.

    Status rules (versioned, deterministic — no magic thresholds):

    - protocol blocking failure                → PARTIAL
    - graded_item_count == 0                   → PARTIAL (measurement unavailable)
    - FAILURE/UNGRADABLE present, graded > 0   → COMPLETED_WITH_WARNINGS (degraded)
    - 32/32 GRADED and no other warning        → COMPLETED
    """

    MEASUREMENT_STATUS_RULES: ClassVar[str] = "v1: graded==0→PARTIAL; any failure/ungradable→warnings; else COMPLETED"

    total_item_count: int = Field(..., ge=0, description="Planned items actually attempted")
    graded_item_count: int = Field(..., ge=0, description="Items with a valid graded measurement (incl. score=0)")
    failure_item_count: int = Field(..., ge=0, description="Items lost to provider/HTTP/execution failures")
    ungradable_item_count: int = Field(..., ge=0, description="Items whose answer could not be graded")
    execution_coverage: float = Field(..., ge=0.0, le=1.0, description="(total - failure) / total")
    grading_coverage: float = Field(..., ge=0.0, le=1.0, description="graded / total")

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _check_consistency(self) -> BenchmarkMeasurementSummary:
        total = self.total_item_count
        if self.graded_item_count + self.failure_item_count + self.ungradable_item_count != total:
            raise ValueError(
                f"graded ({self.graded_item_count}) + failure ({self.failure_item_count}) "
                f"+ ungradable ({self.ungradable_item_count}) must equal total ({total})"
            )
        return self


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
    measurement_summary: BenchmarkMeasurementSummary | None = Field(
        default=None,
        description="Benchmark measurement health; None when the benchmark stage did not run",
    )

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
