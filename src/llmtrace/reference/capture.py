"""ReferenceCaptureService — operator-verified reference capture (§27–§28).

The capture service is a thin orchestration layer.  It MUST reuse
``UnifiedAuditRunner`` for the actual execution — it never writes a second
benchmark chain and never calls ``Provider.complete`` itself::

    ReferenceCaptureService
        → UnifiedAuditRunner
        → RunArtifactRepository
        → qualification (Gate 1–10)
        → ReferenceSnapshotBuilder
        → ReferenceRepository

Trust semantics (§28): a captured run is recorded as
``operator_verified_api_run`` — the operator asserts the endpoint is a
trusted reference source; LLMTrace records declaration + measurement
provenance but does not independently prove endpoint ownership.  The label
``official_api_verified`` is never used.

Dry-run guarantees (§30.1): zero HTTP, zero API-key lookup, zero artifacts,
zero snapshots, zero code execution.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from llmtrace.config import AuditConfig
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.models import UnifiedExecutionPlan, UnifiedRunStatus
from llmtrace.execution.planner import build_unified_execution_plan
from llmtrace.execution.runner import UnifiedAuditRunner
from llmtrace.scoring.reference import ReferenceRepository

from .builder import OPERATOR_VERIFIED_API_RUN, ReferenceSnapshotBuilder
from .models import ReferenceQualificationResult
from .qualification import qualify_reference_run


class ReferenceCaptureStatus(StrEnum):
    """Outcome of one capture attempt."""

    CAPTURED = "CAPTURED"
    QUALIFICATION_REJECTED = "QUALIFICATION_REJECTED"
    RUN_FAILED = "RUN_FAILED"


class ReferenceCaptureResult(BaseModel):
    """Result of one capture attempt, safe to print and to keep in logs.

    Carries machine-readable reason codes when qualification rejected the
    run; the run's artifacts remain available under ``execution_id`` even on
    rejection (§30.2 — a failed reference run is still a measurement fact).
    """

    execution_id: str = Field(..., min_length=1)
    status: ReferenceCaptureStatus = Field(...)
    run_status: UnifiedRunStatus | None = Field(default=None)
    snapshot_id: str | None = Field(default=None, description="Set only on CAPTURED")
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    qualification: ReferenceQualificationResult | None = Field(default=None)

    model_config = {"frozen": True, "extra": "forbid"}


class ReferenceCaptureService:
    """Operator-verified capture: run once, qualify, snapshot if trusted.

    Args:
        reference_dir: Root holding ``snapshots/`` (and later ``sets/``).
        artifact_root: Root of the run artifact repository (``runs/`` inside).
    """

    def __init__(self, *, reference_dir: Path, artifact_root: Path) -> None:
        self._reference_dir = reference_dir
        self._artifact_root = artifact_root

    # -- Planning (side-effect free) ---------------------------------------

    def build_plan(self, config: AuditConfig, target_id: str) -> UnifiedExecutionPlan:
        """Build the unified execution plan (pure; no HTTP, no artifacts)."""
        return build_unified_execution_plan(config, target_id=target_id)

    def snapshot_repository(self) -> ReferenceRepository:
        """Append-only snapshot store under ``<reference_dir>/snapshots``."""
        return ReferenceRepository(directory=self._reference_dir / "snapshots")

    # -- Capture ------------------------------------------------------------

    async def capture(
        self,
        *,
        config: AuditConfig,
        api_key: str,
        target_id: str,
        provider_id: str,
        snapshot_id: str,
        created_by: str,
        max_wall_seconds: float | None = None,
    ) -> ReferenceCaptureResult:
        """Execute one reference run and persist a snapshot if qualified.

        The run always reuses ``UnifiedAuditRunner``; the snapshot is saved
        only after verification and qualification pass (verify → qualify →
        build → save, §15).  A rejected run keeps its artifacts and produces
        no snapshot.
        """
        repository = RunArtifactRepository(self._artifact_root)
        runner = UnifiedAuditRunner(
            config,
            api_key=api_key,
            target_id=target_id,
            repository=repository,
            max_wall_seconds=max_wall_seconds,
        )
        run_result = await runner.run()

        if run_result.status == UnifiedRunStatus.FAILED:
            return ReferenceCaptureResult(
                execution_id=run_result.execution_id,
                status=ReferenceCaptureStatus.RUN_FAILED,
                run_status=run_result.status,
                warnings=run_result.warnings,
            )

        qualification = qualify_reference_run(execution_id=run_result.execution_id, artifact_repository=repository)
        if not qualification.qualified:
            return ReferenceCaptureResult(
                execution_id=run_result.execution_id,
                status=ReferenceCaptureStatus.QUALIFICATION_REJECTED,
                run_status=run_result.status,
                reason_codes=qualification.reason_codes,
                warnings=(*run_result.warnings, *qualification.warnings),
                qualification=qualification,
            )

        builder = ReferenceSnapshotBuilder()
        snapshot = builder.build(
            execution_id=run_result.execution_id,
            artifact_repository=repository,
            reference_repository=self.snapshot_repository(),
            provider_id=provider_id,
            snapshot_id=snapshot_id,
            created_by=created_by,
            source_type=OPERATOR_VERIFIED_API_RUN,
        )
        return ReferenceCaptureResult(
            execution_id=run_result.execution_id,
            status=ReferenceCaptureStatus.CAPTURED,
            run_status=run_result.status,
            snapshot_id=snapshot.snapshot_id,
            qualification=qualification,
        )


__all__: list[str] = ["ReferenceCaptureService", "ReferenceCaptureResult", "ReferenceCaptureStatus"]
