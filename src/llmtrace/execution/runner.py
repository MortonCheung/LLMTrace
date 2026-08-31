"""Unified audit runner — one entry point from API config to artifacts."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from llmtrace.adapters.code_execution import CodeExecutionBackend, create_code_execution_backend
from llmtrace.analysis.behavior_drift import BehaviorDriftEngine, BehaviorDriftResult
from llmtrace.analysis.behavior_models import (
    BehaviorDriftCompatibilityError,
    BehaviorDriftPolicy,
    BehaviorRunSnapshot,
)
from llmtrace.analysis.behavior_snapshot import BehaviorSnapshotBuilder
from llmtrace.benchmarks.models import BenchmarkRunResult
from llmtrace.execution.artifacts import RunArtifactRepository, sha256_of
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import InMemoryEvidenceRecorder
from llmtrace.execution.models import (
    RunArtifactManifest,
    UnifiedExecutionPlan,
    UnifiedRunResult,
    UnifiedRunStatus,
)
from llmtrace.execution.planner import build_unified_execution_plan
from llmtrace.execution.protocol_audit import ProtocolAuditExecutor
from llmtrace.execution.quick_suite import QuickSuiteRunner
from llmtrace.providers.factory import create_provider
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import generate_json_report
from llmtrace.scoring.aggregator import TaskScoringRegistry, aggregate_capability_profile
from llmtrace.scoring.comparison import CapabilityComparator, ComparisonResult
from llmtrace.scoring.errors import ScoringError
from llmtrace.scoring.models import CapabilityProfile
from llmtrace.scoring.policy import CapabilityScoringPolicy
from llmtrace.scoring.reference import ReferenceSnapshot

if TYPE_CHECKING:
    from llmtrace.config import AuditConfig
    from llmtrace.reporting.benchmark_models import BenchmarkReportSection


class PreflightError(Exception):
    """Raised when preflight fails before any API request is sent."""

    error_code = "PREFLIGHT_FAILURE"


class UnifiedRunRequest(BaseModel):
    """Everything the runner needs, decided before execution starts."""

    config: object = Field(..., description="AuditConfig")
    target_id: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1, description="In-memory only; never persisted")

    compare_latest: bool = True
    baseline_snapshot_path: Path | None = None
    reference_snapshot_path: Path | None = None
    max_wall_seconds: float | None = Field(default=None, gt=0)

    model_config = {"frozen": True, "extra": "forbid"}


class UnifiedAuditRunner:
    """Execute the full pipeline: protocol → benchmark → scoring → report.

    Stages::

        PRECHECK → PLAN → PROTOCOL → BENCHMARK → SCORING → SNAPSHOT
        → HISTORY COMPARISON → REFERENCE COMPARISON → REPORT/ARTIFACTS
    """

    def __init__(
        self,
        config: AuditConfig,
        *,
        api_key: str,
        target_id: str,
        repository: RunArtifactRepository,
        code_backend: CodeExecutionBackend | None = None,
        compare_latest: bool = True,
        baseline_snapshot_path: Path | None = None,
        reference_snapshot_path: Path | None = None,
        max_wall_seconds: float | None = None,
    ) -> None:
        # Preflight happens here — constructing a fail-closed backend raises
        # SandboxUnavailableError before any request is sent.
        self._code_backend = code_backend if code_backend is not None else create_code_execution_backend()
        self._config = config
        self._api_key = api_key
        self._target_id = target_id
        self._repository = repository
        self._compare_latest = compare_latest
        self._baseline_snapshot_path = baseline_snapshot_path
        self._reference_snapshot_path = reference_snapshot_path
        self._max_wall_seconds = max_wall_seconds
        self._recorder = InMemoryEvidenceRecorder()
        self._policy = CapabilityScoringPolicy.create_v1()

    async def run(self) -> UnifiedRunResult:
        """Run the whole pipeline; on wall-clock timeout salvage a PARTIAL result."""
        execution_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)

        try:
            if self._max_wall_seconds is not None:
                async with asyncio.timeout(self._max_wall_seconds):
                    return await self._execute(execution_id, started_at)
            return await self._execute(execution_id, started_at)
        except TimeoutError:
            return self._timed_out_result(execution_id, started_at)

    # -- Full pipeline -----------------------------------------------------

    async def _execute(self, execution_id: str, started_at: datetime) -> UnifiedRunResult:
        warnings: list[str] = []
        plan = build_unified_execution_plan(self._config, target_id=self._target_id, policy=self._policy)
        budget = RequestBudget(plan.maximum_requests)
        provider = create_provider(
            self._config,
            self._api_key,
            evidence_recorder=self._recorder,
            request_budget=budget,
        )

        # ---- PROTOCOL ----------------------------------------------------
        protocol_outcome = await ProtocolAuditExecutor(self._config, provider).run()
        audit_result = protocol_outcome.result
        if protocol_outcome.blocking_failure:
            warnings.append(f"protocol blocking failure at stage: {protocol_outcome.blocking_stage}")

        # ---- BENCHMARK / SCORING / SNAPSHOT -------------------------------
        capability_profile = None
        behavior_snapshot = None
        benchmark_plans = []
        benchmark_runs: list[BenchmarkRunResult] = []
        benchmark_sections: list[BenchmarkReportSection] = []

        if not protocol_outcome.blocking_failure:
            suite_runner = QuickSuiteRunner(provider, code_backend=self._code_backend)
            suite_result = await suite_runner.run()
            benchmark_plans = list(suite_result.plans)
            benchmark_runs = list(suite_result.run_results)
            benchmark_sections = list(suite_result.report_sections)

            registry = self._quick_registry()
            capability_profile = aggregate_capability_profile(benchmark_runs, registry, self._policy, strict=True)

            builder = BehaviorSnapshotBuilder()
            behavior_snapshot = builder.build(
                run_results=benchmark_runs,
                profile=capability_profile,
                evidence=list(self._recorder.list()),
                target_id=self._target_id,
                candidate_model_id=self._config.model,
                generation_config=self._generation_config(),
            )

        # ---- HISTORY COMPARISON -------------------------------------------
        behavior_drift = None
        baseline_execution_id: str | None = None
        baseline_snapshot_sha: str | None = None
        if behavior_snapshot is not None:
            if self._baseline_snapshot_path is not None:
                behavior_drift, baseline_execution_id, baseline_snapshot_sha = self._compare_explicit_baseline(
                    behavior_snapshot, warnings
                )
            elif self._compare_latest:
                behavior_drift, baseline_execution_id, baseline_snapshot_sha = self._compare_history(
                    behavior_snapshot, execution_id, warnings
                )
            else:
                warnings.append("history comparison disabled (--no-compare-latest)")

        # ---- REFERENCE COMPARISON ------------------------------------------
        reference_comparison = None
        reference_snapshot_id: str | None = None
        if self._reference_snapshot_path is not None and capability_profile is not None:
            reference_comparison, reference_snapshot_id = self._compare_reference(capability_profile, warnings)

        # ---- STATUS --------------------------------------------------------
        actual_requests = budget.consumed_requests
        status = self._status(warnings, protocol_outcome.blocking_failure, capability_profile is not None)
        unified_evidence = tuple(self._recorder.list())

        result = UnifiedRunResult(
            execution_id=execution_id,
            status=status,
            plan=plan,
            protocol_audit=audit_result,
            benchmark_plans=tuple(benchmark_plans),
            benchmark_runs=tuple(benchmark_runs),
            benchmark_sections=tuple(benchmark_sections),
            capability_profile=capability_profile,
            behavior_snapshot=behavior_snapshot,
            behavior_drift=behavior_drift,
            reference_comparison=reference_comparison,
            evidence=unified_evidence,
            warnings=tuple(warnings),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

        # Unified evidence closure: the report's audit evidence must cover the
        # whole run (protocol + benchmark), not just the probe subset.
        audit_result.evidence = list(unified_evidence)

        self._write_artifacts(
            result,
            actual_requests=actual_requests,
            baseline_execution_id=baseline_execution_id,
            baseline_snapshot_sha=baseline_snapshot_sha,
            reference_snapshot_id=reference_snapshot_id,
        )
        return result

    # -- Plan ----------------------------------------------------------------

    def _plan(self) -> UnifiedExecutionPlan:
        return build_unified_execution_plan(self._config, target_id=self._target_id, policy=self._policy)

    @staticmethod
    def _generation_config() -> dict[str, float | int]:
        from llmtrace.adapters.quick_suite import get_quick_suite_generation_config

        return get_quick_suite_generation_config()

    @staticmethod
    def _quick_registry() -> TaskScoringRegistry:
        from llmtrace.adapters.quick_suite import create_quick_registry

        return create_quick_registry()

    # -- History -------------------------------------------------------------

    def _compare_history(
        self,
        snapshot: BehaviorRunSnapshot,
        execution_id: str,
        warnings: list[str],
    ) -> tuple[BehaviorDriftResult | None, str | None, str | None]:
        """Find the newest compatible prior snapshot; never self-compare."""
        candidates = self._repository.find_behavior_snapshots(
            target_id=self._target_id,
            candidate_model_id=self._config.model,
            exclude_execution_id=execution_id,
        )
        if not candidates:
            # A first run simply has no history yet — an expected state, not
            # a warning that should downgrade the run status.
            return None, None, None

        engine = BehaviorDriftEngine()
        policy = BehaviorDriftPolicy.create_v1()
        for manifest, prior_snapshot in candidates:
            try:
                drift = engine.compare(prior_snapshot, snapshot, policy)
            except BehaviorDriftCompatibilityError as exc:
                warnings.append(f"skipped incompatible baseline {manifest.execution_id}: {exc.error_code}")
                continue
            sha = sha256_of(prior_snapshot.model_dump_json())
            return drift, manifest.execution_id, sha
        # History exists but no candidate was compatible — a real diagnostic.
        warnings.append("no compatible historical baseline found; behavior drift skipped")
        return None, None, None

    def _compare_explicit_baseline(
        self,
        snapshot: BehaviorRunSnapshot,
        warnings: list[str],
    ) -> tuple[BehaviorDriftResult | None, str | None, str | None]:
        if self._baseline_snapshot_path is None:
            return None, None, None
        try:
            raw = self._baseline_snapshot_path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.append(f"baseline snapshot unreadable: {exc}")
            return None, None, None

        try:
            baseline = BehaviorRunSnapshot.model_validate_json(raw)
        except ValueError as exc:
            warnings.append(f"baseline snapshot invalid: {exc}")
            return None, None, None

        try:
            drift = BehaviorDriftEngine().compare(
                baseline,
                snapshot,
                BehaviorDriftPolicy.create_v1(),
            )
        except BehaviorDriftCompatibilityError as exc:
            warnings.append(f"explicit baseline incompatible: {exc.error_code}")
            return None, None, None

        return drift, None, sha256_of(baseline.model_dump_json())

    # -- Reference -------------------------------------------------------------

    def _compare_reference(
        self, profile: CapabilityProfile, warnings: list[str]
    ) -> tuple[ComparisonResult | None, str | None]:
        if self._reference_snapshot_path is None:
            return None, None
        try:
            raw = self._reference_snapshot_path.read_text(encoding="utf-8")
            reference = ReferenceSnapshot.model_validate_json(raw)
        except (OSError, ValueError) as exc:
            warnings.append(f"reference snapshot invalid: {exc}")
            return None, None

        plan = self._plan()
        try:
            comparison = CapabilityComparator().compare(
                reference,
                profile,
                candidate_suite_id=plan.suite_id,
                candidate_suite_version=plan.suite_version,
                candidate_model_id=self._config.model,
            )
        except ScoringError as exc:
            code = getattr(exc, "error_code", type(exc).__name__)
            warnings.append(f"reference comparison skipped: {code}")
            return None, None
        return comparison, reference.snapshot_id

    # -- Status / artifacts ------------------------------------------------------

    @staticmethod
    def _status(warnings: list[str], blocking_failure: bool, has_profile: bool) -> UnifiedRunStatus:
        if blocking_failure:
            return UnifiedRunStatus.PARTIAL
        if warnings:
            return UnifiedRunStatus.COMPLETED_WITH_WARNINGS
        if has_profile:
            return UnifiedRunStatus.COMPLETED
        return UnifiedRunStatus.PARTIAL

    def _timed_out_result(self, execution_id: str, started_at: datetime) -> UnifiedRunResult:
        return UnifiedRunResult(
            execution_id=execution_id,
            status=UnifiedRunStatus.FAILED,
            plan=self._plan(),
            evidence=tuple(self._recorder.list()),
            warnings=("execution exceeded the wall-clock limit and was cancelled",),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    def _write_artifacts(
        self,
        result: UnifiedRunResult,
        *,
        actual_requests: int,
        baseline_execution_id: str | None,
        baseline_snapshot_sha: str | None,
        reference_snapshot_id: str | None,
    ) -> None:
        assert result.protocol_audit is not None
        artifacts: dict[str, str] = {}

        with tempfile.TemporaryDirectory(prefix="llmtrace-report-") as tmpdir:
            json_path = generate_json_report(
                result.protocol_audit,
                Path(tmpdir) / "report.json",
                benchmark_sections=list(result.benchmark_sections) or None,
                reference_comparison=result.reference_comparison,
                behavior_drift=result.behavior_drift,
                capability_profile=result.capability_profile,
                execution_metadata=self._execution_metadata(result, actual_requests),
            )
            html_path = generate_html_report(
                result.protocol_audit,
                Path(tmpdir) / "report.html",
                benchmark_sections=list(result.benchmark_sections) or None,
                reference_comparison=result.reference_comparison,
                behavior_drift=result.behavior_drift,
                capability_profile=result.capability_profile,
            )
            artifacts["report.json"] = json_path.read_text(encoding="utf-8")
            artifacts["report.html"] = html_path.read_text(encoding="utf-8")

        if result.capability_profile is not None:
            artifacts["capability_profile.json"] = result.capability_profile.model_dump_json(indent=2)
        if result.behavior_snapshot is not None:
            artifacts["behavior_snapshot.json"] = result.behavior_snapshot.model_dump_json(indent=2)
        if result.benchmark_runs:
            artifacts["benchmark_runs.json"] = _benchmark_runs_json(result.benchmark_runs)

        manifest = RunArtifactManifest(
            execution_id=result.execution_id,
            report_id=result.protocol_audit.report_id,
            target_id=self._target_id,
            candidate_model_id=self._config.model,
            base_url_redacted=self._redacted_base_url(),
            protocol=str(self._config.protocol.value),
            created_at=result.started_at,
            completed_at=result.finished_at,
            status=result.status,
            suite_id=result.plan.suite_id,
            suite_version=result.plan.suite_version,
            adapter_id="llmtrace-quick-v1",
            adapter_version="0.1.0",
            scoring_policy_id=result.plan.scoring_policy_id,
            scoring_policy_version=result.plan.scoring_policy_version,
            generation_config_sha256=result.plan.generation_config_sha256,
            planned_requests=result.plan.planned_requests,
            actual_requests=actual_requests,
            baseline_execution_id=baseline_execution_id,
            baseline_behavior_snapshot_sha256=baseline_snapshot_sha,
            reference_snapshot_id=reference_snapshot_id,
            warnings=result.warnings,
        )
        self._repository.commit(manifest, artifacts)

    def _execution_metadata(self, result: UnifiedRunResult, actual_requests: int) -> dict[str, object]:
        return {
            "execution_id": result.execution_id,
            "target_id": self._target_id,
            "status": result.status.value,
            "planned_requests": result.plan.planned_requests,
            "actual_requests": actual_requests,
            "maximum_output_token_ceiling": result.plan.maximum_output_token_ceiling,
            "estimated_cost": result.plan.estimated_cost,
            "generation_config_sha256": result.plan.generation_config_sha256,
            "requires_secure_code_sandbox": result.plan.requires_secure_code_sandbox,
            "plan_id": result.plan.plan_id,
        }

    def _redacted_base_url(self) -> str:
        from llmtrace.security.redaction import redact_url

        return redact_url(self._config.base_url)


def _benchmark_runs_json(runs: tuple[BenchmarkRunResult, ...]) -> str:
    import json

    return json.dumps(
        {"runs": [run.model_dump(mode="json") for run in runs]},
        ensure_ascii=False,
        indent=2,
    )
