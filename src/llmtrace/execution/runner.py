"""Unified audit runner — one entry point from API config to artifacts."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from llmtrace.adapters.base import BenchmarkAdapterError
from llmtrace.adapters.code_execution import CodeExecutionBackend, create_code_execution_backend
from llmtrace.analysis.behavior_drift import BehaviorDriftEngine, BehaviorDriftResult
from llmtrace.analysis.behavior_models import (
    BehaviorDriftCompatibilityError,
    BehaviorDriftPolicy,
    BehaviorRunSnapshot,
)
from llmtrace.analysis.behavior_snapshot import BehaviorSnapshotBuilder
from llmtrace.benchmarks.models import BenchmarkRunResult, ItemStatus
from llmtrace.execution.artifacts import ArtifactIntegrityError, RunArtifactRepository, sha256_of
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import InMemoryEvidenceRecorder
from llmtrace.execution.models import (
    BenchmarkMeasurementSummary,
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
from llmtrace.security.redaction import SecretScrubber, extract_url_secret_values

if TYPE_CHECKING:
    from llmtrace.config import AuditConfig
    from llmtrace.reporting.benchmark_models import BenchmarkReportSection


class PreflightError(Exception):
    """Raised when preflight fails before any API request is sent.

    Raised *before* a provider is created, so a preflight failure by
    construction implies zero target HTTP requests, zero consumed budget,
    and zero recorded evidence.
    """

    error_code = "PREFLIGHT_FAILURE"


class SerializationBoundaryError(Exception):
    """Raised when a known secret survives canonical report serialization.

    The report serializers scrub before hashing / template rendering; if a
    known secret still reaches the serialized bytes, the boundary leaked and
    we fail closed instead of persisting stale-hash or leaky artifacts.
    """

    error_code = "SERIALIZATION_BOUNDARY_FAILURE"


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

    Provider lifecycle ownership: this runner is the *single* owner.  One
    provider context wraps both the protocol probes and the Quick Suite, so
    ``provider.client`` is valid for the whole execution and the recorder /
    request budget accumulate across protocol → benchmark as one state.

    Stages::

        PREFLIGHT → PROVIDER OPEN → PROTOCOL → BENCHMARK → SCORING → SNAPSHOT
        → HISTORY COMPARISON → REFERENCE COMPARISON → PROVIDER CLOSE → ARTIFACTS
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
        # Output-boundary scrubber: known secrets are the API key plus every
        # secret value embedded in the endpoint URL (userinfo credentials and
        # sensitive query parameters).  They must never reach a persisted
        # artifact, whichever side (request or response) they leak from.
        self._scrubber = SecretScrubber([api_key, *extract_url_secret_values(config.base_url)])
        self._budget: RequestBudget | None = None

    @property
    def request_budget(self) -> RequestBudget | None:
        """The run's request budget, if execution has started."""
        return self._budget

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
        self._budget = budget

        # ---- PREFLIGHT -----------------------------------------------------
        # Every predictable failure must happen before a provider exists and
        # before the first real HTTP request is sent.
        self._preflight()

        provider = create_provider(
            self._config,
            self._api_key,
            evidence_recorder=self._recorder,
            request_budget=budget,
        )

        # ---- PROTOCOL + BENCHMARK under one provider context ---------------
        # Single lifecycle owner: the runner opens the provider, hands the
        # *open* provider to both executors, and closes it on any exit path
        # (success, exception, timeout, cancellation).
        async with provider:
            protocol_outcome = await ProtocolAuditExecutor(self._config, provider).run_open_provider()
            audit_result = protocol_outcome.result
            if protocol_outcome.blocking_failure:
                warnings.append(f"protocol blocking failure at stage: {protocol_outcome.blocking_stage}")

            # ---- BENCHMARK / SCORING / SNAPSHOT ---------------------------
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

        # ---- MEASUREMENT HEALTH ---------------------------------------------
        measurement = self._measurement_summary(benchmark_runs)
        if measurement is not None:
            if measurement.graded_item_count == 0:
                warnings.append(f"benchmark measurement unavailable: 0 of {measurement.total_item_count} items graded")
            elif measurement.failure_item_count or measurement.ungradable_item_count:
                warnings.append(
                    f"benchmark measurement degraded: {measurement.failure_item_count} failure, "
                    f"{measurement.ungradable_item_count} ungradable, "
                    f"{measurement.graded_item_count} graded of {measurement.total_item_count} items"
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
        status = self._status(warnings, protocol_outcome.blocking_failure, capability_profile is not None, measurement)
        unified_evidence = tuple(self._recorder.list())

        result = UnifiedRunResult(
            execution_id=execution_id,
            status=status,
            plan=plan,
            protocol_audit=audit_result,
            benchmark_plans=tuple(benchmark_plans),
            benchmark_runs=tuple(benchmark_runs),
            benchmark_sections=tuple(benchmark_sections),
            measurement_summary=measurement,
            capability_profile=capability_profile,
            behavior_snapshot=behavior_snapshot,
            behavior_drift=behavior_drift,
            reference_comparison=reference_comparison,
            evidence=unified_evidence,
            warnings=tuple(self._scrubber.scrub_text(w) for w in warnings),
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

    # -- Preflight -------------------------------------------------------------

    def _preflight(self) -> None:
        """Fail closed on every predictable problem, before any HTTP request.

        Checks (I/O + schema validity only — compatibility stays with
        ``BehaviorDriftEngine`` / ``CapabilityComparator`` after the current
        profile exists):

        - Quick Suite manifest + resource integrity (files, counts, hashes);
        - secure code-execution sandbox availability;
        - explicit baseline snapshot exists and parses as a BehaviorRunSnapshot;
        - explicit reference snapshot exists and parses as a ReferenceSnapshot;
        - artifact root can be created and written (probe file, cleaned up).
        """
        from llmtrace.adapters.quick_suite import verify_quick_suite_resources

        try:
            verify_quick_suite_resources()
        except BenchmarkAdapterError as exc:
            raise PreflightError(f"quick suite resources invalid: {exc}") from exc

        if not self._code_backend.is_available():
            raise PreflightError("secure code execution sandbox unavailable; benchmark requires it")

        if self._baseline_snapshot_path is not None:
            path = self._baseline_snapshot_path
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PreflightError(f"baseline snapshot unreadable: {path}") from exc
            try:
                BehaviorRunSnapshot.model_validate_json(raw)
            except ValueError as exc:
                raise PreflightError(f"baseline snapshot is not a valid BehaviorRunSnapshot: {path}") from exc

        if self._reference_snapshot_path is not None:
            path = self._reference_snapshot_path
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PreflightError(f"reference snapshot unreadable: {path}") from exc
            try:
                ReferenceSnapshot.model_validate_json(raw)
            except ValueError as exc:
                raise PreflightError(f"reference snapshot is not a valid ReferenceSnapshot: {path}") from exc

        try:
            self._repository.ensure_writable()
        except OSError as exc:
            raise PreflightError("artifact root is not writable") from exc

    # -- Measurement health ------------------------------------------------------

    @staticmethod
    def _measurement_summary(runs: Sequence[BenchmarkRunResult]) -> BenchmarkMeasurementSummary | None:
        """Deterministic measurement health from the canonical item chain.

        GRADED items with score 0.0 count as *valid* measurements — a wrong
        answer is a measured answer, never a failure.
        """
        graded = failure = ungradable = 0
        for run in runs:
            for attempt in run.task_attempts:
                for item in attempt.item_results:
                    if item.status == ItemStatus.GRADED:
                        graded += 1
                    elif item.status == ItemStatus.FAILURE:
                        failure += 1
                    else:
                        ungradable += 1
        total = graded + failure + ungradable
        if total == 0:
            return None
        return BenchmarkMeasurementSummary(
            total_item_count=total,
            graded_item_count=graded,
            failure_item_count=failure,
            ungradable_item_count=ungradable,
            execution_coverage=(total - failure) / total,
            grading_coverage=graded / total,
        )

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
        """Find the newest compatible, *integrity-verified* prior snapshot.

        For each candidate (newest first): verify the artifact SHA against
        the baseline manifest's recorded hash, parse it, then let the drift
        engine's compatibility gate decide.  A missing / corrupted /
        hash-mismatched / unparseable candidate is skipped with an integrity
        warning and the search continues with older candidates — the current
        run is never crashed by a tampered historical artifact.
        """
        candidates = self._repository.find_behavior_snapshot_candidates(
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
        for manifest in candidates:
            try:
                prior_snapshot = self._repository.load_behavior_snapshot(manifest.execution_id)
            except ArtifactIntegrityError as exc:
                warnings.append(f"skipped corrupted historical baseline {manifest.execution_id}: {exc}")
                continue
            if prior_snapshot is None:
                continue
            try:
                drift = engine.compare(prior_snapshot, snapshot, policy)
            except BehaviorDriftCompatibilityError as exc:
                warnings.append(f"skipped incompatible baseline {manifest.execution_id}: {exc.error_code}")
                continue
            # The baseline manifest's recorded hash IS the historical
            # artifact's real SHA — never recompute from a re-serialization.
            return drift, manifest.execution_id, manifest.artifacts["behavior_snapshot.json"]
        # History exists but no candidate was usable — a real diagnostic.
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

        # Content hash of the file as written — the honest SHA of this
        # baseline artifact (there is no manifest for a file-supplied baseline).
        return drift, None, sha256_of(raw)

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
    def _status(
        warnings: list[str],
        blocking_failure: bool,
        has_profile: bool,
        measurement: BenchmarkMeasurementSummary | None,
    ) -> UnifiedRunStatus:
        if blocking_failure:
            return UnifiedRunStatus.PARTIAL
        # Zero graded items = the benchmark stage produced no usable
        # measurement (e.g. every request failed) — a COMPLETED label would
        # hide total measurement loss behind per-item structures.
        if measurement is not None and measurement.graded_item_count == 0:
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
            warnings=(self._scrubber.scrub_text("execution exceeded the wall-clock limit and was cancelled"),),
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
        scrub = self._scrubber.scrub_text
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
                # Scrub belongs to canonical serialization: report content_hash
                # is computed from the already-scrubbed structure.
                secret_scrubber=self._scrubber,
            )
            html_path = generate_html_report(
                result.protocol_audit,
                Path(tmpdir) / "report.html",
                benchmark_sections=list(result.benchmark_sections) or None,
                reference_comparison=result.reference_comparison,
                behavior_drift=result.behavior_drift,
                capability_profile=result.capability_profile,
                secret_scrubber=self._scrubber,
            )
            json_content = json_path.read_text(encoding="utf-8")
            html_content = html_path.read_text(encoding="utf-8")
            # Defensive boundary check: canonical serialization must already
            # have scrubbed every known secret.  If one survives into the
            # final serialized bytes, the boundary leaked — fail closed rather
            # than persist stale-hash / leaky content.
            if scrub(json_content) != json_content:
                raise SerializationBoundaryError("known secret survived JSON canonical serialization")
            if scrub(html_content) != html_content:
                raise SerializationBoundaryError("known secret survived HTML template rendering")
            artifacts["report.json"] = json_content
            artifacts["report.html"] = html_content

        if result.capability_profile is not None:
            artifacts["capability_profile.json"] = scrub(result.capability_profile.model_dump_json(indent=2))
        if result.behavior_snapshot is not None:
            artifacts["behavior_snapshot.json"] = scrub(result.behavior_snapshot.model_dump_json(indent=2))
        if result.benchmark_runs:
            artifacts["benchmark_runs.json"] = scrub(_benchmark_runs_json(result.benchmark_runs))

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
        metadata: dict[str, object] = {
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
        if result.measurement_summary is not None:
            metadata["benchmark_measurement"] = result.measurement_summary.model_dump(mode="json")
        # Defense in depth: the metadata crosses the persistence boundary.
        return cast("dict[str, object]", self._scrubber.scrub(metadata))

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
