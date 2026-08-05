"""LmEvalAdapter — BenchmarkAdapter for lm-evaluation-harness.

Translates lm-eval task execution into LLMTrace's unified
TaskAttempt and GradeResult models via the Provider-backed bridge.
"""

from __future__ import annotations

from uuid import uuid4

from llmtrace.adapters.base import BenchmarkAdapter
from llmtrace.adapters.lm_eval_bridge import ProviderEvidenceError
from llmtrace.adapters.lm_eval_runner import (
    LmEvalNotInstalledError,
    LmEvalRunner,
    LmEvalSecurityError,
    LmEvalValidationError,
)
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BudgetEstimate,
    CompletionOptions,
    CompletionProvider,
    FailureCategory,
    GradeResult,
    GradeStatus,
    LmEvalMetricResult,
    RunPlan,
    SmokeTaskManifest,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
)

try:
    import lm_eval as _lm_eval_pkg  # noqa: F401

    _LM_EVAL_VERSION: str = getattr(_lm_eval_pkg, "__version__", "0.4.12")
except ImportError:
    _LM_EVAL_VERSION = "unknown"

# ---------------------------------------------------------------------------
# Fixed smoke task identity (single source of truth)
# ---------------------------------------------------------------------------

_SMOKE_MANIFEST = SmokeTaskManifest()

_SMOKE_TASK_SPEC = TaskSpec(
    task_id=_SMOKE_MANIFEST.task_id,
    name="LLMTrace Smoke Task",
    description="Deterministic format-following smoke test for lm-eval adapter validation",
    category="smoke",
    num_samples=4,
)

_KNOWN_TASKS: dict[str, TaskSpec] = {
    _SMOKE_MANIFEST.task_id: _SMOKE_TASK_SPEC,
}


class LmEvalAdapter(BenchmarkAdapter):
    """Adapter for lm-evaluation-harness tasks.

    Connects lm-eval tasks to LLMTrace's Provider → Evidence → Result chain.
    Uses the LmEvalRunner isolation boundary internally.

    Only generate_until tasks are supported in this round.
    """

    def __init__(
        self,
        task_root: str,
        model_name: str = "test-model",
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        self._task_root = task_root
        self._model_name = model_name
        self._generation_kwargs = generation_kwargs or {}

    @property
    def adapter_id(self) -> str:
        return "lm-eval"

    @property
    def adapter_version(self) -> str:
        return _LM_EVAL_VERSION

    # ------------------------------------------------------------------
    # BenchmarkAdapter implementation
    # ------------------------------------------------------------------

    def list_tasks(self) -> list[TaskSpec]:
        """Return the list of known task specs."""
        return list(_KNOWN_TASKS.values())

    def build_plan(
        self,
        suite_id: str,
        suite_version: str,
        source_id: str,
        source_revision: str,
        task_ids: list[str],
    ) -> RunPlan:
        """Build a RunPlan using the shared planner.

        Raises:
            ValueError: If any task_id is unknown or if task_ids is empty.
        """
        from llmtrace.benchmarks.planner import build_plan as _build_plan

        if not task_ids:
            raise ValueError("task_ids must not be empty")

        tasks: list[TaskSpec] = []
        unknown: list[str] = []
        for tid in task_ids:
            spec = _KNOWN_TASKS.get(tid)
            if spec is None:
                unknown.append(tid)
            else:
                tasks.append(spec)

        if unknown:
            raise ValueError(f"Unknown task_ids: {unknown}. Known tasks: {sorted(_KNOWN_TASKS.keys())}")

        return _build_plan(
            suite_id=suite_id,
            suite_version=suite_version,
            source_id=source_id,
            source_revision=source_revision,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            tasks=tasks,
        )

    def estimate_budget(
        self,
        suite_id: str,
        task_ids: list[str],
        max_retries: int = 0,
    ) -> BudgetEstimate:
        """Estimate budget based on known task specs.

        Raises:
            ValueError: If any task_id is unknown or if task_ids is empty.
        """
        if not task_ids:
            raise ValueError("task_ids must not be empty")

        tasks: list[TaskSpec] = []
        unknown: list[str] = []
        for tid in task_ids:
            spec = _KNOWN_TASKS.get(tid)
            if spec is None:
                unknown.append(tid)
            else:
                tasks.append(spec)

        if unknown:
            raise ValueError(f"Unknown task_ids: {unknown}. Known tasks: {sorted(_KNOWN_TASKS.keys())}")

        total = sum(t.num_samples for t in tasks)
        return BudgetEstimate(
            planned_requests=total,
            maximum_requests=total * (1 + max_retries),
            maximum_retries=max_retries,
            assumptions=["lm-eval generate_until: one request per sample"],
        )

    async def run_task(
        self,
        task_spec: TaskSpec,
        provider: CompletionProvider,
    ) -> TaskAttempt:
        """Run a single lm-eval task via the LmEvalRunner.

        On ProviderEvidenceError, captures the structured failure with
        evidence_refs still pointing to the failed Evidence.

        The ``provider`` argument must implement the CompletionProvider protocol.
        """
        attempt_id = str(uuid4())
        try:
            runner = LmEvalRunner(
                provider=provider,
                model_name=self._model_name,
                task_root=self._task_root,
                generation_kwargs=self._generation_kwargs,
            )

            result = runner.run_task(
                task_name=task_spec.task_id,
                num_fewshot=0,
                batch_size=1,
            )

            evidence_ids_raw = result.get("evidence_ids", [])
            evidence_ids: list[object] = list(evidence_ids_raw) if isinstance(evidence_ids_raw, list) else []
            evidence_refs = [str(eid) for eid in evidence_ids]

            # Build controlled LmEvalMetricResult from the raw output
            actual_options_raw: object = result.get("actual_options")
            actual_options = actual_options_raw if isinstance(actual_options_raw, CompletionOptions) else None
            metric_result = _extract_metric_result(result, self.adapter_version, actual_options)

            if metric_result is None:
                return TaskAttempt(
                    attempt_id=attempt_id,
                    source_id=_SMOKE_MANIFEST.source_id,
                    source_revision=_SMOKE_MANIFEST.source_revision,
                    suite_id=_SMOKE_MANIFEST.suite_id,
                    suite_version=_SMOKE_MANIFEST.suite_version,
                    task_id=task_spec.task_id,
                    adapter_id=self.adapter_id,
                    adapter_version=self.adapter_version,
                    status=TaskStatus.FAILURE,
                    evidence_refs=evidence_refs,
                    failure=AdapterFailure(
                        error_code="LM_EVAL_RESULT_INVALID",
                        category=FailureCategory.ADAPTER,
                        message="lm-eval execution produced no valid exact_match metric",
                        retryable=False,
                        details={
                            "task_name": str(result.get("task_name", "unknown")),
                            "lm_eval_version": str(result.get("version", "unknown")),
                        },
                    ),
                )

            return TaskAttempt(
                attempt_id=attempt_id,
                source_id=_SMOKE_MANIFEST.source_id,
                source_revision=_SMOKE_MANIFEST.source_revision,
                suite_id=_SMOKE_MANIFEST.suite_id,
                suite_version=_SMOKE_MANIFEST.suite_version,
                task_id=task_spec.task_id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.SUCCESS,
                evidence_refs=evidence_refs,
                metadata={
                    "metric_result": metric_result.model_dump(),
                },
            )

        except ProviderEvidenceError as exc:
            evidence_refs = [str(exc.evidence_id)]
            return TaskAttempt(
                attempt_id=attempt_id,
                source_id=_SMOKE_MANIFEST.source_id,
                source_revision=_SMOKE_MANIFEST.source_revision,
                suite_id=_SMOKE_MANIFEST.suite_id,
                suite_version=_SMOKE_MANIFEST.suite_version,
                task_id=task_spec.task_id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.FAILURE,
                evidence_refs=evidence_refs,
                failure=AdapterFailure(
                    error_code=exc.error_code,
                    category=exc.category,
                    message=str(exc),
                    retryable=exc.retryable,
                    details={
                        "evidence_id": str(exc.evidence_id),
                        "exception_type": exc.exception_type,
                        "http_status": exc.http_status,
                    },
                ),
            )

        except (LmEvalNotInstalledError, LmEvalSecurityError, LmEvalValidationError) as exc:
            return TaskAttempt(
                attempt_id=attempt_id,
                source_id=_SMOKE_MANIFEST.source_id,
                source_revision=_SMOKE_MANIFEST.source_revision,
                suite_id=_SMOKE_MANIFEST.suite_id,
                suite_version=_SMOKE_MANIFEST.suite_version,
                task_id=task_spec.task_id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.FAILURE,
                failure=AdapterFailure(
                    error_code="LM_EVAL_SETUP_ERROR",
                    category=FailureCategory.ADAPTER,
                    message=str(exc),
                    retryable=False,
                    details={"exception_type": type(exc).__name__},
                ),
            )

        except Exception as exc:
            return TaskAttempt(
                attempt_id=attempt_id,
                source_id=_SMOKE_MANIFEST.source_id,
                source_revision=_SMOKE_MANIFEST.source_revision,
                suite_id=_SMOKE_MANIFEST.suite_id,
                suite_version=_SMOKE_MANIFEST.suite_version,
                task_id=task_spec.task_id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.FAILURE,
                failure=AdapterFailure(
                    error_code="LM_EVAL_RUN_ERROR",
                    category=FailureCategory.ADAPTER,
                    message=f"lm-eval execution failed: {exc}",
                    retryable=False,
                    details={"exception_type": type(exc).__name__},
                ),
            )

    def normalize_result(self, raw_result: dict[str, object]) -> GradeResult:
        """Normalize a raw lm-eval result dict into a GradeResult.

        Only accepts exact_match or exact_match,<filter> metrics.
        Values outside [0, 1] or non-numeric are rejected as UNGRADABLE.
        No fallback to "first numeric metric".
        """
        results: dict[str, object] = raw_result.get("results", {})  # type: ignore[assignment]
        evidence_ids_raw = raw_result.get("evidence_ids", [])
        evidence_ids: list[object] = list(evidence_ids_raw) if isinstance(evidence_ids_raw, list) else []
        task_name = str(raw_result.get("task_name", "unknown"))

        if not results:
            return _ungradable(task_name, "No results found in raw lm-eval output")

        if not isinstance(results, dict):
            return _ungradable(task_name, "Results is not a dict")

        # Strict: only accept exact_match or exact_match,<filter>
        metric_name, raw_score = _find_exact_match(results)

        if metric_name is None:
            return _ungradable(task_name, "No exact_match metric found in results")

        # Strict: value must be within [0, 1]
        if raw_score < 0.0 or raw_score > 1.0:
            return _ungradable(
                task_name,
                f"exact_match value {raw_score} is outside [0, 1]",
            )

        return GradeResult(
            grade_id=str(uuid4()),
            attempt_id=str(raw_result.get("attempt_id", "unknown")),
            source_id=_SMOKE_MANIFEST.source_id,
            source_revision=_SMOKE_MANIFEST.source_revision,
            suite_id=_SMOKE_MANIFEST.suite_id,
            suite_version=_SMOKE_MANIFEST.suite_version,
            task_id=task_name,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            grader_id=metric_name,
            raw_score=raw_score,
            normalized_score=raw_score,
            evidence_refs=[str(e) for e in evidence_ids],
            metadata={
                "lm_eval_version": _LM_EVAL_VERSION,
                "metric_name": metric_name,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_exact_match(results: dict[str, object]) -> tuple[str | None, float]:
    """Find the exact_match metric in flat or nested results.

    Returns (metric_name, score) or (None, 0.0) if not found.
    """
    # Try flat format first
    for key in results:
        if key.startswith("exact_match"):
            val = results[key]
            if isinstance(val, (int, float)):
                return str(key), float(val)
            return None, 0.0

    # Try nested format
    for _task_key, task_metrics in results.items():
        if isinstance(task_metrics, dict):
            for key in task_metrics:
                if key.startswith("exact_match"):
                    val = task_metrics[key]
                    if isinstance(val, (int, float)):
                        return str(key), float(val)
                    return None, 0.0

    return None, 0.0


def _ungradable(task_name: str, error_message: str) -> GradeResult:
    """Create an UNGRADABLE GradeResult."""
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id="unknown",
        source_id=_SMOKE_MANIFEST.source_id,
        source_revision=_SMOKE_MANIFEST.source_revision,
        suite_id=_SMOKE_MANIFEST.suite_id,
        suite_version=_SMOKE_MANIFEST.suite_version,
        task_id=task_name,
        adapter_id="lm-eval",
        adapter_version=_LM_EVAL_VERSION,
        grader_id="exact_match",
        raw_score=0.0,
        normalized_score=0.0,
        status=GradeStatus.UNGRADABLE,
        error_message=error_message,
        evidence_refs=[],
    )


def _extract_metric_result(
    result: dict[str, object],
    adapter_version: str,
    generation_options: CompletionOptions | None,
) -> LmEvalMetricResult | None:
    """Extract a controlled LmEvalMetricResult from runner output.

    Only accepts exact_match or exact_match,<filter> with numeric values
    within [0, 1].  Returns None for any invalid or unparseable result.
    """
    task_results: dict[str, object] = result.get("results", {})  # type: ignore[assignment]
    task_name = str(result.get("task_name", "unknown"))

    if not isinstance(task_results, dict) or not task_results:
        return None

    metric_name: str | None = None
    filter_name: str = "none"
    value: float | None = None

    for _tn, metrics in task_results.items():
        if isinstance(metrics, dict):
            for key, val in metrics.items():
                if key.startswith("exact_match"):
                    metric_name = str(key)
                    if "," in key:
                        _, filter_name = key.split(",", 1)
                    if isinstance(val, (int, float)):
                        value = float(val)
                    else:
                        # Non-numeric exact_match value — invalid
                        return None
                    break
        if metric_name is not None:
            break

    if metric_name is None or value is None:
        return None

    # Strict: value must be within [0, 1]
    if value < 0.0 or value > 1.0:
        return None

    return LmEvalMetricResult(
        task_name=task_name,
        metric_name=metric_name,
        filter_name=filter_name,
        value=value,
        fewshot=0,
        lm_eval_version=adapter_version,
        task_revision="1.0",
        generation_options=generation_options,
    )
