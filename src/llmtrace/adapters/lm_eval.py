"""LmEvalAdapter — BenchmarkAdapter for lm-evaluation-harness.

Translates lm-eval task execution into LLMTrace's unified
TaskAttempt and GradeResult models via the Provider-backed bridge.
"""

from __future__ import annotations

from uuid import uuid4

from llmtrace.adapters.base import BenchmarkAdapter
from llmtrace.adapters.lm_eval_runner import LmEvalNotInstalledError, LmEvalRunner
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BudgetEstimate,
    FailureCategory,
    GradeResult,
    GradeStatus,
    RunPlan,
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
# Fixed smoke task identity (MUST NOT be used in capability scoring)
# ---------------------------------------------------------------------------

_SMOKE_SUITE_ID = "llmtrace_smoke"
_SMOKE_SUITE_VERSION = "1.0.0"
_SMOKE_SOURCE_REVISION = "0000000-smoke"

# ---------------------------------------------------------------------------
# Known task specs for the local smoke test (extensible in future rounds)
# ---------------------------------------------------------------------------

_KNOWN_TASKS = [
    TaskSpec(
        task_id=_SMOKE_SUITE_ID,
        name="LLMTrace Smoke Task",
        description="Deterministic format-following smoke test for lm-eval adapter validation",
        category="smoke",
        num_samples=4,
    ),
]


class LmEvalAdapter(BenchmarkAdapter):
    """Adapter for lm-evaluation-harness tasks.

    Connects lm-eval tasks to LLMTrace's Provider → Evidence → Result chain.
    Uses the LmEvalRunner isolation boundary internally.

    Only generate_until tasks are supported in this round.
    """

    def __init__(
        self,
        include_path: str,
        model_name: str = "test-model",
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        self._include_path = include_path
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
        return list(_KNOWN_TASKS)

    def build_plan(
        self,
        suite_id: str,
        suite_version: str,
        source_id: str,
        source_revision: str,
        task_ids: list[str],
    ) -> RunPlan:
        """Build a RunPlan using the shared planner."""
        from llmtrace.benchmarks.planner import build_plan as _build_plan

        tasks = [t for t in _KNOWN_TASKS if t.task_id in task_ids]
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
        """Estimate budget based on known task specs."""
        tasks = [t for t in _KNOWN_TASKS if t.task_id in task_ids]
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
        provider: object,
    ) -> TaskAttempt:
        """Run a single lm-eval task via the LmEvalRunner.

        On failure (including lm-eval exceptions), returns a TaskAttempt
        with status=FAILURE and a structured AdapterFailure.

        The ``provider`` argument must implement ``complete(model, messages) -> HTTPEvidence``.
        """
        attempt_id = str(uuid4())
        try:
            runner = LmEvalRunner(
                provider=provider,
                model_name=self._model_name,
                include_path=self._include_path,
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

            # Extract metric results for metadata
            task_results: dict[str, object] = result.get("results", {})  # type: ignore[assignment]

            return TaskAttempt(
                attempt_id=attempt_id,
                source_id="lm-eval",
                source_revision=_SMOKE_SOURCE_REVISION,
                suite_id=_SMOKE_SUITE_ID,
                suite_version=_SMOKE_SUITE_VERSION,
                task_id=task_spec.task_id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.SUCCESS,
                evidence_refs=evidence_refs,
                metadata={
                    "lm_eval_version": _LM_EVAL_VERSION,
                    "task_name": task_spec.task_id,
                    "include_path": self._include_path,
                    "output_type": "generate_until",
                    "fewshot": 0,
                    "task_results": task_results,
                },
            )

        except LmEvalNotInstalledError as exc:
            return TaskAttempt(
                attempt_id=attempt_id,
                source_id="lm-eval",
                source_revision=_SMOKE_SOURCE_REVISION,
                suite_id=_SMOKE_SUITE_ID,
                suite_version=_SMOKE_SUITE_VERSION,
                task_id=task_spec.task_id,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.FAILURE,
                failure=AdapterFailure(
                    error_code="LM_EVAL_NOT_INSTALLED",
                    category=FailureCategory.ADAPTER,
                    message=str(exc),
                    retryable=False,
                ),
            )

        except Exception as exc:
            return TaskAttempt(
                attempt_id=attempt_id,
                source_id="lm-eval",
                source_revision=_SMOKE_SOURCE_REVISION,
                suite_id=_SMOKE_SUITE_ID,
                suite_version=_SMOKE_SUITE_VERSION,
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

        Expects raw_result to contain at minimum:
        - results: dict of metric_name → score (per-task metrics)
        - evidence_ids: list of evidence UUIDs
        - task_name: task identifier

        Uses exact_match metric if available, otherwise the first metric.
        """
        results: dict[str, object] = raw_result.get("results", {})  # type: ignore[assignment]
        evidence_ids_raw = raw_result.get("evidence_ids", [])
        evidence_ids: list[object] = list(evidence_ids_raw) if isinstance(evidence_ids_raw, list) else []
        task_name = str(raw_result.get("task_name", "unknown"))

        if not results:
            return GradeResult(
                grade_id=str(uuid4()),
                attempt_id=str(raw_result.get("attempt_id", "unknown")),
                source_id="lm-eval",
                source_revision=_SMOKE_SOURCE_REVISION,
                suite_id=_SMOKE_SUITE_ID,
                suite_version=_SMOKE_SUITE_VERSION,
                task_id=task_name,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                grader_id="exact_match",
                raw_score=0.0,
                normalized_score=0.0,
                status=GradeStatus.UNGRADABLE,
                error_message="No results found in raw lm-eval output",
                evidence_refs=[],
            )

        # The results may be either:
        #   flat:   {"exact_match": 1.0}
        #   nested: {"llmtrace_smoke": {"exact_match": 1.0, ...}}
        # Note: lm-eval may return metric keys like "exact_match,none" (metric,filter format)
        metric_name = "exact_match"
        raw_score: float = 0.0

        if not isinstance(results, dict):
            return GradeResult(
                grade_id=str(uuid4()),
                attempt_id=str(raw_result.get("attempt_id", "unknown")),
                source_id="lm-eval",
                source_revision=_SMOKE_SOURCE_REVISION,
                suite_id=_SMOKE_SUITE_ID,
                suite_version=_SMOKE_SUITE_VERSION,
                task_id=task_name,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                grader_id="exact_match",
                raw_score=0.0,
                normalized_score=0.0,
                status=GradeStatus.UNGRADABLE,
                error_message="Results is not a dict",
                evidence_refs=[],
            )

        # Try flat format first (exact_match or exact_match,filter as a top-level key)
        for key in results:
            if key.startswith("exact_match"):
                val = results[key]
                if isinstance(val, (int, float)):
                    raw_score = float(val)
                    metric_name = str(key)
                    break
        else:
            # Try nested format
            for _task_key, task_metrics in results.items():
                if isinstance(task_metrics, dict):
                    for key in task_metrics:
                        if key.startswith("exact_match"):
                            val = task_metrics[key]
                            if isinstance(val, (int, float)):
                                raw_score = float(val)
                                metric_name = str(key)
                                break
                    else:
                        # Fallback: use the first numeric metric
                        if task_metrics:
                            first_key = next(iter(task_metrics.keys()))
                            first_val = task_metrics[first_key]
                            if isinstance(first_val, (int, float)):
                                metric_name = str(first_key)
                                raw_score = float(first_val)

        normalized_score = max(0.0, min(1.0, raw_score))

        return GradeResult(
            grade_id=str(uuid4()),
            attempt_id=str(raw_result.get("attempt_id", "unknown")),
            source_id="lm-eval",
            source_revision=_SMOKE_SOURCE_REVISION,
            suite_id=_SMOKE_SUITE_ID,
            suite_version=_SMOKE_SUITE_VERSION,
            task_id=task_name,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            grader_id=metric_name,
            raw_score=raw_score,
            normalized_score=normalized_score,
            evidence_refs=[str(e) for e in evidence_ids],
            metadata={
                "lm_eval_version": _LM_EVAL_VERSION,
                "metric_name": metric_name,
                "all_metrics": {
                    str(k): float(v)
                    for k, v in results.items()
                    if not isinstance(v, dict) and isinstance(v, (int, float))
                },
            },
        )
