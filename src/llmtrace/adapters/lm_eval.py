"""LmEvalAdapter — BenchmarkAdapter for lm-evaluation-harness.

Translates lm-eval task execution into LLMTrace's unified
TaskAttempt and GradeResult models via the Provider-backed bridge.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
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
    BenchmarkItemResult,
    BenchmarkTaskDefinition,
    BudgetEstimate,
    CompletionOptions,
    CompletionProvider,
    FailureCategory,
    GradeResult,
    GradeStatus,
    ItemStatus,
    LmEvalMetricResult,
    RunPlan,
    SmokeTaskManifest,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
    aggregate_item_results,
)

try:
    import lm_eval as _lm_eval_pkg  # noqa: F401

    _LM_EVAL_VERSION: str = getattr(_lm_eval_pkg, "__version__", "0.4.12")
except ImportError:
    _LM_EVAL_VERSION = "unknown"

# ---------------------------------------------------------------------------
# Task identity registry — single source of truth for all known tasks
# ---------------------------------------------------------------------------

# Built-in trusted task root — not configurable by callers.
_BUILTIN_RESOURCES = Path(__file__).resolve().parent / "_resources"
BUILTIN_SMOKE_TASK_ROOT = str(_BUILTIN_RESOURCES)

_SMOKE_DEFINITION = BenchmarkTaskDefinition(
    task_id="llmtrace_smoke",
    source_id="lm-eval",
    source_revision="0000000-smoke",
    suite_id="llmtrace_smoke",
    suite_version="1.0.0",
    is_smoke=True,
    capability_score_eligible=False,
)

_GSM8K_DEFINITION = BenchmarkTaskDefinition(
    task_id="gsm8k_subset",
    source_id="gsm8k",
    source_revision="pending-verification",  # See gsm8k_subset.yaml metadata for checksum
    suite_id="llmtrace-v0.2-acceptance",
    suite_version="0.1.0",
    is_smoke=False,
    capability_score_eligible=True,
    metadata={
        "benchmark_source": "openai/gsm8k",
        "upstream_task": "gsm8k",
        "upstream_dataset": "openai/gsm8k",
    },
)

# Canonical task registry — every task_id MUST be registered here.
_TASK_REGISTRY: dict[str, BenchmarkTaskDefinition] = {
    "llmtrace_smoke": _SMOKE_DEFINITION,
    "gsm8k_subset": _GSM8K_DEFINITION,
}

# Legacy: kept for test backward-compatibility.
_SMOKE_MANIFEST = SmokeTaskManifest()

_SMOKE_TASK_SPEC = TaskSpec(
    task_id=_SMOKE_DEFINITION.task_id,
    name="LLMTrace Smoke Task",
    description="Deterministic format-following smoke test for lm-eval adapter validation",
    category="smoke",
    num_samples=4,
)

_GSM8K_SUBSET_SPEC = TaskSpec(
    task_id="gsm8k_subset",
    name="GSM8K Acceptance Subset",
    description="Fixed 8-sample subset of GSM8K grade-school math word problems (openai/gsm8k, MIT license)",
    category="benchmark",
    num_samples=8,
)

_KNOWN_TASKS: dict[str, TaskSpec] = {
    _SMOKE_DEFINITION.task_id: _SMOKE_TASK_SPEC,
    "gsm8k_subset": _GSM8K_SUBSET_SPEC,
}


class LmEvalAdapter(BenchmarkAdapter):
    """Adapter for lm-evaluation-harness tasks.

    Connects lm-eval tasks to LLMTrace's Provider → Evidence → Result chain.
    Uses the LmEvalRunner isolation boundary internally.

    Only generate_until tasks are supported in this round.
    """

    def __init__(
        self,
        model_name: str = "test-model",
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        self._task_root = BUILTIN_SMOKE_TASK_ROOT
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

        Task provenance is read from ``_TASK_REGISTRY`` so every code path
        — success, options-inconsistent, result-invalid, provider-error,
        setup-error, and unexpected exception — returns a TaskAttempt with
        consistent task identity.
        """
        task_id = task_spec.task_id
        defn = _get_task_def(task_id)
        attempt_id = str(uuid4())

        try:
            runner = LmEvalRunner(
                provider=provider,
                model_name=self._model_name,
                task_root=self._task_root,
                generation_kwargs=self._generation_kwargs,
            )

            result = runner.run_task(
                task_name=task_id,
                num_fewshot=0,
                batch_size=1,
            )

            evidence_ids_raw = result.get("evidence_ids", [])
            evidence_ids: list[object] = list(evidence_ids_raw) if isinstance(evidence_ids_raw, list) else []
            evidence_refs = [str(eid) for eid in evidence_ids]

            # Check for options inconsistency
            if result.get("options_inconsistent"):
                return _build_attempt(
                    attempt_id=attempt_id,
                    task_id=task_id,
                    defn=defn,
                    adapter_id=self.adapter_id,
                    adapter_version=self.adapter_version,
                    status=TaskStatus.FAILURE,
                    evidence_refs=evidence_refs,
                    failure=AdapterFailure(
                        error_code="LM_EVAL_OPTIONS_INCONSISTENT",
                        category=FailureCategory.ADAPTER,
                        message="Different requests within this task used inconsistent CompletionOptions",
                        retryable=False,
                        details={"task_name": str(result.get("task_name", "unknown"))},
                    ),
                )

            # Build controlled LmEvalMetricResult from the raw output
            actual_options_raw: object = result.get("actual_options")
            actual_options = actual_options_raw if isinstance(actual_options_raw, CompletionOptions) else None
            metric_result = _extract_metric_result(result, self.adapter_version, actual_options)

            if metric_result is None:
                return _build_attempt(
                    attempt_id=attempt_id,
                    task_id=task_id,
                    defn=defn,
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

            # Extract item-level results for tasks that support them
            item_results: list[BenchmarkItemResult] = []
            sample_results_raw: object = result.get("sample_results")
            if isinstance(sample_results_raw, list) and task_id == "gsm8k_subset":
                sample_results = [s for s in sample_results_raw if isinstance(s, dict)]
                item_results = _grade_gsm8k_items(sample_results, attempt_id, task_id)

            return _build_attempt(
                attempt_id=attempt_id,
                task_id=task_id,
                defn=defn,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.SUCCESS,
                evidence_refs=evidence_refs,
                metric_result=metric_result.model_dump(),
                item_results=item_results,
            )

        except ProviderEvidenceError as exc:
            evidence_refs = [str(exc.evidence_id)]
            return _build_attempt(
                attempt_id=attempt_id,
                task_id=task_id,
                defn=defn,
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
            return _build_attempt(
                attempt_id=attempt_id,
                task_id=task_id,
                defn=defn,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.FAILURE,
                evidence_refs=[],
                failure=AdapterFailure(
                    error_code="LM_EVAL_SETUP_ERROR",
                    category=FailureCategory.ADAPTER,
                    message=str(exc),
                    retryable=False,
                    details={"exception_type": type(exc).__name__},
                ),
            )

        except Exception as exc:
            return _build_attempt(
                attempt_id=attempt_id,
                task_id=task_id,
                defn=defn,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                status=TaskStatus.FAILURE,
                evidence_refs=[],
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

        **Data flow** (v0.3-A):

        1. Extract ``item_results`` and ``planned_item_count`` from ``raw_result``.
        2. Compute ``ItemAggregateResult`` via ``aggregate_item_results()`` — this
           is the **canonical** score source.
        3. Extract lm-eval metric from ``raw_result["results"]``.
        4. Cross-check: if both sources exist and mismatch beyond float tolerance,
           raise ``ValueError("LM_EVAL_ITEM_AGGREGATE_MISMATCH")``.
        5. GradeResult.normalized_score = item aggregate normalized_score.
           If no item results, fall back to lm-eval metric.

        lm-eval metric is NO LONGER the source of truth for the GradeResult score.

        Provenance is read from ``_TASK_REGISTRY`` via the task_name in the
        raw_result, so GradeResult always carries the same identity as the
        corresponding TaskAttempt.
        """
        results: dict[str, object] = raw_result.get("results", {})  # type: ignore[assignment]
        evidence_ids_raw = raw_result.get("evidence_ids", [])
        evidence_ids: list[object] = list(evidence_ids_raw) if isinstance(evidence_ids_raw, list) else []
        task_name = str(raw_result.get("task_name", "unknown"))

        # ----------------------------------------------------------------
        # 1. Derive score from item aggregate (canonical source)
        # ----------------------------------------------------------------
        item_results_raw: object = raw_result.get("item_results", [])
        planned_item_count_raw: object = raw_result.get("planned_item_count")

        item_derived_score: float | None = None
        item_aggregate_meta: dict[str, object] = {}

        if isinstance(item_results_raw, list) and len(item_results_raw) > 0 and planned_item_count_raw is not None:
            try:
                planned = int(planned_item_count_raw)  # type: ignore[call-overload]
            except (TypeError, ValueError):
                planned = None

            if planned is not None:
                items: list[BenchmarkItemResult] = []
                for ir in item_results_raw:
                    if isinstance(ir, BenchmarkItemResult):
                        items.append(ir)
                    elif isinstance(ir, dict):
                        with contextlib.suppress(Exception):
                            items.append(BenchmarkItemResult(**ir))

                if items:
                    aggregate = aggregate_item_results(items, planned_item_count=planned)
                    item_derived_score = aggregate.normalized_score
                    item_aggregate_meta = {
                        "planned_item_count": aggregate.planned_item_count,
                        "graded_item_count": aggregate.graded_item_count,
                        "failure_count": aggregate.failure_count,
                        "ungradable_count": aggregate.ungradable_count,
                        "correct_count": aggregate.correct_count,
                        "wrong_count": aggregate.wrong_count,
                        "coverage": aggregate.coverage,
                        "execution_coverage": aggregate.execution_coverage,
                        "item_aggregate_score": aggregate.normalized_score,
                    }

        # ----------------------------------------------------------------
        # 2. Extract lm-eval metric (cross-check only)
        # ----------------------------------------------------------------
        metric_name: str | None = None
        lm_eval_score: float | None = None
        lm_eval_out_of_bounds: bool = False

        if isinstance(results, dict) and results:
            metric_name, lm_eval_score = _find_exact_match(results)
            if metric_name is not None and lm_eval_score is not None:
                if lm_eval_score < 0.0 or lm_eval_score > 1.0:
                    lm_eval_out_of_bounds = True
                    lm_eval_score = None  # cannot be used for cross-check or score
            else:
                # No valid exact_match found — clear both
                lm_eval_score = None
                metric_name = None

        # ----------------------------------------------------------------
        # 3. Cross-check: item aggregate vs lm-eval metric
        # ----------------------------------------------------------------
        if (
            item_derived_score is not None
            and lm_eval_score is not None
            and not math.isclose(item_derived_score, lm_eval_score, rel_tol=1e-9)
        ):
            raise ValueError(
                f"LM_EVAL_ITEM_AGGREGATE_MISMATCH: "
                f"item-derived score={item_derived_score}, "
                f"lm-eval metric='{metric_name}'={lm_eval_score}"
            )

        # ----------------------------------------------------------------
        # 4. Determine final score
        # ----------------------------------------------------------------
        if item_derived_score is not None:
            final_score = item_derived_score
        elif lm_eval_score is not None:
            final_score = lm_eval_score
        elif lm_eval_out_of_bounds:
            return _ungradable(
                task_name,
                "exact_match value is outside [0, 1]",
            )
        else:
            if not results:
                return _ungradable(task_name, "No results found in raw lm-eval output")
            if not isinstance(results, dict):
                return _ungradable(task_name, "Results is not a dict")
            return _ungradable(task_name, "No exact_match metric found in results")

        # ----------------------------------------------------------------
        # 5. Build GradeResult
        # ----------------------------------------------------------------
        defn = _get_task_def(task_name)

        # Merge metadata: item aggregate info + lm-eval cross-check info
        metadata: dict[str, object] = {
            "lm_eval_version": _LM_EVAL_VERSION,
        }
        metadata.update(item_aggregate_meta)
        if metric_name is not None and lm_eval_score is not None:
            metadata["lm_eval_metric_name"] = metric_name
            metadata["lm_eval_cross_check_score"] = lm_eval_score
            metadata["lm_eval_cross_check_pass"] = item_derived_score is not None

        return GradeResult(
            grade_id=str(uuid4()),
            attempt_id=str(raw_result.get("attempt_id", "unknown")),
            source_id=defn.source_id,
            source_revision=defn.source_revision,
            suite_id=defn.suite_id,
            suite_version=defn.suite_version,
            task_id=task_name,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            grader_id=metric_name if metric_name else "exact_match",
            raw_score=final_score,
            normalized_score=final_score,
            evidence_refs=[str(e) for e in evidence_ids],
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_exact_match_metric(key: str) -> tuple[str, str] | None:
    """Parse a metric key, accepting only exact_match or exact_match,<filter>.

    Accepted:
      "exact_match"  →  ("exact_match", "none")
      "exact_match,none"  →  ("exact_match", "none")
      "exact_match,my_filter"  →  ("exact_match", "my_filter")

    Rejected:
      "exact_match_fake"
      "exact_matching"
      "exact_match,"
      Any other string that merely starts with "exact_match".
    """
    if key == "exact_match":
        return ("exact_match", "none")
    if key.startswith("exact_match,"):
        parts = key.split(",", 1)
        filter_name = parts[1].strip()
        if filter_name:
            return ("exact_match", filter_name)
    return None


def _find_exact_match(results: dict[str, object]) -> tuple[str | None, float]:
    """Find the exact_match metric in flat or nested results.

    Returns (metric_name, score) or (None, 0.0) if not found.
    """

    def _lookup(d: dict[str, object]) -> tuple[str | None, float]:
        for key in d:
            parsed = parse_exact_match_metric(key)
            if parsed is not None:
                val = d[key]
                if isinstance(val, (int, float)):
                    return str(key), float(val)
                return None, 0.0
        return None, 0.0

    # Try flat format first
    mn, score = _lookup(results)
    if mn is not None:
        return mn, score

    # Try nested format
    for _task_key, task_metrics in results.items():
        if isinstance(task_metrics, dict):
            mn, score = _lookup(task_metrics)
            if mn is not None:
                return mn, score

    return None, 0.0


def _ungradable(task_name: str, error_message: str) -> GradeResult:
    """Create an UNGRADABLE GradeResult with provenance from the task registry."""
    defn = _get_task_def(task_name)
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id="unknown",
        source_id=defn.source_id,
        source_revision=defn.source_revision,
        suite_id=defn.suite_id,
        suite_version=defn.suite_version,
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


def _get_task_def(task_id: str) -> BenchmarkTaskDefinition:
    """Look up a task definition from the canonical registry.

    Falls back to the smoke definition for unknown tasks so that
    error paths never return provenance-less objects.
    """
    return _TASK_REGISTRY.get(task_id, _SMOKE_DEFINITION)


def _build_attempt(
    *,
    attempt_id: str,
    task_id: str,
    defn: BenchmarkTaskDefinition,
    adapter_id: str,
    adapter_version: str,
    status: TaskStatus,
    evidence_refs: list[str],
    metric_result: dict[str, object] | None = None,
    failure: AdapterFailure | None = None,
    item_results: list[BenchmarkItemResult] | None = None,
) -> TaskAttempt:
    """Build a TaskAttempt with provenance and metadata from the task definition."""
    return TaskAttempt(
        attempt_id=attempt_id,
        source_id=defn.source_id,
        source_revision=defn.source_revision,
        suite_id=defn.suite_id,
        suite_version=defn.suite_version,
        task_id=task_id,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        status=status,
        evidence_refs=evidence_refs,
        item_results=item_results or [],
        metadata=defn.task_metadata(metric_result=metric_result),
        failure=failure,
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
                parsed = parse_exact_match_metric(key)
                if parsed is not None:
                    metric_name, filter_name = parsed
                    if isinstance(val, (int, float)):
                        value = float(val)
                    else:
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


# ---------------------------------------------------------------------------
# GSM8K item-level grading
# ---------------------------------------------------------------------------

_GSM8K_ANSWER_RE = __import__("re").compile(r"####\s*(-?[\d,.]+)")


def _load_gsm8k_expected_answers() -> list[tuple[str, str]]:
    """Load expected (question, answer) from gsm8k_subset.json.

    Returns a list of (question, expected_answer) tuples in order.
    """
    import json

    data_path = _BUILTIN_RESOURCES / "gsm8k_subset.json"
    samples = json.loads(data_path.read_text())
    return [(s["question"], s["answer"]) for s in samples]


def _extract_gsm8k_final_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K completion.

    Uses the same pattern as the lm-eval strict-match filter:
    #### followed by a number (possibly negative, with commas or decimals).
    """
    m = _GSM8K_ANSWER_RE.search(text)
    if m:
        result = m.group(1)
        return str(result).strip() if result is not None else None
    return None


def _normalize_number(text: str) -> str:
    """Normalize a number string for comparison: remove commas, extra zeros."""
    cleaned = text.replace(",", "")
    try:
        n = float(cleaned)
        if n == int(n):
            return str(int(n))
        return f"{n:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def _grade_gsm8k_items(
    sample_results: list[dict[str, object]],
    attempt_id: str,
    task_id: str,
) -> list[BenchmarkItemResult]:
    """Grade individual GSM8K responses against expected answers.

    Handles both success and failure samples from the bridge layer.
    Failure items (status=failure) are mapped to ItemStatus.FAILURE;
    items with unparseable answers are mapped to ItemStatus.UNGRADABLE.

    Args:
        sample_results: Per-sample data from ProviderBackedLM, each dict
            containing at minimum ``evidence_id``, ``response_text``, ``status``.
        attempt_id: Parent TaskAttempt identifier.
        task_id: Task identifier.

    Returns:
        List of BenchmarkItemResult, one per sample, in order.
    """
    expected = _load_gsm8k_expected_answers()
    items: list[BenchmarkItemResult] = []

    for idx, sample in enumerate(sample_results):
        item_id = f"item-{idx + 1:03d}"
        sample_status = str(sample.get("status", "success"))
        evidence_id = str(sample.get("evidence_id", ""))

        # Provider failure → ItemStatus.FAILURE
        if sample_status == "failure":
            failure_message = str(sample.get("failure_message", "Provider failure"))
            items.append(
                BenchmarkItemResult(
                    item_id=item_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    status=ItemStatus.FAILURE,
                    raw_score=0.0,
                    normalized_score=0.0,
                    evidence_refs=[evidence_id] if evidence_id else [],
                    error_message=failure_message,
                    metadata={
                        "failure_error_code": sample.get("failure_error_code", ""),
                        "failure_category": sample.get("failure_category", ""),
                    },
                )
            )
            continue

        response_text = str(sample.get("response_text", ""))

        # Extract answers
        extracted = _extract_gsm8k_final_answer(response_text)
        if extracted is None:
            # Could not parse answer from model output
            items.append(
                BenchmarkItemResult(
                    item_id=item_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    status=ItemStatus.UNGRADABLE,
                    raw_score=0.0,
                    normalized_score=0.0,
                    evidence_refs=[evidence_id] if evidence_id else [],
                    error_message="Could not extract final answer (#### pattern not found)",
                )
            )
            continue

        # Compare with expected (use the idx-th expected answer if available)
        if idx < len(expected):
            expected_answer = expected[idx][1]
            expected_final = _extract_gsm8k_final_answer(expected_answer)
            if expected_final is None:
                expected_final = expected_answer

            is_correct = _normalize_number(extracted) == _normalize_number(expected_final)
            score = 1.0 if is_correct else 0.0

            items.append(
                BenchmarkItemResult(
                    item_id=item_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    status=ItemStatus.GRADED,
                    raw_score=score,
                    normalized_score=score,
                    evidence_refs=[evidence_id] if evidence_id else [],
                    metadata={
                        "extracted_answer": extracted,
                        "expected_answer": expected_final,
                        "correct": is_correct,
                    },
                )
            )
        else:
            items.append(
                BenchmarkItemResult(
                    item_id=item_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    status=ItemStatus.UNGRADABLE,
                    raw_score=0.0,
                    normalized_score=0.0,
                    evidence_refs=[evidence_id] if evidence_id else [],
                    error_message=f"No expected answer for item index {idx}",
                )
            )

    return items
