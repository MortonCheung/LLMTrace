"""Quick Suite v1 adapter — ARC-Challenge, HumanEval, GSM8K, IFEval.

A single adapter that handles all four Quick Suite task types using
the existing Provider abstraction.  Each task produces 8 item results.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from llmtrace.adapters.base import BenchmarkAdapter, BenchmarkAdapterError
from llmtrace.adapters.code_execution import (
    CodeExecutionBackend,
    SandboxUnavailableError,
    create_code_execution_backend,
)
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkItemResult,
    BenchmarkTaskDefinition,
    BudgetEstimate,
    CompletionOptions,
    FailureCategory,
    GradeResult,
    GradeStatus,
    ItemStatus,
    RunPlan,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
    aggregate_item_results,
)
from llmtrace.models.evidence import HTTPEvidence

if TYPE_CHECKING:
    from llmtrace.providers.base import BaseProvider
    from llmtrace.scoring.aggregator import TaskScoringRegistry

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

_QUICK_TASK_DEFS: dict[str, BenchmarkTaskDefinition] = {
    "arc_challenge_quick_v1": BenchmarkTaskDefinition(
        task_id="arc_challenge_quick_v1",
        source_id="arc_challenge",
        source_revision="ARC-Challenge-2018",
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        adapter_id="llmtrace-quick-v1",
        is_smoke=False,
        requires_item_results=True,
        capability_score_eligible=True,
        metric="accuracy",
        filter_="none",
        metadata={"dimension": "reasoning"},
    ),
    "humaneval_quick_v1": BenchmarkTaskDefinition(
        task_id="humaneval_quick_v1",
        source_id="humaneval",
        source_revision="human-eval-v1-2021",
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        adapter_id="llmtrace-quick-v1",
        is_smoke=False,
        requires_item_results=True,
        capability_score_eligible=True,
        metric="pass@1",
        filter_="none",
        metadata={"dimension": "coding"},
    ),
    "gsm8k_quick_v1": BenchmarkTaskDefinition(
        task_id="gsm8k_quick_v1",
        source_id="gsm8k",
        source_revision="gsm8k-main-2023",
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        adapter_id="llmtrace-quick-v1",
        is_smoke=False,
        requires_item_results=True,
        capability_score_eligible=True,
        metric="exact_match",
        filter_="none",
        metadata={"dimension": "math_science"},
    ),
    "ifeval_quick_v1": BenchmarkTaskDefinition(
        task_id="ifeval_quick_v1",
        source_id="ifeval",
        source_revision="ifeval-v1-2023",
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        adapter_id="llmtrace-quick-v1",
        is_smoke=False,
        requires_item_results=True,
        capability_score_eligible=True,
        metric="constraint_satisfaction",
        filter_="none",
        metadata={"dimension": "instruction_following"},
    ),
}

_QUICK_TASK_SPECS: dict[str, TaskSpec] = {
    "arc_challenge_quick_v1": TaskSpec(
        task_id="arc_challenge_quick_v1",
        name="ARC-Challenge Quick 8",
        description="8 fixed multiple-choice science questions from ARC-Challenge",
        category="reasoning",
        num_samples=8,
        metadata={"scoring": "accuracy"},
    ),
    "humaneval_quick_v1": TaskSpec(
        task_id="humaneval_quick_v1",
        name="HumanEval Quick 8",
        description="8 fixed Python coding problems from HumanEval",
        category="coding",
        num_samples=8,
        metadata={"scoring": "pass@1"},
    ),
    "gsm8k_quick_v1": TaskSpec(
        task_id="gsm8k_quick_v1",
        name="GSM8K Quick 8",
        description="8 fixed math word problems from GSM8K",
        category="math_science",
        num_samples=8,
        metadata={"scoring": "exact_match"},
    ),
    "ifeval_quick_v1": TaskSpec(
        task_id="ifeval_quick_v1",
        name="IFEval Quick 8",
        description="8 fixed instruction-following prompts from IFEval",
        category="instruction_following",
        num_samples=8,
        metadata={"scoring": "constraint_satisfaction"},
    ),
}

# ---------------------------------------------------------------------------
# Canonical generation policy — single source of truth
# ---------------------------------------------------------------------------

# The Quick Suite is a fixed measuring stick: temperature=0, max_tokens=512.
# The generation config that is actually sent, the one hashed into a
# BehaviorRunSnapshot, and the one declared by the execution plan MUST all
# come from here — never from three separate literals.
QUICK_SUITE_GENERATION_CONFIG: dict[str, float | int] = {
    "temperature": 0.0,
    "max_tokens": 512,
}

QUICK_SUITE_BENCHMARK_REQUESTS = 32
QUICK_SUITE_SUITE_ID = "llmtrace_quick_v1"
QUICK_SUITE_SUITE_VERSION = "0.1.0"


def get_quick_suite_completion_options() -> CompletionOptions:
    """Return the canonical Quick Suite CompletionOptions."""
    return CompletionOptions(**QUICK_SUITE_GENERATION_CONFIG)  # type: ignore[arg-type]


def get_quick_suite_generation_config() -> dict[str, float | int]:
    """Return a copy of the canonical Quick Suite generation config dict."""
    return dict(QUICK_SUITE_GENERATION_CONFIG)


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

_RESOURCE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "resources" / "quick_v1"

_TASK_RESOURCE_FILES: dict[str, str] = {
    "arc_challenge_quick_v1": "arc_challenge.json",
    "humaneval_quick_v1": "humaneval.json",
    "gsm8k_quick_v1": "gsm8k.json",
    "ifeval_quick_v1": "ifeval.json",
}


def _load_task_resources(task_id: str) -> dict[str, Any]:
    """Load the resource JSON for a Quick Suite task."""
    resource_file = _TASK_RESOURCE_FILES.get(task_id)
    if resource_file is None:
        raise BenchmarkAdapterError(f"Unknown Quick Suite task: {task_id}")
    path = _RESOURCE_DIR / resource_file
    if not path.exists():
        raise BenchmarkAdapterError(f"Resource file not found: {path}")
    with open(path) as f:
        return cast(dict[str, Any], json.load(f))


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------


def _extract_answer_arc(text: str) -> str | None:
    """Extract the final option letter from an ARC response.

    Looks for patterns like 'A', '(A)', 'Answer: A', etc.
    """
    # Match "(A)", "A.", "answer: A", etc.
    patterns = [
        r"\b([A-D])\b\s*$",  # single letter at end
        r"(?:answer|option)[:\s]*([A-D])\b",  # "answer: A" or "option A"
        r"\(([A-D])\)",  # "(A)"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_answer_gsm8k(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K response."""
    text_clean = text.strip()

    # Look for #### style answer: "#### 42"
    m = re.search(r"####\s*(-?\d+[.,]?\d*)", text_clean)
    if m:
        return m.group(1).replace(",", "")

    # Look for "answer is N" or "= N"
    m = re.search(r"(?:answer|result)\s*(?:is|=)\s*(-?\d+[.,]?\d*)", text_clean, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # Last number in the text
    numbers = re.findall(r"-?\d+[.,]?\d*", text_clean)
    if numbers:
        return cast("str | None", numbers[-1].replace(",", ""))

    return None


def _normalize_numeric(s: str) -> str:
    """Normalize a numeric string for comparison."""
    s = s.strip().replace(",", "").replace(" ", "")
    # Remove leading/trailing zeros sensibly
    try:
        num = float(s)
        if num == int(num):
            return str(int(num))
        return f"{num:.6g}"
    except ValueError:
        return s


def _check_ifeval_constraint(response: str, constraint: dict[str, Any]) -> bool:
    """Check a single IFEval constraint against the response."""
    ctype = constraint.get("type", "")
    args = constraint.get("args", {})

    if ctype == "keyword_frequency":
        keyword = str(args.get("keyword", "")).lower()
        count = int(args.get("count", 1))
        relation = args.get("relation", "eq")
        actual = response.lower().count(keyword)
        if relation == "eq":
            return actual == count
        if relation == "gte":
            return actual >= count
        if relation == "lte":
            return actual <= count
        return False

    if ctype == "word_count":
        min_w = int(args.get("min", 0))
        max_w = float(args.get("max", float("inf")))
        words = response.split()
        return min_w <= len(words) <= max_w

    if ctype == "forbidden_words":
        words = [str(w).lower() for w in args.get("words", [])]
        response_lower = response.lower()
        return not any(w in response_lower for w in words)

    if ctype == "sentence_count":
        count = int(args.get("count", 1))
        relation = args.get("relation", "eq")
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        actual = len(sentences)
        if relation == "eq":
            return actual == count
        if relation == "gte":
            return actual >= count
        return False

    if ctype == "start_with":
        word = str(args.get("word", ""))
        return response.strip().startswith(word)

    if ctype == "bullet_count":
        min_b = int(args.get("min", 1))
        bullets = re.findall(r"^\s*-\s+", response, re.MULTILINE)
        return len(bullets) >= min_b

    if ctype == "paragraph_count":
        count = int(args.get("count", 1))
        relation = args.get("relation", "eq")
        # Paragraphs separated by blank line(s)
        paragraphs = re.split(r"\n\s*\n", response.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        actual = len(paragraphs)
        if relation == "eq":
            return actual == count
        return False

    if ctype == "uppercase_word_count":
        count = int(args.get("count", 1))
        relation = args.get("relation", "eq")
        # Fully uppercase words (at least 2 chars, only A-Z)
        uppercases = re.findall(r"\b[A-Z]{2,}\b", response)
        actual = len(uppercases)
        if relation == "eq":
            return actual == count
        return False

    return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class QuickSuiteAdapter(BenchmarkAdapter):
    """Adapter for the 32-item LLMTrace Quick Suite v1.

    Handles all four task types:
      - ARC-Challenge Quick 8 (multiple choice reasoning)
      - HumanEval Quick 8 (coding, pass@1)
      - GSM8K Quick 8 (math, exact match)
      - IFEval Quick 8 (instruction following, constraint-based)

    Each task produces exactly 8 BenchmarkItemResults with proper
    source_sample_id and input_sha256 identity.
    """

    _ADAPTER_ID = "llmtrace-quick-v1"
    _ADAPTER_VERSION = "0.1.0"

    def __init__(
        self,
        *,
        code_backend: CodeExecutionBackend | None = None,
    ) -> None:
        # Fail closed: without an explicitly injected backend, only a secure
        # sandbox (Docker) is acceptable — never an in-process executor.
        self._code_backend = code_backend if code_backend is not None else create_code_execution_backend()

    # -- Properties ---------------------------------------------------------

    @property
    def adapter_id(self) -> str:
        return self._ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return self._ADAPTER_VERSION

    def get_task_definition(self, task_id: str) -> BenchmarkTaskDefinition:
        """Return the canonical task definition (per-task provenance source)."""
        task_def = _QUICK_TASK_DEFS.get(task_id)
        if task_def is None:
            raise BenchmarkAdapterError(f"Unknown Quick Suite task: {task_id}")
        return task_def

    def validate_resources(self) -> None:
        """Preflight: every task resource must exist with the exact item count."""
        for spec in self.list_tasks():
            resources = _load_task_resources(spec.task_id)
            items = resources.get("items", [])
            if len(items) != spec.num_samples:
                raise BenchmarkAdapterError(
                    f"Task {spec.task_id} has {len(items)} items but num_samples={spec.num_samples}"
                )

    # -- Task listing -------------------------------------------------------

    def list_tasks(self) -> list[TaskSpec]:
        return list(_QUICK_TASK_SPECS.values())

    # -- Planning -----------------------------------------------------------

    def build_plan(
        self,
        suite_id: str,
        suite_version: str,
        source_id: str,
        source_revision: str,
        task_ids: list[str],
    ) -> RunPlan:
        total_samples = 0
        for tid in task_ids:
            spec = _QUICK_TASK_SPECS.get(tid)
            if spec is None:
                raise BenchmarkAdapterError(f"Unknown task: {tid}")
            total_samples += spec.num_samples

        plan_id = hashlib.sha256(
            json.dumps([suite_id, suite_version, source_id, source_revision, task_ids]).encode()
        ).hexdigest()

        budget = BudgetEstimate(
            planned_requests=total_samples,
            maximum_requests=total_samples,
            maximum_retries=0,
        )

        return RunPlan(
            plan_id=plan_id,
            suite_id=suite_id,
            suite_version=suite_version,
            source_id=source_id,
            source_revision=source_revision,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            task_ids=list(task_ids),
            total_samples=total_samples,
            budget=budget,
        )

    def estimate_budget(
        self,
        suite_id: str,
        task_ids: list[str],
        max_retries: int = 0,
    ) -> BudgetEstimate:
        total = 0
        for tid in task_ids:
            spec = _QUICK_TASK_SPECS.get(tid)
            if spec:
                total += spec.num_samples
        return BudgetEstimate(
            planned_requests=total,
            maximum_requests=total * (1 + max_retries),
            maximum_retries=max_retries,
        )

    # -- Task execution -----------------------------------------------------

    async def run_task(
        self,
        task_spec: TaskSpec,
        provider: BaseProvider,
    ) -> TaskAttempt:
        """Execute a Quick Suite task against the given provider."""
        task_id = task_spec.task_id
        task_def = _QUICK_TASK_DEFS.get(task_id)
        if task_def is None:
            raise BenchmarkAdapterError(f"Unknown Quick Suite task: {task_id}")

        resources = _load_task_resources(task_id)
        items_data = resources.get("items", [])
        if len(items_data) != task_spec.num_samples:
            raise BenchmarkAdapterError(
                f"Task {task_id} has {len(items_data)} items but num_samples={task_spec.num_samples}"
            )

        attempt_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        evidence_refs: list[str] = []
        item_results: list[BenchmarkItemResult] = []
        task_failure: AdapterFailure | None = None

        for idx, item_data in enumerate(items_data):
            prompt = item_data["prompt"]
            source_sample_id = item_data["source_sample_id"]
            input_sha256 = item_data["input_sha256"]
            item_id = f"{task_id}:item-{idx:03d}"

            try:
                options = get_quick_suite_completion_options()

                evidence = await provider.complete(
                    model=provider.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options,
                )

                if isinstance(evidence, HTTPEvidence):
                    evidence_id = str(evidence.evidence_id)
                    evidence_refs.append(evidence_id)

                    if evidence.http_status and evidence.http_status >= 400:
                        item_failure = AdapterFailure(
                            error_code=f"HTTP_{evidence.http_status}",
                            category=FailureCategory.PROVIDER,
                            message=evidence.response_text[:500] or f"HTTP {evidence.http_status}",
                            retryable=evidence.http_status in (429, 503),
                        )
                        item_results.append(
                            BenchmarkItemResult(
                                item_id=item_id,
                                task_id=task_id,
                                attempt_id=attempt_id,
                                source_sample_id=source_sample_id,
                                input_sha256=input_sha256,
                                status=ItemStatus.FAILURE,
                                raw_score=0.0,
                                normalized_score=0.0,
                                failure=item_failure,
                                evidence_refs=[evidence_id],
                                grader_id="quick-suite",
                            )
                        )
                        continue

                    response_text = evidence.response_text or ""

                    # Grade based on task type
                    item_result = self._grade_item(
                        task_id=task_id,
                        item_id=item_id,
                        attempt_id=attempt_id,
                        item_data=item_data,
                        source_sample_id=source_sample_id,
                        input_sha256=input_sha256,
                        response_text=response_text,
                        evidence_id=evidence_id,
                    )
                    item_results.append(item_result)
                else:
                    item_results.append(
                        BenchmarkItemResult(
                            item_id=item_id,
                            task_id=task_id,
                            attempt_id=attempt_id,
                            source_sample_id=source_sample_id,
                            input_sha256=input_sha256,
                            status=ItemStatus.FAILURE,
                            raw_score=0.0,
                            normalized_score=0.0,
                            failure=AdapterFailure(
                                error_code="UNEXPECTED_EVIDENCE_TYPE",
                                category=FailureCategory.ADAPTER,
                                message=f"Unexpected evidence type: {type(evidence).__name__}",
                            ),
                            grader_id="quick-suite",
                        )
                    )

            except Exception as exc:
                item_failure = AdapterFailure(
                    error_code=type(exc).__name__.upper(),
                    category=FailureCategory.PROVIDER,
                    message=str(exc)[:500],
                    retryable=False,
                )
                item_results.append(
                    BenchmarkItemResult(
                        item_id=item_id,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        source_sample_id=source_sample_id,
                        input_sha256=input_sha256,
                        status=ItemStatus.FAILURE,
                        raw_score=0.0,
                        normalized_score=0.0,
                        failure=item_failure,
                        grader_id="quick-suite",
                    )
                )

        finished_at = datetime.now(UTC)

        return TaskAttempt(
            attempt_id=attempt_id,
            task_id=task_id,
            status=TaskStatus.SUCCESS,
            evidence_refs=evidence_refs,
            item_results=item_results,
            failure=task_failure,
            started_at=started_at,
            finished_at=finished_at,
            source_id=task_def.source_id,
            source_revision=task_def.source_revision,
            suite_id=task_def.suite_id,
            suite_version=task_def.suite_version,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            metadata=task_def.task_metadata(),
        )

    # -- Per-item grading ---------------------------------------------------

    def _grade_item(
        self,
        *,
        task_id: str,
        item_id: str,
        attempt_id: str,
        item_data: dict[str, Any],
        source_sample_id: str,
        input_sha256: str,
        response_text: str,
        evidence_id: str,
    ) -> BenchmarkItemResult:
        """Grade a single item based on task type."""
        if task_id == "arc_challenge_quick_v1":
            return self._grade_arc(
                item_id,
                attempt_id,
                item_data,
                source_sample_id,
                input_sha256,
                response_text,
                evidence_id,
            )
        if task_id == "gsm8k_quick_v1":
            return self._grade_gsm8k(
                item_id,
                attempt_id,
                item_data,
                source_sample_id,
                input_sha256,
                response_text,
                evidence_id,
            )
        if task_id == "humaneval_quick_v1":
            return self._grade_humaneval(
                item_id,
                attempt_id,
                item_data,
                source_sample_id,
                input_sha256,
                response_text,
                evidence_id,
            )
        if task_id == "ifeval_quick_v1":
            return self._grade_ifeval(
                item_id,
                attempt_id,
                item_data,
                source_sample_id,
                input_sha256,
                response_text,
                evidence_id,
            )
        raise BenchmarkAdapterError(f"No grader for task: {task_id}")

    def _grade_arc(
        self,
        item_id: str,
        attempt_id: str,
        item_data: dict[str, Any],
        source_sample_id: str,
        input_sha256: str,
        response_text: str,
        evidence_id: str,
    ) -> BenchmarkItemResult:
        expected = item_data["answer"].strip().upper()
        extracted = _extract_answer_arc(response_text)
        if extracted is None:
            return BenchmarkItemResult(
                item_id=item_id,
                task_id="arc_challenge_quick_v1",
                attempt_id=attempt_id,
                source_sample_id=source_sample_id,
                input_sha256=input_sha256,
                status=ItemStatus.UNGRADABLE,
                raw_score=0.0,
                normalized_score=0.0,
                error_message="Could not extract option letter from response",
                evidence_refs=[evidence_id],
                grader_id="arc-grader",
            )
        score = 1.0 if extracted == expected else 0.0
        return BenchmarkItemResult(
            item_id=item_id,
            task_id="arc_challenge_quick_v1",
            attempt_id=attempt_id,
            source_sample_id=source_sample_id,
            input_sha256=input_sha256,
            status=ItemStatus.GRADED,
            raw_score=score,
            normalized_score=score,
            evidence_refs=[evidence_id],
            grader_id="arc-grader",
            metadata={"expected": expected, "extracted": extracted},
        )

    def _grade_gsm8k(
        self,
        item_id: str,
        attempt_id: str,
        item_data: dict[str, Any],
        source_sample_id: str,
        input_sha256: str,
        response_text: str,
        evidence_id: str,
    ) -> BenchmarkItemResult:
        expected = _normalize_numeric(str(item_data["answer"]))
        extracted = _extract_answer_gsm8k(response_text)
        if extracted is None:
            return BenchmarkItemResult(
                item_id=item_id,
                task_id="gsm8k_quick_v1",
                attempt_id=attempt_id,
                source_sample_id=source_sample_id,
                input_sha256=input_sha256,
                status=ItemStatus.UNGRADABLE,
                raw_score=0.0,
                normalized_score=0.0,
                error_message="Could not extract numeric answer from response",
                evidence_refs=[evidence_id],
                grader_id="gsm8k-grader",
            )
        extracted_norm = _normalize_numeric(extracted)
        score = 1.0 if extracted_norm == expected else 0.0
        return BenchmarkItemResult(
            item_id=item_id,
            task_id="gsm8k_quick_v1",
            attempt_id=attempt_id,
            source_sample_id=source_sample_id,
            input_sha256=input_sha256,
            status=ItemStatus.GRADED,
            raw_score=score,
            normalized_score=score,
            evidence_refs=[evidence_id],
            grader_id="gsm8k-grader",
            metadata={"expected": expected, "extracted": extracted_norm},
        )

    def _grade_humaneval(
        self,
        item_id: str,
        attempt_id: str,
        item_data: dict[str, Any],
        source_sample_id: str,
        input_sha256: str,
        response_text: str,
        evidence_id: str,
    ) -> BenchmarkItemResult:
        # Check sandbox availability
        if not self._code_backend.is_available():
            return BenchmarkItemResult(
                item_id=item_id,
                task_id="humaneval_quick_v1",
                attempt_id=attempt_id,
                source_sample_id=source_sample_id,
                input_sha256=input_sha256,
                status=ItemStatus.FAILURE,
                raw_score=0.0,
                normalized_score=0.0,
                failure=AdapterFailure(
                    error_code="SANDBOX_UNAVAILABLE",
                    category=FailureCategory.ADAPTER,
                    message="Code execution sandbox is not available",
                    retryable=False,
                ),
                evidence_refs=[evidence_id],
                grader_id="humaneval-grader",
            )

        # Extract the generated code (everything after prompt is the completion)
        code = response_text

        tests = item_data.get("tests", [])
        test_code = "\n".join(tests)

        # Assemble final code: user function + tests
        final_code = f"{code}\n\n# --- tests ---\n{test_code}"

        try:
            result = self._code_backend.execute(final_code, timeout_seconds=10.0)
        except SandboxUnavailableError:
            return BenchmarkItemResult(
                item_id=item_id,
                task_id="humaneval_quick_v1",
                attempt_id=attempt_id,
                source_sample_id=source_sample_id,
                input_sha256=input_sha256,
                status=ItemStatus.FAILURE,
                raw_score=0.0,
                normalized_score=0.0,
                failure=AdapterFailure(
                    error_code="SANDBOX_UNAVAILABLE",
                    category=FailureCategory.ADAPTER,
                    message="Sandbox became unavailable during execution",
                    retryable=False,
                ),
                evidence_refs=[evidence_id],
                grader_id="humaneval-grader",
            )

        if result.timed_out:
            return BenchmarkItemResult(
                item_id=item_id,
                task_id="humaneval_quick_v1",
                attempt_id=attempt_id,
                source_sample_id=source_sample_id,
                input_sha256=input_sha256,
                status=ItemStatus.GRADED,
                raw_score=0.0,
                normalized_score=0.0,
                evidence_refs=[evidence_id],
                grader_id="humaneval-grader",
                metadata={"timeout": True, "timeout_seconds": 10.0},
            )

        score = 1.0 if result.success else 0.0
        return BenchmarkItemResult(
            item_id=item_id,
            task_id="humaneval_quick_v1",
            attempt_id=attempt_id,
            source_sample_id=source_sample_id,
            input_sha256=input_sha256,
            status=ItemStatus.GRADED,
            raw_score=score,
            normalized_score=score,
            evidence_refs=[evidence_id],
            grader_id="humaneval-grader",
            metadata={"tests_passed": score, "stderr": result.stderr[:200] if not result.success else ""},
        )

    def _grade_ifeval(
        self,
        item_id: str,
        attempt_id: str,
        item_data: dict[str, Any],
        source_sample_id: str,
        input_sha256: str,
        response_text: str,
        evidence_id: str,
    ) -> BenchmarkItemResult:
        constraints = item_data.get("constraints", [])
        if not constraints:
            return BenchmarkItemResult(
                item_id=item_id,
                task_id="ifeval_quick_v1",
                attempt_id=attempt_id,
                source_sample_id=source_sample_id,
                input_sha256=input_sha256,
                status=ItemStatus.UNGRADABLE,
                raw_score=0.0,
                normalized_score=0.0,
                error_message="No constraints defined for this item",
                evidence_refs=[evidence_id],
                grader_id="ifeval-grader",
            )

        total = len(constraints)
        satisfied = 0
        constraint_results: list[dict[str, Any]] = []
        for c in constraints:
            result = _check_ifeval_constraint(response_text, c)
            if result:
                satisfied += 1
            constraint_results.append({"constraint_id": c.get("id"), "satisfied": result})

        score = satisfied / total if total > 0 else 0.0
        return BenchmarkItemResult(
            item_id=item_id,
            task_id="ifeval_quick_v1",
            attempt_id=attempt_id,
            source_sample_id=source_sample_id,
            input_sha256=input_sha256,
            status=ItemStatus.GRADED,
            raw_score=score,
            normalized_score=score,
            evidence_refs=[evidence_id],
            grader_id="ifeval-grader",
            metadata={
                "constraint_count": total,
                "satisfied_count": satisfied,
                "constraint_results": constraint_results,
            },
        )

    # -- Result normalization -----------------------------------------------

    def normalize_result(self, raw_result: dict[str, object]) -> GradeResult:
        """Normalize Quick Suite results to a GradeResult.

        Quick Suite always uses ItemResults → aggregate → GradeResult.
        """
        task_id = str(raw_result.get("task_name", raw_result.get("task_id", "")))
        attempt_id = str(raw_result.get("attempt_id", ""))
        item_results_raw = raw_result.get("item_results")
        planned_item_count_raw = raw_result.get("planned_item_count")

        task_def = _QUICK_TASK_DEFS.get(task_id)
        if task_def is None:
            raise BenchmarkAdapterError(f"Unknown Quick Suite task: {task_id}")

        if not item_results_raw or not planned_item_count_raw:
            if task_def.requires_item_results:
                raise ValueError(
                    f"ITEM_RESULTS_REQUIRED: task '{task_id}' requires item_results and planned_item_count"
                )
            # Fallback for non-item tasks (shouldn't happen in Quick Suite)
            return GradeResult(
                grade_id=str(uuid.uuid4()),
                task_id=task_id,
                attempt_id=attempt_id,
                grader_id="quick-suite",
                status=GradeStatus.UNGRADABLE,
                raw_score=0.0,
                normalized_score=0.0,
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
            )

        # Parse item results
        items = []
        if isinstance(item_results_raw, list):
            for ir in item_results_raw:
                if isinstance(ir, dict):
                    items.append(BenchmarkItemResult(**ir))
                elif isinstance(ir, BenchmarkItemResult):
                    items.append(ir)

        planned = int(planned_item_count_raw) if isinstance(planned_item_count_raw, (int, float, str)) else 0

        # Aggregate
        agg = aggregate_item_results(items, planned_item_count=planned)

        lm_eval_results = raw_result.get("results", {})

        return GradeResult(
            grade_id=str(uuid.uuid4()),
            task_id=task_id,
            attempt_id=attempt_id,
            grader_id="quick-suite",
            status=GradeStatus.GRADED if agg.normalized_score is not None else GradeStatus.UNGRADABLE,
            raw_score=agg.normalized_score or 0.0,
            normalized_score=agg.normalized_score or 0.0,
            source_id=task_def.source_id,
            source_revision=task_def.source_revision,
            suite_id=task_def.suite_id,
            suite_version=task_def.suite_version,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            metadata={
                "source_id": task_def.source_id,  # per-task source, not suite-level
                "item_aggregate_score": agg.normalized_score,
                "planned_item_count": planned,
                "graded_item_count": agg.graded_item_count,
                "failure_count": agg.failure_count,
                "ungradable_count": agg.ungradable_count,
                "grading_coverage": agg.grading_coverage,
                "execution_coverage": agg.execution_coverage,
                "lm_eval_metric": lm_eval_results,
            },
        )


# ---------------------------------------------------------------------------
# Quick Suite v1 — TaskScoringRegistry factory
# ---------------------------------------------------------------------------


def create_quick_registry() -> TaskScoringRegistry:
    """Create a TaskScoringRegistry with all four Quick Suite v1 tasks.

    Explicit mapping required by spec section 18:
      arc_challenge_quick_v1  → reasoning,               weight 1.0
      humaneval_quick_v1       → coding,                  weight 1.0
      gsm8k_quick_v1           → math_science,            weight 1.0
      ifeval_quick_v1          → instruction_following,   weight 1.0

    All tasks are capability_score_eligible=True.

    Returns:
        TaskScoringRegistry: Pre-configured registry for Quick Suite v1 tasks.
    """
    from llmtrace.scoring.aggregator import TaskScoringRegistry
    from llmtrace.scoring.models import CapabilityDimension, TaskScoringSpec

    return TaskScoringRegistry(
        specs=[
            TaskScoringSpec(
                task_id="arc_challenge_quick_v1",
                dimension=CapabilityDimension.REASONING,
                task_weight=1.0,
                capability_score_eligible=True,
                source_id="arc_challenge",
                suite_id="llmtrace_quick_v1",
            ),
            TaskScoringSpec(
                task_id="humaneval_quick_v1",
                dimension=CapabilityDimension.CODING,
                task_weight=1.0,
                capability_score_eligible=True,
                source_id="humaneval",
                suite_id="llmtrace_quick_v1",
            ),
            TaskScoringSpec(
                task_id="gsm8k_quick_v1",
                dimension=CapabilityDimension.MATH_SCIENCE,
                task_weight=1.0,
                capability_score_eligible=True,
                source_id="gsm8k",
                suite_id="llmtrace_quick_v1",
            ),
            TaskScoringSpec(
                task_id="ifeval_quick_v1",
                dimension=CapabilityDimension.INSTRUCTION_FOLLOWING,
                task_weight=1.0,
                capability_score_eligible=True,
                source_id="ifeval",
                suite_id="llmtrace_quick_v1",
            ),
        ]
    )
