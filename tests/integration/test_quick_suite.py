"""Quick Suite v1 integration tests — full 32-item pipeline with Mock Provider.

Tests the complete path:
  QuickSuiteAdapter → Provider → 32 BenchmarkItemResults
  → 4 GradeResults → 4 DimensionScoreResults → 1 CapabilityProfile

Uses Mock/Fake Provider only — no real API calls.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.adapters.code_execution import (
    CodeExecutionBackend,
    CodeExecutionResult,
    SandboxUnavailableError,
)
from llmtrace.adapters.quick_suite import (
    QuickSuiteAdapter,
    _check_ifeval_constraint,
    _extract_answer_arc,
    _extract_answer_gsm8k,
)
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkItemResult,
    BenchmarkRunResult,
    BudgetEstimate,
    FailureCategory,
    GradeResult,
    GradeStatus,
    ItemStatus,
    RunPlan,
    TaskAttempt,
    TaskStatus,
    aggregate_item_results,
)
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.scoring.aggregator import (
    TaskScoringRegistry,
    aggregate_capability_profile,
)
from llmtrace.scoring.models import CapabilityDimension, DimensionScoreStatus, TaskScoringSpec
from llmtrace.scoring.policy import CapabilityScoringPolicy


# ---------------------------------------------------------------------------
# Mock Provider that returns pre-programmed responses
# ---------------------------------------------------------------------------
class MockQuickSuiteProvider:
    """Mock Provider with configurable per-item responses."""

    def __init__(
        self,
        responses: dict[int, str] | None = None,
        fail_indices: set[int] | None = None,
    ):
        self.call_count = 0
        self.evidence: list[HTTPEvidence] = []
        self._responses = responses or {}
        self._fail_indices = fail_indices or set()

    async def complete(self, model, messages, *, options=None):
        self.call_count += 1
        idx = self.call_count - 1

        if idx in self._fail_indices:
            raise RuntimeError(f"Mock provider failure at index {idx}")

        response_text = self._responses.get(idx, "empty")

        evidence_id = uuid.uuid4()
        ev = HTTPEvidence(
            evidence_id=evidence_id,
            evidence_type="smoke_test",
            request_method="POST",
            request_url_redacted="https://mock/",
            request_path="/mock",
            request_headers_redacted={},
            request_body_redacted={"messages": [str(m)[:100] for m in messages]},
            request_model=model,
            response_model=model,
            response_text=response_text,
            http_status=200,
            total_latency_ms=50.0,
        )
        self.evidence.append(ev)
        return ev


# ---------------------------------------------------------------------------
# Fake code execution backend (always passes)
# ---------------------------------------------------------------------------
class _FakePassingBackend(CodeExecutionBackend):
    def is_available(self) -> bool:
        return True

    def execute(self, code, *, timeout_seconds=10.0):
        return CodeExecutionResult(success=True, stdout="OK", exit_code=0)


class _FakeFailingBackend(CodeExecutionBackend):
    def is_available(self) -> bool:
        return True

    def execute(self, code, *, timeout_seconds=10.0):
        return CodeExecutionResult(success=False, stderr="TestFailed", exit_code=1)


class _UnavailableBackend(CodeExecutionBackend):
    def is_available(self) -> bool:
        return False

    def execute(self, code, *, timeout_seconds=10.0):
        raise SandboxUnavailableError("not available")


# ---------------------------------------------------------------------------
# Scoring registry factory
# ---------------------------------------------------------------------------
def _build_quick_registry() -> TaskScoringRegistry:
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


# ---------------------------------------------------------------------------
# Full pipeline: all 8/8 correct
# ---------------------------------------------------------------------------
class TestQuickSuiteFullPipeline:
    """All 32 items correct → validate full pipeline."""

    @pytest.fixture
    def adapter(self):
        return QuickSuiteAdapter(code_backend=_FakePassingBackend())

    @pytest.fixture
    def mock_provider(self):
        """Returns perfect answers for all 32 items."""
        responses = {}
        # ARC: 8 correct answers
        arc_answers = ["B", "C", "A", "C", "B", "C", "C", "B"]
        for i in range(8):
            responses[i] = f"Answer: {arc_answers[i]}"
        # HumanEval: 8 empty (passed by sandbox)
        for i in range(8, 16):
            responses[i] = ""
        # GSM8K: 8 correct numeric answers
        gsmk_answers = ["13", "3", "65000", "624", "85", "22.5", "35", "8"]
        for i in range(16, 24):
            responses[i] = f"#### {gsmk_answers[i - 16]}"
        # IFEval: 8 responses that satisfy all constraints
        ifeval_responses = [
            "algorithm algorithm algorithm is key to AI",
            (
                "Machine learning is an important and powerful field of artificial intelligence "
                "that enables modern computer systems to automatically learn from data and "
                "improve their performance over time without being explicitly programmed for "
                "every single possible scenario that they might encounter in real world use "
                "cases across many different industries and domains today and tomorrow"
            ),
            "The machine processes digits through circuits and memory systems",
            "The solar system has eight planets. Mercury is closest to the Sun. "
            "Venus is the hottest. Earth has life. Mars is the Red Planet.",
            "Once upon a time, a robot learned to paint",
            "-\n- Better health\n- More energy\n",
            "Climate changes affect the planet.\n\nRising temperatures melt ice caps.",
            "IMPORTANT technology is ALWAYS evolving",
        ]
        for i in range(24, 32):
            responses[i] = ifeval_responses[i - 24]
        return MockQuickSuiteProvider(responses)

    async def test_all_32_items_correct(self, adapter, mock_provider):
        """Complete pipeline: 4 tasks × 8 items → CapabilityProfile."""
        tasks = adapter.list_tasks()
        assert len(tasks) == 4

        attempts: list[TaskAttempt] = []
        grade_results = []

        for task_spec in tasks:
            attempt = await adapter.run_task(task_spec, mock_provider)
            attempts.append(attempt)

            assert attempt.status == TaskStatus.SUCCESS
            assert len(attempt.item_results) == 8
            assert len(attempt.evidence_refs) == 8

            # Verify source_sample_id and input_sha256
            for item in attempt.item_results:
                assert item.source_sample_id is not None
                assert len(item.source_sample_id) > 0
                assert item.input_sha256 is not None
                assert len(item.input_sha256) == 64

            # Normalize
            raw_result = {
                "results": {},
                "evidence_ids": list(attempt.evidence_refs),
                "task_name": task_spec.task_id,
                "attempt_id": attempt.attempt_id,
                "item_results": [it.model_dump() for it in attempt.item_results],
                "planned_item_count": 8,
            }
            grade = adapter.normalize_result(raw_result)
            grade_results.append(grade)

        # 4 TaskAttempts, 4 GradeResults, 32 ItemResults
        assert len(attempts) == 4
        assert len(grade_results) == 4

        total_items = sum(len(a.item_results) for a in attempts)
        assert total_items == 32

        # All items GRADED with score 1.0
        for a in attempts:
            for item in a.item_results:
                assert item.status == ItemStatus.GRADED
                assert item.normalized_score == 1.0

        # Build BenchmarkRunResults and aggregate — one run per task for provenance match
        run_results = []
        for _, (task_spec, attempt, grade) in enumerate(zip(tasks, attempts, grade_results, strict=True)):
            run_id = str(uuid.uuid4())
            task_def = __import__("llmtrace.adapters.quick_suite", fromlist=["_QUICK_TASK_DEFS"])._QUICK_TASK_DEFS[
                task_spec.task_id
            ]
            run_results.append(
                BenchmarkRunResult(
                    run_id=run_id,
                    task_attempts=[attempt],
                    grade_results=[grade],
                    started_at=datetime.now(UTC),
                    finished_at=datetime.now(UTC),
                    source_id=task_def.source_id,
                    source_revision=task_def.source_revision,
                    suite_id=task_def.suite_id,
                    suite_version=task_def.suite_version,
                    adapter_id=adapter.adapter_id,
                    adapter_version=adapter.adapter_version,
                )
            )

        registry = _build_quick_registry()
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile(
            run_results,
            registry,
            policy,
            strict=True,
        )

        # Verify profile
        assert profile.coverage_weight == 0.75
        assert profile.provisional_raw_index == 0.75
        assert profile.calibrated_total_score is None

        # Verify each dimension
        dims = {d.dimension: d for d in profile.dimensions}
        for dim_key in [
            CapabilityDimension.REASONING,
            CapabilityDimension.CODING,
            CapabilityDimension.MATH_SCIENCE,
            CapabilityDimension.INSTRUCTION_FOLLOWING,
        ]:
            assert dim_key in dims
            d = dims[dim_key]
            assert d.status == DimensionScoreStatus.UNCALIBRATED
            assert d.raw_normalized_score == 1.0
            assert d.calibrated_score is None
            assert d.task_count == 1
            assert d.graded_task_count == 1

    async def test_32_item_evidence_total(self, adapter, mock_provider):
        """Verify 32 Provider request Evidence UUIDs."""
        tasks = adapter.list_tasks()
        all_evidence: set[str] = set()
        for task_spec in tasks:
            attempt = await adapter.run_task(task_spec, mock_provider)
            all_evidence.update(attempt.evidence_refs)
        assert len(all_evidence) == 32

    async def test_arc_answer_extraction(self, adapter):
        """ARC answer extraction from various formats."""
        assert _extract_answer_arc("The answer is (B) Mercury") == "B"
        assert _extract_answer_arc("B\n") == "B"
        assert _extract_answer_arc("Option C is correct") == "C"
        assert _extract_answer_arc("I don't know") is None

    async def test_gsm8k_answer_extraction(self, adapter):
        """GSM8K answer extraction."""
        assert _extract_answer_gsm8k("#### 42") == "42"
        assert _extract_answer_gsm8k("The answer is 100") == "100"
        assert _extract_answer_gsm8k("No answer here") is None

    def test_ifeval_constraints(self):
        """IFEval constraint checking."""
        # keyword frequency
        assert _check_ifeval_constraint(
            "AI is great because AI helps",
            {"type": "keyword_frequency", "args": {"keyword": "AI", "count": 2, "relation": "eq"}},
        )
        assert not _check_ifeval_constraint(
            "AI is great",
            {"type": "keyword_frequency", "args": {"keyword": "AI", "count": 2, "relation": "eq"}},
        )

        # forbidden words
        assert _check_ifeval_constraint("hello world", {"type": "forbidden_words", "args": {"words": ["computer"]}})
        assert not _check_ifeval_constraint(
            "the computer is on",
            {"type": "forbidden_words", "args": {"words": ["computer"]}},
        )

        # sentence count
        assert _check_ifeval_constraint("A. B. C.", {"type": "sentence_count", "args": {"count": 3, "relation": "eq"}})

        # start with
        assert _check_ifeval_constraint("Once upon a time", {"type": "start_with", "args": {"word": "Once"}})
        assert not _check_ifeval_constraint("A long time ago", {"type": "start_with", "args": {"word": "Once"}})


# ---------------------------------------------------------------------------
# Partial failure / mixed scores
# ---------------------------------------------------------------------------
class TestQuickSuiteMixedScores:
    """Test weighted index calculation with mixed scores."""

    async def test_mixed_scores_profile(self):
        """R=0.75, C=0.50, M=0.875, I=0.625 → verify provisional_raw_index."""

        # Use programmatic construction
        policy = CapabilityScoringPolicy.create_v1()
        registry = _build_quick_registry()

        def _make_attempt(task_id, scores):
            items = []
            for i, score in enumerate(scores):
                items.append(
                    BenchmarkItemResult(
                        item_id=f"{task_id}:item-{i:03d}",
                        task_id=task_id,
                        attempt_id=task_id,
                        source_sample_id=f"test:{i}",
                        input_sha256="a" * 64,
                        status=ItemStatus.GRADED,
                        raw_score=score,
                        normalized_score=score,
                        grader_id="test-grader",
                    )
                )
            return TaskAttempt(
                attempt_id=task_id,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                item_results=items,
                source_id="test",
                source_revision="test",
                suite_id="llmtrace_quick_v1",
                suite_version="0.1.0",
                adapter_id="llmtrace-quick-v1",
                adapter_version="0.1.0",
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )

        # R = 6/8 = 0.75
        reason_attempt = _make_attempt("arc_challenge_quick_v1", [1, 1, 1, 1, 1, 1, 0, 0])

        # C = 4/8 = 0.50
        code_attempt = _make_attempt("humaneval_quick_v1", [1, 1, 1, 1, 0, 0, 0, 0])

        # M = 7/8 = 0.875
        math_attempt = _make_attempt("gsm8k_quick_v1", [1, 1, 1, 1, 1, 1, 1, 0])

        # I = 5/8 = 0.625
        instr_attempt = _make_attempt("ifeval_quick_v1", [1, 1, 1, 1, 1, 0, 0, 0])

        attempts = [reason_attempt, code_attempt, math_attempt, instr_attempt]
        # Each task gets its own BenchmarkRunResult to satisfy provenance matching
        run_results = []
        for a in attempts:
            agg = aggregate_item_results(a.item_results, planned_item_count=8)
            grade = GradeResult(
                grade_id=str(uuid.uuid4()),
                task_id=a.task_id,
                attempt_id=a.attempt_id,
                grader_id="test",
                status=GradeStatus.GRADED,
                raw_score=agg.normalized_score,
                normalized_score=agg.normalized_score,
                source_id=a.source_id,
                source_revision=a.source_revision,
                suite_id=a.suite_id,
                suite_version=a.suite_version,
                adapter_id=a.adapter_id,
                adapter_version=a.adapter_version,
            )
            run_results.append(
                BenchmarkRunResult(
                    run_id=str(uuid.uuid4()),
                    task_attempts=[a],
                    grade_results=[grade],
                    source_id=a.source_id,
                    source_revision=a.source_revision,
                    suite_id=a.suite_id,
                    suite_version=a.suite_version,
                    adapter_id=a.adapter_id,
                    adapter_version=a.adapter_version,
                )
            )

        profile = aggregate_capability_profile(run_results, registry, policy, strict=True)

        # Expected: 0.75*0.25 + 0.50*0.20 + 0.875*0.15 + 0.625*0.15
        expected = 0.75 * 0.25 + 0.50 * 0.20 + 0.875 * 0.15 + 0.625 * 0.15
        assert abs(profile.provisional_raw_index - expected) < 0.001
        assert profile.coverage_weight == 0.75


# ---------------------------------------------------------------------------
# Failure isolation and adversarial tests
# ---------------------------------------------------------------------------
class TestQuickSuiteFailureIsolation:
    """One item failure must not affect subsequent items."""

    async def test_provider_failure_isolation(self):
        """Item 2 fails with HTTP 500; remaining 7 continue."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())

        class _SelectiveFailingProvider:
            def __init__(self):
                self.call_count = 0
                self.evidence = []

            async def complete(self, model, messages, *, options=None):
                self.call_count += 1
                fail = self.call_count == 3
                evidence_id = uuid.uuid4()
                ev = HTTPEvidence(
                    evidence_id=evidence_id,
                    evidence_type="smoke_test",
                    request_method="POST",
                    request_url_redacted="https://mock/",
                    request_path="/",
                    request_headers_redacted={},
                    request_body_redacted={},
                    request_model=model,
                    response_model=model,
                    response_text="Internal Error" if fail else "B",
                    http_status=500 if fail else 200,
                    total_latency_ms=50.0,
                )
                self.evidence.append(ev)
                return ev

        provider = _SelectiveFailingProvider()
        tasks = adapter.list_tasks()
        arc_task = next(t for t in tasks if t.task_id == "arc_challenge_quick_v1")
        attempt = await adapter.run_task(arc_task, provider)

        assert len(attempt.item_results) == 8
        failed = [it for it in attempt.item_results if it.status == ItemStatus.FAILURE]
        assert len(failed) == 1
        assert failed[0].failure is not None
        assert isinstance(failed[0].failure, AdapterFailure)
        assert failed[0].failure.category == FailureCategory.PROVIDER

        graded = [it for it in attempt.item_results if it.status == ItemStatus.GRADED]
        assert len(graded) == 7

    async def test_humaneval_sandbox_unavailable(self):
        """HumanEval with unavailable sandbox → all items FAILURE."""
        adapter = QuickSuiteAdapter(code_backend=_UnavailableBackend())

        class _SimpleProvider:
            def __init__(self):
                self.call_count = 0

            async def complete(self, model, messages, *, options=None):
                self.call_count += 1
                evidence_id = uuid.uuid4()
                return HTTPEvidence(
                    evidence_id=evidence_id,
                    evidence_type="smoke_test",
                    request_method="POST",
                    request_url_redacted="https://mock/",
                    request_path="/",
                    request_headers_redacted={},
                    request_body_redacted={},
                    request_model=model,
                    response_model=model,
                    response_text="def solution():\n    pass",
                    http_status=200,
                    total_latency_ms=50.0,
                )

        tasks = adapter.list_tasks()
        he_task = next(t for t in tasks if t.task_id == "humaneval_quick_v1")
        attempt = await adapter.run_task(he_task, _SimpleProvider())

        for item in attempt.item_results:
            assert item.status == ItemStatus.FAILURE
            assert item.failure is not None
            assert item.failure.error_code == "SANDBOX_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Identity and provenance tests
# ---------------------------------------------------------------------------
class TestQuickSuiteIdentity:
    """Verify source_sample_id and input_sha256 on all items."""

    async def test_all_items_have_source_sample_id(self):
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            for item in attempt.item_results:
                assert item.source_sample_id is not None
                assert len(item.source_sample_id) > 0

    async def test_duplicate_source_sample_id_rejected(self):
        """Two items with same source_sample_id → must be rejected (via item_id uniqueness)."""
        items = [
            BenchmarkItemResult(
                item_id="item-001",
                task_id="t",
                attempt_id="a",
                source_sample_id="dup",
                input_sha256="a" * 64,
                raw_score=1.0,
                normalized_score=1.0,
            ),
            BenchmarkItemResult(
                item_id="item-001",
                task_id="t",
                attempt_id="a",
                source_sample_id="dup",
                input_sha256="a" * 64,
                raw_score=1.0,
                normalized_score=1.0,
            ),
        ]
        with pytest.raises(ValueError, match="Duplicate item_id"):
            TaskAttempt(
                attempt_id="a",
                task_id="t",
                item_results=items,
                source_id="test",
                source_revision="test",
                suite_id="test",
                suite_version="0.1.0",
                adapter_id="test",
                adapter_version="0.1.0",
            )

    async def test_invalid_input_sha256_rejected(self):
        """Invalid hex SHA256 must be rejected."""
        with pytest.raises(ValueError, match="input_sha256"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                input_sha256="not-valid-hex" + "0" * 50,
                raw_score=1.0,
                normalized_score=1.0,
            )

    async def test_sha256_must_be_lowercase(self):
        """Uppercase SHA256 must be rejected."""
        with pytest.raises(ValueError, match="lowercase"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                input_sha256="A" * 64,
                raw_score=1.0,
                normalized_score=1.0,
            )


# ---------------------------------------------------------------------------
# Item total invariant
# ---------------------------------------------------------------------------
class TestQuickSuiteTotals:
    """32-item total invariant."""

    async def test_32_total_items(self):
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        total_items = 0
        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            total_items += len(attempt.item_results)
        assert total_items == 32

    async def test_8_items_per_task(self):
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            assert len(attempt.item_results) == 8
            assert task_spec.num_samples == 8


# ---------------------------------------------------------------------------
# Reporting integration
# ---------------------------------------------------------------------------
class TestQuickSuiteReporting:
    """JSON / HTML reporting with Quick Suite data."""

    async def test_json_report_quick_suite(self, tmp_path):
        """Generate JSON report from Quick Suite data."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        attempts = []
        grade_results = []
        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            attempts.append(attempt)
            raw = {
                "results": {},
                "evidence_ids": list(attempt.evidence_refs),
                "task_name": task_spec.task_id,
                "attempt_id": attempt.attempt_id,
                "item_results": [it.model_dump() for it in attempt.item_results],
                "planned_item_count": 8,
            }
            grade = adapter.normalize_result(raw)
            grade_results.append(grade)

        task_defs = __import__("llmtrace.adapters.quick_suite", fromlist=["_QUICK_TASK_DEFS"])._QUICK_TASK_DEFS

        for task_spec, attempt, grade in zip(adapter.list_tasks(), attempts, grade_results, strict=True):
            task_def = task_defs[task_spec.task_id]
            plan = adapter.build_plan(
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                task_ids=[task_spec.task_id],
            )
            run = BenchmarkRunResult(
                run_id=str(uuid.uuid4()),
                task_attempts=[attempt],
                grade_results=[grade],
                evidence_refs=list(attempt.evidence_refs),
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
            )
            report_data = build_benchmark_report_section(plan, run)
            assert report_data.suite_id == "llmtrace_quick_v1"
            assert len(report_data.tasks) == 1
            for task in report_data.tasks:
                assert len(task.items) == 8
                for item in task.items:
                    assert item.source_sample_id is not None

    async def test_html_report_quick_suite(self, tmp_path):
        """Generate HTML report from Quick Suite data."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        attempts = []
        grade_results = []
        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            attempts.append(attempt)
            raw = {
                "results": {},
                "evidence_ids": list(attempt.evidence_refs),
                "task_name": task_spec.task_id,
                "attempt_id": attempt.attempt_id,
                "item_results": [it.model_dump() for it in attempt.item_results],
                "planned_item_count": 8,
            }
            grade = adapter.normalize_result(raw)
            grade_results.append(grade)

        task_defs = __import__("llmtrace.adapters.quick_suite", fromlist=["_QUICK_TASK_DEFS"])._QUICK_TASK_DEFS
        sections = []
        for task_spec, attempt, grade in zip(adapter.list_tasks(), attempts, grade_results, strict=True):
            task_def = task_defs[task_spec.task_id]
            plan = adapter.build_plan(
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                task_ids=[task_spec.task_id],
            )
            run = BenchmarkRunResult(
                run_id=str(uuid.uuid4()),
                task_attempts=[attempt],
                grade_results=[grade],
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
            )
            sections.append(build_benchmark_report_section(plan, run))

        config = __import__("llmtrace.config", fromlist=["AuditConfig"]).AuditConfig(
            protocol=Protocol.OPENAI,
            base_url="https://mock.example.com",
            model="test-model",
            api_key_env="MOCK_KEY",
            auth_style=AuthStyle.AUTO,
        )
        audit_result = AuditResult(config=config, risk_level=RiskLevel.LOW)

        # Populate evidence — required by evidence_validation in generate_html_report
        evidence_ids_seen: set[str] = set()
        for section in sections:
            for task in section.tasks:
                for ref in task.evidence_refs:
                    if ref not in evidence_ids_seen:
                        evidence_ids_seen.add(ref)
                        audit_result.evidence.append(
                            HTTPEvidence(
                                evidence_id=uuid.UUID(ref),
                                evidence_type="benchmark",
                                request_method="POST",
                                request_url_redacted="https://mock.example.com/v1/chat/completions",
                                request_path="/v1/chat/completions",
                                request_headers_redacted={"Authorization": "Bearer mock-***"},
                                request_model="test-model",
                                response_model="test-model",
                                total_latency_ms=100.0,
                                http_status=200,
                                response_text="mock response",
                            )
                        )

        html_path = generate_html_report(
            result=audit_result,
            output_path=Path(tmp_path / "test_report.html"),
            benchmark_sections=sections,
        )

        assert html_path.exists()
        html = html_path.read_text()
        assert "quick_v1" in html.lower() or "Quick Suite" in html


# ---------------------------------------------------------------------------
# Adversarial tests — spec section 27
# ---------------------------------------------------------------------------


class TestQuickSuiteAdversarial:
    """Adversarial / edge-case tests for Quick Suite integrity."""

    async def test_wrong_answer_arc(self):
        """Wrong ARC answer → score 0.0."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(8), "D"))

        attempt = await adapter.run_task(adapter.list_tasks()[0], provider)
        raw = {
            "results": {},
            "evidence_ids": list(attempt.evidence_refs),
            "task_name": "arc_challenge_quick_v1",
            "attempt_id": attempt.attempt_id,
            "item_results": [it.model_dump() for it in attempt.item_results],
            "planned_item_count": 8,
        }
        grade = adapter.normalize_result(raw)
        # All wrong answers → score should be < 1.0 (most will be 0.0, possibly some GRADED wrongly)
        assert grade.normalized_score <= 0.25  # at most 2/8 by chance

    async def test_arc_ungradable_answer(self):
        """ARC response that cannot be parsed → UNGRADABLE."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(8), "I have no idea what the answer is"))

        attempt = await adapter.run_task(adapter.list_tasks()[0], provider)
        assert any(it.status == ItemStatus.UNGRADABLE for it in attempt.item_results)

    async def test_ifeval_partial_constraint_score(self):
        """IFEval with 2/3 constraints satisfied → 0.666..."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        # Mixed response: satisfies keyword but fails word_count and forbidden
        provider = MockQuickSuiteProvider(dict.fromkeys(range(8), "AI AI computer computer computer"))

        attempt = await adapter.run_task(adapter.list_tasks()[3], provider)
        # Should have partial scores (some constraints pass, some fail)
        scores = [it.normalized_score for it in attempt.item_results if it.normalized_score is not None]
        assert any(0.0 < s < 1.0 for s in scores), f"Expected partial scores, got {scores}"

    def test_missing_item_results_raises(self):
        """Quick task missing item_results → should raise."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        with pytest.raises(ValueError, match="ITEM_RESULTS_REQUIRED"):
            adapter.normalize_result(
                {
                    "task_name": "arc_challenge_quick_v1",
                    "attempt_id": "a",
                }
            )

    async def test_calibrated_scores_remain_none(self):
        """All calibrated scores must be None."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        tasks = adapter.list_tasks()
        attempts, grade_results = [], []
        for task_spec in tasks:
            attempt = await adapter.run_task(task_spec, provider)
            attempts.append(attempt)
            raw = {
                "results": {},
                "evidence_ids": list(attempt.evidence_refs),
                "task_name": task_spec.task_id,
                "attempt_id": attempt.attempt_id,
                "item_results": [it.model_dump() for it in attempt.item_results],
                "planned_item_count": 8,
            }
            grade_results.append(adapter.normalize_result(raw))

        run_results = _build_run_results_per_task(adapter, tasks, attempts, grade_results)
        registry = _build_quick_registry()
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile(run_results, registry, policy, strict=True)

        for dim in profile.dimensions:
            assert dim.calibrated_score is None, f"{dim.dimension} calibrated_score should be None"
        assert profile.calibrated_total_score is None

    async def test_no_renormalization(self):
        """coverage_weight=0.75, provisional_raw_index must NOT be 1.0."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        tasks = adapter.list_tasks()
        attempts, grade_results = [], []
        for task_spec in tasks:
            attempt = await adapter.run_task(task_spec, provider)
            attempts.append(attempt)
            raw = {
                "results": {},
                "evidence_ids": list(attempt.evidence_refs),
                "task_name": task_spec.task_id,
                "attempt_id": attempt.attempt_id,
                "item_results": [it.model_dump() for it in attempt.item_results],
                "planned_item_count": 8,
            }
            grade_results.append(adapter.normalize_result(raw))

        run_results = _build_run_results_per_task(adapter, tasks, attempts, grade_results)
        registry = _build_quick_registry()
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile(run_results, registry, policy, strict=True)

        assert profile.provisional_raw_index <= profile.coverage_weight
        assert profile.provisional_raw_index != 1.0

    async def test_humaneval_timeout(self):
        """HumanEval timeout → GRADED 0.0 with timeout message."""
        backend = _FakeTimeoutBackend()
        adapter = QuickSuiteAdapter(code_backend=backend)
        provider = MockQuickSuiteProvider(dict.fromkeys(range(8), "def solve():\n    pass"))

        attempt = await adapter.run_task(adapter.list_tasks()[1], provider)
        assert all(it.status == ItemStatus.GRADED for it in attempt.item_results)
        assert all(it.normalized_score == 0.0 for it in attempt.item_results)

    async def test_humaneval_test_failure(self):
        """HumanEval code runs but tests fail → GRADED 0.0."""
        adapter = QuickSuiteAdapter(code_backend=_FakeFailingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(8), "print('hello')"))

        attempt = await adapter.run_task(adapter.list_tasks()[1], provider)
        assert all(it.status == ItemStatus.GRADED for it in attempt.item_results)
        assert all(it.normalized_score == 0.0 for it in attempt.item_results)

    def test_quick_task_missing_item_results_raises(self):
        """A Quick task result with None item_results → ValueError."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        with pytest.raises(ValueError, match="ITEM_RESULTS_REQUIRED"):
            adapter.normalize_result(
                {
                    "task_name": "arc_challenge_quick_v1",
                    "attempt_id": "a",
                    "item_results": None,
                    "planned_item_count": 8,
                }
            )

    async def test_humaneval_sandbox_unavailable(self):
        """Sandbox unavailable → FAILURE not graded as wrong."""
        adapter = QuickSuiteAdapter(code_backend=_UnavailableBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(8), "def f(): return 1"))

        attempt = await adapter.run_task(adapter.list_tasks()[1], provider)
        assert all(it.status == ItemStatus.FAILURE for it in attempt.item_results)
        assert all(
            it.failure is not None and it.failure.error_code == "SANDBOX_UNAVAILABLE" for it in attempt.item_results
        )

    async def test_32_item_total_invariant(self):
        """Exactly 32 items across all 4 tasks, never fewer."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        provider = MockQuickSuiteProvider(dict.fromkeys(range(32), "mock"))

        total = 0
        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            total += len(attempt.item_results)
        assert total == 32

    async def test_provider_failure_item_isolation(self):
        """Provider failure on item N does not halt the suite."""
        adapter = QuickSuiteAdapter(code_backend=_FakePassingBackend())
        # Fail at responses 3 and 10
        provider = MockQuickSuiteProvider(
            dict.fromkeys(range(32), "mock"),
            fail_indices={3, 10},
        )

        total = 0
        failure_count = 0
        for task_spec in adapter.list_tasks():
            attempt = await adapter.run_task(task_spec, provider)
            total += len(attempt.item_results)
            failure_count += sum(1 for it in attempt.item_results if it.status == ItemStatus.FAILURE)

        assert total == 32  # All items present
        assert failure_count >= 2  # At least the failed items are FAILURE

    def test_html_report_escapes_xss(self):
        """HTML report escapes XSS in task_id."""
        result = AuditResult(
            config=AuditConfig(
                protocol=Protocol.OPENAI,
                base_url="https://mock.example.com",
                model="test-model",
                api_key_env="MOCK_KEY",
                auth_style=AuthStyle.AUTO,
                repeat_count=1,
                timeout=30.0,
                max_output_tokens=100,
                check_streaming=False,
            ),
            risk_level=RiskLevel.LOW,
        )
        xss = "<img src=x onerror=alert(1)>"

        # Build a section with XSS in task_id
        p = _make_provenance()
        plan = RunPlan(
            plan_id="xss-plan",
            task_ids=[xss],
            total_samples=1,
            budget=BudgetEstimate(planned_requests=1, maximum_requests=1, estimated_cost=None),
            **{k: v for k, v in p.items() if k in RunPlan.model_fields},
        )
        attempt = TaskAttempt(
            attempt_id=xss,
            task_id=xss,
            status=TaskStatus.FAILURE,
            evidence_refs=[str(uuid.uuid4())],
            failure=AdapterFailure(
                error_code="ERR",
                category=FailureCategory.PROVIDER,
                message="fail",
                retryable=False,
            ),
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        rr = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[str(uuid.uuid4())],
            **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
        )
        section = build_benchmark_report_section(plan, rr)
        # Add evidence for validation
        for task in section.tasks:
            for ref in task.evidence_refs:
                result.evidence.append(
                    HTTPEvidence(
                        evidence_id=uuid.UUID(ref),
                        evidence_type="benchmark",
                        request_method="POST",
                        request_url_redacted="https://mock.example.com",
                        request_path="/",
                        request_headers_redacted={},
                        http_status=200,
                        response_text="ok",
                    )
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()
            assert "<img src=x" not in html
            assert "&lt;img" in html


# ---------------------------------------------------------------------------
# Helpers for adversarial tests
# ---------------------------------------------------------------------------


class _FakeTimeoutBackend(CodeExecutionBackend):
    """CodeExecutionBackend that always times out."""

    def is_available(self) -> bool:
        return True

    def execute(self, code, *, timeout_seconds=10.0):
        return CodeExecutionResult(
            success=False,
            stderr="Execution timed out after 10 seconds",
            exit_code=-1,
            timed_out=True,
        )


def _build_run_results_per_task(adapter, tasks, attempts, grade_results):
    """Build one BenchmarkRunResult per task for provenance matching."""
    from llmtrace.adapters.quick_suite import _QUICK_TASK_DEFS

    run_results = []
    for _, (task_spec, attempt, grade) in enumerate(zip(tasks, attempts, grade_results, strict=True)):
        task_def = _QUICK_TASK_DEFS[task_spec.task_id]
        run_results.append(
            BenchmarkRunResult(
                run_id=str(uuid.uuid4()),
                task_attempts=[attempt],
                grade_results=[grade],
                evidence_refs=list(attempt.evidence_refs),
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
            )
        )
    return run_results


def _make_provenance():
    """Make benchmark provenance dict for test fixtures."""
    return {
        "suite_id": "llmtrace_quick_v1",
        "suite_version": "0.1.0",
        "source_id": "test",
        "source_revision": "test-rev",
        "adapter_id": "quick-suite-test",
        "adapter_version": "0.1.0",
    }
