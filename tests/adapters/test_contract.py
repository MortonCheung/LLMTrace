"""Contract tests for BenchmarkAdapter protocol.

Includes a FakeBenchmarkAdapter used only for testing (no network calls).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from llmtrace.adapters.base import BenchmarkAdapter
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkRunResult,
    BudgetEstimate,
    FailureCategory,
    GradeResult,
    GradeStatus,
    RunPlan,
    SuiteVersion,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# FakeBenchmarkAdapter (test-only, no network calls)
# ---------------------------------------------------------------------------


class FakeBenchmarkAdapter(BenchmarkAdapter):
    """A fake adapter for testing the protocol contract.

    Does not make any network calls. Returns hardcoded but structurally
    valid outputs to exercise the abstract interface.
    """

    @property
    def adapter_id(self) -> str:
        return "fake-benchmark"

    @property
    def adapter_version(self) -> str:
        return "0.1.0-test"

    def list_tasks(self) -> list[TaskSpec]:
        return [
            TaskSpec(task_id="fake_task_1", name="Fake Task 1", num_samples=10),
            TaskSpec(task_id="fake_task_2", name="Fake Task 2", num_samples=5, category="reasoning"),
        ]

    def build_plan(
        self,
        suite_id: str,
        suite_version: str,
        source_id: str,
        source_revision: str,
        task_ids: list[str],
    ) -> RunPlan:
        tasks = [t for t in self.list_tasks() if t.task_id in task_ids]
        from llmtrace.benchmarks.planner import build_plan as _build

        return _build(
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
        tasks = [t for t in self.list_tasks() if t.task_id in task_ids]
        total = sum(t.num_samples for t in tasks)
        return BudgetEstimate(
            planned_requests=total,
            maximum_requests=total * (1 + max_retries),
            maximum_retries=max_retries,
            assumptions=["fake adapter: no real estimates"],
        )

    async def run_task(self, task_spec: TaskSpec, provider: object) -> TaskAttempt:
        return TaskAttempt(
            attempt_id=str(uuid4()),
            source_id="fake-source",
            source_revision="fake-rev",
            suite_id="fake-suite",
            suite_version="0.1.0",
            task_id=task_spec.task_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
        )

    def normalize_result(self, raw_result: dict[str, object]) -> GradeResult:
        score = float(raw_result.get("score", 0.0))
        if score < 0.0 or score > 1.0:
            raise ValueError("score must be in [0, 1]")
        return GradeResult(
            grade_id=str(uuid4()),
            attempt_id=str(raw_result.get("attempt_id", uuid4())),
            source_id="fake-source",
            source_revision="fake-rev",
            suite_id="fake-suite",
            suite_version="0.1.0",
            task_id=str(raw_result.get("task_id", "unknown")),
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            grader_id="fake-grader",
            raw_score=score,
            normalized_score=score,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFakeAdapterMetadata:
    def test_adapter_id_and_version(self) -> None:
        adapter = FakeBenchmarkAdapter()
        assert adapter.adapter_id == "fake-benchmark"
        assert adapter.adapter_version == "0.1.0-test"

    def test_is_instance_of_benchmark_adapter(self) -> None:
        adapter = FakeBenchmarkAdapter()
        assert isinstance(adapter, BenchmarkAdapter)


class TestFakeAdapterListTasks:
    def test_returns_list_of_task_specs(self) -> None:
        adapter = FakeBenchmarkAdapter()
        tasks = adapter.list_tasks()
        assert len(tasks) == 2
        assert all(isinstance(t, TaskSpec) for t in tasks)
        assert tasks[0].task_id == "fake_task_1"

    def test_tasks_have_positive_samples(self) -> None:
        adapter = FakeBenchmarkAdapter()
        tasks = adapter.list_tasks()
        for t in tasks:
            assert t.num_samples > 0


class TestFakeAdapterBuildPlan:
    def test_build_plan_is_deterministic(self) -> None:
        adapter = FakeBenchmarkAdapter()
        plan1 = adapter.build_plan("s", "1.0.0", "src", "rev", ["fake_task_1"])
        plan2 = adapter.build_plan("s", "1.0.0", "src", "rev", ["fake_task_1"])
        assert plan1.plan_id == plan2.plan_id

    def test_build_plan_filters_tasks(self) -> None:
        adapter = FakeBenchmarkAdapter()
        plan = adapter.build_plan("s", "1.0.0", "src", "rev", ["fake_task_1"])
        assert plan.total_samples == 10

    def test_build_plan_all_tasks(self) -> None:
        adapter = FakeBenchmarkAdapter()
        plan = adapter.build_plan("s", "1.0.0", "src", "rev", ["fake_task_1", "fake_task_2"])
        assert plan.total_samples == 15

    def test_build_plan_returns_run_plan_type(self) -> None:
        adapter = FakeBenchmarkAdapter()
        plan = adapter.build_plan("s", "1.0.0", "src", "rev", ["fake_task_1"])
        assert isinstance(plan, RunPlan)


class TestFakeAdapterEstimateBudget:
    def test_budget_includes_retries(self) -> None:
        adapter = FakeBenchmarkAdapter()
        budget = adapter.estimate_budget("s", ["fake_task_1"], max_retries=2)
        assert budget.planned_requests == 10
        assert budget.maximum_requests == 30

    def test_budget_cost_is_none(self) -> None:
        adapter = FakeBenchmarkAdapter()
        budget = adapter.estimate_budget("s", ["fake_task_1"])
        assert budget.estimated_cost is None

    def test_budget_returns_budget_estimate_type(self) -> None:
        adapter = FakeBenchmarkAdapter()
        budget = adapter.estimate_budget("s", ["fake_task_1"])
        assert isinstance(budget, BudgetEstimate)


class TestFakeAdapterRunTask:
    @pytest.mark.asyncio
    async def test_run_task_returns_task_attempt(self) -> None:
        adapter = FakeBenchmarkAdapter()
        task = TaskSpec(task_id="t", name="T", num_samples=1)
        attempt = await adapter.run_task(task, provider=object())
        assert isinstance(attempt, TaskAttempt)
        assert attempt.status == TaskStatus.SUCCESS
        assert len(attempt.evidence_refs) > 0

    @pytest.mark.asyncio
    async def test_run_task_includes_valid_uuid_evidence_refs(self) -> None:
        adapter = FakeBenchmarkAdapter()
        task = TaskSpec(task_id="t", name="T", num_samples=1)
        attempt = await adapter.run_task(task, provider=object())
        assert len(attempt.evidence_refs) > 0
        # evidence_refs are validated as UUIDs by the model
        from uuid import UUID

        for ref in attempt.evidence_refs:
            UUID(ref)


class TestFakeAdapterNormalizeResult:
    def test_normalize_result_returns_grade_result(self) -> None:
        adapter = FakeBenchmarkAdapter()
        raw = {"score": 0.75, "attempt_id": "a1", "task_id": "t1"}
        grade = adapter.normalize_result(raw)
        assert isinstance(grade, GradeResult)
        assert grade.raw_score == 0.75
        assert grade.normalized_score == 0.75

    def test_normalize_result_rejects_out_of_bounds(self) -> None:
        adapter = FakeBenchmarkAdapter()
        raw = {"score": 1.5, "attempt_id": "a", "task_id": "t"}
        with pytest.raises(ValueError):
            adapter.normalize_result(raw)

    def test_normalize_result_status_is_graded(self) -> None:
        adapter = FakeBenchmarkAdapter()
        raw = {"score": 0.5, "attempt_id": "a", "task_id": "t"}
        grade = adapter.normalize_result(raw)
        assert grade.status == GradeStatus.GRADED


# ============================================================================
# Structured failure contract tests
# ============================================================================


class FailingAdapter(FakeBenchmarkAdapter):
    """An adapter that simulates task failures for contract testing."""

    async def run_task(self, task_spec: TaskSpec, provider: object) -> TaskAttempt:
        failure = AdapterFailure(
            error_code="SIM_FAIL",
            category=FailureCategory.ADAPTER,
            message="Simulated adapter failure",
            retryable=False,
        )
        return TaskAttempt(
            attempt_id=str(uuid4()),
            source_id="fake-source",
            source_revision="fake-rev",
            suite_id="fake-suite",
            suite_version="0.1.0",
            task_id=task_spec.task_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            status=TaskStatus.FAILURE,
            failure=failure,
        )


class TestStructuredFailure:
    @pytest.mark.asyncio
    async def test_failure_returns_structured_error_not_exception(self) -> None:
        """Adapter failures return structured TaskAttempt, not raised exceptions."""
        adapter = FailingAdapter()
        task = TaskSpec(task_id="t", name="T", num_samples=1)
        attempt = await adapter.run_task(task, provider=object())
        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None
        assert attempt.failure.error_code == "SIM_FAIL"
        assert "Simulated" in attempt.failure.message


# ---------------------------------------------------------------------------
# JSON serialization roundtrips
# ---------------------------------------------------------------------------


class TestJsonRoundtrips:
    def test_task_attempt_json_roundtrip(self) -> None:
        uid = str(uuid4())
        ta = TaskAttempt(
            attempt_id=str(uuid4()),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            status=TaskStatus.SUCCESS,
            evidence_refs=[uid],
        )
        data = ta.model_dump_json()
        restored = TaskAttempt.model_validate_json(data)
        assert restored.attempt_id == ta.attempt_id
        assert restored.evidence_refs == ta.evidence_refs

    def test_grade_result_json_roundtrip(self) -> None:
        gr = GradeResult(
            grade_id=str(uuid4()),
            attempt_id=str(uuid4()),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            grader_id="g",
            raw_score=0.8,
            normalized_score=0.8,
        )
        data = gr.model_dump_json()
        restored = GradeResult.model_validate_json(data)
        assert restored.raw_score == 0.8
        assert restored.normalized_score == 0.8

    def test_benchmark_run_result_json_roundtrip(self) -> None:
        result = BenchmarkRunResult(
            run_id=str(uuid4()),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            adapter_id="a",
            adapter_version="v",
        )
        data = result.model_dump_json()
        restored = result.__class__.model_validate_json(data)
        assert restored.run_id == result.run_id

    def test_suite_version_json_roundtrip(self) -> None:
        sv = SuiteVersion(version="2.1.0", notes="Bug fixes")
        data = sv.model_dump_json()
        restored = sv.__class__.model_validate_json(data)
        assert restored.version == "2.1.0"
        assert restored.notes == "Bug fixes"
