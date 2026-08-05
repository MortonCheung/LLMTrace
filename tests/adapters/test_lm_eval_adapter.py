"""Tests for LmEvalAdapter (no lm-eval runtime required)."""

from __future__ import annotations

import pytest

from llmtrace.adapters.lm_eval import LmEvalAdapter
from llmtrace.benchmarks.models import (
    BudgetEstimate,
    GradeStatus,
    RunPlan,
    TaskSpec,
    TaskStatus,
)


class TestLmEvalAdapterMetadata:
    def test_adapter_id(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        assert adapter.adapter_id == "lm-eval"

    def test_adapter_version_is_string(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        assert isinstance(adapter.adapter_version, str)
        assert len(adapter.adapter_version) > 0


class TestLmEvalAdapterListTasks:
    def test_list_tasks_returns_smoke_task(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        tasks = adapter.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "llmtrace_smoke"
        assert tasks[0].num_samples == 4

    def test_list_tasks_has_smoke_category(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        tasks = adapter.list_tasks()
        assert tasks[0].category == "smoke"


class TestLmEvalAdapterBuildPlan:
    def test_build_plan_returns_run_plan(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        plan = adapter.build_plan(
            suite_id="smoke_suite",
            suite_version="1.0.0",
            source_id="lm-eval",
            source_revision="0.4.12",
            task_ids=["llmtrace_smoke"],
        )
        assert isinstance(plan, RunPlan)
        assert plan.total_samples == 4
        assert plan.task_ids == ["llmtrace_smoke"]
        assert plan.budget.planned_requests == 4
        assert plan.adapter_id == "lm-eval"

    def test_build_plan_is_deterministic(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        plan1 = adapter.build_plan("s", "v", "src", "rev", ["llmtrace_smoke"])
        plan2 = adapter.build_plan("s", "v", "src", "rev", ["llmtrace_smoke"])
        assert plan1.plan_id == plan2.plan_id


class TestLmEvalAdapterBudget:
    def test_estimate_budget(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        budget = adapter.estimate_budget("s", ["llmtrace_smoke"])
        assert isinstance(budget, BudgetEstimate)
        assert budget.planned_requests == 4
        assert budget.maximum_requests == 4

    def test_estimate_budget_with_retries(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        budget = adapter.estimate_budget("s", ["llmtrace_smoke"], max_retries=2)
        assert budget.planned_requests == 4
        assert budget.maximum_requests == 12


class TestLmEvalAdapterNormalizeResult:
    def test_normalize_with_exact_match(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 1.0},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "attempt-1",
            }
        )
        assert grade.grader_id == "exact_match"
        assert grade.raw_score == 1.0
        assert grade.normalized_score == 1.0
        assert grade.status == GradeStatus.GRADED

    def test_normalize_with_partial_score(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 0.5},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.normalized_score == 0.5

    def test_normalize_with_no_results(self) -> None:
        adapter = LmEvalAdapter(include_path="/fake")
        grade = adapter.normalize_result(
            {
                "results": {},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.UNGRADABLE
        assert grade.normalized_score == 0.0


class TestLmEvalAdapterRunTaskWithoutLmEval:
    @pytest.mark.asyncio
    async def test_run_task_fails_when_lm_eval_not_installed(self) -> None:
        """When lm-eval is not installed (or unavailable), run_task returns FAILURE."""
        # This test verifies the failure path without needing lm-eval installed.
        # Since we can't uninstall lm-eval in the test env, we test the
        # adapter's structural failure handling.
        adapter = LmEvalAdapter(include_path="/nonexistent/path")
        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)

        # This should fail because the include_path doesn't exist,
        # proving the adapter returns structured failures.
        from tests.adapters.conftest import FakeProvider

        provider = FakeProvider()
        attempt = await adapter.run_task(task, provider)
        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None
        assert attempt.failure.category is not None
        assert len(attempt.failure.error_code) > 0
