"""Tests for LmEvalAdapter (no lm-eval runtime required)."""

from __future__ import annotations

import pytest

from llmtrace.adapters.lm_eval import LmEvalAdapter, parse_exact_match_metric
from llmtrace.benchmarks.models import (
    BudgetEstimate,
    GradeStatus,
    RunPlan,
    SmokeTaskManifest,
    TaskSpec,
    TaskStatus,
)


class TestLmEvalAdapterMetadata:
    def test_adapter_id(self) -> None:
        adapter = LmEvalAdapter()
        assert adapter.adapter_id == "lm-eval"

    def test_adapter_version_is_string(self) -> None:
        adapter = LmEvalAdapter()
        assert isinstance(adapter.adapter_version, str)
        assert len(adapter.adapter_version) > 0


class TestLmEvalAdapterListTasks:
    def test_list_tasks_returns_smoke_task(self) -> None:
        adapter = LmEvalAdapter()
        tasks = adapter.list_tasks()
        assert len(tasks) >= 1
        assert tasks[0].task_id == "llmtrace_smoke"
        assert tasks[0].num_samples == 4

    def test_list_tasks_has_smoke_category(self) -> None:
        adapter = LmEvalAdapter()
        tasks = adapter.list_tasks()
        assert tasks[0].category == "smoke"


class TestSmokeTaskManifest:
    def test_manifest_defaults(self) -> None:
        manifest = SmokeTaskManifest()
        assert manifest.task_id == "llmtrace_smoke"
        assert manifest.suite_id == "llmtrace_smoke"
        assert manifest.metric == "exact_match"
        assert manifest.filter == "none"
        assert manifest.capability_score_eligible is False

    def test_manifest_is_frozen(self) -> None:
        manifest = SmokeTaskManifest()
        with pytest.raises((TypeError, ValueError)):
            manifest.task_id = "other"  # type: ignore[misc]


class TestLmEvalAdapterBuildPlan:
    def test_build_plan_returns_run_plan(self) -> None:
        adapter = LmEvalAdapter()
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
        adapter = LmEvalAdapter()
        plan1 = adapter.build_plan("s", "v", "src", "rev", ["llmtrace_smoke"])
        plan2 = adapter.build_plan("s", "v", "src", "rev", ["llmtrace_smoke"])
        assert plan1.plan_id == plan2.plan_id

    def test_build_plan_rejects_unknown_task_id(self) -> None:
        """Section 7: Unknown task_id must raise an error."""
        adapter = LmEvalAdapter()
        with pytest.raises(ValueError, match="Unknown task_ids"):
            adapter.build_plan("s", "v", "src", "rev", ["nonexistent_task"])

    def test_build_plan_rejects_empty_task_ids(self) -> None:
        """Section 7: Empty task_ids must raise an error."""
        adapter = LmEvalAdapter()
        with pytest.raises(ValueError, match="task_ids must not be empty"):
            adapter.build_plan("s", "v", "src", "rev", [])


class TestLmEvalAdapterBudget:
    def test_estimate_budget(self) -> None:
        adapter = LmEvalAdapter()
        budget = adapter.estimate_budget("s", ["llmtrace_smoke"])
        assert isinstance(budget, BudgetEstimate)
        assert budget.planned_requests == 4
        assert budget.maximum_requests == 4

    def test_estimate_budget_with_retries(self) -> None:
        adapter = LmEvalAdapter()
        budget = adapter.estimate_budget("s", ["llmtrace_smoke"], max_retries=2)
        assert budget.planned_requests == 4
        assert budget.maximum_requests == 12

    def test_estimate_budget_rejects_unknown_task_id(self) -> None:
        """Section 7: Unknown task_id must raise an error in estimate_budget too."""
        adapter = LmEvalAdapter()
        with pytest.raises(ValueError, match="Unknown task_ids"):
            adapter.estimate_budget("s", ["nonexistent"])

    def test_estimate_budget_rejects_empty_task_ids(self) -> None:
        adapter = LmEvalAdapter()
        with pytest.raises(ValueError, match="task_ids must not be empty"):
            adapter.estimate_budget("s", [])


class TestLmEvalAdapterNormalizeResult:
    def test_normalize_with_exact_match(self) -> None:
        adapter = LmEvalAdapter()
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
        adapter = LmEvalAdapter()
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
        adapter = LmEvalAdapter()
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

    def test_normalize_score_out_of_bounds_rejected(self) -> None:
        """Section 6: Scores outside [0, 1] must be UNGRADABLE."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 1.5},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.UNGRADABLE
        assert "outside" in (grade.error_message or "")

    def test_normalize_negative_score_rejected(self) -> None:
        """Section 6: Negative scores must be UNGRADABLE."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": -0.1},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.UNGRADABLE

    def test_normalize_with_exact_match_filter_format(self) -> None:
        """exact_match,none format should be accepted."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"exact_match,none": 0.75},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.GRADED
        assert grade.normalized_score == 0.75

    def test_normalize_non_numeric_metric_ungradable(self) -> None:
        """Non-numeric exact_match value should be UNGRADABLE."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": "not-a-number"},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.UNGRADABLE

    def test_normalize_non_exact_match_metric_ungradable(self) -> None:
        """Section 6: Only exact_match is accepted - no fallback to first metric."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"f1_score": 0.8},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.UNGRADABLE
        assert "No exact_match" in (grade.error_message or "")


class TestParseExactMatchMetric:
    """Section 2: Strict exact_match name parsing."""

    def test_exact_match_plain_accepted(self) -> None:
        result = parse_exact_match_metric("exact_match")
        assert result == ("exact_match", "none")

    def test_exact_match_none_accepted(self) -> None:
        result = parse_exact_match_metric("exact_match,none")
        assert result == ("exact_match", "none")

    def test_exact_match_with_filter_accepted(self) -> None:
        result = parse_exact_match_metric("exact_match,my_filter")
        assert result == ("exact_match", "my_filter")

    def test_exact_match_fake_rejected(self) -> None:
        assert parse_exact_match_metric("exact_match_fake") is None

    def test_exact_matching_rejected(self) -> None:
        assert parse_exact_match_metric("exact_matching") is None

    def test_exact_match_trailing_comma_rejected(self) -> None:
        """exact_match, with empty filter must be rejected."""
        assert parse_exact_match_metric("exact_match,") is None

    def test_f1_score_rejected(self) -> None:
        assert parse_exact_match_metric("f1_score") is None

    def test_empty_string_rejected(self) -> None:
        assert parse_exact_match_metric("") is None

    def test_exact_match_none_normalize_uses_strict_parser(self) -> None:
        """Section 2: normalize_result must use parse_exact_match_metric."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"exact_match_fake": 0.9},
                "evidence_ids": [],
                "task_name": "test",
                "attempt_id": "a",
            }
        )
        assert grade.status == GradeStatus.UNGRADABLE


class TestLmEvalAdapterRunTaskWithoutLmEval:
    @pytest.mark.asyncio
    async def test_run_task_fails_when_lm_eval_not_installed(self) -> None:
        """When lm-eval is not available, run_task returns FAILURE.

        This test is only relevant in the base [dev] environment without lm-eval.
        """
        try:
            import lm_eval  # noqa: F401

            pytest.skip("lm-eval is installed — this test is for the base environment only")
        except ImportError:
            pass

        adapter = LmEvalAdapter()
        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)

        from tests.adapters.conftest import FakeProvider

        provider = FakeProvider()
        attempt = await adapter.run_task(task, provider)
        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None
        assert attempt.failure.category is not None
        assert len(attempt.failure.error_code) > 0
