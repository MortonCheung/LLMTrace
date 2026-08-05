"""Tests for RunPlan planner and BudgetEstimate calculations."""

from __future__ import annotations

from llmtrace.benchmarks.models import TaskSpec
from llmtrace.benchmarks.planner import build_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tasks(*specs: tuple[str, int]) -> list[TaskSpec]:
    return [TaskSpec(task_id=tid, name=tid, num_samples=n) for tid, n in specs]


# ============================================================================
# RunPlan determinism
# ============================================================================


class TestRunPlanDeterminism:
    def test_same_inputs_produce_same_plan(self) -> None:
        """Same inputs produce same plan (except plan_id and created_at)."""
        tasks = _make_tasks(("t1", 50), ("t2", 30))
        plan1 = build_plan(
            suite_id="s",
            suite_version="1.0.0",
            source_id="src",
            source_revision="rev",
            adapter_id="a",
            adapter_version="1.0.0",
            tasks=tasks,
            max_retries=2,
        )
        plan2 = build_plan(
            suite_id="s",
            suite_version="1.0.0",
            source_id="src",
            source_revision="rev",
            adapter_id="a",
            adapter_version="1.0.0",
            tasks=tasks,
            max_retries=2,
        )
        # plan_id and created_at are unique per invocation
        assert plan1.plan_id != plan2.plan_id
        # All other fields must be identical
        assert plan1.suite_id == plan2.suite_id
        assert plan1.suite_version == plan2.suite_version
        assert plan1.source_id == plan2.source_id
        assert plan1.source_revision == plan2.source_revision
        assert plan1.adapter_id == plan2.adapter_id
        assert plan1.adapter_version == plan2.adapter_version
        assert plan1.task_ids == plan2.task_ids
        assert plan1.total_samples == plan2.total_samples
        assert plan1.budget == plan2.budget

    def test_different_tasks_produce_different_plan(self) -> None:
        tasks1 = _make_tasks(("t1", 10))
        tasks2 = _make_tasks(("t2", 20))
        plan1 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks1,
        )
        plan2 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks2,
        )
        assert plan1.total_samples != plan2.total_samples
        assert plan1.task_ids != plan2.task_ids


# ============================================================================
# Request count + retry calculation
# ============================================================================


class TestRequestCountCalculation:
    def test_no_retries_planned_equals_maximum(self) -> None:
        tasks = _make_tasks(("t1", 100))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=0,
        )
        assert plan.budget.planned_requests == 100
        assert plan.budget.maximum_requests == 100
        assert plan.budget.maximum_retries == 0

    def test_with_retries_maximum_includes_retries(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=3,
        )
        assert plan.budget.planned_requests == 10
        assert plan.budget.maximum_requests == 40  # 10 * (1 + 3)

    def test_multiple_samples_with_retries(self) -> None:
        tasks = _make_tasks(("t1", 5), ("t2", 3))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=2,
        )
        total = 5 + 3  # 8
        assert plan.budget.planned_requests == total
        assert plan.budget.maximum_requests == total * 3  # 8 * (1+2) = 24

    def test_custom_requests_per_sample(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            requests_per_sample=3,
            max_retries=1,
        )
        assert plan.budget.planned_requests == 30  # 10 * 3
        assert plan.budget.maximum_requests == 60  # 30 * 2

    def test_zero_samples(self) -> None:
        tasks = _make_tasks(("t1", 0))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=5,
        )
        assert plan.budget.planned_requests == 0
        assert plan.budget.maximum_requests == 0
        assert plan.total_samples == 0


# ============================================================================
# BudgetEstimate
# ============================================================================


class TestBudgetEstimate:
    def test_token_estimates(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            tokens_per_sample_input=100,
            tokens_per_sample_output=50,
        )
        assert plan.budget.estimated_input_tokens == 1000
        assert plan.budget.estimated_output_tokens == 500

    def test_duration_estimate(self) -> None:
        tasks = _make_tasks(("t1", 5))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            duration_per_sample_seconds=3.0,
        )
        assert plan.budget.estimated_duration_seconds == 15.0

    def test_cost_with_pricing(self) -> None:
        tasks = _make_tasks(("t1", 1_000_000))  # 1M samples × 1 token each = 1M tokens
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            tokens_per_sample_input=1,
            tokens_per_sample_output=1,
            price_per_million_input=3.0,
            price_per_million_output=15.0,
        )
        # input: 1M * $3/M = $3, output: 1M * $15/M = $15, total = $18
        assert plan.budget.estimated_cost == 18.0

    def test_cost_none_when_prices_unavailable(self) -> None:
        tasks = _make_tasks(("t1", 100))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            price_per_million_input=None,
            price_per_million_output=None,
        )
        assert plan.budget.estimated_cost is None

    def test_cost_none_when_input_price_missing(self) -> None:
        tasks = _make_tasks(("t1", 100))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            price_per_million_input=None,
            price_per_million_output=15.0,
        )
        assert plan.budget.estimated_cost is None

    def test_cost_none_when_output_price_missing(self) -> None:
        tasks = _make_tasks(("t1", 100))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            price_per_million_input=3.0,
            price_per_million_output=None,
        )
        assert plan.budget.estimated_cost is None

    def test_assumptions_contains_retry_info(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=2,
        )
        assert any("2 retries" in a for a in plan.budget.assumptions)
        assert any("unavailable" in a.lower() for a in plan.budget.assumptions)

    def test_assumptions_contains_pricing_info(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            price_per_million_input=1.0,
            price_per_million_output=2.0,
        )
        assert any("$1.0/M input" in a for a in plan.budget.assumptions)

    def test_total_samples_correct(self) -> None:
        tasks = _make_tasks(("t1", 30), ("t2", 20), ("t3", 50))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
        )
        assert plan.total_samples == 100

    def test_task_ids_in_plan(self) -> None:
        tasks = _make_tasks(("t1", 10), ("t2", 20))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
        )
        assert plan.task_ids == ["t1", "t2"]

    def test_json_roundtrip(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=1,
            price_per_million_input=3.0,
            price_per_million_output=15.0,
        )
        data = plan.model_dump_json()
        restored = plan.__class__.model_validate_json(data)
        assert restored.suite_id == plan.suite_id
        assert restored.budget.planned_requests == plan.budget.planned_requests
        assert restored.budget.estimated_cost == plan.budget.estimated_cost


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_empty_tasks(self) -> None:
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=[],
        )
        assert plan.total_samples == 0
        assert plan.task_ids == []
        assert plan.budget.planned_requests == 0

    def test_large_number_of_samples(self) -> None:
        tasks = _make_tasks(("t1", 100_000))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=1,
            tokens_per_sample_input=200,
            tokens_per_sample_output=100,
        )
        assert plan.total_samples == 100_000
        assert plan.budget.planned_requests == 100_000
        assert plan.budget.maximum_requests == 200_000
        assert plan.budget.estimated_input_tokens == 20_000_000
