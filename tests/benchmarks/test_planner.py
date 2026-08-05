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
# RunPlan determinism – SHA-256 plan_id
# ============================================================================


class TestRunPlanDeterminism:
    def test_same_inputs_produce_same_plan_id(self) -> None:
        """Identical inputs produce identical plan_id and all other fields."""
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
        assert plan1.plan_id == plan2.plan_id
        assert plan1 == plan2

    def test_different_suite_id_produces_different_plan_id(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan1 = build_plan(
            suite_id="suite_a",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
        )
        plan2 = build_plan(
            suite_id="suite_b",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
        )
        assert plan1.plan_id != plan2.plan_id

    def test_different_task_ids_produce_different_plan_id(self) -> None:
        tasks1 = _make_tasks(("t1", 10))
        tasks2 = _make_tasks(("t2", 10))
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
        assert plan1.plan_id != plan2.plan_id

    def test_different_retries_produce_different_plan_id(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan1 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=0,
        )
        plan2 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            max_retries=1,
        )
        assert plan1.plan_id != plan2.plan_id

    def test_different_token_params_produce_different_plan_id(self) -> None:
        tasks = _make_tasks(("t1", 10))
        plan1 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            tokens_per_sample_input=100,
        )
        plan2 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            tokens_per_sample_input=200,
        )
        assert plan1.plan_id != plan2.plan_id

    def test_task_order_affects_plan_id(self) -> None:
        """task_ids order matters for plan_id determinism."""
        tasks1 = _make_tasks(("a", 1), ("b", 1))
        tasks2 = _make_tasks(("b", 1), ("a", 1))
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
        assert plan1.plan_id != plan2.plan_id

    def test_num_samples_change_affects_plan_id(self) -> None:
        """Changing task num_samples changes the plan_id."""
        tasks1 = _make_tasks(("t1", 10))
        tasks2 = _make_tasks(("t1", 20))
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
        assert plan1.plan_id != plan2.plan_id

    def test_price_change_affects_plan_id(self) -> None:
        """Changing pricing parameters changes the plan_id."""
        tasks = _make_tasks(("t1", 10))
        plan1 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            price_per_million_input=3.0,
            price_per_million_output=15.0,
        )
        plan2 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            price_per_million_input=4.0,
            price_per_million_output=15.0,
        )
        assert plan1.plan_id != plan2.plan_id

    def test_duration_change_affects_plan_id(self) -> None:
        """Changing duration_per_sample_seconds changes the plan_id."""
        tasks = _make_tasks(("t1", 10))
        plan1 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            duration_per_sample_seconds=2.0,
        )
        plan2 = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
            duration_per_sample_seconds=3.0,
        )
        assert plan1.plan_id != plan2.plan_id

    def test_plan_has_no_created_at(self) -> None:
        tasks = _make_tasks(("t1", 1))
        plan = build_plan(
            suite_id="s",
            suite_version="v",
            source_id="src",
            source_revision="r",
            adapter_id="a",
            adapter_version="v",
            tasks=tasks,
        )
        assert not hasattr(plan, "created_at")


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
        assert plan.budget.maximum_requests == 40

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
        total = 5 + 3
        assert plan.budget.planned_requests == total
        assert plan.budget.maximum_requests == total * 3

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
        assert plan.budget.planned_requests == 30
        assert plan.budget.maximum_requests == 60

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
        tasks = _make_tasks(("t1", 1_000_000))
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
        assert restored.plan_id == plan.plan_id
        assert restored.budget.planned_requests == plan.budget.planned_requests


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
