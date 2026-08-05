"""Planner for benchmark runs.

Generates deterministic RunPlan and BudgetEstimate objects from
a BenchmarkSuite and its TaskSpec definitions.
"""

from __future__ import annotations

from llmtrace.benchmarks.models import BudgetEstimate, RunPlan, TaskSpec


def build_plan(
    suite_id: str,
    suite_version: str,
    source_id: str,
    source_revision: str,
    adapter_id: str,
    adapter_version: str,
    tasks: list[TaskSpec],
    max_retries: int = 0,
    tokens_per_sample_input: int = 512,
    tokens_per_sample_output: int = 256,
    requests_per_sample: int = 1,
    price_per_million_input: float | None = None,
    price_per_million_output: float | None = None,
    duration_per_sample_seconds: float = 2.0,
) -> RunPlan:
    """Build a deterministic RunPlan from task specifications.

    The plan is deterministic: the same inputs always produce the same plan
    (except for plan_id and created_at, which are unique per invocation).

    Args:
        suite_id: Suite identifier.
        suite_version: Suite version string.
        source_id: Benchmark source identifier.
        source_revision: Source data revision.
        adapter_id: Adapter identifier.
        adapter_version: Adapter version.
        tasks: List of TaskSpec objects.
        max_retries: Maximum retry attempts per request.
        tokens_per_sample_input: Estimated input tokens per sample.
        tokens_per_sample_output: Estimated output tokens per sample.
        requests_per_sample: HTTP requests per sample (default 1).
        price_per_million_input: Price per 1M input tokens, or None if unavailable.
        price_per_million_output: Price per 1M output tokens, or None if unavailable.
        duration_per_sample_seconds: Estimated duration per sample in seconds.

    Returns:
        A deterministic RunPlan with BudgetEstimate.
    """
    total_samples = sum(t.num_samples for t in tasks)
    task_ids = [t.task_id for t in tasks]

    planned_requests = total_samples * requests_per_sample
    maximum_requests = planned_requests * (1 + max_retries)

    estimated_input_tokens = total_samples * tokens_per_sample_input
    estimated_output_tokens = total_samples * tokens_per_sample_output

    estimated_duration_seconds = total_samples * duration_per_sample_seconds

    # Cost calculation: only produce a value if both prices are provided
    estimated_cost: float | None = None
    assumptions: list[str] = [
        f"Each sample requires {requests_per_sample} API request(s)",
        f"Estimated {tokens_per_sample_input} input tokens per sample",
        f"Estimated {tokens_per_sample_output} output tokens per sample",
    ]
    if max_retries > 0:
        assumptions.append(f"Maximum {max_retries} retries per request; worst-case requests = {maximum_requests}")

    if price_per_million_input is not None and price_per_million_output is not None:
        input_cost = (estimated_input_tokens / 1_000_000) * price_per_million_input
        output_cost = (estimated_output_tokens / 1_000_000) * price_per_million_output
        estimated_cost = round(input_cost + output_cost, 6)
        assumptions.append(f"Pricing: ${price_per_million_input}/M input, ${price_per_million_output}/M output tokens")
    else:
        assumptions.append("Pricing information unavailable; estimated_cost is None")

    budget = BudgetEstimate(
        planned_requests=planned_requests,
        maximum_requests=maximum_requests,
        maximum_retries=max_retries,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_duration_seconds=estimated_duration_seconds,
        estimated_cost=estimated_cost,
        assumptions=assumptions,
    )

    return RunPlan(
        suite_id=suite_id,
        suite_version=suite_version,
        source_id=source_id,
        source_revision=source_revision,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        task_ids=task_ids,
        total_samples=total_samples,
        budget=budget,
    )
