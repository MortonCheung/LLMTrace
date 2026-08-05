"""Planner for benchmark runs.

Generates deterministic RunPlan and BudgetEstimate objects from
a BenchmarkSuite and its TaskSpec definitions.  The plan_id is a
SHA-256 digest of the canonicalised input parameters so that
identical inputs always produce the same plan.
"""

from __future__ import annotations

import hashlib
import json

from llmtrace.benchmarks.models import BudgetEstimate, RunPlan, TaskSpec


def _canonical_key(
    suite_id: str,
    suite_version: str,
    source_id: str,
    source_revision: str,
    adapter_id: str,
    adapter_version: str,
    tasks: list[TaskSpec],
    max_retries: int,
    tokens_per_sample_input: int,
    tokens_per_sample_output: int,
    requests_per_sample: int,
    price_per_million_input: float | None,
    price_per_million_output: float | None,
    duration_per_sample_seconds: float,
) -> bytes:
    """Produce a deterministic canonical byte-string for the given inputs.

    Every parameter that influences the plan identity is included so that
    any change produces a different plan_id.  Tasks are hashed as a
    structured list of {task_id, num_samples} (order matters) rather than
    bare task_ids, so changes to sample counts also change the plan_id.
    """
    data = {
        "suite_id": suite_id,
        "suite_version": suite_version,
        "source_id": source_id,
        "source_revision": source_revision,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "tasks": [{"task_id": t.task_id, "num_samples": t.num_samples} for t in tasks],
        "max_retries": max_retries,
        "tokens_per_sample_input": tokens_per_sample_input,
        "tokens_per_sample_output": tokens_per_sample_output,
        "requests_per_sample": requests_per_sample,
        "price_per_million_input": price_per_million_input,
        "price_per_million_output": price_per_million_output,
        "duration_per_sample_seconds": duration_per_sample_seconds,
    }
    return json.dumps(data, sort_keys=True).encode("utf-8")


def _compute_plan_id(canonical_bytes: bytes) -> str:
    return hashlib.sha256(canonical_bytes).hexdigest()


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

    The plan is fully deterministic: identical input parameters always
    produce an identical RunPlan, including the same plan_id.

    *task_ids* order in the plan is the order of the *tasks* argument.
    """
    task_ids = [t.task_id for t in tasks]
    total_samples = sum(t.num_samples for t in tasks)

    planned_requests = total_samples * requests_per_sample
    maximum_requests = planned_requests * (1 + max_retries)

    estimated_input_tokens = total_samples * tokens_per_sample_input
    estimated_output_tokens = total_samples * tokens_per_sample_output

    estimated_duration_seconds: float | None = total_samples * duration_per_sample_seconds

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

    canonical = _canonical_key(
        suite_id=suite_id,
        suite_version=suite_version,
        source_id=source_id,
        source_revision=source_revision,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        tasks=tasks,
        max_retries=max_retries,
        tokens_per_sample_input=tokens_per_sample_input,
        tokens_per_sample_output=tokens_per_sample_output,
        requests_per_sample=requests_per_sample,
        price_per_million_input=price_per_million_input,
        price_per_million_output=price_per_million_output,
        duration_per_sample_seconds=duration_per_sample_seconds,
    )
    plan_id = _compute_plan_id(canonical)

    return RunPlan(
        plan_id=plan_id,
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
