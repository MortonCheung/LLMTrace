"""Shared factories for Behavior Drift tests."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.analysis.behavior_snapshot import BehaviorSnapshotBuilder
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkItemResult,
    BenchmarkRunResult,
    FailureCategory,
    ItemStatus,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)

DEFAULT_GEN_CONFIG: dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}
DEFAULT_SUITE_ID = "llmtrace_quick_v1"
DEFAULT_SUITE_VERSION = "0.1.0"
DEFAULT_SOURCE_ID = "gsm8k"
DEFAULT_SOURCE_REVISION = "gsm8k-main-2023"
DEFAULT_ADAPTER_ID = "llmtrace-quick-v1"
DEFAULT_ADAPTER_VERSION = "0.1.0"
DEFAULT_POLICY_ID = "llmtrace-capability-v1"
DEFAULT_POLICY_VERSION = "0.1.0"

_DIM_WEIGHTS = {
    CapabilityDimension.REASONING: 0.25,
    CapabilityDimension.CODING: 0.20,
    CapabilityDimension.MATH_SCIENCE: 0.15,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 0.15,
}

_ALL_DIMENSIONS = (
    CapabilityDimension.REASONING,
    CapabilityDimension.CODING,
    CapabilityDimension.MATH_SCIENCE,
    CapabilityDimension.INSTRUCTION_FOLLOWING,
)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_evidence(
    *,
    response_text: str = "The answer is 42.",
    response_model: str = "gpt-x",
    finish_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_ms: float = 100.0,
) -> HTTPEvidence:
    return HTTPEvidence(
        evidence_id=uuid4(),
        request_method="POST",
        request_url_redacted="https://api.example.com/v1/chat/completions",
        request_path="/v1/chat/completions",
        request_headers_redacted={},
        response_text=response_text,
        response_model=response_model,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_latency_ms=latency_ms,
        http_status=200,
    )


def make_profile(
    scores: dict[CapabilityDimension, float] | None = None,
    *,
    statuses: dict[CapabilityDimension, DimensionScoreStatus] | None = None,
    coverage_weight: float = 0.75,
) -> CapabilityProfile:
    score_map = scores if scores is not None else dict.fromkeys(_ALL_DIMENSIONS, 1.0)
    status_map = statuses or {}
    return CapabilityProfile(
        scoring_policy_id=DEFAULT_POLICY_ID,
        scoring_policy_version=DEFAULT_POLICY_VERSION,
        dimensions=tuple(
            DimensionScoreResult(
                dimension=d,
                status=status_map.get(d, DimensionScoreStatus.UNCALIBRATED),
                raw_normalized_score=score_map.get(d, 0.0),
                global_weight=_DIM_WEIGHTS[d],
            )
            for d in _ALL_DIMENSIONS
        ),
        coverage_weight=coverage_weight,
    )


def make_snapshot(
    *,
    run_id: str = "run-a",
    target_id: str = "target-api",
    candidate_model_id: str = "gpt-x",
    items: list[dict[str, Any]] | None = None,
    profile: CapabilityProfile | None = None,
    generation_config: dict[str, Any] | None = None,
    suite_id: str = DEFAULT_SUITE_ID,
    suite_version: str = DEFAULT_SUITE_VERSION,
    source_id: str = DEFAULT_SOURCE_ID,
    source_revision: str = DEFAULT_SOURCE_REVISION,
    adapter_id: str = DEFAULT_ADAPTER_ID,
    adapter_version: str = DEFAULT_ADAPTER_VERSION,
) -> BehaviorRunSnapshot:
    """Build a BehaviorRunSnapshot end-to-end through the real builder.

    Each item spec is::

        {
            "task_id": "gsm8k_quick_v1",
            "source_sample_id": "sample-1",
            "status": ItemStatus.GRADED,
            "score": 1.0,
            "response_text": "The answer is 42.",
            "response_model": "gpt-x",
            "finish_reason": "stop",
        }
    """
    if items is None:
        items = [_default_item(i) for i in range(4)]

    evidence_list: list[HTTPEvidence] = []
    attempts: list[TaskAttempt] = []
    attempts_by_task: dict[str, TaskAttempt] = {}

    for spec in items:
        task_id = spec["task_id"]
        sample_id = spec["source_sample_id"]
        attempt = attempts_by_task.get(task_id)
        if attempt is None:
            attempt = TaskAttempt(
                attempt_id=str(uuid4()),
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                source_id=source_id,
                source_revision=source_revision,
                suite_id=suite_id,
                suite_version=suite_version,
                adapter_id=adapter_id,
                adapter_version=adapter_version,
            )
            attempts_by_task[task_id] = attempt
            attempts.append(attempt)

        ev = make_evidence(
            response_text=spec.get("response_text", "The answer is 42."),
            response_model=spec.get("response_model", "gpt-x"),
            finish_reason=spec.get("finish_reason", "stop"),
        )
        evidence_list.append(ev)

        item = BenchmarkItemResult(
            item_id=f"{task_id}-{sample_id}",
            task_id=task_id,
            attempt_id=attempt.attempt_id,
            source_sample_id=sample_id,
            input_sha256=spec.get("input_sha256") or _sha(sample_id),
            status=_item_status(spec),
            raw_score=_item_score(spec),
            normalized_score=_item_score(spec),
            evidence_refs=[str(ev.evidence_id)],
            failure=_item_failure(spec),
            error_message=_item_error_message(spec),
        )
        attempt.item_results.append(item)
        attempt.evidence_refs.append(str(ev.evidence_id))

    run = BenchmarkRunResult(
        run_id=run_id,
        task_attempts=attempts,
        grade_results=[],
        source_id=source_id,
        source_revision=source_revision,
        suite_id=suite_id,
        suite_version=suite_version,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        evidence_refs=[str(ev.evidence_id) for ev in evidence_list],
    )

    builder = BehaviorSnapshotBuilder()
    return builder.build(
        run_results=[run],
        profile=profile if profile is not None else make_profile(),
        evidence=evidence_list,
        target_id=target_id,
        candidate_model_id=candidate_model_id,
        generation_config=generation_config if generation_config is not None else DEFAULT_GEN_CONFIG,
    )


def _default_item(i: int) -> dict[str, Any]:
    return {
        "task_id": "gsm8k_quick_v1",
        "source_sample_id": f"sample-{i}",
        "status": ItemStatus.GRADED,
        "score": 1.0,
        "response_text": f"The answer is {i}.",
        "response_model": "gpt-x",
        "finish_reason": "stop",
    }


def _item_status(spec: dict[str, Any]) -> ItemStatus:
    return spec.get("status", ItemStatus.GRADED)


def _item_score(spec: dict[str, Any]) -> float:
    status = _item_status(spec)
    if status in (ItemStatus.FAILURE, ItemStatus.UNGRADABLE):
        return 0.0
    return float(spec.get("score", 1.0))


def _item_failure(spec: dict[str, Any]) -> AdapterFailure | None:
    if _item_status(spec) != ItemStatus.FAILURE:
        return None
    return AdapterFailure(
        error_code="PROVIDER_TIMEOUT",
        category=FailureCategory.PROVIDER,
        message="simulated provider failure",
    )


def _item_error_message(spec: dict[str, Any]) -> str | None:
    if _item_status(spec) == ItemStatus.UNGRADABLE:
        return spec.get("error_message", "ungradable")
    return spec.get("error_message")
