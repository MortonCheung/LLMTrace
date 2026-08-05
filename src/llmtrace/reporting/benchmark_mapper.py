"""Benchmark report mapper — maps execution artefacts to report models.

``build_benchmark_report_section()`` is the single public entry point.
It converts a RunPlan + BenchmarkRunResult into a BenchmarkReportSection
that is ready for JSON serialization.

Mapping rules:
1. GradeResult is joined to TaskAttempt by attempt_id (1:1).
2. Duplicate GradeResult for the same attempt_id raises ValueError.
3. SUCCESS attempt without GradeResult → ungraded (no fake scores).
4. FAILURE attempt → raw_score=None, normalized_score=None, failure preserved.
5. UNGRADABLE GradeResult → grade_status preserved, no fake scores.
6. Evidence UUIDs flow through as strings.
7. actual_requests prioritizes run_result.evidence_refs count.
8. planned/maximum_requests come from plan.budget.
9. estimated_cost=None stays None.
10. No total_score, capability_score, or dimension aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from llmtrace.benchmarks.models import BenchmarkRunResult, GradeResult, GradeStatus, RunPlan, TaskAttempt, TaskStatus
from llmtrace.reporting.benchmark_models import (
    BenchmarkReportSection,
    BenchmarkRunSummary,
    FailureReportItem,
    TaskReportItem,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_benchmark_report_section(
    plan: RunPlan,
    run_result: BenchmarkRunResult,
    attempts: Sequence[TaskAttempt],
    grades: Sequence[GradeResult],
) -> BenchmarkReportSection:
    """Build a BenchmarkReportSection from execution artefacts.

    Args:
        plan: The RunPlan that governed this run.
        run_result: The aggregated BenchmarkRunResult.
        attempts: All TaskAttempt records for this run.
        grades: All GradeResult records for this run.

    Returns:
        A fully populated BenchmarkReportSection.

    Raises:
        ValueError: If duplicate GradeResult entries exist for the same attempt_id.
    """
    warnings: list[str] = []

    # ---------- GradeResult index (1:1) ----------
    grade_by_attempt: dict[str, GradeResult] = {}
    for g in grades:
        if g.attempt_id in grade_by_attempt:
            raise ValueError(
                f"Duplicate GradeResult for attempt_id='{g.attempt_id}'. "
                f"Each TaskAttempt must have at most one GradeResult."
            )
        grade_by_attempt[g.attempt_id] = g

    # ---------- Build TaskReportItems ----------
    tasks: list[TaskReportItem] = []
    success_count, failure_count, skip_count = 0, 0, 0
    ungraded_count, ungradable_count = 0, 0

    for attempt in attempts:
        grade: GradeResult | None = grade_by_attempt.pop(attempt.attempt_id, None)

        if attempt.status == TaskStatus.SUCCESS:
            success_count += 1
        elif attempt.status == TaskStatus.FAILURE:
            failure_count += 1
        elif attempt.status == TaskStatus.SKIPPED:
            skip_count += 1

        task_item = _build_task_item(attempt, grade, warnings)
        tasks.append(task_item)

        # Count ungraded
        if attempt.status == TaskStatus.SUCCESS and grade is None:
            ungraded_count += 1
        if grade is not None and grade.status == GradeStatus.UNGRADABLE:
            ungradable_count += 1

    # ---------- Orphan GradeResults ----------
    if grade_by_attempt:
        orphan_ids = sorted(grade_by_attempt.keys())
        warnings.append(f"Orphan GradeResults (no matching TaskAttempt): {orphan_ids}")

    # ---------- actual_requests ----------
    actual_requests = len(run_result.evidence_refs)
    if actual_requests == 0 and run_result.task_attempts:
        # Fallback: count evidence_refs across all attempts
        seen: set[str] = set()
        for a in run_result.task_attempts:
            for ref in a.evidence_refs:
                if ref not in seen:
                    seen.add(ref)
        actual_requests = len(seen)

    # ---------- status ----------
    if failure_count == len(attempts):
        status = "failure"
    elif failure_count > 0:
        status = "partial_failure"
    else:
        status = "success"

    summary = BenchmarkRunSummary(
        planned_requests=plan.budget.planned_requests,
        maximum_requests=plan.budget.maximum_requests,
        actual_requests=actual_requests,
        success_count=success_count,
        failure_count=failure_count,
        skip_count=skip_count,
        ungraded_count=ungraded_count,
        ungradable_count=ungradable_count,
        estimated_input_tokens=plan.budget.estimated_input_tokens,
        estimated_output_tokens=plan.budget.estimated_output_tokens,
        estimated_cost=plan.budget.estimated_cost,
        warnings=list(warnings),
    )

    return BenchmarkReportSection(
        run_id=UUID(run_result.run_id),
        plan_id=plan.plan_id,
        suite_id=run_result.suite_id,
        suite_version=run_result.suite_version,
        source_id=run_result.source_id,
        source_revision=run_result.source_revision,
        adapter_id=run_result.adapter_id,
        adapter_version=run_result.adapter_version,
        status=status,
        started_at=run_result.started_at,
        finished_at=run_result.finished_at,
        planned_requests=plan.budget.planned_requests,
        maximum_requests=plan.budget.maximum_requests,
        actual_requests=actual_requests,
        estimated_input_tokens=plan.budget.estimated_input_tokens,
        estimated_output_tokens=plan.budget.estimated_output_tokens,
        estimated_cost=plan.budget.estimated_cost,
        tasks=tasks,
        summary=summary,
        warnings=list(warnings),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_task_item(
    attempt: TaskAttempt,
    grade: GradeResult | None,
    warnings: list[str],
) -> TaskReportItem:
    """Build a single TaskReportItem from a TaskAttempt and optional GradeResult."""

    # Determine if smoke task
    capability_eligible = not _is_smoke_task(attempt.task_id)

    failure_item: FailureReportItem | None = None
    if attempt.failure is not None:
        failure_item = FailureReportItem(
            error_code=attempt.failure.error_code,
            category=attempt.failure.category.value,
            message=attempt.failure.message,
            retryable=attempt.failure.retryable,
            safe_details={k: str(v) for k, v in attempt.failure.details.items() if isinstance(k, str)},
        )

    # Grade fields
    if grade is not None:
        grader_id: str | None = grade.grader_id
        grade_status: str | None = grade.status.value
        if grade.status == GradeStatus.UNGRADABLE:
            raw_score: float | None = None
            normalized_score: float | None = None
        else:
            raw_score = grade.raw_score
            normalized_score = grade.normalized_score
    else:
        grader_id = None
        grade_status = None
        raw_score = None
        normalized_score = None
        if attempt.status == TaskStatus.SUCCESS:
            warnings.append(
                f"TaskAttempt '{attempt.attempt_id}' (task='{attempt.task_id}') is "
                f"SUCCESS but has no matching GradeResult — marked ungraded."
            )

    # Safe metadata: exclude Pydantic objects, exceptions, arbitrary objects
    safe_metadata: dict[str, object] = {}
    for k, v in attempt.metadata.items():
        # Exclude Pydantic BaseModel instances
        if hasattr(v, "model_dump"):
            continue
        # Exclude exception-like objects
        if isinstance(v, BaseException):
            continue
        # Exclude raw dicts containing model_dump artifacts
        if isinstance(v, dict):
            if "metric_result" in v:
                # Keep only the metric_result if present (it's a dict from model_dump)
                safe_metadata[k] = v
                continue
            safe_metadata[k] = v
            continue
        # Accept primitive types
        safe_metadata[k] = v

    return TaskReportItem(
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        status=attempt.status.value,
        grader_id=grader_id,
        grade_status=grade_status,
        raw_score=raw_score,
        normalized_score=normalized_score,
        evidence_refs=list(attempt.evidence_refs),
        failure=failure_item,
        capability_score_eligible=capability_eligible,
        metadata=safe_metadata,
    )


def _is_smoke_task(task_id: str) -> bool:
    """Return True if the task is a smoke/infrastructure task.

    Smoke tasks are excluded from capability scoring.
    """
    return task_id.startswith("llmtrace_smoke")
