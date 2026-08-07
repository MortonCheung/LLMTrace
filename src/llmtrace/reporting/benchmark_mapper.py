"""Benchmark report mapper — maps execution artefacts to report models.

``build_benchmark_report_section()`` is the single public entry point.
It converts a RunPlan + BenchmarkRunResult into a BenchmarkReportSection
that is ready for JSON serialization.

Mapping rules:
1. GradeResult is joined to TaskAttempt by attempt_id (1:1).
2. Duplicate GradeResult for the same attempt_id raises ValueError.
3. ONLY TaskStatus.SUCCESS may carry a GradeResult; anything else
   (FAILURE/SKIPPED/PENDING/RUNNING) with a GradeResult → ValueError.
4. Only GradeStatus.GRADED allows raw_score / normalized_score.
5. SUCCESS without GradeResult → ungraded (no fake scores).
6. FAILURE attempt → raw_score=None, normalized_score=None.
7. UNGRADABLE/ERROR GradeResult → scores None, grade_status preserved.
8. Orphan GradeResult (no matching TaskAttempt) → ValueError.
9. Evidence UUIDs flow through as strings.
10. actual_requests prioritizes run_result.evidence_refs count.
11. planned/maximum_requests come from plan.budget.
12. estimated_cost=None stays None.
13. No total_score, capability_score, or dimension aggregation.
14. Provenance must be consistent across plan, run_result, and all children.
15. Task IDs must exist in plan.task_ids.
"""

from __future__ import annotations

from uuid import UUID

from llmtrace.benchmarks.models import (
    BenchmarkProvenance,
    BenchmarkRunResult,
    GradeResult,
    GradeStatus,
    RunPlan,
    TaskAttempt,
    TaskStatus,
    validate_provenance_consistency,
)
from llmtrace.reporting.benchmark_models import (
    _SENSITIVE_KEY_PATTERNS,
    BenchmarkReportSection,
    BenchmarkReportStatus,
    BenchmarkRunSummary,
    FailureReportItem,
    ReportFailureCategory,
    ReportGradeStatus,
    TaskReportItem,
    TaskReportStatus,
)
from llmtrace.reporting.json_safety import validate_json_mapping

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_benchmark_report_section(
    plan: RunPlan,
    run_result: BenchmarkRunResult,
) -> BenchmarkReportSection:
    """Build a BenchmarkReportSection from a RunPlan and BenchmarkRunResult.

    The single source of truth for attempts and grades is run_result:
      - run_result.task_attempts
      - run_result.grade_results

    Args:
        plan: The RunPlan that governed this run.
        run_result: The aggregated BenchmarkRunResult.

    Returns:
        A fully populated BenchmarkReportSection.

    Raises:
        ValueError: On provenance mismatch, duplicate grades, missing task IDs,
                    orphan GradeResult, grading a non-SUCCESS attempt, etc.
    """
    # ---------- Provenance validation ----------
    _validate_plan_run_result_provenance(plan, run_result)

    attempts = list(run_result.task_attempts)
    grades = list(run_result.grade_results)

    plan_task_ids = set(plan.task_ids)

    # ---------- GradeResult index (1:1, with provenance check) ----------
    grade_by_attempt: dict[str, GradeResult] = {}
    for g in grades:
        if g.attempt_id in grade_by_attempt:
            raise ValueError(
                f"Duplicate GradeResult for attempt_id='{g.attempt_id}'. "
                f"Each TaskAttempt must have at most one GradeResult."
            )
        _check_provenance(run_result, g, "GradeResult")
        grade_by_attempt[g.attempt_id] = g

    # ---------- Build TaskReportItems ----------
    tasks: list[TaskReportItem] = []
    warnings: list[str] = []
    success_count, failure_count, skip_count = 0, 0, 0
    pending_running_count = 0
    ungraded_count, ungradable_count = 0, 0

    for attempt in attempts:
        _check_provenance(run_result, attempt, "TaskAttempt")

        # Validate task_id is in plan
        if attempt.task_id not in plan_task_ids:
            raise ValueError(
                f"TaskAttempt '{attempt.attempt_id}' has task_id='{attempt.task_id}' "
                f"which is not in plan.task_ids={list(plan.task_ids)}"
            )

        # Get and pop the grade
        grade: GradeResult | None = grade_by_attempt.pop(attempt.attempt_id, None)

        # Strict grading: only SUCCESS may carry a GradeResult
        if grade is not None and attempt.status != TaskStatus.SUCCESS:
            raise ValueError(
                f"TaskAttempt '{attempt.attempt_id}' has status={attempt.status.value} "
                f"but a GradeResult (grade_id='{grade.grade_id}') is attached. "
                f"Only TaskStatus.SUCCESS may carry a GradeResult."
            )

        # Validate GradeResult.task_id matches
        if grade is not None and grade.task_id != attempt.task_id:
            raise ValueError(
                f"GradeResult.task_id='{grade.task_id}' does not match "
                f"TaskAttempt.task_id='{attempt.task_id}' for attempt_id='{attempt.attempt_id}'"
            )

        # Count statuses
        if attempt.status == TaskStatus.SUCCESS:
            success_count += 1
        elif attempt.status == TaskStatus.FAILURE:
            failure_count += 1
        elif attempt.status == TaskStatus.SKIPPED:
            skip_count += 1
        elif attempt.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            pending_running_count += 1

        task_item = _build_task_item(attempt, grade, warnings)
        tasks.append(task_item)

        # Count ungraded / ungradable
        if attempt.status == TaskStatus.SUCCESS and grade is None:
            ungraded_count += 1
        if grade is not None and grade.status in (GradeStatus.UNGRADABLE, GradeStatus.ERROR):
            ungradable_count += 1

    # ---------- Orphan GradeResults → ValueError ----------
    if grade_by_attempt:
        orphan_ids = sorted(grade_by_attempt.keys())
        raise ValueError(f"Orphan GradeResult(s) with no matching TaskAttempt: attempt_ids={orphan_ids}")

    # ---------- actual_requests ----------
    actual_requests = len(run_result.evidence_refs)
    if actual_requests == 0 and run_result.task_attempts:
        seen: set[str] = set()
        for a in run_result.task_attempts:
            for ref in a.evidence_refs:
                if ref not in seen:
                    seen.add(ref)
        actual_requests = len(seen)

    # ---------- Overall status ----------
    total = len(attempts)
    if total == 0:
        status = BenchmarkReportStatus.FAILURE
        warnings.append("no_tasks: run_result has zero task_attempts")
    elif pending_running_count > 0:
        status = BenchmarkReportStatus.INCOMPLETE
        warnings.append(f"{pending_running_count} task(s) still in pending/running state")
    elif skip_count == total:
        status = BenchmarkReportStatus.SKIPPED
    elif failure_count == total:
        status = BenchmarkReportStatus.FAILURE
    elif failure_count > 0:
        status = BenchmarkReportStatus.PARTIAL_FAILURE
    else:
        status = BenchmarkReportStatus.SUCCESS

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
# Provenance validation
# ---------------------------------------------------------------------------


def _validate_plan_run_result_provenance(plan: RunPlan, run_result: BenchmarkRunResult) -> None:
    """Validate that plan and run_result share the same provenance fields."""
    fields: list[tuple[str, str, str]] = [
        ("suite_id", plan.suite_id, run_result.suite_id),
        ("suite_version", plan.suite_version, run_result.suite_version),
        ("source_id", plan.source_id, run_result.source_id),
        ("source_revision", plan.source_revision, run_result.source_revision),
        ("adapter_id", plan.adapter_id, run_result.adapter_id),
        ("adapter_version", plan.adapter_version, run_result.adapter_version),
    ]
    for field_name, plan_val, rr_val in fields:
        if plan_val != rr_val:
            raise ValueError(f"Provenance mismatch on '{field_name}': plan has '{plan_val}', run_result has '{rr_val}'")


def _check_provenance(parent: BenchmarkRunResult, child: BenchmarkProvenance, label: str) -> None:
    """Validate that a child (TaskAttempt or GradeResult) shares provenance with run_result."""
    validate_provenance_consistency(parent, child, child_label=label)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_task_item(
    attempt: TaskAttempt,
    grade: GradeResult | None,
    warnings: list[str],
) -> TaskReportItem:
    """Build a single TaskReportItem from a TaskAttempt and optional GradeResult."""

    # Smoke task eligibility via explicit metadata flag
    capability_eligible = not _is_smoke_task_from_metadata(attempt)

    # Status mapping
    try:
        task_status = TaskReportStatus(attempt.status.value)
    except ValueError:
        task_status = TaskReportStatus.FAILURE
        warnings.append(
            f"TaskAttempt '{attempt.attempt_id}' has unknown status '{attempt.status.value}', defaulting to failure"
        )

    # Failure item
    failure_item: FailureReportItem | None = None
    if attempt.failure is not None:
        failure_item = _build_failure_item(attempt)

    # Grade fields — only GRADED on SUCCESS yields scores
    grader_id: str | None = None
    grade_status: ReportGradeStatus | None = None
    raw_score: float | None = None
    normalized_score: float | None = None

    if grade is not None:
        grader_id = grade.grader_id
        if grade.status == GradeStatus.GRADED:
            grade_status = ReportGradeStatus.GRADED
            raw_score = grade.raw_score
            normalized_score = grade.normalized_score
        elif grade.status == GradeStatus.UNGRADABLE:
            grade_status = ReportGradeStatus.UNGRADABLE
        elif grade.status == GradeStatus.ERROR:
            grade_status = ReportGradeStatus.ERROR
        else:
            grade_status = ReportGradeStatus.ERROR
            warnings.append(f"GradeResult for '{grade.attempt_id}' has unknown status '{grade.status.value}'")
    elif attempt.status == TaskStatus.SUCCESS:
        warnings.append(
            f"TaskAttempt '{attempt.attempt_id}' (task='{attempt.task_id}') is "
            f"SUCCESS but has no matching GradeResult — marked ungraded."
        )

    # Validate metadata via json_safety module
    validate_json_mapping(attempt.metadata)

    # Build item-level report items
    from llmtrace.reporting.benchmark_models import ItemReportItem

    item_report_items: list[ItemReportItem] = [
        ItemReportItem(
            item_id=ir.item_id,
            status=ir.status.value,
            raw_score=ir.raw_score,
            normalized_score=ir.normalized_score,
            grader_id=ir.grader_id,
            evidence_refs=list(ir.evidence_refs),
            error_message=ir.error_message,
            metadata=ir.metadata,
        )
        for ir in attempt.item_results
    ]

    return TaskReportItem(
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        status=task_status,
        grader_id=grader_id,
        grade_status=grade_status,
        raw_score=raw_score,
        normalized_score=normalized_score,
        evidence_refs=list(attempt.evidence_refs),
        items=item_report_items,
        failure=failure_item,
        capability_score_eligible=capability_eligible,
        metadata=dict(attempt.metadata),
    )


def _build_failure_item(attempt: TaskAttempt) -> FailureReportItem:
    """Build a FailureReportItem from a TaskAttempt with failure."""
    if attempt.failure is None:
        raise ValueError("Cannot build FailureReportItem: attempt has no failure")

    # Map category
    try:
        category = ReportFailureCategory(attempt.failure.category.value)
    except ValueError:
        category = ReportFailureCategory.UNKNOWN

    # Sanitize safe_details: only allowed keys, redact sensitive ones
    safe_details: dict[str, str] = {}
    for k, v in attempt.failure.details.items():
        if not isinstance(k, str):
            continue
        if _is_sensitive_key(k):
            safe_details[k] = "<REDACTED>"
            continue
        if isinstance(v, bool | int | float | str):
            safe_details[k] = str(v)
        elif v is None:
            safe_details[k] = "null"

    return FailureReportItem(
        error_code=attempt.failure.error_code,
        category=category,
        message=attempt.failure.message,
        retryable=attempt.failure.retryable,
        safe_details=safe_details,
    )


# ---------------------------------------------------------------------------
# Sensitive key detection
# ---------------------------------------------------------------------------


def _is_sensitive_key(key: str) -> bool:
    """Check if a key matches any sensitive pattern."""
    key_lower = key.lower().replace("-", "_")
    return any(pattern in key_lower for pattern in _SENSITIVE_KEY_PATTERNS)


# ---------------------------------------------------------------------------
# Smoke task detection
# ---------------------------------------------------------------------------


def _is_smoke_task_from_metadata(attempt: TaskAttempt) -> bool:
    """Detect smoke tasks via explicit metadata flag."""
    flag = attempt.metadata.get("llmtrace_smoke_task")
    return flag is True
