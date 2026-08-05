"""Benchmark report mapper — maps execution artefacts to report models.

``build_benchmark_report_section()`` is the single public entry point.
It converts a RunPlan + BenchmarkRunResult into a BenchmarkReportSection
that is ready for JSON serialization.

Mapping rules:
1. GradeResult is joined to TaskAttempt by attempt_id (1:1).
2. Duplicate GradeResult for the same attempt_id raises ValueError.
3. Only GRADE == GRADED allows raw_score / normalized_score.
4. SUCCESS attempt without GradeResult → ungraded (no fake scores).
5. FAILURE attempt → raw_score=None, normalized_score=None, failure preserved.
6. UNGRADABLE/ERROR GradeResult → scores None, grade_status preserved.
7. Evidence UUIDs flow through as strings.
8. actual_requests prioritizes run_result.evidence_refs count.
9. planned/maximum_requests come from plan.budget.
10. estimated_cost=None stays None.
11. No total_score, capability_score, or dimension aggregation.
12. Provenance must be consistent across plan, run_result, and all children.
13. Task IDs must exist in plan.task_ids.
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
)
from llmtrace.reporting.benchmark_models import (
    _SENSITIVE_KEY_PATTERNS,
    BenchmarkReportSection,
    BenchmarkReportStatus,
    BenchmarkRunSummary,
    FailureReportItem,
    JsonSafeScalar,
    ReportFailureCategory,
    ReportGradeStatus,
    TaskReportItem,
    TaskReportStatus,
)

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
        ValueError: On provenance mismatch, duplicate grades, missing task IDs, etc.
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
        _validate_provenance(run_result, g, "GradeResult")
        grade_by_attempt[g.attempt_id] = g

    # ---------- Build TaskReportItems ----------
    tasks: list[TaskReportItem] = []
    warnings: list[str] = []
    success_count, failure_count, skip_count = 0, 0, 0
    pending_running_count = 0
    ungraded_count, ungradable_count = 0, 0

    for attempt in attempts:
        _validate_provenance(run_result, attempt, "TaskAttempt")

        # Validate task_id is in plan
        if attempt.task_id not in plan_task_ids:
            raise ValueError(
                f"TaskAttempt '{attempt.attempt_id}' has task_id='{attempt.task_id}' "
                f"which is not in plan.task_ids={list(plan.task_ids)}"
            )

        # Validate GradeResult.task_id matches
        grade: GradeResult | None = grade_by_attempt.pop(attempt.attempt_id, None)
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

    # ---------- Orphan GradeResults ----------
    if grade_by_attempt:
        orphan_ids = sorted(grade_by_attempt.keys())
        warnings.append(f"Orphan GradeResults (no matching TaskAttempt): {orphan_ids}")

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


def _validate_provenance(parent: BenchmarkRunResult, child: BenchmarkProvenance, label: str) -> None:
    """Validate that a child (TaskAttempt or GradeResult) shares provenance with run_result."""
    if child.suite_id != parent.suite_id:
        raise ValueError(f"{label} suite_id mismatch: '{child.suite_id}' != '{parent.suite_id}'")
    if child.suite_version != parent.suite_version:
        raise ValueError(f"{label} suite_version mismatch: '{child.suite_version}' != '{parent.suite_version}'")
    if child.source_id != parent.source_id:
        raise ValueError(f"{label} source_id mismatch: '{child.source_id}' != '{parent.source_id}'")
    if child.source_revision != parent.source_revision:
        raise ValueError(f"{label} source_revision mismatch: '{child.source_revision}' != '{parent.source_revision}'")
    if child.adapter_id != parent.adapter_id:
        raise ValueError(f"{label} adapter_id mismatch: '{child.adapter_id}' != '{parent.adapter_id}'")
    if child.adapter_version != parent.adapter_version:
        raise ValueError(f"{label} adapter_version mismatch: '{child.adapter_version}' != '{parent.adapter_version}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_task_item(
    attempt: TaskAttempt,
    grade: GradeResult | None,
    warnings: list[str],
) -> TaskReportItem:
    """Build a single TaskReportItem from a TaskAttempt and optional GradeResult."""

    # Smoke task eligibility via explicit metadata key, not name guessing
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

    # Grade fields — only GRADED yields scores
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

    # Recursive JSON-safe metadata
    safe_metadata: dict[str, object] = _sanitize_metadata(attempt.metadata)

    return TaskReportItem(
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        status=task_status,
        grader_id=grader_id,
        grade_status=grade_status,
        raw_score=raw_score,
        normalized_score=normalized_score,
        evidence_refs=list(attempt.evidence_refs),
        failure=failure_item,
        capability_score_eligible=capability_eligible,
        metadata=safe_metadata,
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
        # Redact sensitive keys
        if _is_sensitive_key(k):
            safe_details[k] = "<REDACTED>"
            continue
        # Only accept JSON-scalar values
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
# JSON-safe metadata sanitization
# ---------------------------------------------------------------------------

# Types that are explicitly forbidden and must be rejected (not str() converted)
_FORBIDDEN_METADATA_TYPES = (bytes, set, tuple, frozenset, bytearray)


def _sanitize_metadata(raw: dict[str, object]) -> dict[str, object]:
    """Recursively clean metadata into JSON-safe types only.

    Allows: None, bool, int, float, str, list[...], dict[str, ...]
    Rejects (ValueError): Pydantic BaseModel, BaseException, bytes, set, tuple,
                          custom objects, nested non-JSON types.
    """
    result: dict[str, object] = {}
    for k, v in raw.items():
        result[k] = _sanitize_value(v, path=f"metadata['{k}']")
    return result


def _sanitize_value(value: object, path: str = "metadata") -> JsonSafeScalar | list[object] | dict[str, object]:
    """Recursively sanitize a single value into JsonSafeValue.

    Raises ValueError for forbidden types.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value

    # Reject forbidden container types
    if isinstance(value, _FORBIDDEN_METADATA_TYPES):
        raise ValueError(f"{path}: forbidden type {type(value).__name__} — not JSON-safe")

    # Reject Pydantic BaseModel instances
    if hasattr(value, "model_dump"):
        raise ValueError(f"{path}: Pydantic model {type(value).__name__} — not allowed in metadata")

    # Reject exceptions
    if isinstance(value, BaseException):
        raise ValueError(f"{path}: Exception {type(value).__name__} — not allowed in metadata")

    # Reject arbitrary custom objects (anything not in allowed categories)
    if isinstance(value, list):
        return [_sanitize_value(item, path=f"{path}[{i}]") for i, item in enumerate(value)]

    if isinstance(value, dict):
        return {str(dk): _sanitize_value(dv, path=f"{path}.{dk}") for dk, dv in value.items()}

    raise ValueError(f"{path}: unknown type {type(value).__name__} — not JSON-safe")


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


_SMOKE_METADATA_KEY = "llmtrace_smoke_task"


def _is_smoke_task_from_metadata(attempt: TaskAttempt) -> bool:
    """Detect smoke tasks via explicit metadata flag, not name guessing.

    A task is a smoke task if its metadata contains:
      {"llmtrace_smoke_task": True}
    """
    flag = attempt.metadata.get(_SMOKE_METADATA_KEY)
    return flag is True
