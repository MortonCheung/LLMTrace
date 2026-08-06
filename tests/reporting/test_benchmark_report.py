"""Tests for benchmark report models and mapper (v2 — strict).

Covers:
1. Successful task mapping
2. Failed task mapping
3. SUCCESS but missing GradeResult
4. UNGRADABLE GradeResult (scores None)
5. ERROR GradeResult (scores None)
6. Duplicate GradeResult rejection
7. Provenance mismatch rejection (plan vs run_result, child vs parent)
8. GradeResult.task_id mismatch
9. TaskAttempt.task_id not in plan
10. Evidence UUID preservation
11. estimated_cost=None preservation
12. datetime JSON serialization (ISO-8601)
13. JSON roundtrip
14. Recursive JSON-safe metadata — rejects Pydantic, Exception, bytes, custom objects
15. Sensitive key redaction in failure safe_details
16. Smoke task detection via metadata flag
17. Status rules: incomplete, skipped, no_tasks
18. No total_score or capability_score in model fields
19. Strict Golden Test (fixture must pre-exist, full comparison)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel

from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkRunResult,
    BudgetEstimate,
    FailureCategory,
    GradeResult,
    GradeStatus,
    RunPlan,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.benchmark_models import (
    BenchmarkReportSection,
    BenchmarkRunSummary,
)
from llmtrace.reporting.json_safety import sanitize_json_value

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _provenance() -> dict[str, str]:
    return {
        "suite_id": "test-suite",
        "suite_version": "1.0.0",
        "source_id": "test-source",
        "source_revision": "abc123",
        "adapter_id": "lm-eval",
        "adapter_version": "0.4.12",
    }


def _make_run_plan(plan_id: str = "test-plan-001", **overrides: str) -> RunPlan:
    p = _provenance()
    p.update(overrides)
    task_ids_val = p.pop("task_ids", None) if "task_ids" in overrides else None
    task_ids = task_ids_val if task_ids_val is not None else ["task_a", "task_b"]
    # Remove task_ids from p so it doesn't conflict with explicit kwarg
    p.pop("task_ids", None)
    return RunPlan(
        plan_id=plan_id,
        task_ids=task_ids,
        total_samples=4,
        budget=BudgetEstimate(
            planned_requests=4,
            maximum_requests=4,
            estimated_cost=None,
        ),
        **{k: v for k, v in p.items() if k in RunPlan.model_fields},
    )


def _make_success_attempt(
    attempt_id: str,
    task_id: str,
    evidence_refs: list[str] | None = None,
    **overrides: str,
) -> TaskAttempt:
    p = _provenance()
    p.update(overrides)
    return TaskAttempt(
        attempt_id=attempt_id,
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        evidence_refs=evidence_refs or [str(uuid4())],
        **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
    )


def _make_failure_attempt(
    attempt_id: str,
    task_id: str,
    error_code: str = "TEST_ERROR",
    **overrides: str,
) -> TaskAttempt:
    p = _provenance()
    p.update(overrides)
    return TaskAttempt(
        attempt_id=attempt_id,
        task_id=task_id,
        status=TaskStatus.FAILURE,
        evidence_refs=[str(uuid4())],
        failure=AdapterFailure(
            error_code=error_code,
            category=FailureCategory.ADAPTER,
            message="Test failure message",
            retryable=False,
        ),
        **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
    )


def _make_grade(attempt_id: str, task_id: str, raw_score: float = 0.75, **overrides: str) -> GradeResult:
    p = _provenance()
    p.update(overrides)
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=raw_score,
        normalized_score=raw_score,
        **{k: v for k, v in p.items() if k in GradeResult.model_fields},
    )


def _make_ungradable_grade(attempt_id: str, task_id: str, **overrides: str) -> GradeResult:
    p = _provenance()
    p.update(overrides)
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=0.0,
        normalized_score=0.0,
        status=GradeStatus.UNGRADABLE,
        error_message="Cannot grade",
        **{k: v for k, v in p.items() if k in GradeResult.model_fields},
    )


def _make_error_grade(attempt_id: str, task_id: str, **overrides: str) -> GradeResult:
    p = _provenance()
    p.update(overrides)
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=0.0,
        normalized_score=0.0,
        status=GradeStatus.ERROR,
        error_message="Grading error",
        **{k: v for k, v in p.items() if k in GradeResult.model_fields},
    )


def _make_run_result(
    attempts: list[TaskAttempt],
    grades: list[GradeResult],
    run_id: str | None = None,
    evidence_refs: list[str] | None = None,
    **overrides: str,
) -> BenchmarkRunResult:
    p = _provenance()
    p.update(overrides)
    return BenchmarkRunResult(
        run_id=run_id or str(uuid4()),
        task_attempts=attempts,
        grade_results=grades,
        evidence_refs=evidence_refs or [str(uuid4())],
        **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
    )


# ---------------------------------------------------------------------------
# Tests: Successful mapping
# ---------------------------------------------------------------------------


class TestSuccessfulMapping:
    def test_success_task_with_grade(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.8)

        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)
        section = build_benchmark_report_section(plan, run_result)

        assert section.status.value == "success"
        assert len(section.tasks) == 1
        task = section.tasks[0]
        assert task.task_id == "task_a"
        assert task.status.value == "success"
        assert task.grader_id == "exact_match"
        assert task.grade_status is not None
        assert task.grade_status.value == "graded"
        assert task.raw_score == 0.8
        assert task.normalized_score == 0.8
        assert task.failure is None
        assert task.capability_score_eligible is True

    def test_multiple_tasks_all_success(self) -> None:
        plan = _make_run_plan()
        a1 = _make_success_attempt("att-1", "task_a")
        a2 = _make_success_attempt("att-2", "task_b")
        g1 = _make_grade("att-1", "task_a", 0.9)
        g2 = _make_grade("att-2", "task_b", 0.7)

        run_result = _make_run_result([a1, a2], [g1, g2])
        section = build_benchmark_report_section(plan, run_result)

        assert section.status.value == "success"
        assert section.summary.success_count == 2
        assert section.summary.failure_count == 0
        assert len(section.tasks) == 2


# ---------------------------------------------------------------------------
# Tests: Failure mapping
# ---------------------------------------------------------------------------


class TestFailureMapping:
    def test_failure_task_preserves_error(self) -> None:
        plan = _make_run_plan()
        attempt = _make_failure_attempt("att-1", "task_a", "LM_EVAL_OPTIONS_INCONSISTENT")
        run_result = _make_run_result([attempt], [], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        assert section.status.value == "failure"
        task = section.tasks[0]
        assert task.status.value == "failure"
        assert task.raw_score is None
        assert task.normalized_score is None
        assert task.failure is not None
        assert task.failure.error_code == "LM_EVAL_OPTIONS_INCONSISTENT"
        assert task.failure.category.value == "adapter"

    def test_partial_failure_status(self) -> None:
        plan = _make_run_plan()
        a1 = _make_success_attempt("att-1", "task_a")
        a2 = _make_failure_attempt("att-2", "task_b")
        g1 = _make_grade("att-1", "task_a", 1.0)

        run_result = _make_run_result([a1, a2], [g1])
        section = build_benchmark_report_section(plan, run_result)

        assert section.status.value == "partial_failure"
        assert section.summary.success_count == 1
        assert section.summary.failure_count == 1


# ---------------------------------------------------------------------------
# Tests: Ungraded / UNGRADABLE / ERROR
# ---------------------------------------------------------------------------


class TestUngradedHandling:
    def test_success_without_grade_is_ungraded(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        run_result = _make_run_result([attempt], [], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        assert task.status.value == "success"
        assert task.raw_score is None
        assert task.normalized_score is None
        assert task.grader_id is None
        assert task.grade_status is None
        assert section.summary.ungraded_count == 1
        assert any("ungraded" in w for w in section.warnings)

    def test_ungradable_grade_scores_none(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_ungradable_grade("att-1", "task_a")
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        assert task.grade_status is not None
        assert task.grade_status.value == "ungradable"
        assert task.raw_score is None
        assert task.normalized_score is None
        assert section.summary.ungradable_count == 1

    def test_error_grade_scores_none(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_error_grade("att-1", "task_a")
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        assert task.grade_status is not None
        assert task.grade_status.value == "error"
        assert task.raw_score is None
        assert task.normalized_score is None


# ---------------------------------------------------------------------------
# Tests: Provenance validation
# ---------------------------------------------------------------------------


class TestProvenanceValidation:
    def test_plan_run_result_provenance_mismatch(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        run_result = _make_run_result([attempt], [], suite_id="different-suite")
        with pytest.raises(ValueError, match="Provenance mismatch"):
            build_benchmark_report_section(plan, run_result)

    def test_attempt_provenance_mismatch(self) -> None:
        plan = _make_run_plan()
        # Test child vs parent: attempt has wrong suite_id
        attempt_bad = _make_success_attempt("att-1", "task_a", suite_id="wrong")
        run_result = _make_run_result([attempt_bad], [])
        with pytest.raises(ValueError, match="TaskAttempt suite_id mismatch"):
            build_benchmark_report_section(plan, run_result)

    def test_grade_provenance_mismatch(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", suite_id="wrong")
        run_result = _make_run_result([attempt], [grade])
        with pytest.raises(ValueError, match="GradeResult suite_id mismatch"):
            build_benchmark_report_section(plan, run_result)

    def test_grade_task_id_mismatch(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_b")  # different task_id
        run_result = _make_run_result([attempt], [grade])
        with pytest.raises(ValueError, match="does not match"):
            build_benchmark_report_section(plan, run_result)

    def test_attempt_task_id_not_in_plan(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_unknown")
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="not in plan.task_ids"):
            build_benchmark_report_section(plan, run_result)


# ---------------------------------------------------------------------------
# Tests: Duplicate GradeResult
# ---------------------------------------------------------------------------


class TestDuplicateGradeResult:
    def test_duplicate_grade_result_raises(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        g1 = _make_grade("att-1", "task_a", 0.5)
        g2 = _make_grade("att-1", "task_a", 0.9)
        run_result = _make_run_result([attempt], [g1, g2])

        with pytest.raises(ValueError, match="Duplicate GradeResult"):
            build_benchmark_report_section(plan, run_result)


# ---------------------------------------------------------------------------
# Tests: Status rules
# ---------------------------------------------------------------------------


class TestStatusRules:
    def test_all_skipped(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        a1 = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SKIPPED,
            evidence_refs=[],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([a1], [])
        section = build_benchmark_report_section(plan, run_result)
        assert section.status.value == "skipped"

    def test_pending_causes_incomplete(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        a1 = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.PENDING,
            evidence_refs=[],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([a1], [])
        section = build_benchmark_report_section(plan, run_result)
        assert section.status.value == "incomplete"

    def test_zero_tasks_is_failure_with_warning(self) -> None:
        plan = _make_run_plan()
        run_result = _make_run_result([], [])
        section = build_benchmark_report_section(plan, run_result)
        assert section.status.value == "failure"
        assert any("no_tasks" in w for w in section.warnings)


# ---------------------------------------------------------------------------
# Tests: Smoke task
# ---------------------------------------------------------------------------


class TestSmokeTask:
    def test_smoke_task_not_capability_eligible_via_metadata(self) -> None:
        plan = _make_run_plan(task_ids=["llmtrace_smoke"])
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-smoke",
            task_id="llmtrace_smoke",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"llmtrace_smoke_task": True},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        grade = _make_grade("att-smoke", "llmtrace_smoke", 1.0)
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        assert task.capability_score_eligible is False

    def test_no_smoke_metadata_is_capability_eligible(self) -> None:
        plan = _make_run_plan(task_ids=["llmtrace_smoke"])
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-smoke-2",
            task_id="llmtrace_smoke",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={},  # No smoke flag
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        grade = _make_grade("att-smoke-2", "llmtrace_smoke", 1.0)
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        # Without metadata flag, even a task named "llmtrace_smoke" is eligible
        assert task.capability_score_eligible is True


# ---------------------------------------------------------------------------
# Tests: JSON-safe metadata
# ---------------------------------------------------------------------------


class _FakePydanticModel(BaseModel):
    x: int = 0


class _CustomObject:
    pass


class TestMetadataSafety:
    def test_pydantic_model_in_metadata_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"bad": _FakePydanticModel(x=1)},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="Pydantic model"):
            build_benchmark_report_section(plan, run_result)

    def test_exception_in_metadata_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"bad": ValueError("test")},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="Exception"):
            build_benchmark_report_section(plan, run_result)

    def test_custom_object_in_metadata_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"bad": _CustomObject()},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="not JSON-safe"):
            build_benchmark_report_section(plan, run_result)

    def test_bytes_in_metadata_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"bad": b"binary data"},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="forbidden type"):
            build_benchmark_report_section(plan, run_result)

    def test_set_in_metadata_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"bad": {1, 2, 3}},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="forbidden type"):
            build_benchmark_report_section(plan, run_result)

    def test_nested_non_json_value_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={"nested": {"inner": _CustomObject()}},
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [])
        with pytest.raises(ValueError, match="not JSON-safe"):
            build_benchmark_report_section(plan, run_result)

    def test_valid_metadata_passes(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            metadata={
                "str_field": "hello",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "null_field": None,
                "list_field": [1, "two", 3.0, None],
                "dict_field": {"a": 1, "b": "two"},
            },
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        grade = _make_grade("att-1", "task_a", 0.5)
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        assert task.metadata["str_field"] == "hello"
        assert task.metadata["int_field"] == 42
        assert task.metadata["list_field"] == [1, "two", 3.0, None]


# ---------------------------------------------------------------------------
# Tests: Sensitive key redaction
# ---------------------------------------------------------------------------


class TestSensitiveKeys:
    def test_api_key_redacted(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.FAILURE,
            evidence_refs=[str(uuid4())],
            failure=AdapterFailure(
                error_code="TEST",
                category=FailureCategory.ADAPTER,
                message="fail",
                details={"api_key": "sk-12345"},
            ),
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        assert section.tasks[0].failure is not None
        assert section.tasks[0].failure.safe_details.get("api_key") == "<REDACTED>"

    def test_authorization_redacted(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.FAILURE,
            evidence_refs=[str(uuid4())],
            failure=AdapterFailure(
                error_code="TEST",
                category=FailureCategory.ADAPTER,
                message="fail",
                details={"Authorization": "Bearer token"},
            ),
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        run_result = _make_run_result([attempt], [], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        assert section.tasks[0].failure is not None
        assert section.tasks[0].failure.safe_details.get("Authorization") == "<REDACTED>"


# ---------------------------------------------------------------------------
# Tests: Evidence UUID, estimated_cost=None
# ---------------------------------------------------------------------------


class TestEvidenceAndCost:
    def test_evidence_refs_flow_through(self) -> None:
        ev1 = str(uuid4())
        ev2 = str(uuid4())
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a", [ev1, ev2])
        grade = _make_grade("att-1", "task_a", 0.8)
        run_result = _make_run_result([attempt], [grade], evidence_refs=[ev1, ev2])

        section = build_benchmark_report_section(plan, run_result)
        task = section.tasks[0]
        assert ev1 in task.evidence_refs
        assert ev2 in task.evidence_refs

    def test_estimated_cost_none_stays_none(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        assert section.estimated_cost is None
        assert section.summary.estimated_cost is None


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_datetime_is_iso8601(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)
        run_result = _make_run_result(
            [attempt],
            [grade],
            evidence_refs=attempt.evidence_refs,
        )
        run_result.started_at = datetime(2026, 6, 15, 12, 30, 0, tzinfo=UTC)
        run_result.finished_at = datetime(2026, 6, 15, 12, 31, 0, tzinfo=UTC)

        section = build_benchmark_report_section(plan, run_result)
        data = section.model_dump(mode="json")
        assert isinstance(data["started_at"], str)
        assert "2026-06-15T12:30:00" in data["started_at"]

    def test_uuid_is_string_in_json(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)

        section = build_benchmark_report_section(plan, run_result)
        data = section.model_dump(mode="json")
        assert isinstance(data["run_id"], str)

    def test_json_roundtrip(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)
        run_result = _make_run_result(
            [attempt],
            [grade],
            evidence_refs=attempt.evidence_refs,
        )
        run_result.started_at = datetime(2026, 1, 1, tzinfo=UTC)
        run_result.finished_at = datetime(2026, 1, 1, 1, tzinfo=UTC)

        section = build_benchmark_report_section(plan, run_result)
        # Use model_validate_json for roundtrip with string coercion
        restored = BenchmarkReportSection.model_validate_json(section.model_dump_json())
        assert restored.run_id == section.run_id
        assert restored.status == section.status


# ---------------------------------------------------------------------------
# Tests: No total_score / capability_score in model fields
# ---------------------------------------------------------------------------


class TestNoScoringFields:
    def test_no_total_score_or_capability_score(self) -> None:
        fields = BenchmarkReportSection.model_fields
        assert "total_score" not in fields
        assert "capability_score" not in fields
        summary_fields = BenchmarkRunSummary.model_fields
        assert "total_score" not in summary_fields
        assert "capability_score" not in summary_fields


# ---------------------------------------------------------------------------
# Tests: Strict grading rules (only SUCCESS may carry a GradeResult)
# ---------------------------------------------------------------------------


class TestStrictGradingRules:
    def test_failure_with_grade_raises(self) -> None:
        plan = _make_run_plan()
        attempt = _make_failure_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a")
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)
        with pytest.raises(ValueError, match="Only TaskStatus.SUCCESS"):
            build_benchmark_report_section(plan, run_result)

    def test_skipped_with_grade_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SKIPPED,
            evidence_refs=[],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        grade = _make_grade("att-1", "task_a")
        run_result = _make_run_result([attempt], [grade])
        with pytest.raises(ValueError, match="Only TaskStatus.SUCCESS"):
            build_benchmark_report_section(plan, run_result)

    def test_pending_with_grade_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.PENDING,
            evidence_refs=[],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        grade = _make_grade("att-1", "task_a")
        run_result = _make_run_result([attempt], [grade])
        with pytest.raises(ValueError, match="Only TaskStatus.SUCCESS"):
            build_benchmark_report_section(plan, run_result)

    def test_running_with_grade_raises(self) -> None:
        plan = _make_run_plan()
        p = _provenance()
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.RUNNING,
            evidence_refs=[],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        grade = _make_grade("att-1", "task_a")
        run_result = _make_run_result([attempt], [grade])
        with pytest.raises(ValueError, match="Only TaskStatus.SUCCESS"):
            build_benchmark_report_section(plan, run_result)


# ---------------------------------------------------------------------------
# Tests: Orphan GradeResult
# ---------------------------------------------------------------------------


class TestOrphanGradeResult:
    def test_orphan_grade_result_raises(self) -> None:
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-unmatched", "task_a")  # attempt_id not in attempts
        run_result = _make_run_result([attempt], [grade], evidence_refs=attempt.evidence_refs)
        with pytest.raises(ValueError, match="Orphan GradeResult"):
            build_benchmark_report_section(plan, run_result)


# ---------------------------------------------------------------------------
# Tests: sanitize_json_value direct (model-level)
# ---------------------------------------------------------------------------


class TestJsonSafetyModelLevel:
    def test_non_string_dict_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-string dict key"):
            sanitize_json_value({"key": "ok", 1: "bad"}, "test")

    def test_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="datetime-like type"):
            sanitize_json_value(datetime.now(), "test")

    def test_bytearray_raises(self) -> None:
        with pytest.raises(ValueError, match="forbidden type"):
            sanitize_json_value(bytearray(b"test"), "test")

    def test_frozenset_raises(self) -> None:
        with pytest.raises(ValueError, match="forbidden type"):
            sanitize_json_value(frozenset([1, 2]), "test")

    def test_legal_values_pass(self) -> None:
        # None of these should raise
        sanitize_json_value(None, "test")
        sanitize_json_value(True, "test")
        sanitize_json_value(42, "test")
        sanitize_json_value(3.14, "test")
        sanitize_json_value("hello", "test")
        sanitize_json_value([1, "a", None], "test")
        sanitize_json_value({"key": [1, 2]}, "test")

    # --- Non-finite float rejection ---

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite float"):
            sanitize_json_value(float("nan"), "$")

    def test_pos_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite float"):
            sanitize_json_value(float("inf"), "$")

    def test_neg_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite float"):
            sanitize_json_value(float("-inf"), "$")

    # --- Top-level key validation ---

    def test_top_level_int_key_raises(self) -> None:
        from llmtrace.reporting.json_safety import validate_json_mapping

        with pytest.raises(ValueError, match="non-string dict key"):
            validate_json_mapping({1: "a"})  # type: ignore[arg-type]

    def test_nested_int_key_raises(self) -> None:
        from llmtrace.reporting.json_safety import validate_json_mapping

        with pytest.raises(ValueError, match="non-string dict key"):
            validate_json_mapping({"a": {1: "b"}})  # type: ignore[dict-item]

    def test_key_conflict_raises(self) -> None:
        """{1: "a", "1": "b"} would silently overwrite after str() conversion."""
        from llmtrace.reporting.json_safety import validate_json_mapping

        with pytest.raises(ValueError, match="non-string dict key"):
            validate_json_mapping({1: "a", "1": "b"})  # type: ignore[arg-type]

    # --- Additional forbidden types ---

    def test_tuple_raises(self) -> None:
        with pytest.raises(ValueError, match="forbidden type"):
            sanitize_json_value((1, 2), "$")

    # --- TaskReportItem metadata validator: mode="before" ---

    def test_task_report_item_int_key_metadata_raises(self) -> None:
        """TaskReportItem with int key in metadata must fail at construction."""
        from llmtrace.reporting.benchmark_models import TaskReportItem, TaskReportStatus

        with pytest.raises(ValueError, match="non-string dict key"):
            TaskReportItem(
                task_id="test",
                attempt_id="att-1",
                status=TaskReportStatus.SUCCESS,
                metadata={1: "a", "1": "b"},
            )

    def test_task_report_item_metadata_roundtrip(self) -> None:
        """Legal JSON roundtrip through TaskReportItem metadata."""
        from llmtrace.reporting.benchmark_models import TaskReportItem, TaskReportStatus

        data: dict[str, object] = {
            "str_": "hello",
            "int_": 42,
            "float_": 3.14,
            "bool_": True,
            "none_": None,
            "list_": [1, "two", None],
            "dict_": {"nested": "value"},
        }
        item = TaskReportItem(
            task_id="test",
            attempt_id="att-1",
            status=TaskReportStatus.SUCCESS,
            metadata=data,
        )
        assert item.metadata == data


# ---------------------------------------------------------------------------
# Golden Test: strict full comparison
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "benchmark_report_golden.json"


class TestGoldenFixture:
    """Golden test: known inputs → deterministic JSON output.

    The fixture MUST pre-exist.  Any field change causes failure.
    """

    def test_golden_matches_fixture(self, golden_fixture_path: Path) -> None:
        assert golden_fixture_path.exists(), (
            f"Golden fixture not found at {golden_fixture_path}. It must be committed before running this test."
        )

        run_id = "11111111-1111-1111-1111-111111111111"
        ev1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        ev2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        p = _provenance()
        plan = RunPlan(
            plan_id="golden-plan",
            task_ids=["task_a"],
            total_samples=2,
            budget=BudgetEstimate(planned_requests=2, maximum_requests=2, estimated_cost=None),
            **{k: v for k, v in p.items() if k in RunPlan.model_fields},
        )

        a1 = TaskAttempt(
            attempt_id="attempt-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[ev1, ev2],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )

        g1 = GradeResult(
            grade_id="grade-1",
            attempt_id="attempt-1",
            task_id="task_a",
            grader_id="exact_match",
            raw_score=0.75,
            normalized_score=0.75,
            **{k: v for k, v in p.items() if k in GradeResult.model_fields},
        )

        run_result = BenchmarkRunResult(
            run_id=run_id,
            task_attempts=[a1],
            grade_results=[g1],
            evidence_refs=[ev1, ev2],
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
        )

        section = build_benchmark_report_section(plan, run_result)
        actual = json.loads(section.model_dump_json(indent=2))
        expected = json.loads(golden_fixture_path.read_text())

        assert actual == expected, (
            f"Golden fixture mismatch.\n"
            f"Actual:   {json.dumps(actual, indent=2)}\n"
            f"Expected: {json.dumps(expected, indent=2)}"
        )


# ---------------------------------------------------------------------------
# Smoke Golden Test: lm_eval smoke section fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_golden_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "lm_eval_smoke_report_golden.json"


class TestSmokeGoldenFixture:
    """Golden test: known smoke inputs → deterministic BenchmarkReportSection JSON.

    The fixture MUST pre-exist.  Any field change causes failure.
    """

    def test_smoke_golden_matches_fixture(self, smoke_golden_fixture_path: Path) -> None:
        assert smoke_golden_fixture_path.exists(), (
            f"Smoke golden fixture not found at {smoke_golden_fixture_path}. "
            f"It must be committed before running this test."
        )

        run_id = "11111111-1111-1111-1111-111111111111"
        ev1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        ev2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        p = {
            "suite_id": "llmtrace_smoke",
            "suite_version": "1.0.0",
            "source_id": "lm-eval",
            "source_revision": "0000000-smoke",
            "adapter_id": "lm-eval",
            "adapter_version": "0.4.12",
        }

        plan = RunPlan(
            plan_id="smoke-golden-plan",
            task_ids=["llmtrace_smoke"],
            total_samples=4,
            budget=BudgetEstimate(planned_requests=4, maximum_requests=4, estimated_cost=None),
            **{k: v for k, v in p.items() if k in RunPlan.model_fields},
        )

        attempt = TaskAttempt(
            attempt_id="attempt-smoke-001",
            task_id="llmtrace_smoke",
            status=TaskStatus.SUCCESS,
            evidence_refs=[ev1, ev2],
            metadata={
                "llmtrace_smoke_task": True,
                "metric_result": {
                    "task_name": "llmtrace_smoke",
                    "metric_name": "exact_match",
                    "generation_options": {
                        "temperature": 0.0,
                        "until": ["\n"],
                        "do_sample": False,
                    },
                },
            },
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )

        grade = GradeResult(
            grade_id="grade-smoke-001",
            attempt_id="attempt-smoke-001",
            task_id="llmtrace_smoke",
            grader_id="exact_match",
            raw_score=1.0,
            normalized_score=1.0,
            **{k: v for k, v in p.items() if k in GradeResult.model_fields},
        )

        run_result = BenchmarkRunResult(
            run_id=run_id,
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=[ev1, ev2],
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
        )

        section = build_benchmark_report_section(plan, run_result)
        actual = json.loads(section.model_dump_json(indent=2))
        expected = json.loads(smoke_golden_fixture_path.read_text())

        assert actual == expected, (
            f"Smoke golden fixture mismatch.\n"
            f"Actual:   {json.dumps(actual, indent=2)}\n"
            f"Expected: {json.dumps(expected, indent=2)}"
        )
