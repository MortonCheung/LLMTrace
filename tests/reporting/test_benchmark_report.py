"""Tests for benchmark report models and mapper.

Covers:
1. Successful task mapping
2. Failed task mapping
3. SUCCESS but missing GradeResult
4. UNGRADABLE GradeResult
5. Duplicate GradeResult rejection
6. Evidence UUID preservation
7. estimated_cost=None preservation
8. datetime JSON serialization (ISO-8601)
9. JSON roundtrip
10. No Pydantic objects / exceptions in metadata
11. Smoke task marked capability_score_eligible=False
12. No total_score or capability_score in model
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run_plan(plan_id: str = "test-plan-001") -> RunPlan:
    return RunPlan(
        plan_id=plan_id,
        suite_id="test-suite",
        suite_version="1.0.0",
        source_id="test-source",
        source_revision="abc123",
        adapter_id="lm-eval",
        adapter_version="0.4.12",
        task_ids=["task_a", "task_b"],
        total_samples=4,
        budget=BudgetEstimate(
            planned_requests=4,
            maximum_requests=4,
            estimated_input_tokens=None,
            estimated_output_tokens=None,
            estimated_cost=None,
        ),
    )


def _make_success_attempt(attempt_id: str, task_id: str, evidence_refs: list[str] | None = None) -> TaskAttempt:
    return TaskAttempt(
        attempt_id=attempt_id,
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        evidence_refs=evidence_refs or [str(uuid4())],
        source_id="test-source",
        source_revision="abc123",
        suite_id="test-suite",
        suite_version="1.0.0",
        adapter_id="lm-eval",
        adapter_version="0.4.12",
    )


def _make_failure_attempt(attempt_id: str, task_id: str, error_code: str = "TEST_ERROR") -> TaskAttempt:
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
        source_id="test-source",
        source_revision="abc123",
        suite_id="test-suite",
        suite_version="1.0.0",
        adapter_id="lm-eval",
        adapter_version="0.4.12",
    )


def _make_grade(attempt_id: str, task_id: str, raw_score: float = 0.75) -> GradeResult:
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=raw_score,
        normalized_score=raw_score,
        source_id="test-source",
        source_revision="abc123",
        suite_id="test-suite",
        suite_version="1.0.0",
        adapter_id="lm-eval",
        adapter_version="0.4.12",
    )


def _make_ungradable_grade(attempt_id: str, task_id: str) -> GradeResult:
    return GradeResult(
        grade_id=str(uuid4()),
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=0.0,
        normalized_score=0.0,
        status=GradeStatus.UNGRADABLE,
        error_message="Cannot grade",
        source_id="test-source",
        source_revision="abc123",
        suite_id="test-suite",
        suite_version="1.0.0",
        adapter_id="lm-eval",
        adapter_version="0.4.12",
    )


# ---------------------------------------------------------------------------
# Tests: Successful mapping
# ---------------------------------------------------------------------------


class TestSuccessfulMapping:
    def test_success_task_with_grade(self) -> None:
        """A SUCCESS TaskAttempt with a matching GradeResult maps correctly."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.8)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])

        assert section.status == "success"
        assert len(section.tasks) == 1
        task = section.tasks[0]
        assert task.task_id == "task_a"
        assert task.status == "success"
        assert task.grader_id == "exact_match"
        assert task.grade_status == "graded"
        assert task.raw_score == 0.8
        assert task.normalized_score == 0.8
        assert task.failure is None
        assert task.capability_score_eligible is True

    def test_multiple_tasks_all_success(self) -> None:
        """Multiple successful tasks all map correctly."""
        plan = _make_run_plan()
        a1 = _make_success_attempt("att-1", "task_a", [str(uuid4())])
        a2 = _make_success_attempt("att-2", "task_b", [str(uuid4())])
        g1 = _make_grade("att-1", "task_a", 0.9)
        g2 = _make_grade("att-2", "task_b", 0.7)

        all_refs = [str(uuid4())]
        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[a1, a2],
            grade_results=[g1, g2],
            evidence_refs=all_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [a1, a2], [g1, g2])
        assert section.status == "success"
        assert section.summary.success_count == 2
        assert section.summary.failure_count == 0
        assert len(section.tasks) == 2


# ---------------------------------------------------------------------------
# Tests: Failure mapping
# ---------------------------------------------------------------------------


class TestFailureMapping:
    def test_failure_task_preserves_error(self) -> None:
        """A FAILURE TaskAttempt preserves the error and sets scores to None."""
        plan = _make_run_plan()
        attempt = _make_failure_attempt("att-1", "task_a", "LM_EVAL_OPTIONS_INCONSISTENT")

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [])
        assert section.status == "failure"
        task = section.tasks[0]
        assert task.status == "failure"
        assert task.raw_score is None
        assert task.normalized_score is None
        assert task.failure is not None
        assert task.failure.error_code == "LM_EVAL_OPTIONS_INCONSISTENT"
        assert task.failure.category == "adapter"
        assert task.failure.message == "Test failure message"
        assert task.failure.retryable is False

    def test_partial_failure_status(self) -> None:
        """One success + one failure → partial_failure status."""
        plan = _make_run_plan()
        a1 = _make_success_attempt("att-1", "task_a")
        a2 = _make_failure_attempt("att-2", "task_b")
        g1 = _make_grade("att-1", "task_a", 1.0)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[a1, a2],
            grade_results=[g1],
            evidence_refs=[str(uuid4())],
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [a1, a2], [g1])
        assert section.status == "partial_failure"
        assert section.summary.success_count == 1
        assert section.summary.failure_count == 1


# ---------------------------------------------------------------------------
# Tests: Ungraded / UNGRADABLE
# ---------------------------------------------------------------------------


class TestUngradedHandling:
    def test_success_without_grade_is_ungraded(self) -> None:
        """SUCCESS without GradeResult → ungraded, no fake scores, warning added."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [])
        task = section.tasks[0]
        assert task.status == "success"
        assert task.raw_score is None
        assert task.normalized_score is None
        assert task.grader_id is None
        assert task.grade_status is None
        assert section.summary.ungraded_count == 1
        assert len(section.warnings) >= 1
        assert any("ungraded" in w for w in section.warnings)

    def test_ungradable_grade_preserves_status(self) -> None:
        """UNGRADABLE GradeResult → grade_status preserved, scores None."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_ungradable_grade("att-1", "task_a")

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        task = section.tasks[0]
        assert task.grade_status == "ungradable"
        assert task.raw_score is None
        assert task.normalized_score is None
        assert section.summary.ungradable_count == 1


# ---------------------------------------------------------------------------
# Tests: Duplicate GradeResult
# ---------------------------------------------------------------------------


class TestDuplicateGradeResult:
    def test_duplicate_grade_result_raises(self) -> None:
        """Two GradeResults for the same attempt_id → ValueError."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        g1 = _make_grade("att-1", "task_a", 0.5)
        g2 = _make_grade("att-1", "task_a", 0.9)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[g1, g2],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        with pytest.raises(ValueError, match="Duplicate GradeResult"):
            build_benchmark_report_section(plan, run_result, [attempt], [g1, g2])


# ---------------------------------------------------------------------------
# Tests: Evidence UUID
# ---------------------------------------------------------------------------


class TestEvidenceUUID:
    def test_evidence_refs_flow_through(self) -> None:
        """Evidence UUIDs from TaskAttempt are preserved in the report."""
        ev1 = str(uuid4())
        ev2 = str(uuid4())

        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a", [ev1, ev2])
        grade = _make_grade("att-1", "task_a", 0.8)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=[ev1, ev2],
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        task = section.tasks[0]
        assert ev1 in task.evidence_refs
        assert ev2 in task.evidence_refs


# ---------------------------------------------------------------------------
# Tests: estimated_cost=None
# ---------------------------------------------------------------------------


class TestEstimatedCostNone:
    def test_estimated_cost_none_stays_none(self) -> None:
        """When estimated_cost is None, it remains None in the report."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        assert section.estimated_cost is None
        assert section.summary.estimated_cost is None


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_datetime_is_iso8601(self) -> None:
        """datetime fields serialize as ISO-8601 strings."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            started_at=datetime(2026, 6, 15, 12, 30, 0, tzinfo=UTC),
            finished_at=datetime(2026, 6, 15, 12, 31, 0, tzinfo=UTC),
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        data = section.model_dump(mode="json")

        assert isinstance(data["started_at"], str)
        assert "2026-06-15T12:30:00" in data["started_at"]
        assert "Z" in data["started_at"] or "+" in data["started_at"]

    def test_uuid_is_string_in_json(self) -> None:
        """UUID fields serialize as strings in JSON mode."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)
        run_id_str = str(uuid4())

        run_result = BenchmarkRunResult(
            run_id=run_id_str,
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        data = section.model_dump(mode="json")
        assert isinstance(data["run_id"], str)
        assert data["run_id"] == run_id_str

    def test_json_roundtrip(self) -> None:
        """model_dump(mode='json') → json.loads → model_validate: roundtrip works."""
        plan = _make_run_plan()
        attempt = _make_success_attempt("att-1", "task_a")
        grade = _make_grade("att-1", "task_a", 0.5)
        run_id_str = str(uuid4())

        run_result = BenchmarkRunResult(
            run_id=run_id_str,
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        json_str = section.model_dump_json()
        data = json.loads(json_str)

        restored = BenchmarkReportSection.model_validate(data)
        assert restored.run_id == section.run_id
        assert restored.plan_id == section.plan_id
        assert restored.status == section.status
        assert restored.summary.success_count == section.summary.success_count


# ---------------------------------------------------------------------------
# Tests: Metadata safety
# ---------------------------------------------------------------------------


class TestMetadataSafety:
    def test_no_pydantic_or_exception_in_metadata(self) -> None:
        """TaskReportItem.metadata must not contain Pydantic objects or exceptions."""
        plan = _make_run_plan()
        grade = _make_grade("att-1", "task_a", 0.5)

        # Create an attempt with metadata that includes a Pydantic model
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            metadata={
                "metric_result": {"task_name": "task_a", "metric_name": "exact_match", "value": 0.5},
                "extra_info": "ok",
            },
        )

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        task = section.tasks[0]

        # metadata should not contain Pydantic objects or exceptions
        for v in task.metadata.values():
            assert not hasattr(v, "model_dump"), f"Pydantic object leaked: {type(v)}"
            assert not isinstance(v, BaseException), f"Exception leaked: {type(v)}"

    def test_no_total_score_or_capability_score(self) -> None:
        """BenchmarkReportSection must not have total_score or capability_score fields."""
        fields = BenchmarkReportSection.model_fields
        assert "total_score" not in fields
        assert "capability_score" not in fields

        summary_fields = BenchmarkRunSummary.model_fields
        assert "total_score" not in summary_fields
        assert "capability_score" not in summary_fields


# ---------------------------------------------------------------------------
# Tests: Smoke task
# ---------------------------------------------------------------------------


class TestSmokeTask:
    def test_smoke_task_not_capability_eligible(self) -> None:
        """Smoke tasks are marked capability_score_eligible=False."""
        plan = _make_run_plan()
        attempt = TaskAttempt(
            attempt_id="att-smoke",
            task_id="llmtrace_smoke",
            status=TaskStatus.SUCCESS,
            evidence_refs=[str(uuid4())],
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )
        grade = _make_grade("att-smoke", "llmtrace_smoke", 1.0)

        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="test-source",
            source_revision="abc123",
            suite_id="test-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        section = build_benchmark_report_section(plan, run_result, [attempt], [grade])
        task = section.tasks[0]
        assert task.capability_score_eligible is False


# ---------------------------------------------------------------------------
# Golden Test: fixed JSON fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "benchmark_report_golden.json"


class TestGoldenFixture:
    """Golden test: known inputs → deterministic JSON output."""

    def test_golden_matches_fixture(self, golden_fixture_path: Path) -> None:
        """Build a report from fake data and compare against golden fixture."""
        run_id = "11111111-1111-1111-1111-111111111111"
        ev1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        ev2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        plan = RunPlan(
            plan_id="golden-plan",
            suite_id="golden-suite",
            suite_version="1.0.0",
            source_id="golden-source",
            source_revision="abc123",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            task_ids=["task_a"],
            total_samples=2,
            budget=BudgetEstimate(
                planned_requests=2,
                maximum_requests=2,
                estimated_cost=None,
            ),
        )

        a1 = TaskAttempt(
            attempt_id="attempt-1",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[ev1, ev2],
            source_id="golden-source",
            source_revision="abc123",
            suite_id="golden-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        g1 = GradeResult(
            grade_id="grade-1",
            attempt_id="attempt-1",
            task_id="task_a",
            grader_id="exact_match",
            raw_score=0.75,
            normalized_score=0.75,
            source_id="golden-source",
            source_revision="abc123",
            suite_id="golden-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        run_result = BenchmarkRunResult(
            run_id=run_id,
            task_attempts=[a1],
            grade_results=[g1],
            evidence_refs=[ev1, ev2],
            source_id="golden-source",
            source_revision="abc123",
            suite_id="golden-suite",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        )

        section = build_benchmark_report_section(plan, run_result, [a1], [g1])
        actual_json = section.model_dump_json(indent=2)

        # Write the fixture if it doesn't exist (first run)
        if not golden_fixture_path.exists():
            golden_fixture_path.parent.mkdir(parents=True, exist_ok=True)
            golden_fixture_path.write_text(actual_json)
            # Skip comparison on first run when fixture is created
            return

        expected_json = golden_fixture_path.read_text()
        actual = json.loads(actual_json)
        expected = json.loads(expected_json)

        # Compare field by field
        assert actual["run_id"] == expected["run_id"]
        assert actual["plan_id"] == expected["plan_id"]
        assert actual["status"] == expected["status"]
        assert actual["summary"]["success_count"] == expected["summary"]["success_count"]
        assert actual["summary"]["failure_count"] == expected["summary"]["failure_count"]
        assert actual["summary"]["actual_requests"] == expected["summary"]["actual_requests"]
        assert len(actual["tasks"]) == len(expected["tasks"])

        task_a = actual["tasks"][0]
        task_e = expected["tasks"][0]
        assert task_a["task_id"] == task_e["task_id"]
        assert task_a["normalized_score"] == task_e["normalized_score"]
        assert task_a["capability_score_eligible"] == task_e["capability_score_eligible"]
