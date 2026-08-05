"""Tests for benchmark domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmtrace.benchmarks.models import (
    BenchmarkRunResult,
    BenchmarkSource,
    BenchmarkSuite,
    BudgetEstimate,
    DimensionResult,
    GradeResult,
    GradeStatus,
    SuiteVersion,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_task_spec(task_id: str = "task_1", num_samples: int = 100) -> TaskSpec:
    return TaskSpec(task_id=task_id, name=task_id, num_samples=num_samples)


def _make_suite_version(version: str = "1.0.0") -> SuiteVersion:
    return SuiteVersion(version=version)


# ============================================================================
# BenchmarkSource
# ============================================================================


class TestBenchmarkSource:
    def test_valid_creation(self) -> None:
        src = BenchmarkSource(source_id="mmlu", name="MMLU")
        assert src.source_id == "mmlu"
        assert src.name == "MMLU"
        assert src.description == ""
        assert src.url == ""

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkSource(name="MMLU")  # type: ignore[call-arg]

    def test_all_fields(self) -> None:
        src = BenchmarkSource(
            source_id="livebench",
            name="LiveBench",
            description="A live benchmark",
            url="https://livebench.ai",
        )
        assert src.description == "A live benchmark"
        assert src.url == "https://livebench.ai"

    def test_json_roundtrip(self) -> None:
        src = BenchmarkSource(source_id="mmlu", name="MMLU", description="desc")
        data = src.model_dump_json()
        restored = BenchmarkSource.model_validate_json(data)
        assert restored == src

    def test_different_source_ids_are_unique(self) -> None:
        src1 = BenchmarkSource(source_id="a", name="A")
        src2 = BenchmarkSource(source_id="b", name="B")
        assert src1.source_id != src2.source_id
# ============================================================================
# SuiteVersion (immutable)
# ============================================================================


class TestSuiteVersion:
    def test_valid_creation(self) -> None:
        sv = SuiteVersion(version="2.0.0")
        assert sv.version == "2.0.0"

    def test_frozen_immutable(self) -> None:
        sv = SuiteVersion(version="1.0.0")
        with pytest.raises(ValidationError):
            sv.version = "2.0.0"  # type: ignore[misc]

    def test_json_roundtrip_preserves_immutability(self) -> None:
        sv = SuiteVersion(version="1.2.3", notes="Initial release")
        data = sv.model_dump_json()
        restored = SuiteVersion.model_validate_json(data)
        assert restored.version == "1.2.3"
        assert restored.notes == "Initial release"
        with pytest.raises(ValidationError):
            restored.version = "2.0.0"  # type: ignore[misc]


# ============================================================================
# BenchmarkSuite
# ============================================================================


class TestBenchmarkSuite:
    def test_valid_creation(self) -> None:
        tasks = [_make_task_spec("t1"), _make_task_spec("t2")]
        suite = BenchmarkSuite(
            suite_id="mmlu",
            name="MMLU Suite",
            version=_make_suite_version("1.0.0"),
            source_id="mmlu",
            source_revision="abc123",
            tasks=tasks,
        )
        assert suite.suite_id == "mmlu"
        assert len(suite.tasks) == 2

    def test_empty_tasks(self) -> None:
        suite = BenchmarkSuite(
            suite_id="empty",
            name="Empty",
            version=_make_suite_version(),
            source_id="s",
            source_revision="rev",
        )
        assert suite.tasks == []

    def test_json_roundtrip(self) -> None:
        suite = BenchmarkSuite(
            suite_id="mmlu",
            name="MMLU",
            version=_make_suite_version("1.0.0"),
            source_id="mmlu",
            source_revision="abc123",
            tasks=[_make_task_spec("t1", num_samples=50)],
        )
        data = suite.model_dump_json()
        restored = BenchmarkSuite.model_validate_json(data)
        assert restored.suite_id == suite.suite_id
        assert restored.version.version == "1.0.0"
        assert len(restored.tasks) == 1


# ============================================================================
# TaskSpec
# ============================================================================


class TestTaskSpec:
    def test_valid_creation(self) -> None:
        ts = TaskSpec(task_id="mmlu_anatomy", name="Anatomy", num_samples=135)
        assert ts.task_id == "mmlu_anatomy"
        assert ts.num_samples == 135

    def test_negative_samples_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskSpec(task_id="bad", name="Bad", num_samples=-1)

    def test_category(self) -> None:
        ts = TaskSpec(task_id="t", name="T", category="math")
        assert ts.category == "math"

    def test_metadata(self) -> None:
        ts = TaskSpec(task_id="t", name="T", metadata={"difficulty": "hard"})
        assert ts.metadata == {"difficulty": "hard"}


# ============================================================================
# normalized_score boundaries
# ============================================================================


class TestNormalizedScore:
    def test_normalized_score_clamped_to_zero(self) -> None:
        grade = GradeResult(
            attempt_id="a",
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            grader_id="g",
            raw_score=-0.5,
            normalized_score=-0.5,
        )
        assert grade.normalized_score == 0.0

    def test_normalized_score_clamped_to_one(self) -> None:
        grade = GradeResult(
            attempt_id="a",
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            grader_id="g",
            raw_score=1.5,
            normalized_score=1.5,
        )
        assert grade.normalized_score == 1.0

    def test_normalized_score_within_bounds(self) -> None:
        grade = GradeResult(
            attempt_id="a",
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            grader_id="g",
            raw_score=0.75,
            normalized_score=0.75,
        )
        assert grade.normalized_score == 0.75

    def test_dimension_normalized_value_clamped(self) -> None:
        dim = DimensionResult(dimension_id="acc", name="Accuracy", value=1.2, normalized_value=1.2)
        assert dim.normalized_value == 1.0

        dim2 = DimensionResult(dimension_id="acc", name="Accuracy", value=-0.1, normalized_value=-0.1)
        assert dim2.normalized_value == 0.0


# ============================================================================
# Evidence reference fields
# ============================================================================


class TestEvidenceRefs:
    def test_task_attempt_has_evidence_refs(self) -> None:
        ta = TaskAttempt(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            evidence_refs=["uuid-1", "uuid-2"],
        )
        assert ta.evidence_refs == ["uuid-1", "uuid-2"]

    def test_task_attempt_evidence_refs_default_empty(self) -> None:
        ta = TaskAttempt(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
        )
        assert ta.evidence_refs == []

    def test_grade_result_has_evidence_refs(self) -> None:
        gr = GradeResult(
            attempt_id="a",
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            grader_id="g",
            raw_score=0.5,
            normalized_score=0.5,
            evidence_refs=["uuid-3"],
        )
        assert gr.evidence_refs == ["uuid-3"]

    def test_benchmark_run_result_aggregates_evidence_refs(self) -> None:
        result = BenchmarkRunResult(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            adapter_id="a",
            adapter_version="v",
            evidence_refs=["e1", "e2"],
        )
        assert result.evidence_refs == ["e1", "e2"]


# ============================================================================
# GradeResult required fields
# ============================================================================


class TestGradeResult:
    def test_all_required_fields_present(self) -> None:
        gr = GradeResult(
            attempt_id="attempt-1",
            source_id="mmlu",
            source_revision="abc123",
            suite_id="mmlu",
            suite_version="1.0.0",
            task_id="mmlu_anatomy",
            adapter_id="lm-eval",
            adapter_version="0.4.0",
            grader_id="exact_match",
            raw_score=0.85,
            normalized_score=0.85,
        )
        assert gr.source_id == "mmlu"
        assert gr.source_revision == "abc123"
        assert gr.suite_id == "mmlu"
        assert gr.suite_version == "1.0.0"
        assert gr.task_id == "mmlu_anatomy"
        assert gr.adapter_id == "lm-eval"
        assert gr.adapter_version == "0.4.0"
        assert gr.grader_id == "exact_match"
        assert gr.raw_score == 0.85
        assert gr.normalized_score == 0.85

    def test_grade_result_status(self) -> None:
        gr = GradeResult(
            attempt_id="a",
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            grader_id="g",
            raw_score=0.0,
            normalized_score=0.0,
            status=GradeStatus.UNGRADABLE,
            error_message="No valid output to grade",
        )
        assert gr.status == GradeStatus.UNGRADABLE
        assert gr.error_message == "No valid output to grade"


# ============================================================================
# TaskAttempt statuses
# ============================================================================


class TestTaskAttempt:
    def test_default_status_pending(self) -> None:
        ta = TaskAttempt(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
        )
        assert ta.status == TaskStatus.PENDING

    def test_explicit_status(self) -> None:
        ta = TaskAttempt(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            status=TaskStatus.SUCCESS,
        )
        assert ta.status == TaskStatus.SUCCESS

    def test_failure_with_error_message(self) -> None:
        ta = TaskAttempt(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            status=TaskStatus.FAILURE,
            error_message="Connection timeout",
        )
        assert ta.status == TaskStatus.FAILURE
        assert ta.error_message == "Connection timeout"


# ============================================================================
# BenchmarkRunResult
# ============================================================================


class TestBenchmarkRunResult:
    def test_valid_creation(self) -> None:
        result = BenchmarkRunResult(
            source_id="mmlu",
            source_revision="abc123",
            suite_id="mmlu",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.0",
        )
        assert result.run_id is not None
        assert result.task_attempts == []

    def test_error_and_skip_counts_synced(self) -> None:
        attempts = [
            TaskAttempt(
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t1",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.SUCCESS,
            ),
            TaskAttempt(
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t2",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.FAILURE,
            ),
            TaskAttempt(
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t3",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.SKIPPED,
            ),
        ]
        result = BenchmarkRunResult(
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            adapter_id="a",
            adapter_version="v",
            task_attempts=attempts,
        )
        assert result.error_count == 1
        assert result.skip_count == 1

    def test_json_roundtrip(self) -> None:
        result = BenchmarkRunResult(
            source_id="mmlu",
            source_revision="abc123",
            suite_id="mmlu",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.0",
            task_attempts=[
                TaskAttempt(
                    source_id="mmlu",
                    source_revision="abc123",
                    suite_id="mmlu",
                    suite_version="1.0.0",
                    task_id="t1",
                    adapter_id="lm-eval",
                    adapter_version="0.4.0",
                    status=TaskStatus.SUCCESS,
                    evidence_refs=["ev1"],
                ),
            ],
            grade_results=[
                GradeResult(
                    attempt_id="a",
                    source_id="mmlu",
                    source_revision="abc123",
                    suite_id="mmlu",
                    suite_version="1.0.0",
                    task_id="t1",
                    adapter_id="lm-eval",
                    adapter_version="0.4.0",
                    grader_id="exact_match",
                    raw_score=0.9,
                    normalized_score=0.9,
                    evidence_refs=["ev1"],
                ),
            ],
        )
        data = result.model_dump_json()
        restored = BenchmarkRunResult.model_validate_json(data)
        assert restored.source_id == result.source_id
        assert restored.error_count == 0
        assert len(restored.task_attempts) == 1
        assert len(restored.grade_results) == 1


# ============================================================================
# BudgetEstimate
# ============================================================================


class TestBudgetEstimate:
    def test_valid_creation(self) -> None:
        budget = BudgetEstimate(
            planned_requests=100,
            maximum_requests=100,
        )
        assert budget.planned_requests == 100
        assert budget.maximum_requests == 100
        assert budget.estimated_cost is None

    def test_cost_unavailable_when_not_provided(self) -> None:
        budget = BudgetEstimate(planned_requests=50, maximum_requests=50)
        assert budget.estimated_cost is None

    def test_cost_available_when_provided(self) -> None:
        budget = BudgetEstimate(
            planned_requests=100,
            maximum_requests=100,
            estimated_cost=0.05,
        )
        assert budget.estimated_cost == 0.05
        assert budget.currency == "USD"

    def test_assumptions_field(self) -> None:
        budget = BudgetEstimate(
            planned_requests=10,
            maximum_requests=10,
            assumptions=["assume constant rate"],
        )
        assert len(budget.assumptions) == 1

    def test_json_roundtrip(self) -> None:
        budget = BudgetEstimate(
            planned_requests=200,
            maximum_requests=300,
            maximum_retries=1,
            estimated_input_tokens=100000,
            estimated_output_tokens=50000,
            estimated_duration_seconds=600.0,
            estimated_cost=0.50,
            assumptions=["a1", "a2"],
        )
        data = budget.model_dump_json()
        restored = BudgetEstimate.model_validate_json(data)
        assert restored == budget
