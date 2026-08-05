"""Tests for benchmark domain models."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkProvenance,
    BenchmarkRunResult,
    BenchmarkSource,
    BenchmarkSuite,
    BudgetEstimate,
    DimensionResult,
    FailureCategory,
    GradeResult,
    GradeStatus,
    SuiteVersion,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
    validate_evidence_refs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_spec(task_id: str = "task_1", num_samples: int = 100) -> TaskSpec:
    return TaskSpec(task_id=task_id, name=task_id, num_samples=num_samples)


def _make_suite_version(version: str = "1.0.0") -> SuiteVersion:
    return SuiteVersion(version=version)


def _valid_uuid() -> str:
    return str(uuid4())


def _make_failure(**overrides: object) -> AdapterFailure:
    defaults: dict[str, object] = {
        "error_code": "TIMEOUT",
        "category": FailureCategory.TIMEOUT,
        "message": "Request timed out",
        "retryable": True,
    }
    defaults.update(overrides)
    return AdapterFailure(**defaults)  # type: ignore[arg-type]


# ============================================================================
# BenchmarkProvenance – non-empty ID constraints
# ============================================================================


class TestNonEmptyConstraints:
    def test_whitespace_only_source_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkProvenance(
                source_id="   ",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                adapter_id="a",
                adapter_version="v",
            )

    def test_empty_source_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkProvenance(
                source_id="",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                adapter_id="a",
                adapter_version="v",
            )

    def test_empty_suite_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GradeResult(
                grade_id="g",
                attempt_id="a",
                source_id="s",
                source_revision="r",
                suite_id=" ",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                grader_id="g",
                raw_score=0.5,
                normalized_score=0.5,
            )


# ============================================================================
# Extra fields forbidden
# ============================================================================


class TestExtraForbidden:
    def test_source_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkSource(source_id="s", name="N", unknown_field="x")  # type: ignore[call-arg]

    def test_grade_result_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GradeResult(
                grade_id="g",
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
                bogus="yes",  # type: ignore[call-arg]
            )


# ============================================================================
# BenchmarkSource
# ============================================================================


class TestBenchmarkSource:
    def test_valid_creation(self) -> None:
        src = BenchmarkSource(source_id="mmlu", name="MMLU")
        assert src.source_id == "mmlu"
        assert src.name == "MMLU"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkSource(name="MMLU")  # type: ignore[call-arg]

    def test_json_roundtrip(self) -> None:
        src = BenchmarkSource(source_id="mmlu", name="MMLU", description="desc")
        data = src.model_dump_json()
        restored = BenchmarkSource.model_validate_json(data)
        assert restored == src


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


# ============================================================================
# normalized_score – strict validation (no silent clamping)
# ============================================================================


class TestNormalizedScoreStrict:
    def test_score_0_valid(self) -> None:
        gr = GradeResult(
            grade_id="g",
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
        )
        assert gr.normalized_score == 0.0

    def test_score_1_valid(self) -> None:
        gr = GradeResult(
            grade_id="g",
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
            normalized_score=1.0,
        )
        assert gr.normalized_score == 1.0

    def test_score_between_0_and_1_valid(self) -> None:
        gr = GradeResult(
            grade_id="g",
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
        assert gr.normalized_score == 0.75

    def test_score_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GradeResult(
                grade_id="g",
                attempt_id="a",
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                grader_id="g",
                raw_score=-0.2,
                normalized_score=-0.2,
            )

    def test_score_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GradeResult(
                grade_id="g",
                attempt_id="a",
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                grader_id="g",
                raw_score=1.4,
                normalized_score=1.4,
            )

    def test_dimension_normalized_value_rejected_when_negative(self) -> None:
        with pytest.raises(ValidationError):
            DimensionResult(dimension_id="acc", name="Accuracy", value=-0.1, normalized_value=-0.1)

    def test_dimension_normalized_value_rejected_when_above_1(self) -> None:
        with pytest.raises(ValidationError):
            DimensionResult(dimension_id="acc", name="Accuracy", value=1.2, normalized_value=1.2)


# ============================================================================
# Evidence references – UUID validation, dedup
# ============================================================================


class TestEvidenceRefs:
    def test_valid_uuid_refs_accepted(self) -> None:
        uid = _valid_uuid()
        ta = TaskAttempt(
            attempt_id=_valid_uuid(),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            evidence_refs=[uid],
        )
        assert ta.evidence_refs == [uid]

    def test_non_uuid_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskAttempt(
                attempt_id=_valid_uuid(),
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                evidence_refs=["not-a-uuid"],
            )

    def test_duplicate_refs_deduped(self) -> None:
        uid = _valid_uuid()
        ta = TaskAttempt(
            attempt_id=_valid_uuid(),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            evidence_refs=[uid, uid, uid],
        )
        assert ta.evidence_refs == [uid]

    def test_order_preserved(self) -> None:
        u1, u2, u3 = _valid_uuid(), _valid_uuid(), _valid_uuid()
        ta = TaskAttempt(
            attempt_id=_valid_uuid(),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            evidence_refs=[u3, u1, u2, u1],
        )
        assert ta.evidence_refs == [u3, u1, u2]

    def test_grade_result_evidence_refs_uuid_validated(self) -> None:
        uid = _valid_uuid()
        gr = GradeResult(
            grade_id=_valid_uuid(),
            attempt_id=_valid_uuid(),
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
            evidence_refs=[uid],
        )
        assert gr.evidence_refs == [uid]


# ============================================================================
# validate_evidence_refs standalone function
# ============================================================================


class TestValidateEvidenceRefs:
    def test_existing_refs_pass(self) -> None:
        uid = uuid4()
        validate_evidence_refs([str(uid)], {uid})

    def test_non_uuid_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid UUID"):
            validate_evidence_refs(["garbage"], {uuid4()})

    def test_missing_ref_raises(self) -> None:
        uid = uuid4()
        with pytest.raises(ValueError, match="not found in available evidence"):
            validate_evidence_refs([str(uid)], {uuid4()})

    def test_duplicate_refs_ok_if_all_exist(self) -> None:
        uid = uuid4()
        validate_evidence_refs([str(uid), str(uid)], {uid})

    def test_empty_refs_ok(self) -> None:
        validate_evidence_refs([], set())


# ============================================================================
# AdapterFailure
# ============================================================================


class TestAdapterFailure:
    def test_valid_creation(self) -> None:
        af = AdapterFailure(
            error_code="NET_ERR",
            category=FailureCategory.NETWORK,
            message="Connection refused",
            retryable=True,
            details={"host": "example.com"},
        )
        assert af.error_code == "NET_ERR"
        assert af.retryable is True

    def test_default_category_unknown(self) -> None:
        af = AdapterFailure(error_code="E01", message="msg")
        assert af.category == FailureCategory.UNKNOWN

    def test_default_not_retryable(self) -> None:
        af = AdapterFailure(error_code="E01", message="msg")
        assert af.retryable is False

    def test_empty_error_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AdapterFailure(error_code=" ", message="msg")


# ============================================================================
# TaskAttempt with structured failure
# ============================================================================


class TestTaskAttemptWithFailure:
    def test_success_attempt_no_failure(self) -> None:
        ta = TaskAttempt(
            attempt_id=_valid_uuid(),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            status=TaskStatus.SUCCESS,
        )
        assert ta.failure is None

    def test_failure_attempt_must_have_failure(self) -> None:
        failure = _make_failure()
        ta = TaskAttempt(
            attempt_id=_valid_uuid(),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            status=TaskStatus.FAILURE,
            failure=failure,
        )
        assert ta.failure == failure

    def test_failure_status_without_failure_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure must be set"):
            TaskAttempt(
                attempt_id=_valid_uuid(),
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.FAILURE,
            )

    def test_success_status_with_failure_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure must be None"):
            TaskAttempt(
                attempt_id=_valid_uuid(),
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.SUCCESS,
                failure=_make_failure(),
            )

    def test_pending_status_with_failure_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure must be None"):
            TaskAttempt(
                attempt_id=_valid_uuid(),
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.PENDING,
                failure=_make_failure(),
            )

    def test_json_roundtrip_with_failure(self) -> None:
        failure = _make_failure(
            error_code="AUTH_ERR",
            category=FailureCategory.AUTH,
            message="Invalid key",
            retryable=False,
            details={"provider": "openai"},
        )
        ta = TaskAttempt(
            attempt_id=_valid_uuid(),
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            task_id="t",
            adapter_id="a",
            adapter_version="v",
            status=TaskStatus.FAILURE,
            failure=failure,
        )
        data = ta.model_dump_json()
        restored = TaskAttempt.model_validate_json(data)
        assert restored.status == TaskStatus.FAILURE
        assert restored.failure is not None
        assert restored.failure.error_code == "AUTH_ERR"


# ============================================================================
# GradeResult
# ============================================================================


class TestGradeResult:
    def test_all_required_fields_present(self) -> None:
        gr = GradeResult(
            grade_id="grade-1",
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
        assert gr.raw_score == 0.85
        assert gr.normalized_score == 0.85

    def test_grade_result_status(self) -> None:
        gr = GradeResult(
            grade_id="g",
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
# BenchmarkRunResult
# ============================================================================


class TestBenchmarkRunResult:
    def test_valid_creation(self) -> None:
        result = BenchmarkRunResult(
            run_id=_valid_uuid(),
            source_id="mmlu",
            source_revision="abc123",
            suite_id="mmlu",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.0",
        )
        assert result.task_attempts == []

    def test_error_and_skip_counts_synced(self) -> None:
        attempts = [
            TaskAttempt(
                attempt_id=_valid_uuid(),
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
                attempt_id=_valid_uuid(),
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                task_id="t2",
                adapter_id="a",
                adapter_version="v",
                status=TaskStatus.FAILURE,
                failure=_make_failure(),
            ),
            TaskAttempt(
                attempt_id=_valid_uuid(),
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
            run_id=_valid_uuid(),
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
            run_id=_valid_uuid(),
            source_id="mmlu",
            source_revision="abc123",
            suite_id="mmlu",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.0",
        )
        data = result.model_dump_json()
        restored = BenchmarkRunResult.model_validate_json(data)
        assert restored.run_id == result.run_id


# ============================================================================
# BudgetEstimate validation
# ============================================================================


class TestBudgetEstimateValidation:
    def test_valid_creation(self) -> None:
        budget = BudgetEstimate(planned_requests=100, maximum_requests=100)
        assert budget.planned_requests == 100

    def test_maximum_not_equal_to_planned_times_retries_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must equal"):
            BudgetEstimate(planned_requests=10, maximum_requests=25, maximum_retries=1)

    def test_negative_planned_requests_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetEstimate(planned_requests=-1, maximum_requests=0)

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetEstimate(planned_requests=10, maximum_requests=10, maximum_retries=-1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetEstimate(planned_requests=1, maximum_requests=1, estimated_cost=-0.01)

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetEstimate(
                planned_requests=1,
                maximum_requests=1,
                estimated_input_tokens=-1,
            )

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BudgetEstimate(
                planned_requests=1,
                maximum_requests=1,
                estimated_duration_seconds=-1.0,
            )

    def test_maximum_requests_equals_planned_with_zero_retries(self) -> None:
        budget = BudgetEstimate(planned_requests=5, maximum_requests=5, maximum_retries=0)
        assert budget.maximum_requests == 5

    def test_cost_unavailable_when_not_provided(self) -> None:
        budget = BudgetEstimate(planned_requests=50, maximum_requests=50)
        assert budget.estimated_cost is None

    def test_cost_available_when_provided(self) -> None:
        budget = BudgetEstimate(planned_requests=100, maximum_requests=100, estimated_cost=0.05)
        assert budget.estimated_cost == 0.05

    def test_json_roundtrip(self) -> None:
        budget = BudgetEstimate(
            planned_requests=200,
            maximum_requests=400,
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
