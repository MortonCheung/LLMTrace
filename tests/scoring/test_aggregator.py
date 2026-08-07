"""Tests for dimension and capability profile aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkRunResult,
    FailureCategory,
    GradeResult,
    GradeStatus,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.scoring.aggregator import (
    TaskScoringRegistry,
    aggregate_capability_profile,
    aggregate_dimension_score,
)
from llmtrace.scoring.errors import TaskRegistrationError
from llmtrace.scoring.models import (
    CapabilityDimension,
    DimensionScoreStatus,
    TaskScoringSpec,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy

# ---------------------------------------------------------------------------
# Provenance fields shared by BenchmarkRunResult, TaskAttempt, GradeResult
# ---------------------------------------------------------------------------

_PROVENANCE = {
    "source_id": "mmlu",
    "source_revision": "abc123",
    "suite_id": "mmlu",
    "suite_version": "1.0.0",
    "adapter_id": "lm_eval",
    "adapter_version": "0.1.0",
}


# ---------------------------------------------------------------------------
# Helpers for building test BenchmarkRunResult
# ---------------------------------------------------------------------------


def _make_attempt(
    task_id: str,
    status: TaskStatus = TaskStatus.SUCCESS,
    evidence_ids: list[str] | None = None,
    attempt_id: str | None = None,
) -> TaskAttempt:
    failure = None
    if status == TaskStatus.FAILURE:
        failure = AdapterFailure(
            error_code="TEST_ERROR",
            category=FailureCategory.UNKNOWN,
            message="Test-induced failure",
        )
    return TaskAttempt(
        attempt_id=attempt_id or f"att-{task_id}",
        task_id=task_id,
        status=status,
        evidence_refs=evidence_ids or [str(uuid4())],
        failure=failure,
        **_PROVENANCE,
    )


def _make_grade(
    grade_id: str,
    attempt_id: str,
    task_id: str,
    normalized_score: float,
    status: GradeStatus = GradeStatus.GRADED,
    evidence_ids: list[str] | None = None,
    provenance_override: dict | None = None,
) -> GradeResult:
    prov = dict(_PROVENANCE)
    if provenance_override:
        prov.update(provenance_override)
    return GradeResult(
        grade_id=grade_id,
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=normalized_score,
        normalized_score=normalized_score,
        status=status,
        evidence_refs=evidence_ids or [str(uuid4())],
        **prov,
    )


def _make_run(
    run_id: str,
    attempts: list[TaskAttempt],
    grades: list[GradeResult],
    provenance_override: dict | None = None,
) -> BenchmarkRunResult:
    prov = dict(_PROVENANCE)
    if provenance_override:
        prov.update(provenance_override)
    return BenchmarkRunResult(
        run_id=run_id,
        task_attempts=attempts,
        grade_results=grades,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        **prov,
    )


class _FakeCalibrator:
    """Calibrator that returns a fixed value for testing."""

    def __init__(self, value: float) -> None:
        self.value = value

    def calibrate(self, dimension, raw_score, reference_profile=None):
        return self.value


# ---------------------------------------------------------------------------
# Dimension aggregation tests
# ---------------------------------------------------------------------------


class TestDimensionAggregation:
    """Tests for aggregate_dimension_score()."""

    def test_two_tasks_equal_weight(self) -> None:
        """task A=0.8, task B=0.6, both weight=1 → expected 0.7."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a"), _make_attempt("task_b")],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.8),
                _make_grade("g2", "att-task_b", "task_b", 0.6),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()

        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )

        assert abs(result.raw_normalized_score - 0.7) < 1e-9
        assert result.status == DimensionScoreStatus.UNCALIBRATED
        assert result.calibrated_score is None
        assert result.graded_task_count == 2
        assert abs(result.task_coverage - 1.0) < 1e-9

    def test_unequal_weights(self) -> None:
        """task A weight=3 score=0.8, task B weight=1 score=0.6 → 0.75."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=3.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a"), _make_attempt("task_b")],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.8),
                _make_grade("g2", "att-task_b", "task_b", 0.6),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert abs(result.raw_normalized_score - 0.75) < 1e-9

    def test_failure_coverage(self) -> None:
        """One task graded, one failure → score from graded only, coverage=0.5."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [
                _make_attempt("task_a", TaskStatus.SUCCESS),
                _make_attempt("task_b", TaskStatus.FAILURE),
            ],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.8),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert abs(result.raw_normalized_score - 0.8) < 1e-9
        assert abs(result.task_coverage - 0.5) < 1e-9
        assert result.graded_task_count == 1

    def test_smoke_exclusion(self) -> None:
        """Smoke task (capability_score_eligible=False) must not affect score."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(
                    task_id="smoke_task",
                    dimension=CapabilityDimension.REASONING,
                    task_weight=1.0,
                    capability_score_eligible=False,
                ),
            ]
        )
        run = _make_run(
            "run-1",
            [
                _make_attempt("task_a", TaskStatus.SUCCESS),
                _make_attempt("smoke_task", TaskStatus.SUCCESS),
            ],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.5),
                _make_grade("g2", "att-smoke_task", "smoke_task", 1.0),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert abs(result.raw_normalized_score - 0.5) < 1e-9
        assert abs(result.task_coverage - 1.0) < 1e-9
        assert "smoke_task" not in result.source_task_ids

    def test_ungradable_not_in_score_affects_coverage(self) -> None:
        """UNGRADABLE task does not enter score but affects coverage (planned but not graded)."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [
                _make_attempt("task_a", TaskStatus.SUCCESS),
                _make_attempt("task_b", TaskStatus.SUCCESS),
            ],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.8),
                _make_grade("g2", "att-task_b", "task_b", 0.0, status=GradeStatus.UNGRADABLE),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert abs(result.raw_normalized_score - 0.8) < 1e-9
        assert abs(result.task_coverage - 0.5) < 1e-9
        assert result.graded_task_count == 1

    def test_unregistered_task_strict_mode_fails(self) -> None:
        """Strict mode with unregistered task must raise."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a"), _make_attempt("unregistered")],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.8),
                _make_grade("g2", "att-unregistered", "unregistered", 0.5),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(TaskRegistrationError, match="unregistered"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
                strict=True,
            )

    def test_no_eligible_tasks_unavailable(self) -> None:
        """No eligible tasks → UNAVAILABLE."""
        registry = TaskScoringRegistry([])
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [],
            registry,
            policy,
        )
        assert result.status == DimensionScoreStatus.UNAVAILABLE
        assert result.raw_normalized_score == 0.0

    def test_evidence_refs_propagated(self) -> None:
        """Evidence UUIDs from contributing tasks appear in the dimension result."""
        eid = str(uuid4())
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", evidence_ids=[eid])],
            [_make_grade("g1", "att-task_a", "task_a", 0.8, evidence_ids=[eid])],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert eid in result.evidence_refs

    def test_evidence_refs_deduplicated(self) -> None:
        """Duplicate evidence_refs across grade and attempt are deduplicated."""
        eid = str(uuid4())
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", evidence_ids=[eid])],
            [_make_grade("g1", "att-task_a", "task_a", 0.8, evidence_ids=[eid])],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert result.evidence_refs.count(eid) == 1

    def test_global_weight_assigned(self) -> None:
        """Dimension result carries the correct global_weight from policy."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.CODING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.9)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.CODING,
            [run],
            registry,
            policy,
        )
        assert abs(result.global_weight - 0.20) < 1e-9
        assert abs(result.weighted_contribution - 0.9 * 0.20) < 1e-9

    # -- NEW: multi-run graded_task_count --------------------------------

    def test_multi_run_graded_count(self) -> None:
        """Two runs with different tasks, both grading → counts are global."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [_make_grade("g1", "att-a", "task_a", 0.8)],
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_b", attempt_id="att-b")],
            [_make_grade("g2", "att-b", "task_b", 0.6)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run1, run2],
            registry,
            policy,
        )
        assert result.task_count == 2
        assert result.graded_task_count == 2
        assert abs(result.task_coverage - 1.0) < 1e-9

    def test_multi_run_one_failure(self) -> None:
        """Run1 graded, run2 failure → graded_task_count == 1."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", TaskStatus.SUCCESS, attempt_id="att-a")],
            [_make_grade("g1", "att-a", "task_a", 0.8)],
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_b", TaskStatus.FAILURE, attempt_id="att-b")],
            [],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run1, run2],
            registry,
            policy,
        )
        assert result.task_count == 2
        assert result.graded_task_count == 1
        assert abs(result.task_coverage - 0.5) < 1e-9

    # -- NEW: attempt_id pairing errors ----------------------------------

    def test_orphan_grade_result_raises(self) -> None:
        """GradeResult with no matching TaskAttempt → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [_make_grade("orphan", "att-nonexistent", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Orphan GradeResult"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
            )

    def test_grade_attempt_task_id_mismatch_raises(self) -> None:
        """GradeResult.task_id != TaskAttempt.task_id → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [_make_grade("g1", "att-a", "wrong_task_id", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="does not match"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
            )

    def test_duplicate_grade_for_attempt_raises(self) -> None:
        """Two GradeResults for same attempt_id → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [
                _make_grade("g1", "att-a", "task_a", 0.8),
                _make_grade("g2", "att-a", "task_a", 0.9),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate GradeResult"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
            )

    def test_provenance_mismatch_grade_raises(self) -> None:
        """GradeResult with different provenance → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [_make_grade("g1", "att-a", "task_a", 0.8, provenance_override={"source_id": "different"})],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Provenance mismatch"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
            )

    # -- NEW: calibration state machine ----------------------------------

    def test_no_calibration_returns_uncalibrated(self) -> None:
        """Default NoCalibration → status UNCALIBRATED, calibrated_score None."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        assert result.status == DimensionScoreStatus.UNCALIBRATED
        assert result.calibrated_score is None

    def test_fake_calibration_returns_scored(self) -> None:
        """Fake calibrator returning 83.5 → status SCORED, calibrated_score = 83.5."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
            calibrator=_FakeCalibrator(83.5),
        )
        assert result.status == DimensionScoreStatus.SCORED
        assert result.calibrated_score == 83.5

    def test_calibrated_score_out_of_range_raises(self) -> None:
        """Calibrator returning >100 → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
                calibrator=_FakeCalibrator(101.0),
            )

    def test_calibrated_score_negative_raises(self) -> None:
        """Calibrator returning <0 → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
                calibrator=_FakeCalibrator(-0.1),
            )


# ---------------------------------------------------------------------------
# Capability profile tests
# ---------------------------------------------------------------------------


class TestCapabilityProfile:
    """Tests for aggregate_capability_profile()."""

    def test_profile_with_two_dimensions(self) -> None:
        """Only reasoning + coding tested → coverage_weight = 0.45."""
        e1, e2 = str(uuid4()), str(uuid4())
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="reasoning_task", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="coding_task", dimension=CapabilityDimension.CODING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [
                _make_attempt("reasoning_task", evidence_ids=[e1]),
                _make_attempt("coding_task", evidence_ids=[e2]),
            ],
            [
                _make_grade("g1", "att-reasoning_task", "reasoning_task", 0.8, evidence_ids=[e1]),
                _make_grade("g2", "att-coding_task", "coding_task", 0.7, evidence_ids=[e2]),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile(
            [run],
            registry,
            policy,
        )

        assert abs(profile.coverage_weight - 0.45) < 1e-9
        assert abs(profile.provisional_raw_index - 0.34) < 1e-9
        assert profile.calibrated_total_score is None

        assert e1 in profile.evidence_refs
        assert e2 in profile.evidence_refs

        dims_by_name = {d.dimension: d for d in profile.dimensions}
        assert dims_by_name[CapabilityDimension.REASONING].status == DimensionScoreStatus.UNCALIBRATED
        assert dims_by_name[CapabilityDimension.CODING].status == DimensionScoreStatus.UNCALIBRATED
        assert dims_by_name[CapabilityDimension.MATH_SCIENCE].status == DimensionScoreStatus.UNAVAILABLE
        assert dims_by_name[CapabilityDimension.INSTRUCTION_FOLLOWING].status == DimensionScoreStatus.UNAVAILABLE

    def test_coverage_not_renormalized(self) -> None:
        """coverage_weight must NOT become 1.0 when only some dimensions tested."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile(
            [run],
            registry,
            policy,
        )
        assert abs(profile.coverage_weight - 0.25) < 1e-9
        assert abs(profile.provisional_raw_index - 0.20) < 1e-9

    def test_all_calibrated_scores_none(self) -> None:
        """All calibrated_score and calibrated_total_score must be None."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile(
            [run],
            registry,
            policy,
        )
        assert profile.calibrated_total_score is None
        for d in profile.dimensions:
            assert d.calibrated_score is None

    def test_profile_deterministic(self) -> None:
        """Same input must produce identical model_dump_json()."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()

        profile1 = aggregate_capability_profile([run], registry, policy)
        profile2 = aggregate_capability_profile([run], registry, policy)

        assert profile1.model_dump_json() == profile2.model_dump_json()

    def test_profile_is_frozen(self) -> None:
        """CapabilityProfile must be frozen (immutable)."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile([run], registry, policy)
        with pytest.raises((TypeError, ValueError)):
            profile.coverage_weight = 1.0  # type: ignore[misc]

    def test_smoke_does_not_affect_profile(self) -> None:
        """Smoke task must not change provisional_raw_index or dimension scores."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(
                    task_id="smoke",
                    dimension=CapabilityDimension.REASONING,
                    task_weight=1.0,
                    capability_score_eligible=False,
                ),
            ]
        )
        run = _make_run(
            "run-1",
            [
                _make_attempt("task_a", TaskStatus.SUCCESS),
                _make_attempt("smoke", TaskStatus.SUCCESS),
            ],
            [
                _make_grade("g1", "att-task_a", "task_a", 0.5),
                _make_grade("g2", "att-smoke", "smoke", 1.0),
            ],
        )
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile([run], registry, policy)

        dim_reasoning = next(d for d in profile.dimensions if d.dimension == CapabilityDimension.REASONING)
        assert abs(dim_reasoning.raw_normalized_score - 0.5) < 1e-9
        assert abs(profile.provisional_raw_index - 0.5 * 0.25) < 1e-9

    # -- NEW: profile mutation tests -------------------------------------

    def test_dimension_result_is_frozen(self) -> None:
        """DimensionScoreResult must be frozen."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.5)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        with pytest.raises((TypeError, ValueError)):
            result.raw_normalized_score = 0.9  # type: ignore[misc]

    def test_profile_evidence_refs_immutable(self) -> None:
        """evidence_refs on profile is a tuple → cannot be mutated."""
        eid = str(uuid4())
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", evidence_ids=[eid])],
            [_make_grade("g1", "att-task_a", "task_a", 0.5, evidence_ids=[eid])],
        )
        policy = CapabilityScoringPolicy.create_v1()
        profile = aggregate_capability_profile([run], registry, policy)
        # evidence_refs is a tuple, cannot append
        with pytest.raises(AttributeError):
            profile.evidence_refs.append("fake-uuid")  # type: ignore[union-attr]

    def test_dimension_evidence_refs_immutable(self) -> None:
        """evidence_refs on DimensionScoreResult is a tuple."""
        eid = str(uuid4())
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a", evidence_ids=[eid])],
            [_make_grade("g1", "att-task_a", "task_a", 0.5, evidence_ids=[eid])],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run],
            registry,
            policy,
        )
        with pytest.raises(AttributeError):
            result.evidence_refs.append("fake")  # type: ignore[union-attr]

    # -- NEW: cross-run pairing rejection ---------------------------------

    def test_duplicate_attempt_id_within_one_run_raises(self) -> None:
        """Two attempts with the same attempt_id in one run → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [
                _make_attempt("task_a", attempt_id="att-x"),
                _make_attempt("task_a", attempt_id="att-x"),  # duplicate!
            ],
            [],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate attempt_id.*within"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
            )

    def test_duplicate_attempt_id_across_two_runs_raises(self) -> None:
        """Same attempt_id in two different runs → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-x")],
            [_make_grade("g1", "att-x", "task_a", 0.8)],
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_b", attempt_id="att-x")],  # same attempt_id!
            [_make_grade("g2", "att-x", "task_b", 0.6)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate attempt_id.*across"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run1, run2],
                registry,
                policy,
            )

    def test_grade_crosses_run_boundary_raises(self) -> None:
        """Grade in run2 referencing attempt only in run1 → orphan ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [],
        )
        run2 = _make_run(
            "run-2",
            [],
            [_make_grade("g1", "att-a", "task_a", 0.8)],  # grade references run1's attempt
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Orphan GradeResult"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run1, run2],
                registry,
                policy,
            )

    # -- Duplicate attempt_id detection based on TaskAttempt pre-scan ----

    def test_duplicate_attempt_id_across_runs_no_grade_one_side(self) -> None:
        """run1 att-x NO grade, run2 att-x WITH grade → ValueError (pre-scan catches it)."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-x")],
            [],  # NO grade
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_a", attempt_id="att-x")],  # duplicate attempt_id!
            [_make_grade("g1", "att-x", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate attempt_id.*across"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run1, run2],
                registry,
                policy,
            )

    def test_duplicate_attempt_id_across_runs_both_no_grade(self) -> None:
        """run1 att-x NO grade, run2 att-x NO grade → still ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-x")],
            [],  # NO grade
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_a", attempt_id="att-x")],  # duplicate attempt_id!
            [],  # NO grade
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate attempt_id.*across"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run1, run2],
                registry,
                policy,
            )

    def test_score_leakage_prevention(self) -> None:
        """run1 task_a no grade, run2 task_b same attempt_id with grade=1.0 → ValueError.

        Proves run2's grade never enters run1's attempt scoring.
        No DimensionScoreResult is produced.
        """
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-x")],
            [],  # task_a has NO grade
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_b", attempt_id="att-x")],  # same attempt_id!
            [_make_grade("g1", "att-x", "task_b", 1.0)],  # grade with normalized_score=1.0
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate attempt_id.*across"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run1, run2],
                registry,
                policy,
            )

    def test_legitimate_two_run_aggregation_still_passes(self) -> None:
        """Two separate runs with distinct attempt_ids → legitimate aggregation."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                TaskScoringSpec(task_id="task_b", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run1 = _make_run(
            "run-1",
            [_make_attempt("task_a", attempt_id="att-a")],
            [_make_grade("g1", "att-a", "task_a", 0.8)],
        )
        run2 = _make_run(
            "run-2",
            [_make_attempt("task_b", attempt_id="att-b")],
            [_make_grade("g2", "att-b", "task_b", 0.6)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        result = aggregate_dimension_score(
            CapabilityDimension.REASONING,
            [run1, run2],
            registry,
            policy,
        )
        assert result.graded_task_count == 2
        assert abs(result.task_coverage - 1.0) < 1e-9

    # -- NEW: non-finite calibration rejection ----------------------------

    def test_nan_calibration_raises(self) -> None:
        """Calibrator returning NaN → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="non-finite"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
                calibrator=_FakeCalibrator(float("nan")),
            )

    def test_inf_calibration_raises(self) -> None:
        """Calibrator returning +Infinity → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="non-finite"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
                calibrator=_FakeCalibrator(float("inf")),
            )

    def test_neg_inf_calibration_raises(self) -> None:
        """Calibrator returning -Infinity → ValueError."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        run = _make_run(
            "run-1",
            [_make_attempt("task_a")],
            [_make_grade("g1", "att-task_a", "task_a", 0.8)],
        )
        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="non-finite"):
            aggregate_dimension_score(
                CapabilityDimension.REASONING,
                [run],
                registry,
                policy,
                calibrator=_FakeCalibrator(float("-inf")),
            )
