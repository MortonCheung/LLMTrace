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
) -> TaskAttempt:
    failure = None
    if status == TaskStatus.FAILURE:
        failure = AdapterFailure(
            error_code="TEST_ERROR",
            category=FailureCategory.UNKNOWN,
            message="Test-induced failure",
        )
    return TaskAttempt(
        attempt_id=f"att-{task_id}",
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
) -> GradeResult:
    return GradeResult(
        grade_id=grade_id,
        attempt_id=attempt_id,
        task_id=task_id,
        grader_id="exact_match",
        raw_score=normalized_score,
        normalized_score=normalized_score,
        status=status,
        evidence_refs=evidence_ids or [str(uuid4())],
        **_PROVENANCE,
    )


def _make_run(
    run_id: str,
    attempts: list[TaskAttempt],
    grades: list[GradeResult],
) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        run_id=run_id,
        task_attempts=attempts,
        grade_results=grades,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        **_PROVENANCE,
    )


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
        # (0.8*3 + 0.6*1) / (3+1) = 3.0/4 = 0.75
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
        # Smoke excluded → only task_a in the score
        assert abs(result.raw_normalized_score - 0.5) < 1e-9
        assert abs(result.task_coverage - 1.0) < 1e-9  # smoke weight doesn't count toward planned
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
        # eid should appear only once
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

        # coverage_weight: reasoning(0.25) + coding(0.20) = 0.45
        assert abs(profile.coverage_weight - 0.45) < 1e-9
        # provisional_raw_index: 0.8*0.25 + 0.7*0.20 = 0.20 + 0.14 = 0.34
        assert abs(profile.provisional_raw_index - 0.34) < 1e-9
        assert profile.calibrated_total_score is None

        # Evidence propagation
        assert e1 in profile.evidence_refs
        assert e2 in profile.evidence_refs

        # Check dimensions
        dims_by_name = {d.dimension: d for d in profile.dimensions}
        assert dims_by_name[CapabilityDimension.REASONING].status == DimensionScoreStatus.UNCALIBRATED
        assert dims_by_name[CapabilityDimension.CODING].status == DimensionScoreStatus.UNCALIBRATED
        # Untested dimensions should be UNAVAILABLE or INSUFFICIENT_DATA
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
        # Only reasoning → coverage_weight = 0.25, NOT 1.0
        assert abs(profile.coverage_weight - 0.25) < 1e-9
        # provisional_raw_index = 0.8 * 0.25 = 0.20, NOT 0.8
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
        with pytest.raises((TypeError, ValueError)):  # frozen model raises on mutation
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
        assert abs(dim_reasoning.raw_normalized_score - 0.5) < 1e-9  # NOT (0.5+1.0)/2=0.75
        assert abs(profile.provisional_raw_index - 0.5 * 0.25) < 1e-9
