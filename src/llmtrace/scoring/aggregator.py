"""Capability scoring aggregator.

Consumes BenchmarkRunResult directly (not reporting models) to keep
the scoring layer independent of the reporting layer.

Architecture:
    benchmarks
        ↓
    scoring          ← this module
        ↓
    reporting
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Protocol

from llmtrace.benchmarks.models import (
    BenchmarkRunResult,
    GradeStatus,
    TaskStatus,
    validate_provenance_consistency,
)

from .errors import TaskRegistrationError
from .models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
    TaskScoringSpec,
)
from .policy import CapabilityScoringPolicy

# ---------------------------------------------------------------------------
# Future calibration extension point (Section 十五)
# ---------------------------------------------------------------------------


class ScoreCalibrator(Protocol):
    """Protocol for converting a raw_normalized_score into a calibrated score.

    Implementations receive a reference profile to place the raw score
    in context.  The current default (NoCalibration) always returns None
    and all calibrated_score fields remain None.
    """

    def calibrate(
        self,
        dimension: CapabilityDimension,
        raw_score: float,
        reference_profile: object | None = None,
    ) -> float | None:
        """Return a calibrated score or None if calibration is unavailable."""
        ...


class _NoCalibration:
    """Default calibrator: always returns None (no reference data)."""

    def calibrate(
        self,
        dimension: CapabilityDimension,
        raw_score: float,
        reference_profile: object | None = None,
    ) -> float | None:
        return None


# ---------------------------------------------------------------------------
# Task scoring registry
# ---------------------------------------------------------------------------


class TaskScoringRegistry:
    """Explicit, immutable registry of TaskScoringSpec entries.

    Every task that participates in capability scoring must have an
    explicit entry.  Heuristic matching (e.g. ``"math" in task_id``)
    is forbidden.
    """

    __slots__ = ("_specs",)

    def __init__(self, specs: Sequence[TaskScoringSpec] | None = None) -> None:
        """Create a registry from a sequence of TaskScoringSpecs.

        Raises:
            TaskRegistrationError: If the same task_id is registered more
                than once.
            ValueError: If any spec has capability_score_eligible=True but
                the task also appears in a different dimension.
        """
        self._specs: dict[str, TaskScoringSpec] = {}
        if specs:
            for spec in specs:
                if spec.task_id in self._specs:
                    raise TaskRegistrationError(f"Duplicate task_id '{spec.task_id}' in scoring registry")
                self._specs[spec.task_id] = copy.deepcopy(spec)

    def get(self, task_id: str) -> TaskScoringSpec | None:
        """Return a deep copy of the TaskScoringSpec for *task_id*, or None."""
        spec = self._specs.get(task_id)
        return copy.deepcopy(spec) if spec is not None else None

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Any:
        """Iterate over deep copies of all TaskScoringSpec entries."""
        for spec in self._specs.values():
            yield copy.deepcopy(spec)

    def items(self) -> Any:
        """Return (task_id, deep-copied TaskScoringSpec) pairs."""
        for tid, spec in self._specs.items():
            yield (tid, copy.deepcopy(spec))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_evidence_refs(refs: list[str]) -> list[str]:
    """Deduplicate evidence UUIDs, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _resolve_spec(
    registry: TaskScoringRegistry,
    task_id: str,
    strict: bool,
) -> TaskScoringSpec | None:
    """Resolve a task_id to its TaskScoringSpec.

    Raises:
        TaskRegistrationError: If *strict* and task_id is not registered.
    """
    spec = registry.get(task_id)
    if spec is None and strict:
        raise TaskRegistrationError(
            f"Task '{task_id}' is not registered in the scoring registry. "
            f"All scored tasks must have an explicit TaskScoringSpec."
        )
    return spec


def _count_eligible_tasks(
    run_results: Sequence[BenchmarkRunResult],
    registry: TaskScoringRegistry,
    dimension: CapabilityDimension,
) -> int:
    """Count eligible tasks for a dimension across all runs."""
    count = 0
    for run in run_results:
        for a in run.task_attempts:
            spec = registry.get(a.task_id)
            if spec is not None and spec.dimension == dimension and spec.capability_score_eligible:
                count += 1
    return count


# ---------------------------------------------------------------------------
# GradeResult ↔ TaskAttempt pairing (by attempt_id)
# ---------------------------------------------------------------------------


def _build_attempt_index(
    run_results: Sequence[BenchmarkRunResult],
) -> dict[str, tuple[BenchmarkRunResult, Any]]:  # Any = TaskAttempt
    """Build a global index of attempt_id → (run_result, TaskAttempt)."""
    index: dict[str, tuple[BenchmarkRunResult, Any]] = {}
    for run in run_results:
        for attempt in run.task_attempts:
            index[attempt.attempt_id] = (run, attempt)
    return index


def _pair_grades_with_attempts(
    run_results: Sequence[BenchmarkRunResult],
) -> dict[str, tuple[BenchmarkRunResult, Any, Any]]:  # (run, attempt, grade)
    """Pair GradeResult → TaskAttempt by attempt_id with full validation.

    Returns:
        attempt_id → (run_result, TaskAttempt, GradeResult)

    Raises:
        ValueError: On orphan GradeResult, duplicate GradeResult,
                    attempt_id/task_id mismatch, or provenance mismatch.
    """
    attempt_index = _build_attempt_index(run_results)
    paired: dict[str, tuple[BenchmarkRunResult, Any, Any]] = {}

    for run in run_results:
        for grade in run.grade_results:
            aid = grade.attempt_id

            # Orphan check
            if aid not in attempt_index:
                raise ValueError(
                    f"Orphan GradeResult: attempt_id='{aid}' (grade_id='{grade.grade_id}') has no matching TaskAttempt"
                )

            # Duplicate check
            if aid in paired:
                existing_grade = paired[aid][2]
                raise ValueError(
                    f"Duplicate GradeResult for attempt_id='{aid}': "
                    f"grade_ids='{grade.grade_id}' and '{existing_grade.grade_id}'"
                )

            # task_id consistency
            _, attempt = attempt_index[aid]
            if grade.task_id != attempt.task_id:
                raise ValueError(
                    f"GradeResult.task_id='{grade.task_id}' does not match "
                    f"TaskAttempt.task_id='{attempt.task_id}' for attempt_id='{aid}'"
                )

            # Provenance: grade vs run
            validate_provenance_consistency(run, grade, child_label="GradeResult")

            # Provenance: attempt vs run
            validate_provenance_consistency(run, attempt, child_label="TaskAttempt")

            paired[aid] = (run, attempt, grade)

    return paired


# ---------------------------------------------------------------------------
# Dimension aggregation
# ---------------------------------------------------------------------------


def aggregate_dimension_score(
    dimension: CapabilityDimension,
    run_results: Sequence[BenchmarkRunResult],
    registry: TaskScoringRegistry,
    policy: CapabilityScoringPolicy,
    *,
    calibrator: ScoreCalibrator | None = None,
    strict: bool = True,
) -> DimensionScoreResult:
    """Aggregate a single capability dimension from benchmark run results.

    Only tasks that are:
      - registered with *dimension* in the registry,
      - capability_score_eligible = True,
      - status = SUCCESS, and
      - graded with GradeStatus.GRADED and normalized_score is not None

    contribute to the dimension score.

    GradeResult is paired with TaskAttempt by attempt_id (not task_id).

    Args:
        dimension: Target capability dimension.
        run_results: BenchmarkRunResult instances to aggregate from.
        registry: Task-to-dimension mapping registry.
        policy: Scoring policy (weights, coverage minimum, etc.).
        calibrator: Optional calibrator (default = NoCalibration → None).
        strict: If True, unregistered tasks raise TaskRegistrationError.

    Returns:
        DimensionScoreResult with aggregate scores and evidence.

    Raises:
        TaskRegistrationError: If *strict* and unregistered tasks encountered.
        ValueError: On orphan GradeResult, duplicate GradeResult,
                    attempt_id/task_id mismatch, or provenance mismatch.
    """
    if calibrator is None:
        calibrator = _NoCalibration()

    global_weight = policy.weight_for(dimension)
    warnings: list[str] = []

    # ---------- Build attempt_id-level grade index ----------
    paired = _pair_grades_with_attempts(run_results)

    # ---------- Aggregate across all runs ----------
    planned_weight_sum = 0.0
    graded_weight_sum = 0.0
    weighted_score_sum = 0.0
    evidence_refs: list[str] = []
    source_task_ids: list[str] = []
    graded_attempt_ids: list[str] = []

    for run in run_results:
        for attempt in run.task_attempts:
            spec = _resolve_spec(registry, attempt.task_id, strict)
            if spec is None:
                continue
            if spec.dimension != dimension:
                continue
            if not spec.capability_score_eligible:
                warnings.append(
                    f"Task '{attempt.task_id}' is capability_score_eligible=False; excluding from dimension score"
                )
                continue

            planned_weight_sum += spec.task_weight

            # Check if this attempt has a valid grade
            grade_entry = paired.get(attempt.attempt_id)
            if grade_entry is None:
                continue  # no grade → cannot participate

            _, _, grade = grade_entry
            if attempt.status != TaskStatus.SUCCESS:
                continue
            if grade.status != GradeStatus.GRADED:
                continue

            graded_weight_sum += spec.task_weight
            weighted_score_sum += grade.normalized_score * spec.task_weight
            evidence_refs.extend(grade.evidence_refs)
            evidence_refs.extend(attempt.evidence_refs)
            source_task_ids.append(attempt.task_id)
            graded_attempt_ids.append(attempt.attempt_id)

    # ---------- Determine status ----------
    eligible_count = _count_eligible_tasks(run_results, registry, dimension)

    if planned_weight_sum == 0.0 or eligible_count == 0:
        return DimensionScoreResult(
            dimension=dimension,
            status=DimensionScoreStatus.UNAVAILABLE,
            raw_normalized_score=0.0,
            calibrated_score=None,
            task_count=0,
            graded_task_count=0,
            task_coverage=0.0,
            global_weight=global_weight,
            weighted_contribution=0.0,
            evidence_refs=(),
            source_task_ids=(),
            warnings=tuple(warnings),
        )

    raw_normalized_score = weighted_score_sum / graded_weight_sum if graded_weight_sum > 0 else 0.0
    task_coverage = graded_weight_sum / planned_weight_sum if planned_weight_sum > 0 else 0.0

    if graded_weight_sum == 0.0:
        return DimensionScoreResult(
            dimension=dimension,
            status=DimensionScoreStatus.INSUFFICIENT_DATA,
            raw_normalized_score=0.0,
            calibrated_score=None,
            task_count=eligible_count,
            graded_task_count=0,
            task_coverage=0.0,
            global_weight=global_weight,
            weighted_contribution=0.0,
            evidence_refs=(),
            source_task_ids=(),
            warnings=tuple(warnings),
        )

    # Coverage below minimum → INSUFFICIENT_DATA
    if task_coverage < policy.minimum_dimension_coverage:
        return DimensionScoreResult(
            dimension=dimension,
            status=DimensionScoreStatus.INSUFFICIENT_DATA,
            raw_normalized_score=raw_normalized_score,
            calibrated_score=None,
            task_count=eligible_count,
            graded_task_count=len(graded_attempt_ids),
            task_coverage=task_coverage,
            global_weight=global_weight,
            weighted_contribution=raw_normalized_score * global_weight,
            evidence_refs=tuple(_normalize_evidence_refs(evidence_refs)),
            source_task_ids=tuple(sorted(set(source_task_ids))),
            warnings=tuple(warnings),
        )

    # ---------- Calibration state machine ----------
    calibrated = calibrator.calibrate(dimension, raw_normalized_score)

    if calibrated is not None:
        # Validate calibration range
        if calibrated < 0.0 or calibrated > 100.0:
            raise ValueError(f"Calibrated score for {dimension.value} is {calibrated}, must be in [0, 100]")
        status = DimensionScoreStatus.SCORED
    else:
        status = DimensionScoreStatus.UNCALIBRATED

    return DimensionScoreResult(
        dimension=dimension,
        status=status,
        raw_normalized_score=raw_normalized_score,
        calibrated_score=calibrated,
        task_count=eligible_count,
        graded_task_count=len(graded_attempt_ids),
        task_coverage=task_coverage,
        global_weight=global_weight,
        weighted_contribution=raw_normalized_score * global_weight,
        evidence_refs=tuple(_normalize_evidence_refs(evidence_refs)),
        source_task_ids=tuple(sorted(set(source_task_ids))),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Capability profile aggregation
# ---------------------------------------------------------------------------


def aggregate_capability_profile(
    run_results: Sequence[BenchmarkRunResult],
    registry: TaskScoringRegistry,
    policy: CapabilityScoringPolicy,
    *,
    calibrator: ScoreCalibrator | None = None,
    strict: bool = True,
) -> CapabilityProfile:
    """Aggregate a full CapabilityProfile from benchmark run results.

    Computes per-dimension scores for all enabled dimensions and
    assembles them into an immutable profile.

    Args:
        run_results: BenchmarkRunResult instances to aggregate from.
        registry: Task-to-dimension mapping registry.
        policy: Scoring policy.
        calibrator: Optional calibrator (default = NoCalibration).
        strict: If True, unregistered tasks raise TaskRegistrationError.

    Returns:
        Immutable CapabilityProfile.
    """
    if calibrator is None:
        calibrator = _NoCalibration()

    dimensions: list[DimensionScoreResult] = []
    all_evidence: list[str] = []
    all_warnings: list[str] = [policy.description]

    for dimension in CapabilityDimension:
        if not policy.is_enabled(dimension):
            continue
        dim_result = aggregate_dimension_score(
            dimension,
            run_results,
            registry,
            policy,
            calibrator=calibrator,
            strict=strict,
        )
        dimensions.append(dim_result)
        all_evidence.extend(dim_result.evidence_refs)
        all_warnings.extend(dim_result.warnings)

    # Compute profile-level aggregates
    coverage_weight = sum(
        d.global_weight
        for d in dimensions
        if d.status not in (DimensionScoreStatus.UNAVAILABLE, DimensionScoreStatus.INSUFFICIENT_DATA)
    )

    provisional_raw_index = sum(
        d.weighted_contribution
        for d in dimensions
        if d.status not in (DimensionScoreStatus.UNAVAILABLE, DimensionScoreStatus.INSUFFICIENT_DATA)
    )

    return CapabilityProfile(
        profile_version="0.1.0",
        scoring_policy_id=policy.policy_id,
        scoring_policy_version=policy.policy_version,
        dimensions=tuple(dimensions),
        coverage_weight=coverage_weight,
        calibrated_total_score=None,  # Always None until Reference Calibration
        provisional_raw_index=provisional_raw_index,
        evidence_refs=tuple(_normalize_evidence_refs(all_evidence)),
        warnings=tuple(all_warnings),
    )
