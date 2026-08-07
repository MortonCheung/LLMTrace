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
        """Return the TaskScoringSpec for *task_id*, or None if unregistered."""
        return self._specs.get(task_id)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Any:
        return iter(self._specs.values())

    def items(self) -> Any:
        return self._specs.items()


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


def _count_graded_tasks(
    run_results: Sequence[BenchmarkRunResult],
    registry: TaskScoringRegistry,
    dimension: CapabilityDimension,
    graded_task_ids: set[str],
) -> int:
    """Count successfully graded tasks for a dimension."""
    count = 0
    for run in run_results:
        for a in run.task_attempts:
            spec = registry.get(a.task_id)
            if (
                spec is not None
                and spec.dimension == dimension
                and spec.capability_score_eligible
                and a.status == TaskStatus.SUCCESS
                and a.task_id in graded_task_ids
            ):
                count += 1
    return count


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
    """
    if calibrator is None:
        calibrator = _NoCalibration()

    global_weight = policy.weight_for(dimension)
    warnings: list[str] = []

    # Collect eligible + graded tasks for this dimension
    planned_weight_sum = 0.0
    graded_weight_sum = 0.0
    weighted_score_sum = 0.0
    evidence_refs: list[str] = []
    source_task_ids: list[str] = []

    for run in run_results:
        graded_map: dict[str, float] = {}  # task_id → normalized_score
        graded_evidence: dict[str, list[str]] = {}  # task_id → evidence_refs

        for grade in run.grade_results:
            spec = _resolve_spec(registry, grade.task_id, strict)
            if spec is None:
                continue
            if spec.dimension != dimension:
                continue
            if not spec.capability_score_eligible:
                warnings.append(
                    f"Task '{grade.task_id}' is capability_score_eligible=False; excluding from dimension score"
                )
                continue
            if grade.status != GradeStatus.GRADED:
                continue

            graded_map[grade.task_id] = grade.normalized_score
            graded_evidence[grade.task_id] = list(grade.evidence_refs)

        # Process attempts
        for attempt in run.task_attempts:
            spec = _resolve_spec(registry, attempt.task_id, strict)
            if spec is None:
                continue
            if spec.dimension != dimension:
                continue
            if not spec.capability_score_eligible:
                continue

            planned_weight_sum += spec.task_weight

            if attempt.status == TaskStatus.SUCCESS and attempt.task_id in graded_map:
                score = graded_map[attempt.task_id]
                weighted_score_sum += score * spec.task_weight
                graded_weight_sum += spec.task_weight
                evidence_refs.extend(graded_evidence.get(attempt.task_id, []))
                evidence_refs.extend(attempt.evidence_refs)
                source_task_ids.append(attempt.task_id)

    # Determine status
    if planned_weight_sum == 0.0:
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
            evidence_refs=[],
            source_task_ids=[],
            warnings=warnings,
        )

    raw_normalized_score = weighted_score_sum / graded_weight_sum if graded_weight_sum > 0 else 0.0
    task_coverage = graded_weight_sum / planned_weight_sum

    if graded_weight_sum == 0.0:
        return DimensionScoreResult(
            dimension=dimension,
            status=DimensionScoreStatus.INSUFFICIENT_DATA,
            raw_normalized_score=0.0,
            calibrated_score=None,
            task_count=_count_eligible_tasks(run_results, registry, dimension),
            graded_task_count=0,
            task_coverage=0.0,
            global_weight=global_weight,
            weighted_contribution=0.0,
            evidence_refs=[],
            source_task_ids=[],
            warnings=warnings,
        )

    # Determine status
    status = DimensionScoreStatus.UNCALIBRATED if policy.calibration_required else DimensionScoreStatus.SCORED

    if task_coverage < policy.minimum_dimension_coverage:
        status = DimensionScoreStatus.INSUFFICIENT_DATA

    calibrated = calibrator.calibrate(dimension, raw_normalized_score)

    return DimensionScoreResult(
        dimension=dimension,
        status=status,
        raw_normalized_score=raw_normalized_score,
        calibrated_score=calibrated,
        task_count=_count_eligible_tasks(run_results, registry, dimension),
        graded_task_count=_count_graded_tasks(run_results, registry, dimension, set(graded_map.keys())),
        task_coverage=task_coverage,
        global_weight=global_weight,
        weighted_contribution=raw_normalized_score * global_weight,
        evidence_refs=_normalize_evidence_refs(evidence_refs),
        source_task_ids=sorted(set(source_task_ids)),
        warnings=warnings,
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
        dimensions=dimensions,
        coverage_weight=coverage_weight,
        calibrated_total_score=None,  # Always None until Reference Calibration
        provisional_raw_index=provisional_raw_index,
        evidence_refs=_normalize_evidence_refs(all_evidence),
        warnings=all_warnings,
    )
