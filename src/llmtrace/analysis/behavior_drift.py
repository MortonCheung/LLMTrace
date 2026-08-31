"""Behavior Drift engine — compare two BehaviorRunSnapshots under a versioned policy.

This module is a *separate* drift concept from ``analysis/drift.py`` (protocol
/ operational report drift).  Here we answer: given two runs of the same
target API under identical generation conditions, what behavior changed?

Pipeline::

    compatibility gate (fail closed)
        → stable item alignment (BehaviorItemKey)
        → detector pipeline (Outcome / Status / Output / Operational)
        → dimension drift (raw_normalized_score delta)
        → BehaviorDriftResult

Detectors are pluggable, deterministic, side-effect-free, and never decide the
final drift level by themselves — they only emit local ``BehaviorSignal`` s.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llmtrace.benchmarks.models import ItemStatus
from llmtrace.scoring.comparison import comparable_dimensions
from llmtrace.scoring.models import CapabilityDimension

from .behavior_models import (
    BehaviorAdapterMismatchError,
    BehaviorCoverageMismatchError,
    BehaviorDriftLevel,
    BehaviorDriftPolicy,
    BehaviorItemKey,
    BehaviorItemObservation,
    BehaviorItemSetMismatchError,
    BehaviorRunSnapshot,
    BehaviorScoringPolicyMismatchError,
    BehaviorSourceMismatchError,
    BehaviorSuiteMismatchError,
    BehaviorSuiteVersionMismatchError,
    GenerationConfigMismatchError,
)

# ---------------------------------------------------------------------------
# Detector protocol and signal
# ---------------------------------------------------------------------------


class BehaviorSignal(BaseModel):
    """Local signal produced by a single detector for a single item."""

    detector_id: str = Field(..., min_length=1, description="Detector identifier")
    detector_version: str = Field(..., min_length=1, description="Detector version")
    changed: bool = Field(..., description="Whether this detector observed a change")
    summary: str = Field(..., description="Human-readable signal summary")
    details: dict[str, Any] = Field(default_factory=dict, description="Machine-readable signal details")

    model_config = {"frozen": True, "extra": "forbid"}


@runtime_checkable
class BehaviorDetector(Protocol):
    """Minimal detector plugin contract.

    A detector is deterministic, side-effect-free, reads no files, sends no
    requests, never mutates a snapshot, and never decides the final drift
    level — it only produces a local ``BehaviorSignal``.
    """

    detector_id: str
    detector_version: str

    def detect(self, baseline: BehaviorItemObservation, current: BehaviorItemObservation) -> BehaviorSignal:
        """Compare one aligned item pair and return a signal."""
        ...


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class OutcomeChangeDetector:
    """Detect whether the graded outcome (normalized_score) changed."""

    detector_id = "outcome_change"
    detector_version = "0.1.0"

    def detect(self, baseline: BehaviorItemObservation, current: BehaviorItemObservation) -> BehaviorSignal:
        delta = current.normalized_score - baseline.normalized_score
        changed = baseline.normalized_score != current.normalized_score
        return BehaviorSignal(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            changed=changed,
            summary="outcome changed" if changed else "outcome unchanged",
            details={
                "baseline_score": baseline.normalized_score,
                "current_score": current.normalized_score,
                "delta": delta,
            },
        )


class StatusChangeDetector:
    """Detect transitions between GRADED / UNGRADABLE / FAILURE."""

    detector_id = "status_change"
    detector_version = "0.1.0"

    def detect(self, baseline: BehaviorItemObservation, current: BehaviorItemObservation) -> BehaviorSignal:
        changed = baseline.status != current.status
        return BehaviorSignal(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            changed=changed,
            summary=f"status changed {baseline.status.value} → {current.status.value}"
            if changed
            else "status unchanged",
            details={
                "baseline_status": baseline.status.value,
                "current_status": current.status.value,
            },
        )


class OutputChangeDetector:
    """Detect whether the canonicalized output text changed.

    An output change is an *observation*, never proof of a model switch.
    """

    detector_id = "output_change"
    detector_version = "0.1.0"

    def detect(self, baseline: BehaviorItemObservation, current: BehaviorItemObservation) -> BehaviorSignal:
        changed = baseline.output_text_sha256 != current.output_text_sha256
        return BehaviorSignal(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            changed=changed,
            summary="output text changed" if changed else "output text unchanged",
            details={
                "baseline_output_sha256": baseline.output_text_sha256,
                "current_output_sha256": current.output_text_sha256,
                "baseline_output_length": baseline.output_length,
                "current_output_length": current.output_length,
            },
        )


class OperationalChangeDetector:
    """Detect changes in auxiliary operational signals.

    Latency is recorded in details but does NOT flip ``changed`` — a latency
    change alone is not evidence of a model switch.
    """

    detector_id = "operational_change"
    detector_version = "0.1.0"

    def detect(self, baseline: BehaviorItemObservation, current: BehaviorItemObservation) -> BehaviorSignal:
        model_changed = baseline.response_model != current.response_model
        finish_changed = baseline.finish_reason != current.finish_reason
        tokens_changed = (
            baseline.input_tokens != current.input_tokens or baseline.output_tokens != current.output_tokens
        )
        changed = model_changed or finish_changed or tokens_changed
        return BehaviorSignal(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            changed=changed,
            summary="operational signal changed" if changed else "operational signals unchanged",
            details={
                "baseline_response_model": baseline.response_model,
                "current_response_model": current.response_model,
                "baseline_finish_reason": baseline.finish_reason,
                "current_finish_reason": current.finish_reason,
                "baseline_input_tokens": baseline.input_tokens,
                "current_input_tokens": current.input_tokens,
                "baseline_output_tokens": baseline.output_tokens,
                "current_output_tokens": current.output_tokens,
                "baseline_latency_ms": baseline.latency_ms,
                "current_latency_ms": current.latency_ms,
            },
        )


_DEFAULT_DETECTORS: tuple[BehaviorDetector, ...] = (
    OutcomeChangeDetector(),
    StatusChangeDetector(),
    OutputChangeDetector(),
    OperationalChangeDetector(),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class ItemDriftResult(BaseModel):
    """Drift result for one aligned item pair."""

    key: BehaviorItemKey = Field(..., description="Stable item identity")
    baseline_status: ItemStatus = Field(..., description="Baseline item status")
    current_status: ItemStatus = Field(..., description="Current item status")
    baseline_score: float = Field(..., ge=0.0, le=1.0, description="Baseline normalized_score")
    current_score: float = Field(..., ge=0.0, le=1.0, description="Current normalized_score")
    score_delta: float = Field(..., description="current_score - baseline_score")
    outcome_changed: bool = Field(..., description="Graded outcome changed")
    status_changed: bool = Field(..., description="Item status changed")
    output_changed: bool = Field(..., description="Canonicalized output text changed")
    operational_changed: bool = Field(..., description="Auxiliary operational signal changed")
    signals: tuple[BehaviorSignal, ...] = Field(default_factory=tuple, description="Detector signals")
    baseline_evidence_refs: tuple[str, ...] = Field(default_factory=tuple, description="Baseline evidence UUIDs")
    current_evidence_refs: tuple[str, ...] = Field(default_factory=tuple, description="Current evidence UUIDs")

    model_config = {"frozen": True, "extra": "forbid"}


class DimensionDriftResult(BaseModel):
    """Drift for one capability dimension (raw_normalized_score delta)."""

    dimension: CapabilityDimension = Field(..., description="Capability dimension")
    baseline_score: float = Field(..., ge=0.0, le=1.0, description="Baseline raw_normalized_score")
    current_score: float = Field(..., ge=0.0, le=1.0, description="Current raw_normalized_score")
    delta: float = Field(..., description="current - baseline")
    absolute_delta: float = Field(..., ge=0.0, description="abs(current - baseline)")

    model_config = {"frozen": True, "extra": "forbid"}


class BehaviorDriftResult(BaseModel):
    """Result of comparing two BehaviorRunSnapshots under a policy."""

    baseline_run_id: str = Field(..., min_length=1, description="Baseline run id")
    current_run_id: str = Field(..., min_length=1, description="Current run id")
    target_id: str = Field(..., min_length=1, description="Target label")
    candidate_model_id: str = Field(..., min_length=1, description="Candidate model label")
    suite_id: str = Field(..., min_length=1, description="Suite identifier")
    suite_version: str = Field(..., min_length=1, description="Suite version")
    policy_id: str = Field(..., min_length=1, description="Policy identifier")
    policy_version: str = Field(..., min_length=1, description="Policy version")

    total_items: int = Field(..., ge=0, description="Total aligned items")
    graded_overlap_count: int = Field(..., ge=0, description="Items graded in both runs")
    graded_overlap_ratio: float = Field(..., ge=0.0, le=1.0, description="graded_overlap_count / total_items")
    outcome_changed_count: int = Field(..., ge=0, description="Graded items whose outcome changed")
    outcome_changed_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="outcome_changed_count / graded_overlap_count"
    )
    status_changed_count: int = Field(..., ge=0, description="Items whose status changed")
    status_changed_ratio: float = Field(..., ge=0.0, le=1.0, description="status_changed_count / total_items")
    output_changed_count: int = Field(..., ge=0, description="Graded items whose output text changed")
    output_changed_ratio: float = Field(..., ge=0.0, le=1.0, description="output_changed_count / graded_overlap_count")
    response_model_change_count: int = Field(..., ge=0, description="Items whose response_model changed")
    finish_reason_change_count: int = Field(..., ge=0, description="Items whose finish_reason changed")

    dimension_diffs: tuple[DimensionDriftResult, ...] = Field(default_factory=tuple, description="Per-dimension drift")
    item_diffs: tuple[ItemDriftResult, ...] = Field(default_factory=tuple, description="Per-item drift")
    drift_level: BehaviorDriftLevel = Field(..., description="Assigned drift level")
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Non-fatal warnings")

    model_config = {"frozen": True, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BehaviorDriftEngine:
    """Compare two BehaviorRunSnapshots, fail-closed on incompatibility."""

    def __init__(self, detectors: tuple[BehaviorDetector, ...] | None = None) -> None:
        self._detectors = detectors if detectors is not None else _DEFAULT_DETECTORS

    def compare(
        self,
        baseline: BehaviorRunSnapshot,
        current: BehaviorRunSnapshot,
        policy: BehaviorDriftPolicy,
    ) -> BehaviorDriftResult:
        """Compare *baseline* against *current* under *policy*.

        Raises:
            BehaviorDriftCompatibilityError: If the two snapshots are not
                comparable.  An incomparable pair is NOT an inconclusive
                result — it fails closed.
        """
        self._check_compatibility(baseline, current)

        current_by_key = {obs.key: obs for obs in current.items}

        item_diffs: list[ItemDriftResult] = []
        graded_overlap = 0
        outcome_changed = 0
        output_changed = 0
        status_changed = 0
        response_model_changed = 0
        finish_reason_changed = 0

        for b_obs in baseline.items:
            c_obs = current_by_key[b_obs.key]  # guaranteed present by the gate

            signals = tuple(det.detect(b_obs, c_obs) for det in self._detectors)
            signal_map = {s.detector_id: s for s in signals}

            outcome_signal = signal_map["outcome_change"]
            status_signal = signal_map["status_change"]
            output_signal = signal_map["output_change"]
            operational_signal = signal_map["operational_change"]

            both_graded = b_obs.status == ItemStatus.GRADED and c_obs.status == ItemStatus.GRADED
            if both_graded:
                graded_overlap += 1
                if outcome_signal.changed:
                    outcome_changed += 1
                if output_signal.changed:
                    output_changed += 1

            if status_signal.changed:
                status_changed += 1
            if b_obs.response_model != c_obs.response_model:
                response_model_changed += 1
            if b_obs.finish_reason != c_obs.finish_reason:
                finish_reason_changed += 1

            item_diffs.append(
                ItemDriftResult(
                    key=b_obs.key,
                    baseline_status=b_obs.status,
                    current_status=c_obs.status,
                    baseline_score=b_obs.normalized_score,
                    current_score=c_obs.normalized_score,
                    score_delta=c_obs.normalized_score - b_obs.normalized_score,
                    outcome_changed=both_graded and outcome_signal.changed,
                    status_changed=status_signal.changed,
                    output_changed=output_signal.changed,
                    operational_changed=operational_signal.changed,
                    signals=signals,
                    baseline_evidence_refs=b_obs.evidence_refs,
                    current_evidence_refs=c_obs.evidence_refs,
                )
            )

        total_items = len(baseline.items)
        graded_overlap_ratio = graded_overlap / total_items if total_items else 0.0
        outcome_changed_ratio = outcome_changed / graded_overlap if graded_overlap else 0.0
        output_changed_ratio = output_changed / graded_overlap if graded_overlap else 0.0
        status_changed_ratio = status_changed / total_items if total_items else 0.0

        dimension_diffs = self._compute_dimension_diffs(baseline, current)

        drift_level, warnings = self._classify(
            policy=policy,
            graded_overlap_ratio=graded_overlap_ratio,
            outcome_changed_ratio=outcome_changed_ratio,
            outcome_changed_count=outcome_changed,
            output_changed_count=output_changed,
            status_changed_count=status_changed,
            response_model_changed=response_model_changed,
            finish_reason_changed=finish_reason_changed,
            dimension_diffs=dimension_diffs,
        )

        return BehaviorDriftResult(
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            target_id=baseline.target_id,
            candidate_model_id=baseline.candidate_model_id,
            suite_id=baseline.suite_id,
            suite_version=baseline.suite_version,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            total_items=total_items,
            graded_overlap_count=graded_overlap,
            graded_overlap_ratio=graded_overlap_ratio,
            outcome_changed_count=outcome_changed,
            outcome_changed_ratio=outcome_changed_ratio,
            status_changed_count=status_changed,
            status_changed_ratio=status_changed_ratio,
            output_changed_count=output_changed,
            output_changed_ratio=output_changed_ratio,
            response_model_change_count=response_model_changed,
            finish_reason_change_count=finish_reason_changed,
            dimension_diffs=dimension_diffs,
            item_diffs=tuple(item_diffs),
            drift_level=drift_level,
            warnings=tuple(warnings),
        )

    # -- Gate --------------------------------------------------------------

    @staticmethod
    def _check_compatibility(baseline: BehaviorRunSnapshot, current: BehaviorRunSnapshot) -> None:
        """Fail closed unless every comparability axis matches, in order."""
        if current.suite_id != baseline.suite_id:
            raise BehaviorSuiteMismatchError(
                f"suite_id mismatch: baseline '{baseline.suite_id}' != current '{current.suite_id}'"
            )
        if current.suite_version != baseline.suite_version:
            raise BehaviorSuiteVersionMismatchError(
                f"suite_version mismatch: baseline '{baseline.suite_version}' != current '{current.suite_version}'"
            )
        if current.source_ids != baseline.source_ids or current.source_revisions != baseline.source_revisions:
            raise BehaviorSourceMismatchError(
                f"source mismatch: baseline ids={baseline.source_ids} revs={baseline.source_revisions} "
                f"!= current ids={current.source_ids} revs={current.source_revisions}"
            )
        if current.adapter_id != baseline.adapter_id or current.adapter_version != baseline.adapter_version:
            raise BehaviorAdapterMismatchError(
                f"adapter mismatch: baseline '{baseline.adapter_id}'/{baseline.adapter_version} "
                f"!= current '{current.adapter_id}'/{current.adapter_version}"
            )
        if current.scoring_policy_id != baseline.scoring_policy_id:
            raise BehaviorScoringPolicyMismatchError(
                f"scoring_policy_id mismatch: baseline '{baseline.scoring_policy_id}' "
                f"!= current '{current.scoring_policy_id}'"
            )
        if current.scoring_policy_version != baseline.scoring_policy_version:
            raise BehaviorScoringPolicyMismatchError(
                f"scoring_policy_version mismatch: baseline '{baseline.scoring_policy_version}' "
                f"!= current '{current.scoring_policy_version}'"
            )
        if current.generation_config_sha256 != baseline.generation_config_sha256:
            raise GenerationConfigMismatchError(
                f"generation_config mismatch: baseline '{baseline.generation_config_sha256}' "
                f"!= current '{current.generation_config_sha256}'"
            )

        baseline_keys = {obs.key for obs in baseline.items}
        current_keys = {obs.key for obs in current.items}
        if baseline_keys != current_keys:
            raise BehaviorItemSetMismatchError(
                f"stable item set mismatch: baseline has {len(baseline_keys)} items, "
                f"current has {len(current_keys)} items"
            )

        baseline_dims = set(comparable_dimensions(baseline.capability_profile))
        current_dims = set(comparable_dimensions(current.capability_profile))
        if baseline_dims != current_dims:
            raise BehaviorCoverageMismatchError(
                f"coverage mismatch: baseline comparable dimensions "
                f"{sorted(d.value for d in baseline_dims)} != current {sorted(d.value for d in current_dims)}"
            )

    # -- Dimension drift ---------------------------------------------------

    @staticmethod
    def _compute_dimension_diffs(
        baseline: BehaviorRunSnapshot, current: BehaviorRunSnapshot
    ) -> tuple[DimensionDriftResult, ...]:
        current_by_dim = {d.dimension: d for d in current.capability_profile.dimensions}
        diffs: list[DimensionDriftResult] = []
        for b_dim in baseline.capability_profile.dimensions:
            c_dim = current_by_dim.get(b_dim.dimension)
            if c_dim is None:
                continue
            delta = c_dim.raw_normalized_score - b_dim.raw_normalized_score
            diffs.append(
                DimensionDriftResult(
                    dimension=b_dim.dimension,
                    baseline_score=b_dim.raw_normalized_score,
                    current_score=c_dim.raw_normalized_score,
                    delta=delta,
                    absolute_delta=abs(delta),
                )
            )
        return tuple(diffs)

    # -- Classification ----------------------------------------------------

    @staticmethod
    def _classify(
        *,
        policy: BehaviorDriftPolicy,
        graded_overlap_ratio: float,
        outcome_changed_ratio: float,
        outcome_changed_count: int,
        output_changed_count: int,
        status_changed_count: int,
        response_model_changed: int,
        finish_reason_changed: int,
        dimension_diffs: tuple[DimensionDriftResult, ...],
    ) -> tuple[BehaviorDriftLevel, list[str]]:
        warnings: list[str] = []

        if graded_overlap_ratio < policy.minimum_graded_overlap_ratio:
            warnings.append(
                f"graded overlap {graded_overlap_ratio:.3f} below minimum "
                f"{policy.minimum_graded_overlap_ratio:.3f}; not enough comparable data"
            )
            return BehaviorDriftLevel.INCONCLUSIVE, warnings

        material_dimension = any(d.absolute_delta >= policy.material_dimension_delta for d in dimension_diffs)
        material_outcome = outcome_changed_ratio >= policy.material_outcome_change_ratio

        if material_dimension or material_outcome:
            if material_dimension:
                warnings.append("a capability dimension delta reached the MATERIAL threshold")
            if material_outcome:
                warnings.append("outcome change ratio reached the MATERIAL threshold")
            return BehaviorDriftLevel.MATERIAL_DRIFT, warnings

        if (
            outcome_changed_count > 0
            or output_changed_count > 0
            or status_changed_count > 0
            or response_model_changed > 0
            or finish_reason_changed > 0
        ):
            return BehaviorDriftLevel.OBSERVED_DRIFT, warnings

        return BehaviorDriftLevel.NO_SIGNIFICANT_DRIFT, warnings
