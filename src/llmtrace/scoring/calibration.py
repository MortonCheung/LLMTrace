"""Reference-relative calibration: Trusted ReferenceSet → formal 0–100 scores.

Design contract (v0.4-B):

    0   = random / chance floor
    50  = reference median (identity-level)
    90  = reference P90 (identity-level)
    100 = suite ceiling (coverage_weight for total; 1.0 per dimension)

Reference identity = ``(provider_id, model_id)``.  Multiple snapshots of the
same identity are aggregated by median to prevent repeated采样 from moving
the distribution.

ponytail: v1 piecewise calibration assumes a curated reference set with
>=5 distinct identities; upgrade to isotonic/IRT only when enough stable
reference data exists to validate a more complex model.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmtrace.reference.reference_set import ReferenceSet, ReferenceSetMember

from llmtrace.scoring.models import (
    CalibrationProvenance,
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy

# ---------------------------------------------------------------------------
# Calibration Policy v1
# ---------------------------------------------------------------------------


class ReferenceCalibrationPolicy:
    """Versioned, immutable calibration policy.

    v1 rules are locked: piecewise linear, P90 flagship, >=5 identities,
    same-model median aggregation, clamp out-of-range.

    Changing these rules requires a new policy version with different
    rule/threshold/config — not just a relabelled id.
    """

    __slots__ = (
        "_policy_id",
        "_policy_version",
        "_minimum_distinct_reference_identities",
        "_flagship_quantile",
        "_description",
    )

    def __init__(
        self,
        *,
        policy_id: str,
        policy_version: str,
        minimum_distinct_reference_identities: int = 5,
        flagship_quantile: float = 0.90,
        description: str = "",
    ) -> None:
        self._policy_id = policy_id
        self._policy_version = policy_version
        self._minimum_distinct_reference_identities = minimum_distinct_reference_identities
        self._flagship_quantile = flagship_quantile
        self._description = description

    @classmethod
    def create_v1(cls) -> ReferenceCalibrationPolicy:
        return cls(
            policy_id="llmtrace-reference-calibration-v1",
            policy_version="0.1.0",
            minimum_distinct_reference_identities=5,
            flagship_quantile=0.90,
            description=(
                "Reference-relative piecewise linear calibration v1.  "
                "50=median, 90=P90 flagship, 100=suite ceiling, "
                "0=random floor.  Same-model repeated snapshots use median."
            ),
        )

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def minimum_distinct_reference_identities(self) -> int:
        return self._minimum_distinct_reference_identities

    @property
    def flagship_quantile(self) -> float:
        return self._flagship_quantile

    @property
    def description(self) -> str:
        return self._description

    def __repr__(self) -> str:
        return (
            f"ReferenceCalibrationPolicy(id={self._policy_id!r}, "
            f"v={self._policy_version!r}, min_identities={self._minimum_distinct_reference_identities})"
        )


# ---------------------------------------------------------------------------
# Calibration errors
# ---------------------------------------------------------------------------


class CalibrationError(Exception):
    """Base exception for calibration failures."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class ReferenceSetIntegrityFailureError(CalibrationError):
    """ReferenceSet member verification failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="REFERENCE_SET_INTEGRITY_FAILURE")


class ReferenceSetIncompatibleError(CalibrationError):
    """ReferenceSet is incompatible with the candidate."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="REFERENCE_SET_INCOMPATIBLE")


class UntrustedReferenceSourceError(CalibrationError):
    """A ReferenceSet member is not from operator_verified_api_run."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="UNTRUSTED_REFERENCE_SOURCE")


class InsufficientReferenceIdentitiesError(CalibrationError):
    """Fewer than minimum distinct reference identities."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="INSUFFICIENT_REFERENCE_IDENTITIES")


class InsufficientCalibrationSpreadError(CalibrationError):
    """Reference values lack enough spread for calibration."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="INSUFFICIENT_CALIBRATION_SPREAD")


class CalibrationSaturatedError(CalibrationError):
    """P90 anchor >= ceiling, making 90 indistinguishable from 100."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="CALIBRATION_SATURATED")


class CandidateIncompatibleError(CalibrationError):
    """Candidate profile is incompatible with the calibration context."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="CANDIDATE_INCOMPATIBLE")


# ---------------------------------------------------------------------------
# Quantile (pure stdlib — no numpy/scipy)
# ---------------------------------------------------------------------------


def _linear_quantile(values: list[float], q: float) -> float:
    """Deterministic linear-interpolation quantile.

    Rules:
        sorted values
        position = (n - 1) * q
        lower = floor(position)
        upper = ceil(position)
        fraction = position - lower
        linear interpolation between values[lower] and values[upper]
    """
    if not values:
        raise ValueError("cannot compute quantile of empty list")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    position = (n - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_vals[lower]
    fraction = position - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])


# ---------------------------------------------------------------------------
# Reference identity aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceIdentity:
    """One distinct (provider_id, model_id) after median aggregation."""

    provider_id: str
    model_id: str
    dimension_raws: dict[CapabilityDimension, float]
    total_raw: float
    snapshot_count: int


def aggregate_reference_identities(
    members: tuple[ReferenceSetMember, ...],
    profiles: dict[str, CapabilityProfile],
) -> list[ReferenceIdentity]:
    """Group snapshots by (provider_id, model_id) and aggregate with median.

    ``profiles`` maps ``snapshot_id → CapabilityProfile`` for every member.
    """
    groups: dict[tuple[str, str], dict[str, list[tuple[dict[CapabilityDimension, float], float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for member in members:
        profile = profiles.get(member.snapshot_id)
        if profile is None:
            continue
        key = (member.provider_id, member.model_id)
        dim_raws: dict[CapabilityDimension, float] = {}
        for dim_result in profile.dimensions:
            if dim_result.status in (DimensionScoreStatus.SCORED, DimensionScoreStatus.UNCALIBRATED):
                dim_raws[dim_result.dimension] = dim_result.raw_normalized_score
        groups[key][member.snapshot_id].append((dim_raws, profile.provisional_raw_index))

    identities: list[ReferenceIdentity] = []
    for (provider_id, model_id), snapshots_map in groups.items():
        all_dim_raws: dict[CapabilityDimension, list[float]] = defaultdict(list)
        all_total_raws: list[float] = []
        total_count = 0
        for _snap_id, entries in snapshots_map.items():
            for dim_raws, total_raw in entries:
                for dim, raw in dim_raws.items():
                    all_dim_raws[dim].append(raw)
                all_total_raws.append(total_raw)
                total_count += 1

        median_dim_raws: dict[CapabilityDimension, float] = {}
        for dim, raws in all_dim_raws.items():
            median_dim_raws[dim] = statistics.median(raws)
        median_total = statistics.median(all_total_raws) if all_total_raws else 0.0

        identities.append(
            ReferenceIdentity(
                provider_id=provider_id,
                model_id=model_id,
                dimension_raws=median_dim_raws,
                total_raw=median_total,
                snapshot_count=total_count,
            )
        )

    return identities


# ---------------------------------------------------------------------------
# Internal calibration curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationAnchor:
    """Four calibration anchor points for one score axis."""

    x0: float
    x50: float
    x90: float
    x100: float


@dataclass(frozen=True)
class AxisCalibrationCurve:
    """Piecewise linear calibration curve for one axis (dimension or total)."""

    anchor: CalibrationAnchor

    def calibrate(self, raw: float) -> float:
        """Map a raw score to 0–100 using piecewise linear interpolation."""
        a = self.anchor
        if raw <= a.x0:
            return 0.0
        if raw >= a.x100:
            return 100.0
        if raw <= a.x50:
            return _interpolate(raw, a.x0, 0.0, a.x50, 50.0)
        if raw <= a.x90:
            return _interpolate(raw, a.x50, 50.0, a.x90, 90.0)
        return _interpolate(raw, a.x90, 90.0, a.x100, 100.0)


def _interpolate(x: float, x_lo: float, y_lo: float, x_hi: float, y_hi: float) -> float:
    """Linear interpolation between two points, clamped."""
    if x_hi == x_lo:
        return y_lo
    t = (x - x_lo) / (x_hi - x_lo)
    t = max(0.0, min(1.0, t))
    return y_lo + t * (y_hi - y_lo)


@dataclass(frozen=True)
class CalibrationCurveBundle:
    """All calibration curves for one calibration context."""

    dimension_curves: dict[CapabilityDimension, AxisCalibrationCurve]
    total_curve: AxisCalibrationCurve


# ---------------------------------------------------------------------------
# Build calibration context from ReferenceSet + Policy
# ---------------------------------------------------------------------------


def build_calibration_curves(
    identities: list[ReferenceIdentity],
    scoring_policy: CapabilityScoringPolicy,
    calibration_policy: ReferenceCalibrationPolicy,
    random_floors: dict[CapabilityDimension, float],
) -> CalibrationCurveBundle:
    """Build calibration curves from aggregated reference identities.

    Raises CalibrationError subclasses on any precondition failure.
    """
    min_identities = calibration_policy.minimum_distinct_reference_identities
    if len(identities) < min_identities:
        raise InsufficientReferenceIdentitiesError(
            f"need >= {min_identities} distinct reference identities, got {len(identities)}"
        )

    dimension_curves: dict[CapabilityDimension, AxisCalibrationCurve] = {}

    for dimension in scoring_policy.enabled_dimensions:
        raws = [
            identity.dimension_raws.get(dimension, 0.0)
            for identity in identities
            if dimension in identity.dimension_raws
        ]
        if len(raws) < min_identities:
            raise InsufficientReferenceIdentitiesError(
                f"dimension {dimension.value}: need >= {min_identities} identities with data, got {len(raws)}"
            )

        unique_raws = len({round(r, 9) for r in raws})
        if unique_raws < 3:
            raise InsufficientCalibrationSpreadError(
                f"dimension {dimension.value}: only {unique_raws} distinct raw values, need >= 3"
            )

        floor = random_floors.get(dimension, 0.0)
        x50 = statistics.median(raws)
        x90 = _linear_quantile(raws, calibration_policy.flagship_quantile)
        x100 = 1.0

        if x50 <= floor:
            raise InsufficientCalibrationSpreadError(
                f"dimension {dimension.value}: median {x50:.6f} <= random floor {floor:.6f}"
            )
        if x90 >= x100:
            raise CalibrationSaturatedError(
                f"dimension {dimension.value}: P90 {x90:.6f} >= ceiling {x100:.6f}"
            )

        anchor = CalibrationAnchor(x0=floor, x50=x50, x90=x90, x100=x100)
        dimension_curves[dimension] = AxisCalibrationCurve(anchor=anchor)

    total_raws = [identity.total_raw for identity in identities]
    unique_total = len({round(r, 9) for r in total_raws})
    if unique_total < 3:
        raise InsufficientCalibrationSpreadError(
            f"total: only {unique_total} distinct raw values, need >= 3"
        )

    total_floor = sum(
        random_floors.get(d, 0.0) * scoring_policy.weight_for(d)
        for d in scoring_policy.enabled_dimensions
    )
    total_x50 = statistics.median(total_raws)
    total_x90 = _linear_quantile(total_raws, calibration_policy.flagship_quantile)
    total_x100 = scoring_policy.coverage_weight_for(*scoring_policy.enabled_dimensions)

    if total_x50 <= total_floor:
        raise InsufficientCalibrationSpreadError(
            f"total: median {total_x50:.6f} <= random floor {total_floor:.6f}"
        )
    if total_x90 >= total_x100:
        raise CalibrationSaturatedError(
            f"total: P90 {total_x90:.6f} >= ceiling {total_x100:.6f}"
        )

    total_anchor = CalibrationAnchor(x0=total_floor, x50=total_x50, x90=total_x90, x100=total_x100)

    return CalibrationCurveBundle(
        dimension_curves=dimension_curves,
        total_curve=AxisCalibrationCurve(anchor=total_anchor),
    )


# ---------------------------------------------------------------------------
# Calibrate a CapabilityProfile
# ---------------------------------------------------------------------------


def calibrate_capability_profile(
    raw_profile: CapabilityProfile,
    curves: CalibrationCurveBundle,
    scoring_policy: CapabilityScoringPolicy,
    calibration_policy: ReferenceCalibrationPolicy,
    reference_set: ReferenceSet,
    reference_identity_count: int,
) -> CapabilityProfile:
    """Apply calibration curves to a raw CapabilityProfile.

    Returns a new CapabilityProfile with calibrated scores filled in.
    Raw scores and provisional_raw_index are preserved unchanged.
    """
    calibrated_dims: list[DimensionScoreResult] = []
    all_warnings: list[str] = list(raw_profile.warnings)

    for dim_result in raw_profile.dimensions:
        if dim_result.status in (DimensionScoreStatus.UNAVAILABLE, DimensionScoreStatus.INSUFFICIENT_DATA):
            calibrated_dims.append(dim_result)
            continue

        curve = curves.dimension_curves.get(dim_result.dimension)
        if curve is None:
            calibrated_dims.append(dim_result)
            continue

        calibrated_value = curve.calibrate(dim_result.raw_normalized_score)
        calibrated_dims.append(
            DimensionScoreResult(
                dimension=dim_result.dimension,
                status=DimensionScoreStatus.SCORED,
                raw_normalized_score=dim_result.raw_normalized_score,
                calibrated_score=calibrated_value,
                task_count=dim_result.task_count,
                graded_task_count=dim_result.graded_task_count,
                task_coverage=dim_result.task_coverage,
                global_weight=dim_result.global_weight,
                weighted_contribution=dim_result.weighted_contribution,
                evidence_refs=dim_result.evidence_refs,
                source_task_ids=dim_result.source_task_ids,
                warnings=dim_result.warnings,
            )
        )

    calibrated_total = curves.total_curve.calibrate(raw_profile.provisional_raw_index)

    provenance = CalibrationProvenance(
        policy_id=calibration_policy.policy_id,
        policy_version=calibration_policy.policy_version,
        method="piecewise_linear_v1",
        reference_set_id=reference_set.reference_set_id,
        reference_set_version=reference_set.reference_set_version,
        reference_set_content_sha256=reference_set.content_sha256,
        reference_identity_count=reference_identity_count,
        coverage_weight=raw_profile.coverage_weight,
    )

    return CapabilityProfile(
        profile_version="0.2.0",
        scoring_policy_id=raw_profile.scoring_policy_id,
        scoring_policy_version=raw_profile.scoring_policy_version,
        dimensions=tuple(calibrated_dims),
        coverage_weight=raw_profile.coverage_weight,
        calibrated_total_score=calibrated_total,
        provisional_raw_index=raw_profile.provisional_raw_index,
        evidence_refs=raw_profile.evidence_refs,
        warnings=tuple(all_warnings),
        metadata=raw_profile.metadata,
        calibration=provenance,
    )


__all__ = [
    "ReferenceCalibrationPolicy",
    "CalibrationError",
    "ReferenceSetIntegrityFailureError",
    "ReferenceSetIncompatibleError",
    "UntrustedReferenceSourceError",
    "InsufficientReferenceIdentitiesError",
    "InsufficientCalibrationSpreadError",
    "CalibrationSaturatedError",
    "CandidateIncompatibleError",
    "_linear_quantile",
    "ReferenceIdentity",
    "aggregate_reference_identities",
    "CalibrationAnchor",
    "AxisCalibrationCurve",
    "CalibrationCurveBundle",
    "build_calibration_curves",
    "calibrate_capability_profile",
]
