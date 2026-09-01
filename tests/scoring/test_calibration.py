"""Tests for Reference Calibration (scoring/calibration.py) — v0.4-B."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmtrace.reference.reference_set import ReferenceSet, ReferenceSetMember
from llmtrace.scoring.calibration import (
    AxisCalibrationCurve,
    CalibrationAnchor,
    CalibrationCurveBundle,
    CalibrationSaturatedError,
    InsufficientCalibrationSpreadError,
    InsufficientReferenceIdentitiesError,
    ReferenceCalibrationPolicy,
    ReferenceIdentity,
    _linear_quantile,
    aggregate_reference_identities,
    build_calibration_curves,
    calibrate_capability_profile,
)
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLICY = CapabilityScoringPolicy.create_v1()
_CAL_POLICY = ReferenceCalibrationPolicy.create_v1()
_DEFAULT_FLOORS: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: 0.25,
    CapabilityDimension.CODING: 0.0,
    CapabilityDimension.MATH_SCIENCE: 0.0,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 0.0,
}


def _make_profile(
    dim_scores: dict[CapabilityDimension, float] | None = None,
    total_raw: float = 0.5,
    *,
    coverage_weight: float = 0.75,
) -> CapabilityProfile:
    """Build a CapabilityProfile for testing calibration."""
    dims: list[DimensionScoreResult] = []
    for dim in CapabilityDimension:
        if dim in _POLICY.enabled_dimensions:
            score = (dim_scores or {}).get(dim, 0.5)
            dims.append(
                DimensionScoreResult(
                    dimension=dim,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=score,
                    task_count=8,
                    graded_task_count=8,
                    task_coverage=1.0,
                    global_weight=_POLICY.weight_for(dim),
                    weighted_contribution=score * _POLICY.weight_for(dim),
                    evidence_refs=(),
                    source_task_ids=(f"task-{dim.value}",),
                    warnings=(),
                )
            )
        else:
            dims.append(
                DimensionScoreResult(
                    dimension=dim,
                    status=DimensionScoreStatus.UNAVAILABLE,
                    raw_normalized_score=0.0,
                )
            )
    return CapabilityProfile(
        scoring_policy_id=_POLICY.policy_id,
        scoring_policy_version=_POLICY.policy_version,
        dimensions=tuple(dims),
        coverage_weight=coverage_weight,
        provisional_raw_index=total_raw,
        evidence_refs=(),
        warnings=(),
    )


def _make_identity(
    provider_id: str,
    model_id: str,
    dim_raws: dict[CapabilityDimension, float],
    total_raw: float = 0.5,
    snapshot_count: int = 1,
) -> ReferenceIdentity:
    return ReferenceIdentity(
        provider_id=provider_id,
        model_id=model_id,
        dimension_raws=dim_raws,
        total_raw=total_raw,
        snapshot_count=snapshot_count,
    )


def _make_member(snapshot_id: str, provider_id: str, model_id: str) -> ReferenceSetMember:
    return ReferenceSetMember(
        snapshot_id=snapshot_id,
        snapshot_sha256="a" * 64,
        model_id=model_id,
        provider_id=provider_id,
        execution_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        capability_profile_sha256="c" * 64,
    )


def _make_ref_set(member_count: int = 5) -> ReferenceSet:
    members = []
    for i in range(member_count):
        members.append(_make_member(f"snap-{i}", f"provider-{i}", f"model-{i}"))
    return ReferenceSet(
        reference_set_id="test-set",
        reference_set_version="1",
        created_at=datetime.now(UTC),
        suite_id="llmtrace-quick-suite-v1",
        suite_version="0.1.0",
        suite_content_sha256="d" * 64,
        adapter_id="llmtrace-quick-v1",
        adapter_version="0.1.0",
        scoring_policy_id=_POLICY.policy_id,
        scoring_policy_version=_POLICY.policy_version,
        generation_config_sha256="e" * 64,
        qualification_policy_id="llmtrace_reference_qualification_v1",
        qualification_policy_version="0.1.0",
        members=tuple(members),
        content_sha256="f" * 64,
    )


# ---------------------------------------------------------------------------
# ReferenceCalibrationPolicy
# ---------------------------------------------------------------------------


class TestReferenceCalibrationPolicy:
    def test_create_v1(self) -> None:
        policy = ReferenceCalibrationPolicy.create_v1()
        assert policy.policy_id == "llmtrace-reference-calibration-v1"
        assert policy.policy_version == "0.1.0"
        assert policy.minimum_distinct_reference_identities == 5
        assert policy.flagship_quantile == 0.90

    def test_repr(self) -> None:
        policy = ReferenceCalibrationPolicy.create_v1()
        assert "min_identities=5" in repr(policy)


# ---------------------------------------------------------------------------
# _linear_quantile
# ---------------------------------------------------------------------------


class TestLinearQuantile:
    def test_single_value(self) -> None:
        assert _linear_quantile([0.5], 0.5) == 0.5

    def test_median_of_odd(self) -> None:
        assert _linear_quantile([0.1, 0.5, 0.9], 0.5) == 0.5

    def test_median_of_even(self) -> None:
        result = _linear_quantile([0.1, 0.2, 0.3, 0.4], 0.5)
        assert abs(result - 0.25) < 1e-9

    def test_p90(self) -> None:
        values = list(range(1, 11))
        result = _linear_quantile([float(v) for v in values], 0.90)
        assert result >= 9.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _linear_quantile([], 0.5)

    def test_q0_returns_min(self) -> None:
        assert _linear_quantile([0.3, 0.7, 0.1], 0.0) == 0.1

    def test_q1_returns_max(self) -> None:
        assert _linear_quantile([0.3, 0.7, 0.1], 1.0) == 0.7


# ---------------------------------------------------------------------------
# aggregate_reference_identities
# ---------------------------------------------------------------------------


class TestAggregateReferenceIdentities:
    def _base_dim_raws(self, score: float) -> dict[CapabilityDimension, float]:
        return {dim: score for dim in CapabilityDimension if dim in _POLICY.enabled_dimensions}

    def test_single_identity(self) -> None:
        members = (_make_member("s1", "p1", "m1"),)
        profile = _make_profile(total_raw=0.6)
        profiles = {"s1": profile}
        identities = aggregate_reference_identities(members, profiles)
        assert len(identities) == 1
        assert identities[0].provider_id == "p1"
        assert identities[0].model_id == "m1"
        assert identities[0].snapshot_count == 1

    def test_same_identity_median(self) -> None:
        members = (
            _make_member("s1", "p1", "m1"),
            _make_member("s2", "p1", "m1"),
        )
        p1 = _make_profile(total_raw=0.4)
        p2 = _make_profile(total_raw=0.6)
        identities = aggregate_reference_identities(members, {"s1": p1, "s2": p2})
        assert len(identities) == 1
        assert identities[0].snapshot_count == 2
        assert identities[0].total_raw == 0.5

    def test_different_identities(self) -> None:
        members = (
            _make_member("s1", "p1", "m1"),
            _make_member("s2", "p2", "m2"),
        )
        p1 = _make_profile(total_raw=0.3)
        p2 = _make_profile(total_raw=0.7)
        identities = aggregate_reference_identities(members, {"s1": p1, "s2": p2})
        assert len(identities) == 2

    def test_missing_profile_skipped(self) -> None:
        members = (_make_member("s1", "p1", "m1"),)
        identities = aggregate_reference_identities(members, {})
        assert len(identities) == 0


# ---------------------------------------------------------------------------
# build_calibration_curves
# ---------------------------------------------------------------------------


class TestBuildCalibrationCurves:
    def _make_identities(
        self,
        scores: list[float],
    ) -> list[ReferenceIdentity]:
        """Build 5+ distinct identities with different total_raw scores."""
        identities = []
        for i, score in enumerate(scores):
            dim_raws = {dim: score for dim in CapabilityDimension if dim in _POLICY.enabled_dimensions}
            identities.append(
                _make_identity(
                    provider_id=f"provider-{i}",
                    model_id=f"model-{i}",
                    dim_raws=dim_raws,
                    total_raw=score,
                )
            )
        return identities

    def test_insufficient_identities(self) -> None:
        identities = self._make_identities([0.1, 0.2, 0.3])
        with pytest.raises(InsufficientReferenceIdentitiesError):
            build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)

    def test_insufficient_spread(self) -> None:
        identities = self._make_identities([0.5] * 5)
        with pytest.raises(InsufficientCalibrationSpreadError):
            build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)

    def test_saturated_p90(self) -> None:
        identities = self._make_identities([0.26, 0.27, 0.99, 0.995, 0.999])
        with pytest.raises(CalibrationSaturatedError):
            build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)

    def test_median_below_floor(self) -> None:
        identities = self._make_identities([0.1, 0.15, 0.2, 0.22, 0.24])
        with pytest.raises(InsufficientCalibrationSpreadError):
            build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)

    def test_valid_curves(self) -> None:
        identities = self._make_identities([0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.72])
        bundle = build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)
        assert isinstance(bundle, CalibrationCurveBundle)
        assert len(bundle.dimension_curves) == len(_POLICY.enabled_dimensions)
        assert bundle.total_curve is not None

    def test_curve_mapping(self) -> None:
        identities = self._make_identities([0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.72])
        bundle = build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)

        reasoning_curve = bundle.dimension_curves[CapabilityDimension.REASONING]
        score_50 = reasoning_curve.calibrate(reasoning_curve.anchor.x50)
        assert abs(score_50 - 50.0) < 1e-9

        score_0 = reasoning_curve.calibrate(reasoning_curve.anchor.x0)
        assert abs(score_0 - 0.0) < 1e-9

        score_100 = reasoning_curve.calibrate(reasoning_curve.anchor.x100)
        assert abs(score_100 - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# AxisCalibrationCurve
# ---------------------------------------------------------------------------


class TestAxisCalibrationCurve:
    def test_below_floor(self) -> None:
        curve = AxisCalibrationCurve(anchor=CalibrationAnchor(x0=0.25, x50=0.5, x90=0.8, x100=1.0))
        assert curve.calibrate(0.0) == 0.0

    def test_above_ceiling(self) -> None:
        curve = AxisCalibrationCurve(anchor=CalibrationAnchor(x0=0.0, x50=0.5, x90=0.8, x100=1.0))
        assert curve.calibrate(1.0) == 100.0

    def test_interior_interpolation(self) -> None:
        curve = AxisCalibrationCurve(anchor=CalibrationAnchor(x0=0.0, x50=0.5, x90=0.8, x100=1.0))
        score = curve.calibrate(0.65)
        assert 50.0 < score < 90.0

    def test_exact_50(self) -> None:
        curve = AxisCalibrationCurve(anchor=CalibrationAnchor(x0=0.0, x50=0.5, x90=0.8, x100=1.0))
        assert abs(curve.calibrate(0.5) - 50.0) < 1e-9

    def test_exact_90(self) -> None:
        curve = AxisCalibrationCurve(anchor=CalibrationAnchor(x0=0.0, x50=0.5, x90=0.8, x100=1.0))
        assert abs(curve.calibrate(0.8) - 90.0) < 1e-9


# ---------------------------------------------------------------------------
# calibrate_capability_profile
# ---------------------------------------------------------------------------


class TestCalibrateCapabilityProfile:
    def _build_curves(self) -> CalibrationCurveBundle:
        scores = [0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.72]
        identities = []
        for i, score in enumerate(scores):
            dim_raws = {dim: score for dim in CapabilityDimension if dim in _POLICY.enabled_dimensions}
            identities.append(
                _make_identity(f"p{i}", f"m{i}", dim_raws, score)
            )
        return build_calibration_curves(identities, _POLICY, _CAL_POLICY, _DEFAULT_FLOORS)

    def test_calibration_fills_scores(self) -> None:
        curves = self._build_curves()
        raw_profile = _make_profile(total_raw=0.6)
        ref_set = _make_ref_set()
        calibrated = calibrate_capability_profile(
            raw_profile, curves, _POLICY, _CAL_POLICY, ref_set, reference_identity_count=7
        )
        assert calibrated.calibrated_total_score is not None
        assert 0.0 <= calibrated.calibrated_total_score <= 100.0

    def test_calibration_provenance(self) -> None:
        curves = self._build_curves()
        raw_profile = _make_profile(total_raw=0.6)
        ref_set = _make_ref_set()
        calibrated = calibrate_capability_profile(
            raw_profile, curves, _POLICY, _CAL_POLICY, ref_set, reference_identity_count=7
        )
        assert calibrated.calibration is not None
        assert calibrated.calibration.policy_id == "llmtrace-reference-calibration-v1"
        assert calibrated.calibration.reference_set_id == "test-set"
        assert calibrated.calibration.reference_identity_count == 7
        assert calibrated.calibration.coverage_weight == 0.75

    def test_raw_scores_preserved(self) -> None:
        curves = self._build_curves()
        raw_profile = _make_profile(total_raw=0.6)
        ref_set = _make_ref_set()
        calibrated = calibrate_capability_profile(
            raw_profile, curves, _POLICY, _CAL_POLICY, ref_set, reference_identity_count=7
        )
        assert calibrated.provisional_raw_index == 0.6

    def test_unavailable_dimensions_unchanged(self) -> None:
        curves = self._build_curves()
        raw_profile = _make_profile(total_raw=0.6)
        ref_set = _make_ref_set()
        calibrated = calibrate_capability_profile(
            raw_profile, curves, _POLICY, _CAL_POLICY, ref_set, reference_identity_count=7
        )
        for dim_result in calibrated.dimensions:
            if dim_result.status == DimensionScoreStatus.UNAVAILABLE:
                assert dim_result.calibrated_score is None

    def test_profile_version_bumped(self) -> None:
        curves = self._build_curves()
        raw_profile = _make_profile(total_raw=0.6)
        ref_set = _make_ref_set()
        calibrated = calibrate_capability_profile(
            raw_profile, curves, _POLICY, _CAL_POLICY, ref_set, reference_identity_count=7
        )
        assert calibrated.profile_version == "0.2.0"
