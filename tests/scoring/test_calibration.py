"""Reference Calibration tests (v0.4-B §22): math properties, fail-closed, claimed gap.

Covers:
- monotonicity / determinism / bounds / exact anchor mapping / interpolation /
  lower & upper extrapolation (§22 Calibration 数学性质)
- fail-closed: too few identities, zero spread, insufficient per-dimension
  references, saturation (§22 Fail Closed)
- total calibration consumes ``provisional_raw_index`` and never renormalizes
  the 0.75 Quick Suite coverage (§22 Total Calibration)
- claimed model gap: matching / non-matching / missing / ambiguous, per-dimension
  and total gaps (§22 Claimed Gap)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from llmtrace.scoring.calibration import (
    AmbiguousClaimedModelError,
    AxisCalibrationCurve,
    CalibrationAnchor,
    CalibrationCurveBundle,
    InsufficientCalibrationSpreadError,
    InsufficientReferenceIdentitiesError,
    ReferenceCalibrationPolicy,
    ReferenceIdentity,
    aggregate_reference_identities,
    build_calibration_curves,
    calibrate_capability_profile,
    compute_claimed_model_gap,
)
from llmtrace.scoring.models import (
    CalibrationProvenance,
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy

SCORING_POLICY = CapabilityScoringPolicy.create_v1()
CALIBRATION_POLICY = ReferenceCalibrationPolicy.create_v1()

_DIMENSIONS = (
    CapabilityDimension.REASONING,
    CapabilityDimension.CODING,
    CapabilityDimension.MATH_SCIENCE,
    CapabilityDimension.INSTRUCTION_FOLLOWING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _floors() -> dict[CapabilityDimension, float]:
    """Random floors: reasoning 0.25, others 0.0 (Quick Suite v1)."""
    return {
        CapabilityDimension.REASONING: 0.25,
        CapabilityDimension.CODING: 0.0,
        CapabilityDimension.MATH_SCIENCE: 0.0,
        CapabilityDimension.INSTRUCTION_FOLLOWING: 0.0,
    }


def _identity(
    provider_id: str,
    model_id: str,
    dim_raws: dict[CapabilityDimension, float],
    total_raw: float,
    snapshot_count: int = 1,
) -> ReferenceIdentity:
    return ReferenceIdentity(
        provider_id=provider_id,
        model_id=model_id,
        dimension_raws=dict(dim_raws),
        total_raw=total_raw,
        snapshot_count=snapshot_count,
    )


def _six_level_identities() -> list[ReferenceIdentity]:
    """A reference universe spanning low → flagship across all four dimensions."""
    levels = [
        ("openai", "gpt-x-low", 0.30, 0.25, 0.35, 0.40),
        ("openai", "gpt-x-midlow", 0.45, 0.40, 0.50, 0.52),
        ("openai", "gpt-x-mid", 0.60, 0.55, 0.65, 0.62),
        ("openai", "gpt-x-midhigh", 0.73, 0.68, 0.78, 0.74),
        ("openai", "gpt-x-high", 0.90, 0.85, 0.92, 0.88),
        ("openai", "gpt-x-flagship", 0.94, 0.90, 0.96, 0.93),
    ]
    identities = []
    for provider, model, r, c, m, i in levels:
        dim_raws = {
            CapabilityDimension.REASONING: r,
            CapabilityDimension.CODING: c,
            CapabilityDimension.MATH_SCIENCE: m,
            CapabilityDimension.INSTRUCTION_FOLLOWING: i,
        }
        total = sum(dim_raws[d] * SCORING_POLICY.weight_for(d) for d in _DIMENSIONS)
        identities.append(_identity(provider, model, dim_raws, total))
    return identities


def _curves(identities: list[ReferenceIdentity]) -> CalibrationCurveBundle:
    return build_calibration_curves(identities, SCORING_POLICY, CALIBRATION_POLICY, _floors())


def _raw_profile(
    dim_raws: dict[CapabilityDimension, float],
    *,
    calibrated: bool = False,
    total_calibrated: float | None = None,
) -> CapabilityProfile:
    """Build a raw (optionally calibrated) candidate profile."""
    dims = []
    for d in _DIMENSIONS:
        dims.append(
            DimensionScoreResult(
                dimension=d,
                status=DimensionScoreStatus.SCORED,
                raw_normalized_score=dim_raws[d],
                task_count=8,
                graded_task_count=8,
                task_coverage=1.0,
                global_weight=SCORING_POLICY.weight_for(d),
            )
        )
    provisional = sum(dim_raws[d] * SCORING_POLICY.weight_for(d) for d in _DIMENSIONS)
    return CapabilityProfile(
        scoring_policy_id=SCORING_POLICY.policy_id,
        scoring_policy_version=SCORING_POLICY.policy_version,
        dimensions=tuple(dims),
        coverage_weight=SCORING_POLICY.coverage_weight_for(*_DIMENSIONS),
        provisional_raw_index=provisional,
        calibrated_total_score=total_calibrated,
        calibration=_provenance() if calibrated else None,
    )


def _provenance() -> CalibrationProvenance:
    return CalibrationProvenance(
        policy_id=CALIBRATION_POLICY.policy_id,
        policy_version=CALIBRATION_POLICY.policy_version,
        method="piecewise_linear_v1",
        reference_set_id="test-reference-set",
        reference_set_version="0.1.0",
        reference_set_content_sha256="a" * 64,
        reference_identity_count=6,
        coverage_weight=0.75,
    )


def _fake_reference_set() -> Any:
    """Minimal duck-typed ReferenceSet for calibrate_capability_profile."""
    from llmtrace.reference.reference_set import ReferenceSet, ReferenceSetMember

    member = ReferenceSetMember(
        snapshot_id="snap-1",
        snapshot_sha256="b" * 64,
        model_id="gpt-x-mid",
        provider_id="openai",
        capability_profile_sha256="c" * 64,
    )
    set_ = ReferenceSet(
        reference_set_id="test-reference-set",
        reference_set_version="0.1.0",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        suite_content_sha256="d" * 64,
        adapter_id="llmtrace-quick-v1",
        adapter_version="0.1.0",
        scoring_policy_id=SCORING_POLICY.policy_id,
        scoring_policy_version=SCORING_POLICY.policy_version,
        generation_config_sha256="e" * 64,
        qualification_policy_id="llmtrace-reference-qualification-v1",
        qualification_policy_version="0.1.0",
        members=(member,),
        content_sha256="f" * 64,
    )
    return set_


def _calibrate(dim_raws: dict[CapabilityDimension, float]) -> CapabilityProfile:
    identities = _six_level_identities()
    return calibrate_capability_profile(
        _raw_profile(dim_raws),
        _curves(identities),
        SCORING_POLICY,
        CALIBRATION_POLICY,
        _fake_reference_set(),
        len(identities),
    )


# ---------------------------------------------------------------------------
# AxisCalibrationCurve math properties
# ---------------------------------------------------------------------------


class TestAxisCalibrationCurve:
    ANCHOR = CalibrationAnchor(x0=0.25, x50=0.60, x90=0.94, x100=1.0)

    def test_exact_anchor_mapping(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        assert curve.calibrate(0.25) == pytest.approx(0.0)
        assert curve.calibrate(0.60) == pytest.approx(50.0)
        assert curve.calibrate(0.94) == pytest.approx(90.0)
        assert curve.calibrate(1.0) == pytest.approx(100.0)

    def test_bounds(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        for raw in [-10.0, -0.1, 0.0, 0.3, 0.7, 0.95, 1.0, 1.5, 100.0]:
            score = curve.calibrate(raw)
            assert 0.0 <= score <= 100.0

    def test_monotonicity(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        prev = -1.0
        raw = 0.0
        while raw <= 1.0001:
            score = curve.calibrate(raw)
            assert score >= prev
            prev = score
            raw += 0.013

    def test_lower_extrapolation_clamps_to_zero(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        assert curve.calibrate(-5.0) == 0.0
        assert curve.calibrate(self.ANCHOR.x0) == 0.0

    def test_upper_extrapolation_clamps_to_hundred(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        assert curve.calibrate(2.0) == 100.0
        assert curve.calibrate(self.ANCHOR.x100) == 100.0

    def test_interpolation_between_anchors(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        # Midpoint between x0 and x50 maps to 25
        mid = (self.ANCHOR.x0 + self.ANCHOR.x50) / 2
        assert curve.calibrate(mid) == pytest.approx(25.0)
        # Midpoint between x50 and x90 maps to 70
        mid = (self.ANCHOR.x50 + self.ANCHOR.x90) / 2
        assert curve.calibrate(mid) == pytest.approx(70.0)

    def test_determinism(self) -> None:
        curve = AxisCalibrationCurve(anchor=self.ANCHOR)
        assert curve.calibrate(0.77) == curve.calibrate(0.77)
        assert curve.calibrate(0.77) == curve.calibrate(0.77)


# ---------------------------------------------------------------------------
# build_calibration_curves
# ---------------------------------------------------------------------------


class TestBuildCalibrationCurves:
    def test_success_builds_all_dimension_and_total_curves(self) -> None:
        bundle = _curves(_six_level_identities())
        for d in _DIMENSIONS:
            assert d in bundle.dimension_curves
        assert bundle.total_curve is not None

    def test_too_few_identities_fail_closed(self) -> None:
        identities = _six_level_identities()[:4]
        with pytest.raises(InsufficientReferenceIdentitiesError):
            build_calibration_curves(identities, SCORING_POLICY, CALIBRATION_POLICY, _floors())

    def test_dimension_insufficient_references_fail_closed(self) -> None:
        identities = _six_level_identities()
        # Drop reasoning data from two identities → only 4 remain with data
        for identity in identities[:2]:
            identity.dimension_raws.pop(CapabilityDimension.REASONING)
        with pytest.raises(InsufficientReferenceIdentitiesError):
            build_calibration_curves(identities, SCORING_POLICY, CALIBRATION_POLICY, _floors())

    def test_zero_spread_fail_closed(self) -> None:
        identities = [
            _identity("p", f"m{i}", dict.fromkeys(_DIMENSIONS, 0.5), 0.375) for i in range(5)
        ]
        with pytest.raises(InsufficientCalibrationSpreadError):
            build_calibration_curves(identities, SCORING_POLICY, CALIBRATION_POLICY, _floors())

    def test_median_at_floor_fail_closed(self) -> None:
        # Reasoning floor is 0.25 — all raw scores at the floor leave no room
        identities = [
            _identity(
                "p",
                f"m{i}",
                {
                    CapabilityDimension.REASONING: 0.25,
                    CapabilityDimension.CODING: 0.1 + i * 0.1,
                    CapabilityDimension.MATH_SCIENCE: 0.1 + i * 0.1,
                    CapabilityDimension.INSTRUCTION_FOLLOWING: 0.1 + i * 0.1,
                },
                0.1,
            )
            for i in range(5)
        ]
        with pytest.raises(InsufficientCalibrationSpreadError):
            build_calibration_curves(identities, SCORING_POLICY, CALIBRATION_POLICY, _floors())

    def test_determinism_same_inputs_same_curves(self) -> None:
        identities = _six_level_identities()
        b1 = _curves(list(identities))
        b2 = _curves(list(identities))
        for d in _DIMENSIONS:
            assert b1.dimension_curves[d].anchor == b2.dimension_curves[d].anchor
        assert b1.total_curve.anchor == b2.total_curve.anchor


# ---------------------------------------------------------------------------
# calibrate_capability_profile
# ---------------------------------------------------------------------------


class TestCalibrateProfile:
    def test_all_dimensions_calibrated_in_bounds(self) -> None:
        profile = _calibrate(dict.fromkeys(_DIMENSIONS, 0.75))
        for dim in profile.dimensions:
            assert dim.calibrated_score is not None
            assert 0.0 <= dim.calibrated_score <= 100.0

    def test_total_calibrated_in_bounds(self) -> None:
        profile = _calibrate(dict.fromkeys(_DIMENSIONS, 0.75))
        assert profile.calibrated_total_score is not None
        assert 0.0 <= profile.calibrated_total_score <= 100.0

    def test_raw_scores_preserved(self) -> None:
        raws = {
            CapabilityDimension.REASONING: 0.75,
            CapabilityDimension.CODING: 0.70,
            CapabilityDimension.MATH_SCIENCE: 0.80,
            CapabilityDimension.INSTRUCTION_FOLLOWING: 0.74,
        }
        raw_profile = _raw_profile(raws)
        profile = calibrate_capability_profile(
            raw_profile,
            _curves(_six_level_identities()),
            SCORING_POLICY,
            CALIBRATION_POLICY,
            _fake_reference_set(),
            6,
        )
        assert profile.provisional_raw_index == raw_profile.provisional_raw_index
        for dim in profile.dimensions:
            assert dim.raw_normalized_score == pytest.approx(raws[dim.dimension])

    def test_provisional_raw_index_is_total_curve_input(self) -> None:
        """Total calibration consumes provisional_raw_index (§9)."""
        identities = _six_level_identities()
        bundle = _curves(identities)
        raws = {
            CapabilityDimension.REASONING: 0.80,
            CapabilityDimension.CODING: 0.75,
            CapabilityDimension.MATH_SCIENCE: 0.85,
            CapabilityDimension.INSTRUCTION_FOLLOWING: 0.78,
        }
        profile = calibrate_capability_profile(
            _raw_profile(raws),
            bundle,
            SCORING_POLICY,
            CALIBRATION_POLICY,
            _fake_reference_set(),
            6,
        )
        assert profile.calibrated_total_score == pytest.approx(
            bundle.total_curve.calibrate(profile.provisional_raw_index)
        )

    def test_no_renormalization_of_075_coverage(self) -> None:
        """The total curve ceiling is coverage_weight (0.75), not 1.0 (§9).

        The flagship reference (provisional_raw_index ≈ full raw weighted sum
        under the 0.75 coverage) maps to ~90, and a raw index of 0.75 maps to
        exactly 100.  A renormalized curve would push 0.75 well below 100.
        """
        bundle = _curves(_six_level_identities())
        # Full-marks candidate: raw index == coverage weight == 0.75
        assert bundle.total_curve.calibrate(0.75) == pytest.approx(100.0)
        assert bundle.total_curve.anchor.x100 == pytest.approx(0.75)

    def test_acceptance_example_low_candidate_high_ordering(self) -> None:
        """§23 acceptance: Low < Candidate < High on every dimension and total."""
        low = {d: 0.30 if d is CapabilityDimension.REASONING else 0.25 for d in _DIMENSIONS}
        low[CapabilityDimension.MATH_SCIENCE] = 0.35
        low[CapabilityDimension.INSTRUCTION_FOLLOWING] = 0.40
        candidate = {
            CapabilityDimension.REASONING: 0.75,
            CapabilityDimension.CODING: 0.70,
            CapabilityDimension.MATH_SCIENCE: 0.80,
            CapabilityDimension.INSTRUCTION_FOLLOWING: 0.74,
        }
        high = dict.fromkeys(_DIMENSIONS, 0.9)
        high[CapabilityDimension.CODING] = 0.85
        high[CapabilityDimension.MATH_SCIENCE] = 0.92
        high[CapabilityDimension.INSTRUCTION_FOLLOWING] = 0.88

        p_low = _calibrate(low)
        p_cand = _calibrate(candidate)
        p_high = _calibrate(high)

        assert p_low.calibrated_total_score is not None
        assert p_cand.calibrated_total_score is not None
        assert p_high.calibrated_total_score is not None
        assert p_low.calibrated_total_score < p_cand.calibrated_total_score < p_high.calibrated_total_score

        for d in _DIMENSIONS:
            low_dim = next(x for x in p_low.dimensions if x.dimension is d)
            cand_dim = next(x for x in p_cand.dimensions if x.dimension is d)
            high_dim = next(x for x in p_high.dimensions if x.dimension is d)
            assert low_dim.calibrated_score is not None
            assert cand_dim.calibrated_score is not None
            assert high_dim.calibrated_score is not None
            assert low_dim.calibrated_score <= cand_dim.calibrated_score <= high_dim.calibrated_score

    def test_provenance_recorded(self) -> None:
        profile = _calibrate(dict.fromkeys(_DIMENSIONS, 0.75))
        assert profile.calibration is not None
        assert profile.calibration.policy_id == CALIBRATION_POLICY.policy_id
        assert profile.calibration.policy_version == CALIBRATION_POLICY.policy_version
        assert profile.calibration.method == "piecewise_linear_v1"
        assert profile.calibration.reference_set_id == "test-reference-set"
        assert profile.calibration.reference_identity_count == 6

    def test_determinism(self) -> None:
        p1 = _calibrate(dict.fromkeys(_DIMENSIONS, 0.77))
        p2 = _calibrate(dict.fromkeys(_DIMENSIONS, 0.77))
        assert p1.calibrated_total_score == p2.calibrated_total_score
        for a, b in zip(p1.dimensions, p2.dimensions, strict=True):
            assert a.calibrated_score == b.calibrated_score


# ---------------------------------------------------------------------------
# aggregate_reference_identities
# ---------------------------------------------------------------------------


class TestAggregateIdentities:
    def test_same_model_multiple_snapshots_aggregated_by_median(self) -> None:
        from llmtrace.reference.reference_set import ReferenceSetMember

        members = tuple(
            ReferenceSetMember(
                snapshot_id=f"snap-{i}",
                snapshot_sha256=f"{i}" * 64,
                model_id="gpt-x-mid",
                provider_id="openai",
                capability_profile_sha256="a" * 64,
            )
            for i in range(3)
        )
        raws = [0.55, 0.60, 0.65]
        profiles: dict[str, CapabilityProfile] = {}
        for i, raw in enumerate(raws):
            dim_raws = dict.fromkeys(_DIMENSIONS, raw)
            profiles[f"snap-{i}"] = _raw_profile(dim_raws)

        identities = aggregate_reference_identities(members, profiles)
        assert len(identities) == 1
        assert identities[0].model_id == "gpt-x-mid"
        assert identities[0].snapshot_count == 3
        # median of 0.55/0.60/0.65 per dimension
        assert identities[0].dimension_raws[CapabilityDimension.REASONING] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Claimed model gap
# ---------------------------------------------------------------------------


class TestClaimedModelGap:
    def _context(self) -> tuple[list[ReferenceIdentity], CalibrationCurveBundle, CapabilityProfile]:
        identities = _six_level_identities()
        bundle = _curves(identities)
        candidate = _calibrate(
            {
                CapabilityDimension.REASONING: 0.75,
                CapabilityDimension.CODING: 0.70,
                CapabilityDimension.MATH_SCIENCE: 0.80,
                CapabilityDimension.INSTRUCTION_FOLLOWING: 0.74,
            }
        )
        return identities, bundle, candidate

    def test_matching_claimed_model(self) -> None:
        identities, bundle, candidate = self._context()
        gap = compute_claimed_model_gap(
            claimed_model="gpt-x-mid",
            identities=identities,
            curves=bundle,
            candidate_profile=candidate,
        )
        assert gap is not None
        assert gap.claimed_model_id == "gpt-x-mid"
        assert gap.reference_model_id == "gpt-x-mid"
        assert 0.0 <= gap.reference_total_score <= 100.0
        assert gap.candidate_total_score == candidate.calibrated_total_score
        assert gap.total_delta == pytest.approx(gap.candidate_total_score - gap.reference_total_score)

    def test_non_matching_claimed_model_returns_none(self) -> None:
        identities, bundle, candidate = self._context()
        gap = compute_claimed_model_gap(
            claimed_model="not-in-reference-set",
            identities=identities,
            curves=bundle,
            candidate_profile=candidate,
        )
        assert gap is None

    def test_matching_is_exact_no_fuzzy(self) -> None:
        identities, bundle, candidate = self._context()
        for claimed in ["GPT-X-MID", "gpt-x-mid ", "gpt-x-mi", "gpt-x"]:
            assert (
                compute_claimed_model_gap(
                    claimed_model=claimed,
                    identities=identities,
                    curves=bundle,
                    candidate_profile=candidate,
                )
                is None
            )

    def test_ambiguous_claimed_model_fails_closed(self) -> None:
        identities, bundle, candidate = self._context()
        # Two providers both offering gpt-x-mid → ambiguous, refuse to pick
        identities.append(_identity("another-provider", "gpt-x-mid", dict.fromkeys(_DIMENSIONS, 0.5), 0.375))
        with pytest.raises(AmbiguousClaimedModelError):
            compute_claimed_model_gap(
                claimed_model="gpt-x-mid",
                identities=identities,
                curves=bundle,
                candidate_profile=candidate,
            )

    def test_dimension_gaps_present_and_consistent(self) -> None:
        identities, bundle, candidate = self._context()
        gap = compute_claimed_model_gap(
            claimed_model="gpt-x-mid",
            identities=identities,
            curves=bundle,
            candidate_profile=candidate,
        )
        assert gap is not None
        assert len(gap.dimension_gaps) == len(_DIMENSIONS)
        for dg in gap.dimension_gaps:
            assert dg.delta == pytest.approx(dg.candidate_score - dg.reference_score)
            assert 0.0 <= dg.candidate_score <= 100.0
            assert 0.0 <= dg.reference_score <= 100.0

    def test_reference_score_uses_same_curves(self) -> None:
        """The claimed model's reference score is computed with the same curve.

        A candidate exactly matching the reference's raw scores must show a
        zero gap — identity of inputs through the same mapping.
        """
        identities = _six_level_identities()
        bundle = _curves(identities)
        mid = next(i for i in identities if i.model_id == "gpt-x-mid")
        candidate = _calibrate(mid.dimension_raws)
        gap = compute_claimed_model_gap(
            claimed_model="gpt-x-mid",
            identities=identities,
            curves=bundle,
            candidate_profile=candidate,
        )
        assert gap is not None
        assert gap.total_delta == pytest.approx(0.0, abs=1e-9)
        for dg in gap.dimension_gaps:
            assert dg.delta == pytest.approx(0.0, abs=1e-9)
