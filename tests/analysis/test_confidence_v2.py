"""Four-component confidence bundle (Task 32).

v2 must not collapse the four components into one number, and must not invent
thresholds: every level comes from the versioned v1 ``ConfidencePolicy`` or
routing v2's ``minimum_samples``.  These tests pin the fail-closed paths — a
missing measurement, an unvalidated policy, an absent claimed reference — and
confirm no fabricated probability is exposed.
"""

from __future__ import annotations

import pytest

from llmtrace.analysis.confidence import ConfidenceLevel
from llmtrace.analysis.confidence_v2 import (
    ConfidenceBundle,
    ConfidenceComponent,
    build_confidence_bundle,
    calibration_confidence,
    fingerprint_confidence,
    measurement_confidence,
    routing_confidence,
)
from llmtrace.analysis.routing import RoutingStabilityLevel
from llmtrace.execution.models import BenchmarkMeasurementSummary
from llmtrace.fingerprint.matcher import (
    FingerprintMatchEntry,
    FingerprintMatchResult,
    FingerprintVerificationResult,
)
from llmtrace.fingerprint.models import INVALID_OUTCOME, FingerprintMatchStatus, ProbeDistribution
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy, build_fingerprint_policy
from llmtrace.fingerprint.routing import RoutingAssessmentV2
from llmtrace.fingerprint.temporal import TemporalFingerprintStatus
from llmtrace.scoring.models import CalibrationProvenance, CapabilityProfile

from .conftest import make_profile

_SET_SHA = "d" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measurement(
    *,
    total: int = 32,
    graded: int = 32,
    failures: int = 0,
    ungradable: int = 0,
) -> BenchmarkMeasurementSummary:
    assert graded + failures + ungradable == total
    return BenchmarkMeasurementSummary(
        total_item_count=total,
        graded_item_count=graded,
        failure_item_count=failures,
        ungradable_item_count=ungradable,
        execution_coverage=(total - failures) / total,
        grading_coverage=graded / total,
    )


def _calibrated_profile(
    *,
    identity_count: int = 6,
    coverage_weight: float = 0.75,
    set_id: str = "reference-set-a",
    set_sha: str = _SET_SHA,
) -> CapabilityProfile:
    provenance = CalibrationProvenance(
        policy_id="llmtrace-calibration-v1",
        policy_version="0.1.0",
        method="reference-normalization",
        reference_set_id=set_id,
        reference_set_version="0.1.0",
        reference_set_content_sha256=set_sha,
        reference_identity_count=identity_count,
        coverage_weight=coverage_weight,
    )
    return make_profile(coverage_weight=coverage_weight).model_copy(update={"calibration": provenance})


def _distribution(*, sample_count: int, invalid_count: int) -> ProbeDistribution:
    valid = sample_count - invalid_count
    return ProbeDistribution(
        probe_id="probe-a",
        support=("a", "b", INVALID_OUTCOME),
        counts={"a": valid, "b": 0, INVALID_OUTCOME: invalid_count},
        probabilities={"a": valid / sample_count, "b": 0.0, INVALID_OUTCOME: invalid_count / sample_count},
        sample_count=sample_count,
        invalid_count=invalid_count,
    )


def _policy(*, validated: bool) -> FingerprintDecisionPolicy:
    judged = (
        {
            "distance_threshold": 0.2,
            "top1_accuracy": 1.0,
            "top3_accuracy": 1.0,
            "true_positive_rate": 0.95,
            "false_accept_rate": 0.01,
        }
        if validated
        else {}
    )
    return build_fingerprint_policy(
        policy_id="test-fingerprint-policy",
        policy_version="0.1.0",
        fingerprint_set_id="test-identity-set",
        fingerprint_set_content_sha256="b" * 64,
        suite_content_sha256="c" * 64,
        validated=validated,
        minimum_comparable_probes=6,
        max_far_target=0.05,
        identity_count=2,
        held_out_capture_count=4,
        **judged,
    )


def _verification(
    policy: FingerprintDecisionPolicy | None,
    *,
    status: FingerprintMatchStatus = FingerprintMatchStatus.RANKED_ONLY,
    claim_verdict_produced: bool = False,
) -> FingerprintVerificationResult:
    return FingerprintVerificationResult(
        match_status=status,
        claim_verdict_produced=claim_verdict_produced,
        policy=policy,
        minimum_comparable_probes=6,
        reference_identity_count=2,
    )


def _match(
    *,
    claimed_model_id: str | None = "my-real-model",
    claimed_reference_distance: float | None = 0.1,
) -> FingerprintMatchResult:
    return FingerprintMatchResult(
        entries=(
            FingerprintMatchEntry(
                model_id="my-real-model",
                provider_id="openai",
                distance=0.1,
                similarity=0.9,
                comparable_probes=6,
            ),
        ),
        status=FingerprintMatchStatus.RANKED_ONLY,
        claimed_model_id=claimed_model_id,
        claimed_reference_distance=claimed_reference_distance,
    )


def _assessment(
    *,
    level: RoutingStabilityLevel = RoutingStabilityLevel.STABLE,
    item_count: int = 10,
    coverage: float = 1.0,
    temporal_status: TemporalFingerprintStatus = TemporalFingerprintStatus.AVAILABLE,
) -> RoutingAssessmentV2:
    return RoutingAssessmentV2(
        level=level,
        policy_id="llmtrace-routing-v2",
        policy_version="2.0.0",
        item_count=item_count,
        reported_model_count=int(item_count * coverage),
        response_model_coverage=coverage,
        distinct_response_models=1,
        dominant_model_ratio=1.0,
        failure_ratio=0.0,
        temporal_status=temporal_status,
        signals=(),
    )


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


class TestMeasurementConfidence:
    def test_missing_measurement_is_unavailable(self) -> None:
        component = measurement_confidence(None)
        assert component.level is ConfidenceLevel.UNAVAILABLE
        assert component.limitations

    def test_zero_graded_items_is_unavailable(self) -> None:
        component = measurement_confidence(_measurement(graded=0, ungradable=32))
        assert component.level is ConfidenceLevel.UNAVAILABLE

    def test_fully_graded_clean_measurement_is_high(self) -> None:
        component = measurement_confidence(_measurement())
        assert component.level is ConfidenceLevel.HIGH
        assert component.limitations == ()

    def test_provider_failures_at_the_versioned_threshold_are_low(self) -> None:
        # 8/32 == 25%, exactly the versioned degraded-failure threshold.
        component = measurement_confidence(_measurement(graded=24, failures=8))
        assert component.level is ConfidenceLevel.LOW

    def test_mostly_lost_measurement_is_low(self) -> None:
        component = measurement_confidence(_measurement(graded=13, failures=19))
        assert component.level is ConfidenceLevel.LOW

    def test_ungradable_remainder_is_medium(self) -> None:
        component = measurement_confidence(_measurement(graded=30, ungradable=2))
        assert component.level is ConfidenceLevel.MEDIUM

    def test_mostly_incomplete_measurement_is_low(self) -> None:
        component = measurement_confidence(_measurement(graded=10, ungradable=22))
        assert component.level is ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


class TestCalibrationConfidence:
    def test_missing_profile_is_unavailable(self) -> None:
        assert calibration_confidence(None).level is ConfidenceLevel.UNAVAILABLE

    def test_uncalibrated_profile_is_unavailable(self) -> None:
        component = calibration_confidence(make_profile())
        assert component.level is ConfidenceLevel.UNAVAILABLE
        assert any("unanchored" in limitation for limitation in component.limitations)

    def test_unchecked_reference_compatibility_is_a_limitation(self) -> None:
        component = calibration_confidence(_calibrated_profile())
        assert component.level is ConfidenceLevel.HIGH
        assert any("was not checked" in limitation for limitation in component.limitations)

    def test_matching_expected_reference_set_is_high_without_a_limitation(self) -> None:
        component = calibration_confidence(
            _calibrated_profile(),
            expected_reference_set_id="reference-set-a",
            expected_reference_set_content_sha256=_SET_SHA,
        )
        assert component.level is ConfidenceLevel.HIGH
        assert component.limitations == ()

    def test_mismatched_reference_set_is_low(self) -> None:
        component = calibration_confidence(
            _calibrated_profile(),
            expected_reference_set_content_sha256="e" * 64,
        )
        assert component.level is ConfidenceLevel.LOW
        assert any("does not match" in reason for reason in component.reasons)

    def test_few_reference_identities_is_medium(self) -> None:
        component = calibration_confidence(_calibrated_profile(identity_count=3))
        assert component.level is ConfidenceLevel.MEDIUM

    def test_weak_calibration_coverage_is_medium(self) -> None:
        component = calibration_confidence(_calibrated_profile(coverage_weight=0.3))
        assert component.level is ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


class TestFingerprintConfidence:
    def test_no_evidence_is_unavailable(self) -> None:
        component = fingerprint_confidence(verification=None, match=None)
        assert component.level is ConfidenceLevel.UNAVAILABLE
        assert component.limitations

    def test_unvalidated_policy_only_ranks(self) -> None:
        component = fingerprint_confidence(verification=_verification(_policy(validated=False)), match=_match())
        assert component.level is ConfidenceLevel.LOW
        assert any("UNVALIDATED" in limitation for limitation in component.limitations)

    def test_validated_policy_with_full_coverage_is_high(self) -> None:
        component = fingerprint_confidence(
            verification=_verification(_policy(validated=True)),
            match=_match(),
            candidate=(_distribution(sample_count=8, invalid_count=0),),
        )
        assert component.level is ConfidenceLevel.HIGH

    def test_absent_claimed_reference_is_low(self) -> None:
        component = fingerprint_confidence(
            verification=_verification(_policy(validated=True)),
            match=_match(claimed_model_id="ghost-model", claimed_reference_distance=None),
            candidate=(_distribution(sample_count=8, invalid_count=0),),
        )
        assert component.level is ConfidenceLevel.LOW
        assert any("no capture in the fingerprint set" in reason for reason in component.reasons)

    @pytest.mark.parametrize(
        ("invalid_count", "expected"),
        [(5, ConfidenceLevel.LOW), (3, ConfidenceLevel.MEDIUM), (1, ConfidenceLevel.HIGH)],
    )
    def test_invalid_samples_lower_the_component(self, invalid_count: int, expected: ConfidenceLevel) -> None:
        component = fingerprint_confidence(
            verification=_verification(_policy(validated=True)),
            match=_match(),
            candidate=(_distribution(sample_count=8, invalid_count=invalid_count),),
        )
        assert component.level is expected

    def test_match_without_a_policy_records_the_missing_policy(self) -> None:
        component = fingerprint_confidence(verification=None, match=_match())
        assert component.level is ConfidenceLevel.LOW
        assert any("no decision policy was available" in limitation for limitation in component.limitations)

    def test_claim_verdict_is_surfaced_in_the_reasons(self) -> None:
        verification = _verification(
            _policy(validated=True),
            status=FingerprintMatchStatus.CONSISTENT_WITH_CLAIM,
            claim_verdict_produced=True,
        )

        component = fingerprint_confidence(
            verification=verification,
            match=_match(),
            candidate=(_distribution(sample_count=8, invalid_count=0),),
        )

        assert component.level is ConfidenceLevel.HIGH
        assert any("claim consistency" in reason for reason in component.reasons)

    def test_no_claimed_model_is_recorded_as_a_limitation(self) -> None:
        # No claim was supplied, so no claim comparison happened; the limitation
        # must say so rather than staying silent.
        component = fingerprint_confidence(
            verification=_verification(_policy(validated=False)),
            match=_match(claimed_model_id=None, claimed_reference_distance=None),
            candidate=(_distribution(sample_count=8, invalid_count=0),),
        )

        assert component.level is ConfidenceLevel.LOW
        assert any("no claimed model was supplied" in limitation for limitation in component.limitations)

    def test_unvalidated_policy_without_any_observation_still_names_the_missing_policy(self) -> None:
        # Nothing else contributes a reason, so the policy gap must be reported
        # instead of leaving ``reasons`` empty.
        component = fingerprint_confidence(verification=_verification(_policy(validated=False)))

        assert component.level is ConfidenceLevel.LOW
        assert component.reasons == ("no validated decision policy is available",)


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


class TestRoutingConfidence:
    def test_no_assessment_is_unavailable(self) -> None:
        assert routing_confidence(None).level is ConfidenceLevel.UNAVAILABLE

    def test_below_the_routing_sample_minimum_is_low(self) -> None:
        component = routing_confidence(_assessment(item_count=4))
        assert component.level is ConfidenceLevel.LOW

    def test_low_response_model_coverage_is_low(self) -> None:
        component = routing_confidence(_assessment(coverage=0.5))
        assert component.level is ConfidenceLevel.LOW

    def test_temporal_without_a_validated_policy_is_medium(self) -> None:
        component = routing_confidence(_assessment(), validated_policy_available=False)
        assert component.level is ConfidenceLevel.MEDIUM
        assert any("not backed by a validated policy" in limitation for limitation in component.limitations)

    def test_suspicious_routing_is_medium(self) -> None:
        component = routing_confidence(
            _assessment(level=RoutingStabilityLevel.SUSPICIOUS),
            validated_policy_available=True,
        )
        assert component.level is ConfidenceLevel.MEDIUM
        assert any("mixed routing is possible" in reason for reason in component.reasons)

    def test_insufficient_routing_evidence_is_low(self) -> None:
        component = routing_confidence(
            _assessment(level=RoutingStabilityLevel.INSUFFICIENT_DATA),
            validated_policy_available=True,
        )
        assert component.level is ConfidenceLevel.LOW

    def test_stable_routing_with_validated_temporal_is_high(self) -> None:
        component = routing_confidence(_assessment(), validated_policy_available=True)
        assert component.level is ConfidenceLevel.HIGH


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


class TestConfidenceBundle:
    def test_absent_inputs_are_all_unavailable(self) -> None:
        bundle = build_confidence_bundle()

        assert isinstance(bundle, ConfidenceBundle)
        for component in (bundle.measurement, bundle.calibration, bundle.fingerprint, bundle.routing):
            assert component.level is ConfidenceLevel.UNAVAILABLE

    def test_full_inputs_produce_four_independent_components(self) -> None:
        bundle = build_confidence_bundle(
            measurement=_measurement(),
            capability_profile=_calibrated_profile(),
            verification=_verification(_policy(validated=True)),
            match=_match(),
            candidate=(_distribution(sample_count=8, invalid_count=0),),
            routing=_assessment(),
        )

        assert bundle.measurement.level is ConfidenceLevel.HIGH
        assert bundle.calibration.level is ConfidenceLevel.HIGH
        assert bundle.fingerprint.level is ConfidenceLevel.HIGH
        # The bundle derives policy availability from the verification record.
        assert bundle.routing.level is ConfidenceLevel.HIGH

    def test_components_expose_no_probability_field(self) -> None:
        assert set(ConfidenceComponent.model_fields) == {"level", "reasons", "limitations"}
