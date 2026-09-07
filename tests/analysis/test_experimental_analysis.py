"""Confidence v1 / Behavior Similarity v1 / Routing Stability v1 (§34–§42).

Experimental, deterministic, rule-based labels over already-verified facts.
These tests pin the fail-closed behavior: a missing measurement, a missing
reported model identifier, or an identity with no comparable dimension must
never fabricate a verdict.
"""

from __future__ import annotations

import pytest

from llmtrace.analysis.confidence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    ConfidencePolicy,
)
from llmtrace.analysis.routing import (
    RoutingAssessment,
    RoutingStabilityLevel,
    assess_routing_stability,
)
from llmtrace.analysis.similarity import (
    BehaviorFeatureVector,
    assess_behavior_similarity,
)
from llmtrace.benchmarks.models import ItemStatus
from llmtrace.execution.models import BenchmarkMeasurementSummary
from llmtrace.scoring.models import (
    CalibrationProvenance,
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreStatus,
)

from .conftest import make_profile, make_snapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_measurement(
    *,
    total: int = 32,
    graded: int = 32,
    failures: int = 0,
    ungradable: int = 0,
) -> BenchmarkMeasurementSummary:
    """Build a consistent BenchmarkMeasurementSummary."""
    assert graded + failures + ungradable == total
    return BenchmarkMeasurementSummary(
        total_item_count=total,
        graded_item_count=graded,
        failure_item_count=failures,
        ungradable_item_count=ungradable,
        execution_coverage=(total - failures) / total,
        grading_coverage=graded / total,
    )


def make_calibrated_profile(*, identities: int = 5) -> CapabilityProfile:
    """CapabilityProfile carrying formal reference calibration."""
    profile = make_profile(
        statuses=dict.fromkeys(
            (
                CapabilityDimension.REASONING,
                CapabilityDimension.CODING,
                CapabilityDimension.MATH_SCIENCE,
                CapabilityDimension.INSTRUCTION_FOLLOWING,
            ),
            DimensionScoreStatus.SCORED,
        )
    )
    return profile.model_copy(
        update={
            "calibration": CalibrationProvenance(
                policy_id="llmtrace-reference-calibration-v1",
                policy_version="1.0.0",
                method="piecewise-anchored",
                reference_set_id="ref-set-quick-v1",
                reference_set_version="0.1.0",
                reference_set_content_sha256="a" * 64,
                reference_identity_count=identities,
                coverage_weight=0.75,
            )
        }
    )


def _graded_spec(i: int, *, model: str = "gpt-x") -> dict:
    return {
        "task_id": "gsm8k_quick_v1",
        "source_sample_id": f"sample-{i}",
        "status": ItemStatus.GRADED,
        "score": 1.0,
        "response_model": model,
    }


def _failed_spec(i: int, *, model: str = "gpt-x") -> dict:
    spec = _graded_spec(i, model=model)
    spec["status"] = ItemStatus.FAILURE
    spec["score"] = 0.0
    return spec


# ---------------------------------------------------------------------------
# Confidence v1
# ---------------------------------------------------------------------------


class TestConfidencePolicy:
    def test_no_measurement_is_unavailable(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(measurement=None, capability_profile=None)
        assert isinstance(assessment, ConfidenceAssessment)
        assert assessment.level == ConfidenceLevel.UNAVAILABLE
        assert assessment.measurement_available is False
        assert assessment.reasons

    def test_zero_graded_items_is_unavailable(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=make_measurement(total=8, graded=0, failures=8),
            capability_profile=None,
        )
        assert assessment.level == ConfidenceLevel.UNAVAILABLE

    @pytest.mark.parametrize(
        "measurement",
        [
            make_measurement(total=10, graded=4, failures=6),  # coverage 0.4
            make_measurement(total=8, graded=6, failures=2),  # failure ratio 0.25
        ],
    )
    def test_degraded_measurement_is_low(self, measurement: BenchmarkMeasurementSummary) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=measurement, capability_profile=None
        )
        assert assessment.level == ConfidenceLevel.LOW

    def test_unanchored_mostly_complete_is_medium(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=make_measurement(total=8, graded=8),
            capability_profile=None,
        )
        assert assessment.level == ConfidenceLevel.MEDIUM
        assert assessment.calibrated is False

    def test_unanchored_incomplete_is_low(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=make_measurement(total=8, graded=5, ungradable=3),
            capability_profile=None,
        )
        assert assessment.level == ConfidenceLevel.LOW

    def test_full_healthy_calibrated_is_high(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=make_measurement(total=32, graded=32),
            capability_profile=make_calibrated_profile(identities=5),
        )
        assert assessment.level == ConfidenceLevel.HIGH
        assert assessment.calibrated is True
        assert assessment.reference_identity_count == 5

    def test_calibrated_but_few_identities_is_medium(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=make_measurement(total=32, graded=32),
            capability_profile=make_calibrated_profile(identities=3),
        )
        assert assessment.level == ConfidenceLevel.MEDIUM

    def test_calibrated_but_provider_failure_is_medium(self) -> None:
        assessment = ConfidencePolicy.create_v1().assess(
            measurement=make_measurement(total=32, graded=31, failures=1),
            capability_profile=make_calibrated_profile(identities=5),
        )
        assert assessment.level == ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# Behavior Similarity v1
# ---------------------------------------------------------------------------


class TestBehaviorSimilarity:
    def test_identical_behavior_is_1_0(self) -> None:
        candidate = BehaviorFeatureVector.from_snapshot(make_snapshot(run_id="run-a"), provider_id="openai")
        reference = BehaviorFeatureVector.from_snapshot(make_snapshot(run_id="run-b"), provider_id="openai")
        result = assess_behavior_similarity(
            candidate, [("gpt-x", "openai", reference)]
        )
        assert result.unavailable is False
        assert result.entries[0].similarity == pytest.approx(1.0)
        assert result.entries[0].comparable_dimensions == 4

    def test_full_score_gap_lowers_similarity(self) -> None:
        candidate = BehaviorFeatureVector.from_snapshot(
            make_snapshot(run_id="run-candidate"), provider_id="openai"
        )
        reference_profile = make_profile(scores=dict.fromkeys(
            (
                CapabilityDimension.REASONING,
                CapabilityDimension.CODING,
                CapabilityDimension.MATH_SCIENCE,
                CapabilityDimension.INSTRUCTION_FOLLOWING,
            ),
            0.0,
        ))
        reference = BehaviorFeatureVector.from_snapshot(
            make_snapshot(run_id="run-reference", profile=reference_profile),
            provider_id="openai",
        )
        result = assess_behavior_similarity(candidate, [("weak-model", "openai", reference)])
        assert result.unavailable is False
        # 0.6 weight on dimension scores is fully different; latency/tokens match.
        assert result.entries[0].similarity == pytest.approx(0.4)

    def test_no_shared_dimension_fails_closed(self) -> None:
        candidate = BehaviorFeatureVector(
            model_id="gpt-x",
            provider_id="openai",
            item_count=8,
            graded_ratio=1.0,
            dimension_scores={CapabilityDimension.REASONING.value: 0.8},
        )
        reference = BehaviorFeatureVector(
            model_id="some-other-model",
            provider_id="openai",
            item_count=8,
            graded_ratio=1.0,
            dimension_scores={CapabilityDimension.CODING.value: 0.8},
        )
        result = assess_behavior_similarity(candidate, [("some-other-model", "openai", reference)])
        assert result.unavailable is True
        assert result.unavailable_reason is not None

    def test_name_match_alone_never_fabricates_similarity(self) -> None:
        """Same model id with zero comparable dimensions is still unavailable."""
        candidate = BehaviorFeatureVector(
            model_id="gpt-x",
            provider_id="openai",
            item_count=8,
            graded_ratio=1.0,
            dimension_scores={CapabilityDimension.REASONING.value: 0.8},
        )
        reference = BehaviorFeatureVector(
            model_id="gpt-x",
            provider_id="openai",
            item_count=8,
            graded_ratio=1.0,
            dimension_scores={CapabilityDimension.CODING.value: 0.8},
        )
        result = assess_behavior_similarity(candidate, [("gpt-x", "openai", reference)])
        assert result.unavailable is True

    def test_entries_are_ranked_by_similarity(self) -> None:
        candidate = BehaviorFeatureVector.from_snapshot(
            make_snapshot(run_id="run-c"), provider_id="openai"
        )
        far_profile = make_profile(scores=dict.fromkeys(
            (
                CapabilityDimension.REASONING,
                CapabilityDimension.CODING,
                CapabilityDimension.MATH_SCIENCE,
                CapabilityDimension.INSTRUCTION_FOLLOWING,
            ),
            0.0,
        ))
        near = BehaviorFeatureVector.from_snapshot(
            make_snapshot(run_id="run-near"), provider_id="openai"
        )
        far = BehaviorFeatureVector.from_snapshot(
            make_snapshot(run_id="run-far", profile=far_profile), provider_id="openai"
        )
        result = assess_behavior_similarity(
            candidate,
            [("far-model", "openai", far), ("near-model", "openai", near)],
        )
        assert [entry.model_id for entry in result.entries] == ["near-model", "far-model"]


# ---------------------------------------------------------------------------
# Routing Stability v1
# ---------------------------------------------------------------------------


class TestRoutingStability:
    def test_single_model_zero_failures_is_stable(self) -> None:
        snapshot = make_snapshot(
            run_id="run-stable",
            items=[_graded_spec(i, model="gpt-x") for i in range(10)],
        )
        assessment = assess_routing_stability(snapshot)
        assert isinstance(assessment, RoutingAssessment)
        assert assessment.level == RoutingStabilityLevel.STABLE

    def test_single_model_few_failures_is_mostly_stable(self) -> None:
        snapshot = make_snapshot(
            run_id="run-mostly",
            items=[_graded_spec(i) for i in range(9)] + [_failed_spec(9)],
        )
        assessment = assess_routing_stability(snapshot)
        assert assessment.level == RoutingStabilityLevel.MOSTLY_STABLE

    def test_single_model_many_failures_is_suspicious(self) -> None:
        snapshot = make_snapshot(
            run_id="run-failover",
            items=[_graded_spec(i) for i in range(6)] + [_failed_spec(i) for i in range(6, 8)],
        )
        assessment = assess_routing_stability(snapshot)
        assert assessment.level == RoutingStabilityLevel.SUSPICIOUS
        assert "failover" in " ".join(assessment.reasons)

    def test_dominant_model_with_stray_identity_is_mostly_stable(self) -> None:
        snapshot = make_snapshot(
            run_id="run-dominant",
            items=[_graded_spec(i, model="gpt-x") for i in range(9)]
            + [_graded_spec(9, model="gpt-y")],
        )
        assessment = assess_routing_stability(snapshot)
        assert assessment.level == RoutingStabilityLevel.MOSTLY_STABLE

    def test_no_dominant_model_is_suspicious(self) -> None:
        snapshot = make_snapshot(
            run_id="run-mixed",
            items=[_graded_spec(i, model="gpt-x") for i in range(5)]
            + [_graded_spec(i, model="gpt-y") for i in range(5, 8)],
        )
        assessment = assess_routing_stability(snapshot)
        assert assessment.level == RoutingStabilityLevel.SUSPICIOUS
        assert any("mixed routing is possible" in reason for reason in assessment.reasons)

    def test_too_few_samples_is_insufficient(self) -> None:
        snapshot = make_snapshot(
            run_id="run-few",
            items=[_graded_spec(i) for i in range(5)],
        )
        assessment = assess_routing_stability(snapshot)
        assert assessment.level == RoutingStabilityLevel.INSUFFICIENT_DATA

    def test_no_reported_model_is_insufficient(self) -> None:
        items = [_graded_spec(i) for i in range(10)]
        for spec in items:
            spec["response_model"] = None
        snapshot = make_snapshot(run_id="run-anon", items=items)
        assessment = assess_routing_stability(snapshot)
        assert assessment.level == RoutingStabilityLevel.INSUFFICIENT_DATA
