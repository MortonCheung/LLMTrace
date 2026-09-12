"""Routing v2 signal strengths and verdicts (Task 28 / Task 31 / Task 54).

The scenarios here are exactly the six the spec lists: a stable single
identifier, a stray identifier, an explicit model split, a validated temporal
reference switch, latency variance alone, and too few samples.  Two
properties are pinned hardest:

* only *strong* evidence yields ``Suspicious`` — dispersion alone never does;
* temporal evidence supports nothing unless a **validated** decision policy
  (and, for divergence, its reference baseline) is present (Rule 2).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from llmtrace.analysis.behavior_models import (
    BehaviorItemKey,
    BehaviorItemObservation,
    BehaviorRunSnapshot,
    output_text_sha256,
)
from llmtrace.analysis.routing import RoutingStabilityLevel
from llmtrace.benchmarks.models import ItemStatus
from llmtrace.fingerprint.matcher import FingerprintMatchEntry, FingerprintMatchResult
from llmtrace.fingerprint.models import FingerprintMatchStatus, FingerprintSuite
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy, build_fingerprint_policy
from llmtrace.fingerprint.routing import (
    SIGNAL_LATENCY_DISPERSION,
    SIGNAL_MULTIPLE_RESPONSE_MODELS,
    SIGNAL_PROVIDER_FAILURE,
    SIGNAL_REFERENCE_MATCH_SWITCH,
    SIGNAL_TEMPORAL_DIVERGENCE,
    SIGNAL_TOKEN_DISPERSION,
    RoutingAssessmentV2,
    RoutingEvidenceStrength,
    RoutingSignal,
    RoutingV2Policy,
    assess_routing_v2,
)
from llmtrace.fingerprint.temporal import (
    MINIMUM_TEMPORAL_REPETITIONS,
    TemporalFingerprint,
    TemporalFingerprintStatus,
    build_temporal_fingerprint,
)
from tests.analysis.conftest import make_profile

from .conftest import MODEL_ID, make_reference_snapshot, make_reference_snapshot_with_rounds

FULL_FLIP = (0, 0, 0, 0, 1, 1, 1, 1)
NO_FLIP = (0,) * 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _item(
    index: int,
    *,
    response_model: str | None,
    status: ItemStatus = ItemStatus.GRADED,
    latency_ms: float | None = 100.0,
    output_tokens: int | None = 20,
) -> BehaviorItemObservation:
    text = f"answer-{index}"
    return BehaviorItemObservation(
        key=BehaviorItemKey(
            task_id="gsm8k_quick_v1",
            source_sample_id=f"sample-{index}",
            input_sha256=_sha(f"sample-{index}"),
        ),
        status=status,
        raw_score=1.0,
        normalized_score=1.0,
        output_text_sha256=output_text_sha256(text),
        output_length=len(text),
        response_model=response_model,
        latency_ms=latency_ms,
        output_tokens=output_tokens,
    )


def _snapshot(items: Sequence[BehaviorItemObservation], *, run_id: str = "run-routing") -> BehaviorRunSnapshot:
    return BehaviorRunSnapshot(
        run_id=run_id,
        target_id="target-api",
        candidate_model_id=MODEL_ID,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        adapter_id="llmtrace-quick-v1",
        adapter_version="0.1.0",
        scoring_policy_id="llmtrace-capability-v1",
        scoring_policy_version="0.1.0",
        generation_config_sha256=_sha("generation-config"),
        capability_profile=make_profile(),
        items=tuple(items),
    )


def _uniform_snapshot(
    count: int,
    *,
    response_model: str | None = MODEL_ID,
    latency_ms: float | None = 100.0,
    output_tokens: int | None = 20,
) -> BehaviorRunSnapshot:
    return _snapshot(
        [
            _item(
                index,
                response_model=response_model,
                latency_ms=latency_ms,
                output_tokens=output_tokens,
            )
            for index in range(count)
        ]
    )


def _match(model_id: str, *, provider_id: str = "openai", distance: float = 0.05) -> FingerprintMatchResult:
    return FingerprintMatchResult(
        entries=(
            FingerprintMatchEntry(
                model_id=model_id,
                provider_id=provider_id,
                distance=distance,
                similarity=1.0 - distance,
                comparable_probes=6,
            ),
        ),
        status=FingerprintMatchStatus.RANKED_ONLY,
    )


def _policy(
    *,
    suite: FingerprintSuite,
    validated: bool,
    temporal_divergence_baseline: float | None = None,
) -> FingerprintDecisionPolicy:
    judged: dict[str, float] = {}
    if validated:
        judged = {
            "distance_threshold": 0.2,
            "top1_accuracy": 1.0,
            "top3_accuracy": 1.0,
            "true_positive_rate": 0.95,
            "false_accept_rate": 0.01,
        }
    return build_fingerprint_policy(
        policy_id="test-routing-policy",
        policy_version="0.1.0",
        fingerprint_set_id="test-identity-set",
        fingerprint_set_content_sha256="b" * 64,
        suite_content_sha256=suite.content_sha256,
        validated=validated,
        minimum_comparable_probes=len(suite.probes),
        max_far_target=0.05,
        identity_count=2,
        held_out_capture_count=4,
        temporal_divergence_baseline=temporal_divergence_baseline,
        **judged,
    )


def _observed(assessment: RoutingAssessmentV2, strength: RoutingEvidenceStrength) -> list[str]:
    return [signal.signal_id for signal in assessment.signals if signal.strength is strength and signal.observed]


def _signal(assessment: RoutingAssessmentV2, signal_id: str) -> RoutingSignal:
    return next(signal for signal in assessment.signals if signal.signal_id == signal_id)


def _temporal(
    suite: FingerprintSuite,
    *,
    choice_index_by_round: Sequence[int],
    snapshot_id: str,
) -> TemporalFingerprint:
    snapshot = make_reference_snapshot_with_rounds(
        suite=suite,
        snapshot_id=snapshot_id,
        model_id=MODEL_ID,
        choice_index_by_round=choice_index_by_round,
    )
    return build_temporal_fingerprint(
        snapshot=snapshot,
        suite=suite,
        minimum_comparable_probes=len(suite.probes),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stable_temporal(suite: FingerprintSuite) -> TemporalFingerprint:
    """An ``AVAILABLE`` fingerprint whose two windows agree (divergence 0.0)."""
    temporal = _temporal(suite, choice_index_by_round=NO_FLIP, snapshot_id="candidate-stable")
    assert temporal.status is TemporalFingerprintStatus.AVAILABLE
    return temporal


@pytest.fixture
def drifted_temporal(suite: FingerprintSuite) -> TemporalFingerprint:
    """An ``AVAILABLE`` fingerprint whose windows disagree completely."""
    temporal = _temporal(suite, choice_index_by_round=FULL_FLIP, snapshot_id="candidate-drifted")
    assert temporal.status is TemporalFingerprintStatus.AVAILABLE
    return temporal


@pytest.fixture
def unavailable_temporal(suite: FingerprintSuite) -> TemporalFingerprint:
    """A capture with too few rounds to form two windows."""
    snapshot = make_reference_snapshot(
        suite=suite,
        snapshot_id="candidate-short",
        model_id=MODEL_ID,
        repetitions=MINIMUM_TEMPORAL_REPETITIONS - 2,
    )
    temporal = build_temporal_fingerprint(
        snapshot=snapshot,
        suite=suite,
        minimum_comparable_probes=len(suite.probes),
    )
    assert temporal.status is TemporalFingerprintStatus.UNAVAILABLE
    return temporal


@pytest.fixture
def validated_policy(suite: FingerprintSuite) -> FingerprintDecisionPolicy:
    """A validated policy that carries no reference temporal baseline."""
    return _policy(suite=suite, validated=True)


@pytest.fixture
def validated_policy_with_baseline(suite: FingerprintSuite) -> FingerprintDecisionPolicy:
    """A validated policy whose reference drift baseline is 0.5."""
    return _policy(suite=suite, validated=True, temporal_divergence_baseline=0.5)


@pytest.fixture
def unvalidated_policy(suite: FingerprintSuite) -> FingerprintDecisionPolicy:
    return _policy(suite=suite, validated=False)


# ---------------------------------------------------------------------------
# Task 54 scenarios
# ---------------------------------------------------------------------------


class TestRoutingV2Scenarios:
    def test_single_identifier_with_stable_temporal_is_stable(
        self,
        stable_temporal: TemporalFingerprint,
        validated_policy_with_baseline: FingerprintDecisionPolicy,
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=stable_temporal,
            fingerprint_policy=validated_policy_with_baseline,
        )

        assert assessment.level is RoutingStabilityLevel.STABLE
        assert assessment.distinct_response_models == 1
        assert assessment.response_model_coverage == pytest.approx(1.0)
        assert assessment.temporal_status is TemporalFingerprintStatus.AVAILABLE
        assert _observed(assessment, RoutingEvidenceStrength.STRONG) == []
        assert assessment.limitations == ()

    def test_stray_identifier_keeps_mostly_stable(
        self, validated_policy_with_baseline: FingerprintDecisionPolicy
    ) -> None:
        items = [_item(index, response_model=MODEL_ID) for index in range(9)]
        items.append(_item(9, response_model="stray-model"))

        assessment = assess_routing_v2(
            snapshot=_snapshot(items),
            temporal=None,
            fingerprint_policy=validated_policy_with_baseline,
        )

        assert assessment.level is RoutingStabilityLevel.MOSTLY_STABLE
        assert assessment.distinct_response_models == 2
        assert assessment.dominant_model_ratio == pytest.approx(0.9)
        # 90% share is at the policy ceiling, so this is a stray — not a split.
        assert _signal(assessment, SIGNAL_MULTIPLE_RESPONSE_MODELS).observed is False
        assert _observed(assessment, RoutingEvidenceStrength.STRONG) == []

    def test_model_split_is_suspicious(self, validated_policy_with_baseline: FingerprintDecisionPolicy) -> None:
        items = [_item(index, response_model=MODEL_ID) for index in range(4)]
        items.extend(_item(index, response_model="other-model") for index in range(4, 8))

        assessment = assess_routing_v2(
            snapshot=_snapshot(items),
            temporal=None,
            fingerprint_policy=validated_policy_with_baseline,
        )

        assert assessment.level is RoutingStabilityLevel.SUSPICIOUS
        assert _signal(assessment, SIGNAL_MULTIPLE_RESPONSE_MODELS).observed is True
        assert any("mixed routing is possible" in reason for reason in assessment.reasons)
        assert _observed(assessment, RoutingEvidenceStrength.STRONG) == [SIGNAL_MULTIPLE_RESPONSE_MODELS]

    def test_validated_reference_switch_is_suspicious(
        self,
        stable_temporal: TemporalFingerprint,
        validated_policy: FingerprintDecisionPolicy,
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=stable_temporal,
            window_matches=(_match(MODEL_ID), _match("other-model")),
            fingerprint_policy=validated_policy,
        )

        assert assessment.level is RoutingStabilityLevel.SUSPICIOUS
        assert _signal(assessment, SIGNAL_REFERENCE_MATCH_SWITCH).observed is True
        assert _observed(assessment, RoutingEvidenceStrength.STRONG) == [SIGNAL_REFERENCE_MATCH_SWITCH]

    def test_latency_dispersion_alone_never_claims_mixed_routing(
        self,
        stable_temporal: TemporalFingerprint,
        validated_policy_with_baseline: FingerprintDecisionPolicy,
    ) -> None:
        # Only the latency varies; identifiers, tokens and failures are flat.
        items = [_item(index, response_model=MODEL_ID, latency_ms=10.0 if index % 2 else 1000.0) for index in range(10)]

        assessment = assess_routing_v2(
            snapshot=_snapshot(items),
            temporal=stable_temporal,
            fingerprint_policy=validated_policy_with_baseline,
        )

        latency = _signal(assessment, SIGNAL_LATENCY_DISPERSION)
        assert latency.observed is True
        assert latency.strength is RoutingEvidenceStrength.SUPPORTING
        # The supporting signal is reported, but it must not upgrade the label.
        assert _observed(assessment, RoutingEvidenceStrength.STRONG) == []
        assert assessment.level is RoutingStabilityLevel.MOSTLY_STABLE

    def test_too_few_samples_is_insufficient_data(
        self, validated_policy_with_baseline: FingerprintDecisionPolicy
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(4),
            temporal=None,
            fingerprint_policy=validated_policy_with_baseline,
        )

        assert assessment.level is RoutingStabilityLevel.INSUFFICIENT_DATA
        assert assessment.item_count == 4
        assert any("at least 8" in reason for reason in assessment.reasons)

    def test_no_reported_identifier_is_insufficient_data(
        self, validated_policy_with_baseline: FingerprintDecisionPolicy
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10, response_model=None),
            temporal=None,
            fingerprint_policy=validated_policy_with_baseline,
        )

        assert assessment.level is RoutingStabilityLevel.INSUFFICIENT_DATA
        assert assessment.response_model_coverage == pytest.approx(0.0)
        assert assessment.dominant_model_ratio is None
        assert any("no model identifier" in reason for reason in assessment.reasons)

    def test_unavailable_temporal_cannot_confirm_stability(
        self,
        unavailable_temporal: TemporalFingerprint,
        validated_policy_with_baseline: FingerprintDecisionPolicy,
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=unavailable_temporal,
            fingerprint_policy=validated_policy_with_baseline,
        )

        assert assessment.temporal_status is TemporalFingerprintStatus.UNAVAILABLE
        assert assessment.level is RoutingStabilityLevel.MOSTLY_STABLE
        assert any("unavailable" in limitation for limitation in assessment.limitations)


# ---------------------------------------------------------------------------
# Dispersion availability
# ---------------------------------------------------------------------------


class TestDispersionAvailability:
    def test_missing_series_marks_dispersion_unavailable(
        self, validated_policy_with_baseline: FingerprintDecisionPolicy
    ) -> None:
        # No item carries a latency or token count: neither series reaches two
        # samples, so no coefficient exists and the signal must stay unobserved.
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10, latency_ms=None, output_tokens=None),
            temporal=None,
            fingerprint_policy=validated_policy_with_baseline,
        )

        for signal_id in (SIGNAL_LATENCY_DISPERSION, SIGNAL_TOKEN_DISPERSION):
            signal = _signal(assessment, signal_id)
            assert signal.observed is False
            assert signal.strength is RoutingEvidenceStrength.SUPPORTING
            assert "fewer than 2 samples or a zero mean" in signal.reason
        assert _observed(assessment, RoutingEvidenceStrength.SUPPORTING) == []

    def test_zero_mean_series_marks_dispersion_unavailable(
        self, validated_policy_with_baseline: FingerprintDecisionPolicy
    ) -> None:
        # Samples exist, but a zero mean leaves the coefficient undefined.
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10, latency_ms=0.0, output_tokens=0),
            temporal=None,
            fingerprint_policy=validated_policy_with_baseline,
        )

        for signal_id in (SIGNAL_LATENCY_DISPERSION, SIGNAL_TOKEN_DISPERSION):
            signal = _signal(assessment, signal_id)
            assert signal.observed is False
            assert "dispersion unavailable" in signal.reason


# ---------------------------------------------------------------------------
# Rule 2 gates and caller errors
# ---------------------------------------------------------------------------


class TestRoutingV2PolicyGates:
    def test_validated_baseline_breach_is_strong_evidence(
        self,
        drifted_temporal: TemporalFingerprint,
        validated_policy_with_baseline: FingerprintDecisionPolicy,
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=drifted_temporal,
            fingerprint_policy=validated_policy_with_baseline,
        )

        signal = _signal(assessment, SIGNAL_TEMPORAL_DIVERGENCE)
        assert signal.observed is True
        assert signal.strength is RoutingEvidenceStrength.STRONG
        assert assessment.level is RoutingStabilityLevel.SUSPICIOUS

    def test_validated_policy_without_a_baseline_records_a_limitation(
        self,
        drifted_temporal: TemporalFingerprint,
        validated_policy: FingerprintDecisionPolicy,
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=drifted_temporal,
            fingerprint_policy=validated_policy,
        )

        assert _signal(assessment, SIGNAL_TEMPORAL_DIVERGENCE).observed is False
        assert any("no temporal baseline" in limitation for limitation in assessment.limitations)

    def test_missing_policy_suppresses_every_temporal_signal(self, drifted_temporal: TemporalFingerprint) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=drifted_temporal,
            fingerprint_policy=None,
        )

        assert _signal(assessment, SIGNAL_TEMPORAL_DIVERGENCE).observed is False
        assert any("no validated decision policy" in limitation for limitation in assessment.limitations)
        assert _observed(assessment, RoutingEvidenceStrength.STRONG) == []

    def test_unvalidated_policy_suppresses_the_reference_switch(
        self,
        stable_temporal: TemporalFingerprint,
        unvalidated_policy: FingerprintDecisionPolicy,
    ) -> None:
        assessment = assess_routing_v2(
            snapshot=_uniform_snapshot(10),
            temporal=stable_temporal,
            window_matches=(_match(MODEL_ID), _match("other-model")),
            fingerprint_policy=unvalidated_policy,
        )

        signal = _signal(assessment, SIGNAL_REFERENCE_MATCH_SWITCH)
        assert signal.observed is False
        assert "no validated decision policy" in signal.reason
        assert assessment.level is not RoutingStabilityLevel.SUSPICIOUS

    def test_window_match_count_must_match_the_temporal_windows(
        self,
        stable_temporal: TemporalFingerprint,
        validated_policy: FingerprintDecisionPolicy,
    ) -> None:
        with pytest.raises(ValueError, match="must be empty or match"):
            assess_routing_v2(
                snapshot=_uniform_snapshot(10),
                temporal=stable_temporal,
                window_matches=(_match(MODEL_ID),),
                fingerprint_policy=validated_policy,
            )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestRoutingV2Models:
    def test_default_policy_is_versioned(self) -> None:
        policy = RoutingV2Policy.create_v2()

        assert policy.policy_id == "llmtrace-routing-v2"
        assert policy.policy_version == "2.0.0"
        assert policy.minimum_samples == 8

    def test_assessment_rejects_duplicate_signal_ids(self) -> None:
        signal = RoutingSignal(
            signal_id=SIGNAL_PROVIDER_FAILURE,
            strength=RoutingEvidenceStrength.SUPPORTING,
            observed=False,
            reason="0/8 provider failures",
        )

        with pytest.raises(ValueError, match="must be unique"):
            RoutingAssessmentV2(
                level=RoutingStabilityLevel.STABLE,
                policy_id="llmtrace-routing-v2",
                policy_version="2.0.0",
                item_count=8,
                reported_model_count=8,
                response_model_coverage=1.0,
                distinct_response_models=1,
                dominant_model_ratio=1.0,
                failure_ratio=0.0,
                temporal_status=TemporalFingerprintStatus.AVAILABLE,
                signals=(signal, signal),
            )
