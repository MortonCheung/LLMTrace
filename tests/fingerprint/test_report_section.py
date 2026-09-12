"""Task 44 —— fingerprint 报告段的序列化覆盖测试.

这一层只做"已经产生的证据 → JSON / HTML 共用的一个 dict"的拼装，因此测试关心的是：

- Task 44 的字段清单一个不少地出现在段里；
- 逐 probe / 聚合 JSD 明确标注它描述的是哪一次比较（claimed reference 还是
  top-ranked），数字不会错配到别的参考（Rule 3）；
- 缺 policy / 缺 routing / 缺 confidence 时如实写 ``None``，不伪造标签（Rule 2）；
- 没有任何 fingerprint 产物时整段消失，非 fingerprint run 的报告结构不变（Rule 1）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtrace.analysis.confidence_v2 import build_confidence_bundle
from llmtrace.analysis.routing import RoutingStabilityLevel
from llmtrace.fingerprint.aggregation import aggregate_fingerprint_reference
from llmtrace.fingerprint.matcher import (
    FingerprintMatcher,
    FingerprintMatchResult,
    FingerprintVerificationResult,
)
from llmtrace.fingerprint.models import (
    FingerprintMatchStatus,
    FingerprintProfile,
    FingerprintSourceRole,
    FingerprintSuite,
    repetitions_for_profile,
)
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot, build_fingerprint_snapshot
from llmtrace.fingerprint.routing import RoutingAssessmentV2
from llmtrace.fingerprint.temporal import TemporalFingerprintStatus
from llmtrace.reporting.fingerprint_report import (
    BASIS_CLAIMED_REFERENCE,
    BASIS_TOP_RANKED,
    build_fingerprint_section,
)

from .conftest import (
    MODEL_ID,
    TARGET_ID,
    FingerprintFixture,
    make_observations,
    make_reference_snapshot,
    publish_reference_set,
    publish_validated_policy,
)

STANDARD_REPETITIONS = repetitions_for_profile(FingerprintProfile.STANDARD)

#: Task 44 要求必须可读的字段清单（顶层键）。
TASK_44_FIELDS = {
    "suite",
    "reference_set",
    "decision_policy",
    "repetitions",
    "probe_coverage",
    "invalid_outcome_count",
    "per_probe_jsd",
    "aggregate_jsd",
    "top_k",
    "claimed_model_id",
    "claimed_reference_distance",
    "verdict",
    "routing",
    "confidence",
}


@pytest.fixture
def references(
    tmp_path: Path, suite: FingerprintSuite
) -> tuple[FingerprintFixture, tuple[FingerprintReferenceSnapshot, ...]]:
    """A two-identity reference set, with the member captures kept for aggregation."""
    snapshots = (
        make_reference_snapshot(
            suite=suite,
            snapshot_id="ref-claimed-capture-1",
            model_id=MODEL_ID,
            repetitions=STANDARD_REPETITIONS,
        ),
        make_reference_snapshot(
            suite=suite,
            snapshot_id="ref-impostor-capture-1",
            model_id="other-model",
            repetitions=STANDARD_REPETITIONS,
            choice_index=1,
        ),
    )
    fixture = publish_reference_set(data_root=tmp_path / "appdata", suite=suite, snapshots=snapshots)
    return fixture, snapshots


def _candidate(
    suite: FingerprintSuite,
    *,
    choice_index: int = 0,
    valid: bool = True,
) -> FingerprintReferenceSnapshot:
    """This run's capture, labelled as the audited endpoint (never a reference)."""
    return build_fingerprint_snapshot(
        snapshot_id="candidate-test-run",
        model_id=MODEL_ID,
        provider_id=f"target:{TARGET_ID}",
        source_role=FingerprintSourceRole.CANDIDATE_CAPTURE,
        suite=suite,
        repetitions=STANDARD_REPETITIONS,
        observations=make_observations(
            suite,
            STANDARD_REPETITIONS,
            choice_index=choice_index,
            valid=valid,
            ref_prefix="candidate",
        ),
    )


def _match(
    suite: FingerprintSuite,
    snapshots: tuple[FingerprintReferenceSnapshot, ...],
    candidate: FingerprintReferenceSnapshot,
    *,
    claimed_model_id: str | None = MODEL_ID,
    policy: FingerprintDecisionPolicy | None = None,
) -> FingerprintMatchResult:
    references = tuple(aggregate_fingerprint_reference(snapshots=[snapshot], suite=suite) for snapshot in snapshots)
    return FingerprintMatcher(suite=suite).match(
        candidate=candidate.distributions,
        references=references,
        minimum_comparable_probes=len(suite.probes),
        claimed_model_id=claimed_model_id,
        policy=policy,
    )


def _verification(
    match: FingerprintMatchResult, *, policy: FingerprintDecisionPolicy | None = None
) -> FingerprintVerificationResult:
    return FingerprintVerificationResult(
        match_status=match.status,
        claim_verdict_produced=match.status
        in (FingerprintMatchStatus.CONSISTENT_WITH_CLAIM, FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM),
        policy=policy,
        minimum_comparable_probes=match.entries[0].comparable_probes if match.entries else 1,
        reference_identity_count=len(match.entries),
    )


class TestFingerprintSectionFields:
    """Task 44 的字段清单必须完整出现在段里，且逐项可复核."""

    def test_every_task_44_field_is_present(
        self,
        suite: FingerprintSuite,
        references: tuple[FingerprintFixture, tuple[FingerprintReferenceSnapshot, ...]],
    ) -> None:
        fixture, snapshots = references
        policy = publish_validated_policy(fixture)
        candidate = _candidate(suite)
        match = _match(suite, snapshots, candidate, policy=policy)

        section = build_fingerprint_section(
            match=match,
            verification=_verification(match, policy=policy),
            snapshot=candidate,
            suite=suite,
            reference_set=fixture.reference_set,
        )

        assert section is not None
        assert set(section) >= TASK_44_FIELDS
        assert section["available"] is True
        assert section["experimental"] is True
        assert "not cryptographic proof" in str(section["disclaimer"])

        assert section["suite"] == {
            "suite_id": suite.suite_id,
            "suite_version": suite.suite_version,
            "content_sha256": suite.content_sha256,
            "normalization_policy_id": suite.normalization_policy_id,
            "normalization_policy_version": suite.normalization_policy_version,
        }
        assert section["reference_set"] == {
            "fingerprint_set_id": fixture.reference_set.fingerprint_set_id,
            "fingerprint_set_version": fixture.reference_set.fingerprint_set_version,
            "content_sha256": fixture.reference_set.content_sha256,
            "member_count": 2,
        }
        assert section["decision_policy"] == {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "validated": True,
            "minimum_comparable_probes": len(suite.probes),
            "distance_threshold": policy.distance_threshold,
        }
        assert section["repetitions"] == STANDARD_REPETITIONS

        assert section["probe_coverage"] == {
            "total_probes": len(suite.probes),
            "compared_probes": len(suite.probes),
            "ratio": 1.0,
        }
        assert section["invalid_outcome_count"] == 0
        assert section["candidate_sample_count"] == len(suite.probes) * STANDARD_REPETITIONS

        # The capture answers exactly like the claimed reference, so every probe
        # is comparable with zero divergence.
        assert section["aggregate_jsd"] == pytest.approx(0.0)
        assert section["jsd_reference"] == {
            "basis": BASIS_CLAIMED_REFERENCE,
            "model_id": MODEL_ID,
            "provider_id": "openai",
            "comparable_probes": len(suite.probes),
        }
        per_probe = section["per_probe_jsd"]
        assert isinstance(per_probe, list)
        assert [entry["probe_id"] for entry in per_probe] == [probe.probe_id for probe in suite.probes]
        assert all(entry["comparable"] is True for entry in per_probe)
        assert all(entry["distance"] == pytest.approx(0.0) for entry in per_probe)
        assert all(entry["candidate_invalid_count"] == 0 for entry in per_probe)

        top_k = section["top_k"]
        assert isinstance(top_k, list)
        assert [entry["rank"] for entry in top_k] == list(range(1, len(top_k) + 1))
        assert top_k[0]["rank"] == 1
        assert top_k[0]["model_id"] == MODEL_ID
        assert top_k[0]["similarity"] == pytest.approx(1.0)

        assert section["claimed_model_id"] == MODEL_ID
        assert section["claimed_reference_distance"] == pytest.approx(0.0)
        verdict = section["verdict"]
        assert isinstance(verdict, dict)
        assert verdict["match_status"] == FingerprintMatchStatus.CONSISTENT_WITH_CLAIM.value
        assert verdict["claim_verdict_produced"] is True
        assert verdict["reference_identity_count"] == 2

        # Not supplied → reported as absent instead of invented (Rule 2).
        assert section["routing"] is None
        assert section["confidence"] is None

    def test_invalid_outcomes_are_counted_without_shrinking_the_denominator(
        self,
        suite: FingerprintSuite,
    ) -> None:
        candidate = _candidate(suite, valid=False)
        section = build_fingerprint_section(match=None, verification=None, snapshot=candidate, suite=suite)

        assert section is not None
        assert section["invalid_outcome_count"] == len(suite.probes) * STANDARD_REPETITIONS
        assert section["candidate_sample_count"] == len(suite.probes) * STANDARD_REPETITIONS

    def test_top_ranked_basis_when_no_claim_was_compared(
        self,
        suite: FingerprintSuite,
        references: tuple[FingerprintFixture, tuple[FingerprintReferenceSnapshot, ...]],
    ) -> None:
        _, snapshots = references
        candidate = _candidate(suite)
        match = _match(suite, snapshots, candidate, claimed_model_id=None)

        section = build_fingerprint_section(
            match=match,
            verification=_verification(match),
            snapshot=candidate,
            suite=suite,
        )

        assert section is not None
        assert match.status == FingerprintMatchStatus.RANKED_ONLY
        assert section["claimed_model_id"] is None
        assert section["claimed_reference_distance"] is None
        # The numbers describe the top-ranked comparison, and say so.
        jsd_reference = section["jsd_reference"]
        assert isinstance(jsd_reference, dict)
        assert jsd_reference["basis"] == BASIS_TOP_RANKED
        assert jsd_reference["model_id"] == match.entries[0].model_id

    def test_routing_and_confidence_are_serialized_when_supplied(
        self,
        suite: FingerprintSuite,
    ) -> None:
        candidate = _candidate(suite)
        routing = RoutingAssessmentV2(
            level=RoutingStabilityLevel.STABLE,
            policy_id="test-routing-v2",
            policy_version="2.0.0",
            item_count=32,
            reported_model_count=32,
            response_model_coverage=1.0,
            distinct_response_models=1,
            dominant_model_ratio=1.0,
            failure_ratio=0.0,
            temporal_status=TemporalFingerprintStatus.UNAVAILABLE,
            signals=(),
        )

        section = build_fingerprint_section(
            match=None,
            verification=None,
            snapshot=candidate,
            suite=suite,
            routing=routing,
            confidence=build_confidence_bundle(routing=routing),
        )

        assert section is not None
        assert section["routing"] == routing.model_dump(mode="json")
        serialized_confidence = section["confidence"]
        assert isinstance(serialized_confidence, dict)
        # Four independent components, never merged into one number (Task 32).
        assert set(serialized_confidence) == {"measurement", "calibration", "fingerprint", "routing"}

    def test_section_disappears_without_any_fingerprint_artifact(self) -> None:
        assert build_fingerprint_section(match=None, verification=None, snapshot=None) is None

    def test_capture_only_run_reports_no_verdict(self, suite: FingerprintSuite) -> None:
        candidate = _candidate(suite)
        section = build_fingerprint_section(match=None, verification=None, snapshot=candidate, suite=suite)

        assert section is not None
        assert section["decision_policy"] is None
        assert section["jsd_reference"] is None
        assert section["per_probe_jsd"] == []
        assert section["aggregate_jsd"] is None
        assert section["top_k"] == []
        verdict = section["verdict"]
        assert isinstance(verdict, dict)
        assert verdict["match_status"] is None
        assert verdict["claim_verdict_produced"] is False
