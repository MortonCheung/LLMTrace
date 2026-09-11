"""Task 51 — matcher tests.

Every entry on the Task 51 checklist is asserted here.  ``FingerprintMatcher``
was previously exercised only through the runner and the held-out validation
path, so the list itself was never verified:

    closest identity ranked first
    no shared probes → unavailable
    multiple captures same identity aggregated
    claimed model absent → inconclusive
    unvalidated policy → no consistency verdict
    validated policy → consistent
    validated policy → inconsistent

Two spec wordings are pinned to the implemented (honest) behaviour:

* "no shared probes → unavailable" — a library with no comparable reference
  yields ``INCONCLUSIVE`` ("no comparable reference"); when a reference *is*
  present but shares no probe with the candidate the matcher fails closed with
  ``FingerprintMatchError`` rather than silently ranking nothing.
* "claimed model absent → inconclusive" — the matcher still has a ranking of the
  references it does hold, so the honest status is ``RANKED_ONLY`` (ranking, no
  verdict).  ``INCONCLUSIVE`` is reserved for the case with no ranking at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from llmtrace.fingerprint.aggregation import aggregate_fingerprint_reference
from llmtrace.fingerprint.matcher import (
    FingerprintMatchEntry,
    FingerprintMatcher,
    FingerprintMatchError,
    FingerprintMatchResult,
    FingerprintVerificationResult,
)
from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    FingerprintMatchStatus,
    FingerprintProfile,
    FingerprintSourceRole,
    FingerprintSuite,
    ProbeDistribution,
    repetitions_for_profile,
)
from llmtrace.fingerprint.policy import build_fingerprint_policy
from llmtrace.fingerprint.reference import build_fingerprint_snapshot

from .conftest import CAPTURED_AT, make_observations, make_reference_snapshot

_REPETITIONS = repetitions_for_profile(FingerprintProfile.STANDARD)


def _candidate(
    suite: FingerprintSuite,
    *,
    choice_index: int = 0,
    snapshot_id: str = "candidate-capture-1",
) -> tuple[ProbeDistribution, ...]:
    """The audited endpoint's capture, built as a CANDIDATE_CAPTURE snapshot."""
    snapshot = build_fingerprint_snapshot(
        snapshot_id=snapshot_id,
        model_id="audited-model",
        provider_id="openai",
        source_role=FingerprintSourceRole.CANDIDATE_CAPTURE,
        suite=suite,
        repetitions=_REPETITIONS,
        observations=make_observations(suite, _REPETITIONS, choice_index=choice_index),
        captured_at=CAPTURED_AT,
    )
    return snapshot.distributions


def _reference(
    suite: FingerprintSuite,
    *,
    model_id: str,
    choice_indexes: Sequence[int],
    provider_id: str = "openai",
):
    """A reference identity aggregated from one capture per choice index."""
    snapshots = tuple(
        make_reference_snapshot(
            suite=suite,
            snapshot_id=f"{provider_id}-{model_id}-capture-{index}",
            model_id=model_id,
            provider_id=provider_id,
            repetitions=_REPETITIONS,
            choice_index=choice_index,
        )
        for index, choice_index in enumerate(choice_indexes)
    )
    return aggregate_fingerprint_reference(snapshots=snapshots, suite=suite)


def _policy(suite: FingerprintSuite, *, validated: bool, distance_threshold: float = 0.2):
    return build_fingerprint_policy(
        policy_id="test-policy",
        policy_version="0.1.0",
        fingerprint_set_id="test-identity-set",
        fingerprint_set_content_sha256="a" * 64,
        suite_content_sha256=suite.content_sha256,
        validated=validated,
        minimum_comparable_probes=len(suite.probes),
        max_far_target=0.05,
        identity_count=2,
        held_out_capture_count=4,
        distance_threshold=distance_threshold if validated else None,
        top1_accuracy=1.0,
        top3_accuracy=1.0,
        true_positive_rate=0.95 if validated else None,
        false_accept_rate=0.01 if validated else None,
    )


def _entry(model_id: str, distance: float, *, provider_id: str = "openai") -> FingerprintMatchEntry:
    return FingerprintMatchEntry(
        model_id=model_id,
        provider_id=provider_id,
        distance=distance,
        similarity=1.0 - distance,
        comparable_probes=1,
    )


class TestRanking:
    def test_the_closest_identity_is_ranked_first(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (
            _reference(suite, model_id="model-a", choice_indexes=(0,)),
            _reference(suite, model_id="model-b", choice_indexes=(1,)),
        )

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
        )

        assert [entry.model_id for entry in result.entries] == ["model-a", "model-b"]
        assert result.entries[0].distance == 0.0
        assert result.entries[1].distance == pytest.approx(1.0)

    def test_similarity_is_one_minus_distance(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (_reference(suite, model_id="model-b", choice_indexes=(1,)),)

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
        )

        entry = result.entries[0]
        assert entry.similarity == pytest.approx(1.0 - entry.distance)
        assert entry.comparable_probes == len(suite.probes)

    def test_ranking_alone_is_not_a_verdict(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (_reference(suite, model_id="model-a", choice_indexes=(0,)),)

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
        )

        assert result.status is FingerprintMatchStatus.RANKED_ONLY


class TestAggregationInMatching:
    def test_multiple_captures_of_one_identity_are_averaged(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        # The same identity answered choice 0 in one capture and choice 1 in the
        # next; equal capture weight makes the aggregate 50/50.
        reference = _reference(suite, model_id="model-a", choice_indexes=(0, 1))

        assert reference.snapshot_count == 2
        assert reference.snapshot_ids == ("openai-model-a-capture-0", "openai-model-a-capture-1")

        probe = suite.probes[0]
        aggregated = reference.distributions[0]
        assert aggregated.probabilities[probe.choices[0]] == pytest.approx(0.5)
        assert aggregated.probabilities[probe.choices[1]] == pytest.approx(0.5)
        assert aggregated.sample_count == _REPETITIONS * 2

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=(reference,),
            minimum_comparable_probes=len(suite.probes),
        )

        assert len(result.entries) == 1
        # JSD((1, 0, 0, 0, 0) || (0.5, 0.5, 0, 0, 0)) over the 4 choices + __INVALID__.
        assert result.entries[0].distance == pytest.approx(0.3112781244591329)


class TestNoComparableReference:
    def test_an_empty_reference_library_is_inconclusive(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)

        result = matcher.match(
            candidate=_candidate(suite),
            references=(),
            minimum_comparable_probes=1,
        )

        assert result.entries == ()
        assert result.status is FingerprintMatchStatus.INCONCLUSIVE
        assert result.claimed_reference_distance is None

    def test_a_candidate_sharing_no_probes_with_the_suite_cannot_be_ranked(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        foreign = tuple(
            ProbeDistribution(
                probe_id=f"foreign-{index}",
                support=("x", "y", INVALID_OUTCOME),
                counts={"x": 8, "y": 0, INVALID_OUTCOME: 0},
                probabilities={"x": 1.0, "y": 0.0, INVALID_OUTCOME: 0.0},
                sample_count=8,
                invalid_count=0,
            )
            for index in range(2)
        )

        with pytest.raises(FingerprintMatchError, match="cannot be ranked"):
            matcher.match(
                candidate=foreign,
                references=(_reference(suite, model_id="model-a", choice_indexes=(0,)),),
                minimum_comparable_probes=1,
            )


class TestClaimVerdicts:
    def test_an_unvalidated_policy_never_produces_a_consistency_verdict(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (_reference(suite, model_id="model-a", choice_indexes=(0,)),)

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
            claimed_model_id="model-a",
            policy=_policy(suite, validated=False),
        )

        # The claimed reference is a perfect match, but an unvalidated policy
        # may not turn that into a verdict (Rule 2).
        assert result.claimed_reference_distance == pytest.approx(0.0)
        assert result.status is FingerprintMatchStatus.RANKED_ONLY

    def test_a_validated_policy_marks_a_close_claimed_reference_as_consistent(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (_reference(suite, model_id="model-a", choice_indexes=(0,)),)

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
            claimed_model_id="model-a",
            policy=_policy(suite, validated=True, distance_threshold=0.2),
        )

        assert result.status is FingerprintMatchStatus.CONSISTENT_WITH_CLAIM
        assert result.policy_id == "test-policy"
        assert result.policy_version == "0.1.0"

    def test_a_validated_policy_marks_a_distant_claimed_reference_as_inconsistent(
        self, suite: FingerprintSuite
    ) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (_reference(suite, model_id="model-a", choice_indexes=(1,)),)

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
            claimed_model_id="model-a",
            policy=_policy(suite, validated=True, distance_threshold=0.2),
        )

        assert result.claimed_reference_distance == pytest.approx(1.0)
        assert result.status is FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM

    def test_a_claimed_model_without_a_reference_yields_no_verdict(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (_reference(suite, model_id="model-a", choice_indexes=(0,)),)

        result = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
            claimed_model_id="never-captured-model",
            policy=_policy(suite, validated=True, distance_threshold=0.2),
        )

        assert result.claimed_reference_distance is None
        assert result.status is FingerprintMatchStatus.RANKED_ONLY

    def test_the_claimed_provider_narrows_the_reference_pairing(self, suite: FingerprintSuite) -> None:
        matcher = FingerprintMatcher(suite=suite)
        references = (
            _reference(suite, model_id="model-a", choice_indexes=(0,), provider_id="openai"),
            _reference(suite, model_id="model-a", choice_indexes=(1,), provider_id="other"),
        )
        policy = _policy(suite, validated=True, distance_threshold=0.2)

        without_provider = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
            claimed_model_id="model-a",
            policy=policy,
        )
        with_provider = matcher.match(
            candidate=_candidate(suite, choice_index=0),
            references=references,
            minimum_comparable_probes=len(suite.probes),
            claimed_model_id="model-a",
            claimed_provider_id="other",
            policy=policy,
        )

        # Without a provider the closest capture of the label wins (0.0 → consistent);
        # pinning the provider selects the distant capture instead (1.0 → inconsistent).
        assert without_provider.claimed_reference_distance == pytest.approx(0.0)
        assert without_provider.status is FingerprintMatchStatus.CONSISTENT_WITH_CLAIM
        assert with_provider.claimed_reference_distance == pytest.approx(1.0)
        assert with_provider.status is FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM


class TestMatcherFailClosed:
    def test_the_minimum_must_be_at_least_one(self, suite: FingerprintSuite) -> None:
        with pytest.raises(FingerprintMatchError, match="minimum_comparable_probes must be >= 1"):
            FingerprintMatcher(suite=suite).match(
                candidate=_candidate(suite),
                references=(),
                minimum_comparable_probes=0,
            )

    def test_a_policy_with_a_different_minimum_is_rejected(self, suite: FingerprintSuite) -> None:
        with pytest.raises(FingerprintMatchError, match="contradicts policy"):
            FingerprintMatcher(suite=suite).match(
                candidate=_candidate(suite),
                references=(),
                minimum_comparable_probes=1,
                policy=_policy(suite, validated=True),
            )

    def test_a_policy_validated_against_another_suite_is_rejected(self, suite: FingerprintSuite) -> None:
        foreign = build_fingerprint_policy(
            policy_id="foreign-policy",
            policy_version="0.1.0",
            fingerprint_set_id="test-identity-set",
            fingerprint_set_content_sha256="a" * 64,
            suite_content_sha256="a" * 64,
            validated=True,
            minimum_comparable_probes=len(suite.probes),
            max_far_target=0.05,
            identity_count=2,
            held_out_capture_count=4,
            distance_threshold=0.2,
            true_positive_rate=0.95,
            false_accept_rate=0.01,
        )

        with pytest.raises(FingerprintMatchError, match="not the suite being matched"):
            FingerprintMatcher(suite=suite).match(
                candidate=_candidate(suite),
                references=(),
                minimum_comparable_probes=len(suite.probes),
                policy=foreign,
            )

    def test_duplicate_reference_identities_are_rejected(self, suite: FingerprintSuite) -> None:
        reference = _reference(suite, model_id="model-a", choice_indexes=(0,))

        with pytest.raises(FingerprintMatchError, match="duplicate fingerprint reference identity"):
            FingerprintMatcher(suite=suite).match(
                candidate=_candidate(suite),
                references=(reference, reference),
                minimum_comparable_probes=len(suite.probes),
            )

    def test_a_claimed_provider_without_a_claimed_model_is_rejected(self, suite: FingerprintSuite) -> None:
        with pytest.raises(FingerprintMatchError, match="claimed_provider_id requires claimed_model_id"):
            FingerprintMatcher(suite=suite).match(
                candidate=_candidate(suite),
                references=(),
                minimum_comparable_probes=1,
                claimed_provider_id="openai",
            )


class TestMatchResultModel:
    def test_entries_must_be_sorted_by_ascending_distance(self) -> None:
        with pytest.raises(ValidationError, match="must be sorted by ascending"):
            FingerprintMatchResult(
                entries=(_entry("model-a", 0.5), _entry("model-b", 0.2)),
                status=FingerprintMatchStatus.RANKED_ONLY,
            )

    def test_entries_must_reference_distinct_identities(self) -> None:
        with pytest.raises(ValidationError, match="must reference distinct identities"):
            FingerprintMatchResult(
                entries=(_entry("model-a", 0.2), _entry("model-a", 0.2)),
                status=FingerprintMatchStatus.RANKED_ONLY,
            )

    def test_a_claim_verdict_must_name_the_policy_it_came_from(self) -> None:
        with pytest.raises(ValidationError, match="must name the validated policy"):
            FingerprintMatchResult(
                entries=(_entry("model-a", 0.2),),
                status=FingerprintMatchStatus.CONSISTENT_WITH_CLAIM,
                claimed_model_id="model-a",
                claimed_reference_distance=0.2,
            )


class TestVerificationGate:
    def test_a_claim_verdict_requires_a_validated_policy(self) -> None:
        with pytest.raises(ValidationError, match="validated decision policy"):
            FingerprintVerificationResult(
                match_status=FingerprintMatchStatus.CONSISTENT_WITH_CLAIM,
                claim_verdict_produced=True,
                policy=None,
                minimum_comparable_probes=1,
                reference_identity_count=1,
            )

    def test_claim_verdict_produced_must_match_the_status(self) -> None:
        with pytest.raises(ValidationError, match="contradicts match_status"):
            FingerprintVerificationResult(
                match_status=FingerprintMatchStatus.RANKED_ONLY,
                claim_verdict_produced=True,
                minimum_comparable_probes=1,
                reference_identity_count=1,
            )
