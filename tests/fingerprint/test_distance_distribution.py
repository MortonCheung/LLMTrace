"""Task 50 — distance / distribution unit tests.

The Task 50 checklist is asserted directly here.  These modules were previously
reached only *indirectly* (through validation / temporal / reference), so the
list itself was never verified:

    JSD identical == 0
    JSD disjoint == 1
    support order canonical
    invalid outcome retained
    distribution sums to 1
    no samples fail closed
    probe coverage enforced
    hash tampering rejected
    fixture production reject
    incompatible set rejected
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmtrace.fingerprint.distance import (
    FingerprintDistance,
    FingerprintDistanceError,
    ProbeDistance,
    compute_fingerprint_distance,
    js_divergence,
)
from llmtrace.fingerprint.distribution import build_probe_distribution
from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    AggregatedProbeDistribution,
    FingerprintProbe,
    FingerprintSampleObservation,
    FingerprintSourceRole,
    FingerprintSuite,
    ProbeDistribution,
)
from llmtrace.fingerprint.reference import build_fingerprint_snapshot
from llmtrace.fingerprint.reference_set import (
    FingerprintReferenceSetBuilder,
    FingerprintReferenceSetCompatibilityError,
    FingerprintReferenceSetError,
    FingerprintReferenceSetIntegrityError,
    FixtureFingerprintReferenceError,
)
from llmtrace.fingerprint.suite import FingerprintSuiteIntegrityError, verify_fingerprint_suite

from .conftest import CAPTURED_AT, FingerprintFixture, make_observations, make_reference_snapshot

_SUPPORT = ("a", "b", INVALID_OUTCOME)


def _probe(probe_id: str, *, weight: float = 1.0) -> FingerprintProbe:
    return FingerprintProbe(
        probe_id=probe_id,
        prompt=f"pick a letter for {probe_id}",
        choices=("a", "b"),
        weight=weight,
    )


def _observation(probe_id: str, outcome: str, *, valid: bool = True) -> FingerprintSampleObservation:
    return FingerprintSampleObservation(
        probe_id=probe_id,
        sequence_index=0,
        round_index=0,
        outcome=outcome,
        valid=valid,
        response_body_sha256="a" * 64,
        evidence_ref=f"fixture:{probe_id}:{outcome}",
    )


def _dist(probe_id: str, counts: dict[str, int], *, support: tuple[str, ...] = _SUPPORT) -> ProbeDistribution:
    total = sum(counts.values())
    return ProbeDistribution(
        probe_id=probe_id,
        support=support,
        counts=counts,
        probabilities={item: (counts[item] / total if total else 0.0) for item in support},
        sample_count=total,
        invalid_count=counts.get(INVALID_OUTCOME, 0),
    )


def _certain(probe_id: str, choice: str) -> ProbeDistribution:
    """A distribution that always answers *choice*."""
    return _dist(probe_id, {item: (10 if item == choice else 0) for item in _SUPPORT})


class TestJsDivergence:
    def test_identical_distributions_are_zero(self) -> None:
        assert js_divergence((1.0, 0.0), (1.0, 0.0)) == 0.0
        assert js_divergence((0.5, 0.5), (0.5, 0.5)) == 0.0

    def test_disjoint_distributions_are_one(self) -> None:
        assert js_divergence((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)

    def test_partial_overlap_stays_inside_the_unit_interval(self) -> None:
        value = js_divergence((0.75, 0.25), (0.25, 0.75))

        assert 0.0 < value < 1.0

    def test_is_symmetric(self) -> None:
        p, q = (0.9, 0.1), (0.2, 0.8)

        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="distribution lengths differ"):
            js_divergence((1.0, 0.0), (1.0, 0.0, 0.0))


class TestBuildProbeDistribution:
    def test_support_is_the_choices_plus_the_reserved_outcome(self, suite: FingerprintSuite) -> None:
        probe = suite.probes[0]

        distribution = build_probe_distribution(probe, make_observations(suite, 3))

        assert distribution.support == (*probe.choices, INVALID_OUTCOME)
        assert distribution.sample_count == 3

    def test_invalid_outcome_is_retained_and_the_denominator_is_not_shrunk(self) -> None:
        probe = _probe("letter")
        observations = (
            _observation("letter", "a"),
            _observation("letter", INVALID_OUTCOME, valid=False),
        )

        distribution = build_probe_distribution(probe, observations)

        # The failed request stays in the denominator: 2 samples, not 1.
        assert distribution.sample_count == 2
        assert distribution.invalid_count == 1
        assert distribution.counts[INVALID_OUTCOME] == 1
        assert distribution.probabilities[INVALID_OUTCOME] == pytest.approx(0.5)
        assert sum(distribution.probabilities.values()) == pytest.approx(1.0)

    def test_observations_of_other_probes_are_not_counted(self) -> None:
        probe = _probe("letter")

        distribution = build_probe_distribution(probe, (_observation("letter", "a"), _observation("other", "b")))

        assert distribution.sample_count == 1
        assert distribution.counts["a"] == 1
        assert distribution.counts["b"] == 0

    def test_zero_samples_keep_the_probe_with_zero_probabilities(self) -> None:
        probe = _probe("letter")

        distribution = build_probe_distribution(probe, ())

        assert distribution.sample_count == 0
        assert distribution.support == ("a", "b", INVALID_OUTCOME)
        assert all(probability == 0.0 for probability in distribution.probabilities.values())


class TestSupportOrderIsCanonical:
    def test_ordered_probabilities_follow_the_support_not_dict_insertion_order(self) -> None:
        distribution = ProbeDistribution(
            probe_id="letter",
            support=("a", "b", INVALID_OUTCOME),
            counts={"a": 6, "b": 3, INVALID_OUTCOME: 1},
            # Deliberately inserted in a different order than the support.
            probabilities={INVALID_OUTCOME: 0.1, "b": 0.3, "a": 0.6},
            sample_count=10,
            invalid_count=1,
        )

        assert distribution.ordered_probabilities() == (0.6, 0.3, 0.1)

    def test_aggregated_ordered_probabilities_follow_the_support(self) -> None:
        aggregated = AggregatedProbeDistribution(
            probe_id="letter",
            support=("a", "b", INVALID_OUTCOME),
            probabilities={"b": 0.25, INVALID_OUTCOME: 0.0, "a": 0.75},
            snapshot_ids=("capture-1",),
            snapshot_count=1,
            sample_count=10,
        )

        assert aggregated.ordered_probabilities() == (0.75, 0.25, 0.0)

    def test_a_reordered_support_is_recorded_as_not_comparable(self) -> None:
        probes = (_probe("aligned"), _probe("reordered"))
        candidate = (_certain("aligned", "a"), _dist("reordered", {"a": 10, "b": 0, INVALID_OUTCOME: 0}))
        reference = (
            _certain("aligned", "a"),
            _dist("reordered", {"a": 10, "b": 0, INVALID_OUTCOME: 0}, support=("b", "a", INVALID_OUTCOME)),
        )

        distance = compute_fingerprint_distance(
            probes=probes, candidate=candidate, reference=reference, minimum_comparable_probes=1
        )

        notes = {entry.probe_id: entry.note for entry in distance.per_probe if not entry.comparable}
        assert notes == {"reordered": "support_mismatch"}


class TestComputeFingerprintDistance:
    def test_identical_captures_have_zero_distance(self) -> None:
        distributions = (_certain("letter", "a"),)

        distance = compute_fingerprint_distance(
            probes=(_probe("letter"),),
            candidate=distributions,
            reference=distributions,
            minimum_comparable_probes=1,
        )

        assert distance.distance == 0.0
        assert distance.comparable_probes == 1
        assert distance.per_probe[0].comparable is True

    def test_disjoint_captures_have_unit_distance(self) -> None:
        distance = compute_fingerprint_distance(
            probes=(_probe("letter"),),
            candidate=(_certain("letter", "a"),),
            reference=(_certain("letter", "b"),),
            minimum_comparable_probes=1,
        )

        assert distance.distance == pytest.approx(1.0)

    def test_a_missing_reference_distribution_is_not_comparable(self) -> None:
        probes = (_probe("present"), _probe("absent"))

        distance = compute_fingerprint_distance(
            probes=probes,
            candidate=(_certain("present", "a"), _certain("absent", "a")),
            reference=(_certain("present", "a"),),
            minimum_comparable_probes=1,
        )

        notes = {entry.probe_id: entry.note for entry in distance.per_probe if not entry.comparable}
        assert notes == {"absent": "missing_reference_distribution"}

    def test_empty_samples_fail_closed(self) -> None:
        probes = (_probe("probe-a"), _probe("probe-b"))

        distance = compute_fingerprint_distance(
            probes=probes,
            candidate=(_dist("probe-a", {"a": 0, "b": 0, INVALID_OUTCOME: 0}), _certain("probe-b", "a")),
            reference=(_certain("probe-a", "a"), _certain("probe-b", "a")),
            minimum_comparable_probes=1,
        )

        assert distance.per_probe[0].note == "empty_candidate_samples"

    def test_an_unnormalized_distribution_is_not_comparable(self) -> None:
        probes = (_probe("broken"), _probe("aligned"))
        # `_dist` normalizes, so overwrite the probabilities post-construction.
        broken = _dist("broken", {"a": 5, "b": 5, INVALID_OUTCOME: 0}).model_copy(
            update={"probabilities": {"a": 5.0, "b": 5.0, INVALID_OUTCOME: 0.0}}
        )

        distance = compute_fingerprint_distance(
            probes=probes,
            candidate=(broken, _certain("aligned", "a")),
            reference=(_certain("broken", "a"), _certain("aligned", "a")),
            minimum_comparable_probes=1,
        )

        assert distance.per_probe[0].note == "unnormalized_candidate_distribution"

    def test_probe_coverage_below_the_minimum_fails_closed(self) -> None:
        probes = tuple(_probe(f"probe-{index}") for index in range(3))
        candidate = tuple(_certain(probe.probe_id, "a") for probe in probes)

        with pytest.raises(FingerprintDistanceError, match="below the policy minimum"):
            compute_fingerprint_distance(
                probes=probes,
                candidate=candidate,
                reference=(_certain("probe-0", "a"),),
                minimum_comparable_probes=2,
            )

    def test_the_minimum_must_be_at_least_one(self) -> None:
        with pytest.raises(FingerprintDistanceError, match="minimum_comparable_probes must be >= 1"):
            compute_fingerprint_distance(
                probes=(_probe("letter"),),
                candidate=(_certain("letter", "a"),),
                reference=(_certain("letter", "a"),),
                minimum_comparable_probes=0,
            )

    def test_duplicate_probe_ids_are_rejected(self) -> None:
        with pytest.raises(FingerprintDistanceError, match="duplicate probe_id in candidate distributions"):
            compute_fingerprint_distance(
                probes=(_probe("letter"),),
                candidate=(_certain("letter", "a"), _certain("letter", "b")),
                reference=(_certain("letter", "a"),),
                minimum_comparable_probes=1,
            )

    def test_coverage_and_sample_counts_are_reported(self) -> None:
        probes = (_probe("probe-a"), _probe("probe-b"))

        distance = compute_fingerprint_distance(
            probes=probes,
            candidate=(_dist("probe-a", {"a": 3, "b": 1, INVALID_OUTCOME: 0}), _certain("probe-b", "a")),
            reference=(_dist("probe-a", {"a": 2, "b": 2, INVALID_OUTCOME: 0}), _certain("probe-b", "a")),
            minimum_comparable_probes=2,
        )

        assert distance.comparable_probes == 2
        assert distance.sample_count_candidate == 4 + 10
        assert distance.sample_count_reference == 4 + 10
        assert len(distance.per_probe) == 2

    def test_probe_weights_are_honoured(self) -> None:
        probes = (_probe("heavy", weight=3.0), _probe("light", weight=1.0))

        distance = compute_fingerprint_distance(
            probes=probes,
            candidate=(_certain("heavy", "a"), _certain("light", "a")),
            reference=(_certain("heavy", "b"), _certain("light", "a")),
            minimum_comparable_probes=2,
        )

        # heavy contributes distance 1.0 at weight 3, light 0.0 at weight 1.
        assert distance.distance == pytest.approx(0.75)


class TestProbeDistanceModel:
    def test_a_comparable_probe_needs_a_distance(self) -> None:
        with pytest.raises(ValidationError, match="must carry a distance value"):
            ProbeDistance(probe_id="letter", weight=1.0, comparable=True)

    def test_a_comparable_probe_must_not_carry_a_note(self) -> None:
        with pytest.raises(ValidationError, match="must not carry a note"):
            ProbeDistance(probe_id="letter", weight=1.0, comparable=True, distance=0.1, note="why")

    def test_a_non_comparable_probe_must_not_carry_a_distance(self) -> None:
        with pytest.raises(ValidationError, match="must not carry a distance value"):
            ProbeDistance(probe_id="letter", weight=1.0, comparable=False, distance=0.1, note="why")

    def test_a_non_comparable_probe_must_explain_itself(self) -> None:
        with pytest.raises(ValidationError, match="must explain itself via note"):
            ProbeDistance(probe_id="letter", weight=1.0, comparable=False)


class TestFingerprintDistanceModel:
    @staticmethod
    def _comparable(probe_id: str, distance: float) -> ProbeDistance:
        return ProbeDistance(probe_id=probe_id, weight=1.0, comparable=True, distance=distance)

    def test_comparable_probes_must_match_the_per_probe_entries(self) -> None:
        with pytest.raises(ValidationError, match="must equal the number of comparable per_probe entries"):
            FingerprintDistance(
                distance=0.0,
                comparable_probes=2,
                sample_count_candidate=1,
                sample_count_reference=1,
                per_probe=(self._comparable("letter", 0.0),),
            )

    def test_a_distance_requires_at_least_one_comparable_probe(self) -> None:
        with pytest.raises(ValidationError, match="requires at least one comparable probe"):
            FingerprintDistance(
                distance=0.0,
                comparable_probes=0,
                sample_count_candidate=0,
                sample_count_reference=0,
                per_probe=(
                    ProbeDistance(probe_id="letter", weight=1.0, comparable=False, note="empty_candidate_samples"),
                ),
            )


class TestReferenceTrustBoundary:
    def test_test_fixtures_never_enter_a_reference_set(self, suite: FingerprintSuite) -> None:
        snapshot = build_fingerprint_snapshot(
            snapshot_id="fixture-capture-1",
            model_id="some-model",
            provider_id="openai",
            source_role=FingerprintSourceRole.TEST_FIXTURE,
            suite=suite,
            repetitions=1,
            observations=make_observations(suite, 1),
            captured_at=CAPTURED_AT,
        )

        with pytest.raises(FixtureFingerprintReferenceError, match="test fixtures never enter"):
            FingerprintReferenceSetBuilder().build(
                fingerprint_set_id="fixture-set",
                fingerprint_set_version="0.1.0",
                snapshots=(snapshot,),
                snapshot_sha256s={},
            )

    def test_candidate_captures_never_enter_a_reference_set(self, suite: FingerprintSuite) -> None:
        snapshot = build_fingerprint_snapshot(
            snapshot_id="candidate-capture-1",
            model_id="audited-model",
            provider_id="openai",
            source_role=FingerprintSourceRole.CANDIDATE_CAPTURE,
            suite=suite,
            repetitions=1,
            observations=make_observations(suite, 1),
            captured_at=CAPTURED_AT,
        )

        with pytest.raises(FingerprintReferenceSetError, match="never reference material"):
            FingerprintReferenceSetBuilder().build(
                fingerprint_set_id="candidate-set",
                fingerprint_set_version="0.1.0",
                snapshots=(snapshot,),
                snapshot_sha256s={},
            )

    def test_members_that_disagree_on_repetitions_are_rejected(self, suite: FingerprintSuite) -> None:
        first = make_reference_snapshot(suite=suite, snapshot_id="a-capture-1", model_id="model-a", repetitions=8)
        second = make_reference_snapshot(suite=suite, snapshot_id="b-capture-1", model_id="model-b", repetitions=4)

        with pytest.raises(FingerprintReferenceSetCompatibilityError, match="is incompatible with"):
            FingerprintReferenceSetBuilder().build(
                fingerprint_set_id="mixed-set",
                fingerprint_set_version="0.1.0",
                snapshots=(first, second),
                snapshot_sha256s={first.snapshot_id: "a" * 64, second.snapshot_id: "b" * 64},
            )

    def test_a_tampered_suite_hash_is_rejected(self, suite: FingerprintSuite) -> None:
        with pytest.raises(FingerprintSuiteIntegrityError, match="content hash mismatch"):
            verify_fingerprint_suite(suite.model_copy(update={"content_sha256": "0" * 64}))

    def test_a_tampered_reference_set_hash_is_rejected(self, fingerprint_reference: FingerprintFixture) -> None:
        tampered = fingerprint_reference.reference_set.model_copy(update={"content_sha256": "0" * 64})

        with pytest.raises(FingerprintReferenceSetIntegrityError, match="content hash mismatch"):
            tampered.verify_content_hash()
