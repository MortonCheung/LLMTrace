"""Fingerprint policy validation（Task 18 / Task 52）.

Held-out validation is the only path allowed to turn fingerprint distances into
a production threshold (Rule 2 / Rule 6). These tests pin the three things that
make it trustworthy:

* the held-out capture never scores itself (``own_distance`` would collapse if
  it leaked into its own reference library),
* the minimum conditions are fail-closed and publish no accuracy numbers,
* a FAR budget that no candidate threshold can honour yields a policy without
  a threshold instead of a lenient one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from llmtrace.fingerprint.models import (
    FingerprintProfile,
    FingerprintSampleObservation,
    FingerprintSourceRole,
    FingerprintSuite,
    repetitions_for_profile,
)
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot, build_fingerprint_snapshot
from llmtrace.fingerprint.reference_set import FingerprintReferenceSet
from llmtrace.fingerprint.validation import (
    MINIMUM_CAPTURES_PER_IDENTITY,
    MINIMUM_DISTINCT_IDENTITIES,
    FingerprintHoldoutOutcome,
    FingerprintValidationError,
    FingerprintValidationReport,
    select_distance_threshold,
    validate_fingerprint_policy,
)

from .conftest import CAPTURED_AT, publish_reference_set

# ---------------------------------------------------------------------------
# Fixture captures
# ---------------------------------------------------------------------------

_CAPTURED_AT: datetime = CAPTURED_AT
_REPETITIONS = repetitions_for_profile(FingerprintProfile.STANDARD)

_ALL_ONE = (1, 1, 1, 1, 1, 1)
_ALL_TWO = (2, 2, 2, 2, 2, 2)

#: Per-round probe choices: 8 rounds × 6 probes.
_Rounds = tuple[tuple[int, ...], ...]
_Captures = tuple[_Rounds, ...]
_Specs = tuple[tuple[str, _Captures], ...]


def _constant(choices: tuple[int, ...]) -> _Rounds:
    """The same answer to every probe, repeated for every round."""
    return (choices,) * _REPETITIONS


#: model-a's two captures agree on nothing: holding either one out must score it
#: against the *other* capture only, so ``own_distance`` is the maximum 1.0.
#: Had the held-out capture leaked into its own reference the distance would
#: collapse to ~0.311 (JSD between a point mass and a 50/50 mixture).
_DISTINCT_BEHAVIOUR_SPECS: _Specs = (
    ("model-a", (_constant((0, 0, 0, 0, 0, 0)), _constant((1, 2, 3, 1, 2, 3)))),
    ("model-b", (_constant(_ALL_ONE), _constant(_ALL_ONE))),
    ("model-c", (_constant(_ALL_TWO), _constant(_ALL_TWO))),
    ("model-d", (_constant((3, 3, 3, 3, 3, 3)), _constant((3, 3, 3, 3, 3, 3)))),
    ("model-e", (_constant((0, 2, 0, 2, 0, 2)), _constant((0, 2, 0, 2, 0, 2)))),
)

#: model-b's first capture answers one way in rounds 0-3 and another way in
#: rounds 4-7, so the within-reference temporal windows differ maximally.
_DRIFTING_B_CAPTURE: _Rounds = (_ALL_ONE,) * 4 + (_ALL_TWO,) * 4

_DRIFTING_SPECS: _Specs = (
    ("model-a", (_constant((0, 0, 0, 0, 0, 0)), _constant((1, 2, 3, 1, 2, 3)))),
    ("model-b", (_DRIFTING_B_CAPTURE, _constant(_ALL_ONE))),
    ("model-c", (_constant(_ALL_TWO), _constant(_ALL_TWO))),
    ("model-d", (_constant((3, 3, 3, 3, 3, 3)), _constant((3, 3, 3, 3, 3, 3)))),
    ("model-e", (_constant((0, 2, 0, 2, 0, 2)), _constant((0, 2, 0, 2, 0, 2)))),
)

#: model-b and model-c behave identically, so the closest impostor distance is
#: 0.0 and no threshold can keep the false accept rate at 0.05.
_DUPLICATE_BEHAVIOUR_SPECS: _Specs = (
    ("model-a", (_constant((0, 0, 0, 0, 0, 0)), _constant((1, 2, 3, 1, 2, 3)))),
    ("model-b", (_constant(_ALL_ONE), _constant(_ALL_ONE))),
    ("model-c", (_constant(_ALL_ONE), _constant(_ALL_ONE))),
    ("model-d", (_constant((3, 3, 3, 3, 3, 3)), _constant((3, 3, 3, 3, 3, 3)))),
    ("model-e", (_constant((0, 2, 0, 2, 0, 2)), _constant((0, 2, 0, 2, 0, 2)))),
)

#: Only four distinct identities — below the Step 18.1 floor.
_FOUR_IDENTITY_SPECS: _Specs = _DISTINCT_BEHAVIOUR_SPECS[:4]

#: Five identities with a single capture each — no held-out comparison exists.
_ONE_CAPTURE_SPECS: _Specs = tuple((model_id, (captures[0],)) for model_id, captures in _DISTINCT_BEHAVIOUR_SPECS)


@dataclass(frozen=True)
class _Case:
    """A published reference set plus the snapshots it was built from."""

    suite: FingerprintSuite
    reference_set: FingerprintReferenceSet
    snapshots: tuple[FingerprintReferenceSnapshot, ...]


def _capture(
    suite: FingerprintSuite,
    *,
    snapshot_id: str,
    model_id: str,
    rounds: _Rounds,
) -> FingerprintReferenceSnapshot:
    """Build a trusted-reference capture that answers ``rounds`` verbatim."""
    assert len(rounds) == _REPETITIONS
    observations: list[FingerprintSampleObservation] = []
    for round_index, choices in enumerate(rounds):
        assert len(choices) == len(suite.probes)
        for sequence_index, probe in enumerate(suite.probes):
            observations.append(
                FingerprintSampleObservation(
                    probe_id=probe.probe_id,
                    sequence_index=sequence_index,
                    round_index=round_index,
                    outcome=probe.choices[choices[sequence_index]],
                    valid=True,
                    response_body_sha256="a" * 64,
                    evidence_ref=f"{snapshot_id}:{probe.probe_id}:{round_index}",
                )
            )

    return build_fingerprint_snapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id="openai",
        source_role=FingerprintSourceRole.TRUSTED_REFERENCE,
        suite=suite,
        repetitions=_REPETITIONS,
        observations=tuple(observations),
        captured_at=_CAPTURED_AT,
    )


def _publish(tmp_path: Path, suite: FingerprintSuite, specs: _Specs) -> _Case:
    snapshots = tuple(
        _capture(suite, snapshot_id=f"{model_id}-{index + 1}", model_id=model_id, rounds=rounds)
        for model_id, captures in specs
        for index, rounds in enumerate(captures)
    )
    fixture = publish_reference_set(data_root=tmp_path / "appdata", suite=suite, snapshots=snapshots)
    return _Case(suite=suite, reference_set=fixture.reference_set, snapshots=snapshots)


def _validate(
    case: _Case,
    *,
    snapshots: Sequence[FingerprintReferenceSnapshot] | None = None,
    reference_set: FingerprintReferenceSet | None = None,
    max_far_target: float = 0.05,
    minimum_comparable_probes: int | None = None,
    minimum_distinct_identities: int = MINIMUM_DISTINCT_IDENTITIES,
    minimum_captures_per_identity: int = MINIMUM_CAPTURES_PER_IDENTITY,
) -> tuple[FingerprintDecisionPolicy, FingerprintValidationReport]:
    return validate_fingerprint_policy(
        reference_set=reference_set if reference_set is not None else case.reference_set,
        snapshots=snapshots if snapshots is not None else case.snapshots,
        suite=case.suite,
        policy_id="fixture-policy",
        policy_version="0.1.0",
        minimum_comparable_probes=(
            minimum_comparable_probes if minimum_comparable_probes is not None else len(case.suite.probes)
        ),
        max_far_target=max_far_target,
        minimum_distinct_identities=minimum_distinct_identities,
        minimum_captures_per_identity=minimum_captures_per_identity,
    )


def _by_snapshot_id(report: FingerprintValidationReport) -> dict[str, FingerprintHoldoutOutcome]:
    return {outcome.snapshot_id: outcome for outcome in report.outcomes}


@pytest.fixture
def distinct_case(tmp_path: Path, suite: FingerprintSuite) -> _Case:
    return _publish(tmp_path, suite, _DISTINCT_BEHAVIOUR_SPECS)


@pytest.fixture
def drifting_case(tmp_path: Path, suite: FingerprintSuite) -> _Case:
    return _publish(tmp_path, suite, _DRIFTING_SPECS)


@pytest.fixture
def duplicate_behaviour_case(tmp_path: Path, suite: FingerprintSuite) -> _Case:
    return _publish(tmp_path, suite, _DUPLICATE_BEHAVIOUR_SPECS)


# ---------------------------------------------------------------------------
# select_distance_threshold
# ---------------------------------------------------------------------------


class TestSelectDistanceThreshold:
    def test_maximises_true_positive_rate_within_the_budget(self) -> None:
        selection = select_distance_threshold([0.1, 0.2], [0.7, 0.8], max_far=0.0)

        # 0.1 gives TPR 0.5 / FAR 0.0; 0.2 gives TPR 1.0 / FAR 0.0 and wins.
        assert selection == pytest.approx((0.2, 1.0, 0.0))

    def test_true_positive_rate_is_the_share_of_same_within_threshold(self) -> None:
        selection = select_distance_threshold([0.1, 0.2, 0.3], [0.9], max_far=1.0)

        # Every same distance is at or below 0.3; the 0.9 candidate adds FAR 1.0
        # without improving TPR, so the conservative boundary is kept.
        assert selection == pytest.approx((0.3, 1.0, 0.0))

    def test_false_accept_rate_is_the_share_of_impostors_within_threshold(self) -> None:
        selection = select_distance_threshold([0.5], [0.4, 0.6], max_far=0.5)

        # 0.4 -> TPR 0.0 / FAR 0.5 (budget exactly met); 0.5 -> TPR 1.0 / FAR 0.5
        # wins; 0.6 would push FAR to 1.0 and is rejected.
        assert selection == pytest.approx((0.5, 1.0, 0.5))

    def test_false_accept_budget_can_cost_true_positive_rate(self) -> None:
        selection = select_distance_threshold([0.1, 0.6], [0.2], max_far=0.0)

        # Accepting the 0.6 same distance would also accept the 0.2 impostor.
        assert selection == pytest.approx((0.1, 0.5, 0.0))

    def test_threshold_is_inclusive_of_distances_on_the_boundary(self) -> None:
        assert select_distance_threshold([0.5], [0.5], max_far=0.0) is None
        assert select_distance_threshold([0.5], [0.5], max_far=1.0) == pytest.approx((0.5, 1.0, 1.0))

    def test_returns_none_when_no_candidate_satisfies_the_budget(self) -> None:
        assert select_distance_threshold([0.9], [0.0], max_far=0.0) is None

    @pytest.mark.parametrize(
        ("same", "impostor"),
        [([], [0.1]), ([0.1], []), ([], [])],
    )
    def test_returns_none_for_empty_inputs(self, same: list[float], impostor: list[float]) -> None:
        assert select_distance_threshold(same, impostor, max_far=1.0) is None

    def test_is_deterministic_and_order_independent(self) -> None:
        first = select_distance_threshold([0.2, 0.6], [0.3, 0.4], max_far=1.0)
        shuffled = select_distance_threshold([0.6, 0.2], [0.4, 0.3], max_far=1.0)

        # TPR reaches 1.0 only at 0.6, which also admits both impostors (FAR 1.0).
        assert first == pytest.approx((0.6, 1.0, 1.0))
        assert shuffled == first

    @pytest.mark.parametrize("max_far", [-0.1, 1.1])
    def test_rejects_out_of_range_budget(self, max_far: float) -> None:
        with pytest.raises(FingerprintValidationError, match="max_far must be in"):
            select_distance_threshold([0.1], [0.2], max_far=max_far)

    @pytest.mark.parametrize(
        ("same", "impostor"),
        [([1.5], [0.1]), ([0.1], [-0.1])],
    )
    def test_rejects_distances_outside_the_unit_interval(self, same: list[float], impostor: list[float]) -> None:
        with pytest.raises(FingerprintValidationError, match="distances must be in"):
            select_distance_threshold(same, impostor, max_far=1.0)


# ---------------------------------------------------------------------------
# Leave-one-capture-out outcomes
# ---------------------------------------------------------------------------


class TestHeldOutOutcomes:
    def test_every_capture_is_held_out_exactly_once(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)

        assert len(report.outcomes) == len(distinct_case.snapshots) == report.capture_count == 10
        assert {outcome.snapshot_id for outcome in report.outcomes} == {
            snapshot.snapshot_id for snapshot in distinct_case.snapshots
        }

    def test_held_out_capture_is_excluded_from_its_own_reference(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)
        outcomes = _by_snapshot_id(report)

        # model-a's two captures share no answer; scoring either one against a
        # library that still contained it would report ~0.311 instead of 1.0.
        assert outcomes["model-a-1"].own_distance == pytest.approx(1.0)
        assert outcomes["model-a-2"].own_distance == pytest.approx(1.0)

    def test_identical_captures_have_zero_own_distance(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)
        outcomes = _by_snapshot_id(report)

        for snapshot_id in ("model-b-1", "model-b-2", "model-c-1", "model-c-2", "model-d-1", "model-e-2"):
            assert outcomes[snapshot_id].own_distance == pytest.approx(0.0)

    def test_closest_identity_is_ranked_first(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)
        outcomes = _by_snapshot_id(report)

        for snapshot_id in ("model-b-1", "model-c-1", "model-d-1", "model-e-1"):
            outcome = outcomes[snapshot_id]
            assert outcome.own_distance < outcome.nearest_impostor_distance
            assert outcome.top1_correct is True

    def test_accuracy_is_the_mean_of_the_outcomes(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)
        outcomes = report.outcomes

        assert report.top1_accuracy == pytest.approx(sum(o.top1_correct for o in outcomes) / len(outcomes))
        assert report.top3_accuracy == pytest.approx(sum(o.top3_correct for o in outcomes) / len(outcomes))
        assert report.top1_accuracy == pytest.approx(0.8)
        assert report.top3_accuracy == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Validated policy
# ---------------------------------------------------------------------------


class TestValidatedPolicy:
    def test_publishes_threshold_and_rates(self, distinct_case: _Case) -> None:
        policy, _ = _validate(distinct_case)

        assert policy.validated is True
        assert policy.distance_threshold == pytest.approx(0.0)
        assert policy.true_positive_rate == pytest.approx(0.8)
        assert policy.false_accept_rate == pytest.approx(0.0)
        assert policy.identity_count == 5
        assert policy.held_out_capture_count == 10

    def test_report_records_the_minimum_conditions_it_met(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)

        assert report.meets_minimum_conditions is True
        assert report.insufficient_reason is None
        assert report.identity_count == 5
        assert report.capture_count == 10
        assert report.minimum_distinct_identities == MINIMUM_DISTINCT_IDENTITIES == 5
        assert report.minimum_captures_per_identity == MINIMUM_CAPTURES_PER_IDENTITY == 2
        assert report.minimum_comparable_probes == len(distinct_case.suite.probes)
        assert report.max_far_target == pytest.approx(0.05)

    def test_policy_is_bound_to_the_set_and_suite_it_was_validated_on(self, distinct_case: _Case) -> None:
        policy, report = _validate(distinct_case)

        assert policy.fingerprint_set_id == distinct_case.reference_set.fingerprint_set_id
        assert policy.fingerprint_set_content_sha256 == distinct_case.reference_set.content_sha256
        assert policy.suite_content_sha256 == distinct_case.suite.content_sha256
        assert report.fingerprint_set_content_sha256 == distinct_case.reference_set.content_sha256
        assert report.suite_content_sha256 == distinct_case.suite.content_sha256

    def test_validation_is_deterministic(self, distinct_case: _Case) -> None:
        first_policy, first_report = _validate(distinct_case)
        second_policy, second_report = _validate(distinct_case)

        assert first_policy.distance_threshold == second_policy.distance_threshold
        assert first_policy.true_positive_rate == second_policy.true_positive_rate
        assert first_policy.false_accept_rate == second_policy.false_accept_rate
        assert first_report.outcomes == second_report.outcomes
        assert first_report.top1_accuracy == second_report.top1_accuracy

    def test_temporal_baseline_travels_with_a_validated_policy(self, drifting_case: _Case) -> None:
        policy, report = _validate(drifting_case)

        assert policy.validated is True
        assert len(report.within_reference_temporal_distances) == 10
        assert max(report.within_reference_temporal_distances) == pytest.approx(1.0)
        assert policy.temporal_divergence_baseline == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fail-closed minimum conditions
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_fewer_than_five_identities_is_unvalidated(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        case = _publish(tmp_path, suite, _FOUR_IDENTITY_SPECS)
        policy, report = _validate(case)

        assert policy.validated is False
        assert policy.distance_threshold is None
        assert policy.true_positive_rate is None
        assert policy.false_accept_rate is None
        assert policy.held_out_capture_count == 0
        assert report.meets_minimum_conditions is False
        assert report.outcomes == ()
        assert report.top1_accuracy is None
        assert report.top3_accuracy is None
        assert report.identity_count == 4
        assert report.capture_count == 8
        assert report.insufficient_reason is not None
        assert "distinct model identities" in report.insufficient_reason

    def test_one_capture_per_identity_is_unvalidated(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        case = _publish(tmp_path, suite, _ONE_CAPTURE_SPECS)
        policy, report = _validate(case)

        assert policy.validated is False
        assert policy.distance_threshold is None
        assert report.meets_minimum_conditions is False
        assert report.outcomes == ()
        assert report.identity_count == 5
        assert report.capture_count == 5
        assert report.insufficient_reason is not None
        assert "independent captures" in report.insufficient_reason

    def test_no_acceptable_threshold_is_unvalidated_but_conditions_held(self, duplicate_behaviour_case: _Case) -> None:
        policy, report = _validate(duplicate_behaviour_case)

        # The set is large enough, so held-out validation did run and did record
        # the measured distances — it just could not honour the FAR budget.
        assert report.meets_minimum_conditions is True
        assert report.capture_count == 10
        assert report.outcomes != ()
        assert len(report.within_reference_temporal_distances) == 10
        assert report.insufficient_reason is not None
        assert "no candidate threshold satisfies" in report.insufficient_reason

        assert policy.validated is False
        assert policy.distance_threshold is None
        # A baseline is a threshold-adjacent claim: it does not ship without one.
        assert policy.temporal_divergence_baseline is None


# ---------------------------------------------------------------------------
# Fail-closed inputs
# ---------------------------------------------------------------------------


class TestFailClosedInputs:
    def test_minimum_comparable_probes_must_be_positive(self, distinct_case: _Case) -> None:
        with pytest.raises(FingerprintValidationError, match="minimum_comparable_probes"):
            _validate(distinct_case, minimum_comparable_probes=0)

    def test_minimum_condition_parameters_must_be_positive(self, distinct_case: _Case) -> None:
        with pytest.raises(FingerprintValidationError, match="minimum validation conditions"):
            _validate(distinct_case, minimum_distinct_identities=0)

    def test_missing_snapshot_is_rejected(self, distinct_case: _Case) -> None:
        with pytest.raises(FingerprintValidationError, match="disagree"):
            _validate(distinct_case, snapshots=distinct_case.snapshots[:-1])

    def test_unexpected_snapshot_is_rejected(self, distinct_case: _Case) -> None:
        extra = _capture(
            distinct_case.suite,
            snapshot_id="model-f-1",
            model_id="model-f",
            rounds=_constant(_ALL_ONE),
        )

        with pytest.raises(FingerprintValidationError, match="disagree"):
            _validate(distinct_case, snapshots=(*distinct_case.snapshots, extra))

    def test_identity_mismatch_between_set_and_snapshot_is_rejected(self, distinct_case: _Case) -> None:
        renamed = _capture(
            distinct_case.suite,
            snapshot_id="model-a-1",
            model_id="renamed-model",
            rounds=_constant((0, 0, 0, 0, 0, 0)),
        )

        with pytest.raises(FingerprintValidationError, match="is recorded as"):
            _validate(distinct_case, snapshots=(renamed, *distinct_case.snapshots[1:]))

    def test_set_captured_with_another_suite_is_rejected(self, distinct_case: _Case) -> None:
        provisional = distinct_case.reference_set.model_copy(
            update={"suite_content_sha256": "f" * 64, "content_sha256": "0" * 64}
        )
        tampered = provisional.model_copy(update={"content_sha256": provisional.compute_content_sha256()})
        tampered.verify_content_hash()

        with pytest.raises(FingerprintValidationError, match="not the suite being validated"):
            _validate(distinct_case, reference_set=tampered)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class TestValidationRecords:
    def test_report_and_outcomes_are_frozen(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)

        with pytest.raises(ValueError, match="frozen"):
            report.top1_accuracy = 1.0  # type: ignore[misc]

        with pytest.raises(ValueError, match="frozen"):
            report.outcomes[0].top1_correct = False  # type: ignore[misc]

    def test_outcome_rejects_unknown_fields(self, distinct_case: _Case) -> None:
        _, report = _validate(distinct_case)
        outcome = report.outcomes[0]

        with pytest.raises(ValueError, match="extra_forbidden"):
            outcome.__class__(**outcome.model_dump(), unexpected="nope")
