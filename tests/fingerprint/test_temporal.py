"""Temporal fingerprint windows and the reference temporal baseline (Task 29 / Task 30).

Every input here is fixture data — deterministic observations, no network, no
writes outside the fixtures.  The tests pin three fail-closed properties:

* a capture with too few rounds is ``UNAVAILABLE`` and carries no number
  ("no drift estimate from two samples");
* windows that are not comparable never yield a divergence;
* the reference baseline is the *maximum* within-reference drift and is
  ``None`` — not 0 — when no capture could contribute one.
"""

from __future__ import annotations

import pytest

from llmtrace.fingerprint.models import INVALID_OUTCOME, FingerprintSuite, ProbeDistribution
from llmtrace.fingerprint.reference import FingerprintReferenceError
from llmtrace.fingerprint.temporal import (
    MINIMUM_ROUNDS_PER_WINDOW,
    MINIMUM_TEMPORAL_REPETITIONS,
    FingerprintTemporalError,
    TemporalFingerprint,
    TemporalFingerprintStatus,
    TemporalWindow,
    build_temporal_fingerprint,
    collect_within_reference_temporal_distances,
    split_temporal_windows,
    temporal_divergence_baseline,
)

from .conftest import MODEL_ID, make_reference_snapshot, make_reference_snapshot_with_rounds

#: First four rounds answer choice 0, last four choice 1 — a full flip.
FULL_FLIP = (0, 0, 0, 0, 1, 1, 1, 1)
#: Every round answers the same choice — one identity that does not drift.
NO_FLIP = (0,) * 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _distribution(probe_id: str, *, sample_count: int = 4, invalid_count: int = 0) -> ProbeDistribution:
    valid = sample_count - invalid_count
    return ProbeDistribution(
        probe_id=probe_id,
        support=("a", "b", INVALID_OUTCOME),
        counts={"a": valid, "b": 0, INVALID_OUTCOME: invalid_count},
        probabilities={
            "a": valid / sample_count if sample_count else 0.0,
            "b": 0.0,
            INVALID_OUTCOME: invalid_count / sample_count if sample_count else 0.0,
        },
        sample_count=sample_count,
        invalid_count=invalid_count,
    )


def _window(
    window_id: str = "rounds_0_1",
    *,
    start: int = 0,
    end: int = 2,
    probe_ids: tuple[str, ...] = ("probe-a",),
) -> TemporalWindow:
    return TemporalWindow(
        window_id=window_id,
        round_start=start,
        round_end=end,
        distributions=tuple(_distribution(probe_id) for probe_id in probe_ids),
    )


@pytest.fixture
def available_temporal(suite: FingerprintSuite) -> TemporalFingerprint:
    """A real, ``AVAILABLE`` temporal fingerprint (an undrifted capture)."""
    snapshot = make_reference_snapshot(
        suite=suite,
        snapshot_id="ref-stable",
        model_id=MODEL_ID,
        repetitions=MINIMUM_TEMPORAL_REPETITIONS * 2,
    )
    result = build_temporal_fingerprint(
        snapshot=snapshot,
        suite=suite,
        minimum_comparable_probes=len(suite.probes),
    )
    assert result.status is TemporalFingerprintStatus.AVAILABLE
    return result


# ---------------------------------------------------------------------------
# split_temporal_windows
# ---------------------------------------------------------------------------


class TestSplitTemporalWindows:
    def test_standard_repetitions_split_into_two_halves(self) -> None:
        assert split_temporal_windows(8) == ((0, 4), (4, 8))

    def test_minimum_repetitions_is_enough(self) -> None:
        assert split_temporal_windows(4) == ((0, 2), (2, 4))

    def test_odd_rounds_leave_the_remainder_in_the_second_window(self) -> None:
        assert split_temporal_windows(7) == ((0, 3), (3, 7))

    @pytest.mark.parametrize("repetitions", [0, 1, 2, 3])
    def test_below_the_minimum_yields_no_windows(self, repetitions: int) -> None:
        assert split_temporal_windows(repetitions) is None

    @pytest.mark.parametrize("repetitions", range(4, 18))
    def test_every_window_always_spans_the_minimum_rounds(self, repetitions: int) -> None:
        split = split_temporal_windows(repetitions)
        assert split is not None
        first, second = split
        assert first[0] == 0
        assert second[1] == repetitions
        assert first[1] == second[0]
        assert first[1] - first[0] >= MINIMUM_ROUNDS_PER_WINDOW
        assert second[1] - second[0] >= MINIMUM_ROUNDS_PER_WINDOW


# ---------------------------------------------------------------------------
# TemporalWindow / TemporalFingerprint validators
# ---------------------------------------------------------------------------


class TestTemporalWindowModel:
    def test_round_count_is_the_open_interval_span(self) -> None:
        assert _window(start=4, end=8).round_count == 4

    def test_a_window_shorter_than_the_minimum_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must span at least"):
            _window(start=0, end=1)

    def test_a_window_without_distributions_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one probe distribution"):
            TemporalWindow(window_id="rounds_0_1", round_start=0, round_end=2, distributions=())

    def test_duplicate_probe_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate probe_id"):
            _window(probe_ids=("probe-a", "probe-a"))


class TestTemporalFingerprintModel:
    def test_available_requires_exactly_two_windows(self, available_temporal: TemporalFingerprint) -> None:
        with pytest.raises(ValueError, match="exactly 2 windows"):
            TemporalFingerprint(
                status=TemporalFingerprintStatus.AVAILABLE,
                repetitions=available_temporal.repetitions,
                windows=available_temporal.windows[:1],
                divergence=available_temporal.divergence,
            )

    def test_available_requires_its_divergence(self, available_temporal: TemporalFingerprint) -> None:
        with pytest.raises(ValueError, match="aggregate divergence"):
            TemporalFingerprint(
                status=TemporalFingerprintStatus.AVAILABLE,
                repetitions=available_temporal.repetitions,
                windows=available_temporal.windows,
                divergence=None,
            )

    def test_available_must_not_carry_an_unavailability_reason(self, available_temporal: TemporalFingerprint) -> None:
        with pytest.raises(ValueError, match="unavailability reason"):
            TemporalFingerprint(
                status=TemporalFingerprintStatus.AVAILABLE,
                repetitions=available_temporal.repetitions,
                windows=available_temporal.windows,
                divergence=available_temporal.divergence,
                reason="but also this",
            )

    def test_unavailable_must_not_carry_windows(self, available_temporal: TemporalFingerprint) -> None:
        with pytest.raises(ValueError, match="must not carry windows"):
            TemporalFingerprint(
                status=TemporalFingerprintStatus.UNAVAILABLE,
                repetitions=available_temporal.repetitions,
                windows=available_temporal.windows,
                reason="too few rounds",
            )

    def test_unavailable_must_not_carry_a_divergence(self, available_temporal: TemporalFingerprint) -> None:
        with pytest.raises(ValueError, match="must not carry a divergence"):
            TemporalFingerprint(
                status=TemporalFingerprintStatus.UNAVAILABLE,
                repetitions=available_temporal.repetitions,
                divergence=available_temporal.divergence,
                reason="too few rounds",
            )

    def test_unavailable_must_explain_itself(self) -> None:
        with pytest.raises(ValueError, match="via reason"):
            TemporalFingerprint(status=TemporalFingerprintStatus.UNAVAILABLE, repetitions=2)


# ---------------------------------------------------------------------------
# build_temporal_fingerprint
# ---------------------------------------------------------------------------


class TestBuildTemporalFingerprint:
    def test_undrifted_capture_has_zero_divergence(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot_with_rounds(
            suite=suite,
            snapshot_id="ref-undrifted",
            model_id=MODEL_ID,
            choice_index_by_round=NO_FLIP,
        )

        temporal = build_temporal_fingerprint(
            snapshot=snapshot,
            suite=suite,
            minimum_comparable_probes=len(suite.probes),
        )

        assert temporal.status is TemporalFingerprintStatus.AVAILABLE
        assert temporal.divergence is not None
        assert temporal.divergence.distance == pytest.approx(0.0)
        assert temporal.divergence.comparable_probes == len(suite.probes)
        assert [window.round_count for window in temporal.windows] == [4, 4]
        assert [window.window_id for window in temporal.windows] == ["rounds_0_3", "rounds_4_7"]

    def test_capture_that_flips_between_windows_diverges(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot_with_rounds(
            suite=suite,
            snapshot_id="ref-flipped",
            model_id=MODEL_ID,
            choice_index_by_round=FULL_FLIP,
        )

        temporal = build_temporal_fingerprint(
            snapshot=snapshot,
            suite=suite,
            minimum_comparable_probes=len(suite.probes),
        )

        assert temporal.status is TemporalFingerprintStatus.AVAILABLE
        assert temporal.divergence is not None
        # Every probe moved from one point mass to a different one → JSD 1.0.
        assert temporal.divergence.distance == pytest.approx(1.0)

    def test_too_few_rounds_is_unavailable_and_carries_no_number(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot(
            suite=suite,
            snapshot_id="ref-short",
            model_id=MODEL_ID,
            repetitions=MINIMUM_TEMPORAL_REPETITIONS - 1,
        )

        temporal = build_temporal_fingerprint(
            snapshot=snapshot,
            suite=suite,
            minimum_comparable_probes=len(suite.probes),
        )

        assert temporal.status is TemporalFingerprintStatus.UNAVAILABLE
        assert temporal.windows == ()
        assert temporal.divergence is None
        assert temporal.reason is not None
        assert "two temporal windows" in temporal.reason
        assert str(MINIMUM_TEMPORAL_REPETITIONS) in temporal.reason

    def test_incomparable_windows_are_unavailable(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot(
            suite=suite,
            snapshot_id="ref-undrifted",
            model_id=MODEL_ID,
            repetitions=MINIMUM_TEMPORAL_REPETITIONS * 2,
        )

        temporal = build_temporal_fingerprint(
            snapshot=snapshot,
            suite=suite,
            minimum_comparable_probes=len(suite.probes) + 1,
        )

        assert temporal.status is TemporalFingerprintStatus.UNAVAILABLE
        assert temporal.divergence is None
        assert temporal.reason is not None
        assert "not comparable" in temporal.reason

    def test_minimum_comparable_probes_must_be_positive(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot_with_rounds(
            suite=suite,
            snapshot_id="ref-undrifted",
            model_id=MODEL_ID,
            choice_index_by_round=NO_FLIP,
        )

        with pytest.raises(FingerprintTemporalError, match="minimum_comparable_probes"):
            build_temporal_fingerprint(snapshot=snapshot, suite=suite, minimum_comparable_probes=0)

    def test_capture_declared_against_another_suite_is_rejected(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot_with_rounds(
            suite=suite,
            snapshot_id="ref-other-suite",
            model_id=MODEL_ID,
            choice_index_by_round=NO_FLIP,
        )
        # Self-consistent but declared against a different suite version.
        incompatible = snapshot.model_copy(update={"suite_version": "9.9.9"})
        incompatible = incompatible.model_copy(update={"content_sha256": incompatible.compute_content_sha256()})

        with pytest.raises(FingerprintTemporalError, match="not compatible with suite"):
            build_temporal_fingerprint(
                snapshot=incompatible,
                suite=suite,
                minimum_comparable_probes=len(suite.probes),
            )

    def test_tampered_capture_fails_its_content_hash(self, suite: FingerprintSuite) -> None:
        snapshot = make_reference_snapshot_with_rounds(
            suite=suite,
            snapshot_id="ref-tampered",
            model_id=MODEL_ID,
            choice_index_by_round=NO_FLIP,
        )
        tampered = snapshot.model_copy(update={"model_id": "somebody-else"})

        with pytest.raises(FingerprintReferenceError, match="content hash mismatch"):
            build_temporal_fingerprint(
                snapshot=tampered,
                suite=suite,
                minimum_comparable_probes=len(suite.probes),
            )


# ---------------------------------------------------------------------------
# collect_within_reference_temporal_distances / temporal_divergence_baseline
# ---------------------------------------------------------------------------


class TestCollectWithinReferenceTemporalDistances:
    def test_sorted_by_snapshot_id_and_skips_captures_without_windows(self, suite: FingerprintSuite) -> None:
        snapshots = (
            make_reference_snapshot_with_rounds(
                suite=suite,
                snapshot_id="ref-b-flipped",
                model_id=MODEL_ID,
                choice_index_by_round=FULL_FLIP,
            ),
            make_reference_snapshot_with_rounds(
                suite=suite,
                snapshot_id="ref-a-undrifted",
                model_id=MODEL_ID,
                choice_index_by_round=NO_FLIP,
            ),
            make_reference_snapshot(
                suite=suite,
                snapshot_id="ref-c-too-short",
                model_id=MODEL_ID,
                repetitions=MINIMUM_TEMPORAL_REPETITIONS - 2,
            ),
        )

        distances = collect_within_reference_temporal_distances(
            snapshots=snapshots,
            suite=suite,
            minimum_comparable_probes=len(suite.probes),
        )

        # Ordered by snapshot_id, and the short capture contributed no number.
        assert distances == pytest.approx((0.0, 1.0))
        assert len(distances) == 2

    def test_no_snapshots_yield_no_distances(self, suite: FingerprintSuite) -> None:
        assert (
            collect_within_reference_temporal_distances(
                snapshots=(), suite=suite, minimum_comparable_probes=len(suite.probes)
            )
            == ()
        )


class TestTemporalDivergenceBaseline:
    def test_takes_the_maximum_of_the_within_reference_drifts(self) -> None:
        assert temporal_divergence_baseline((0.0, 1.0, 0.25)) == pytest.approx(1.0)

    def test_single_distance_is_its_own_baseline(self) -> None:
        assert temporal_divergence_baseline((0.3,)) == pytest.approx(0.3)

    def test_no_distances_yield_no_baseline(self) -> None:
        assert temporal_divergence_baseline(()) is None
