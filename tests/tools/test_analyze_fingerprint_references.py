"""Task 29 — tests for tools/analyze_fingerprint_references.py.

The analysis tool is read-only research code over the fingerprint snapshot
repository: these tests cover the Task 29 checklist — within/between splits,
None summaries, invalid rates, per-probe separation, deterministic ordering,
tampered-snapshot rejection and the guarantee that fixture data can never be
mistaken for a real campaign. No test here touches a real endpoint.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.fingerprint.distance import compute_fingerprint_distance
from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    FingerprintSampleObservation,
    FingerprintSourceRole,
)
from llmtrace.fingerprint.policy import build_fingerprint_policy
from llmtrace.fingerprint.reference import (
    FingerprintReferenceError,
    build_fingerprint_snapshot,
)
from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.suite import default_fingerprint_suite_path, load_fingerprint_suite

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "analyze_fingerprint_references.py"
_spec = importlib.util.spec_from_file_location("analyze_fingerprint_references", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
sys.modules["analyze_fingerprint_references"] = tool
_spec.loader.exec_module(tool)

SUITE = load_fingerprint_suite()
CAPTURED_AT = datetime(2026, 9, 13, tzinfo=UTC)


def observations_for(
    *,
    choice_index: int,
    repetitions: int,
    prefix: str,
    invalid_rounds: frozenset[int] = frozenset(),
) -> tuple[FingerprintSampleObservation, ...]:
    """Deterministic observations: one answer per (probe, round), some INVALID."""
    observations: list[FingerprintSampleObservation] = []
    for round_index in range(repetitions):
        for sequence_index, probe in enumerate(SUITE.probes):
            invalid = round_index in invalid_rounds
            observations.append(
                FingerprintSampleObservation(
                    probe_id=probe.probe_id,
                    sequence_index=sequence_index,
                    round_index=round_index,
                    outcome=INVALID_OUTCOME if invalid else probe.choices[choice_index],
                    valid=not invalid,
                    response_body_sha256="a" * 64,
                    evidence_ref=f"{prefix}:{probe.probe_id}:{round_index}",
                )
            )
    return tuple(observations)


def make_snapshot(
    *,
    snapshot_id: str,
    model_id: str,
    choice_index: int,
    repetitions: int = 8,
    provider_id: str = "official-provider",
    role: FingerprintSourceRole = FingerprintSourceRole.OFFICIAL_BASELINE,
    invalid_rounds: frozenset[int] = frozenset(),
    suite=None,
):
    """A trusted-reference-style capture that always answers one fixed choice."""
    return build_fingerprint_snapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        source_role=role,
        suite=suite or SUITE,
        repetitions=repetitions,
        observations=observations_for(
            choice_index=choice_index,
            repetitions=repetitions,
            prefix=snapshot_id,
            invalid_rounds=invalid_rounds,
        ),
        captured_at=CAPTURED_AT,
    )


def loaded_repository(tmp_path: Path) -> FingerprintRepository:
    return FingerprintRepository.load(data_root=tmp_path / "data")


# ---------------------------------------------------------------------------
# Task 29 checklist: within / between splits
# ---------------------------------------------------------------------------


def test_same_identity_pair_counts_as_within() -> None:
    left = make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0)
    right = make_snapshot(snapshot_id="a-02", model_id="model-a", choice_index=0)

    split = tool.split_pairwise_distances([left, right], suite=SUITE)

    assert split.within == [pytest.approx(0.0)]
    assert split.between == []
    assert split.skipped == []


def test_different_identity_pair_counts_as_between() -> None:
    left = make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0)
    right = make_snapshot(snapshot_id="b-01", model_id="model-b", choice_index=1)

    split = tool.split_pairwise_distances([left, right], suite=SUITE)

    assert split.within == []
    assert len(split.between) == 1
    assert split.between[0] == pytest.approx(1.0)


def test_same_model_different_provider_is_between() -> None:
    """Identity is (provider_id, model_id): same model via another provider differs (Task 6)."""
    left = make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0, provider_id="p1")
    right = make_snapshot(snapshot_id="a-02", model_id="model-a", choice_index=0, provider_id="p2")

    split = tool.split_pairwise_distances([left, right], suite=SUITE)

    assert split.within == []
    assert split.between == [pytest.approx(0.0)]


def test_pair_distance_reuses_production_distance() -> None:
    left = make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0)
    right = make_snapshot(
        snapshot_id="b-01",
        model_id="model-b",
        choice_index=1,
        invalid_rounds=frozenset({0, 1}),
    )

    production = compute_fingerprint_distance(
        probes=SUITE.probes,
        candidate=left.distributions,
        reference=right.distributions,
        minimum_comparable_probes=len(SUITE.probes),
    )

    assert tool.pair_distance(left, right, suite=SUITE) == pytest.approx(production.distance)


# ---------------------------------------------------------------------------
# Task 29 checklist: None summaries
# ---------------------------------------------------------------------------


def test_no_within_pairs_yields_none_summary() -> None:
    left = make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0)
    right = make_snapshot(snapshot_id="b-01", model_id="model-b", choice_index=1)

    split = tool.split_pairwise_distances([left, right], suite=SUITE)
    summary = tool.separation_summary(split.within, split.between)

    assert split.between == [pytest.approx(1.0)]
    assert all(value is None for value in summary.values())


def test_no_between_pairs_yields_none_summary() -> None:
    left = make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0)
    right = make_snapshot(snapshot_id="a-02", model_id="model-a", choice_index=0)

    split = tool.split_pairwise_distances([left, right], suite=SUITE)
    summary = tool.separation_summary(split.within, split.between)

    assert split.within == [pytest.approx(0.0)]
    assert all(value is None for value in summary.values())


# ---------------------------------------------------------------------------
# Task 29 checklist: invalid rate + per-probe separation
# ---------------------------------------------------------------------------


def test_probe_invalid_rate() -> None:
    snapshot = make_snapshot(
        snapshot_id="a-01",
        model_id="model-a",
        choice_index=0,
        repetitions=8,
        invalid_rounds=frozenset({0, 1, 2, 3}),
    )

    assert tool.invalid_rate(snapshot, "symbol-choice") == pytest.approx(0.5)
    assert tool.invalid_rate(snapshot, "prime-choice") == pytest.approx(0.5)
    assert tool.invalid_rate(snapshot, "missing-probe") is None


def test_aggregate_invalid_rate_across_snapshots() -> None:
    first = make_snapshot(
        snapshot_id="a-01",
        model_id="model-a",
        choice_index=0,
        invalid_rounds=frozenset({0, 1}),
    )
    second = make_snapshot(
        snapshot_id="a-02",
        model_id="model-a",
        choice_index=0,
        invalid_rounds=frozenset(),
    )

    # 2 invalid out of 16 samples per probe.
    assert tool.aggregate_invalid_rate([first, second], "symbol-choice") == pytest.approx(0.125)


def test_per_probe_separation() -> None:
    identities = [
        make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0),
        make_snapshot(snapshot_id="a-02", model_id="model-a", choice_index=0),
        make_snapshot(snapshot_id="b-01", model_id="model-b", choice_index=1),
        make_snapshot(snapshot_id="b-02", model_id="model-b", choice_index=1),
    ]

    split = tool.split_pairwise_distances(identities, suite=SUITE)

    for probe in SUITE.probes:
        within = split.probe_within[probe.probe_id]
        between = split.probe_between[probe.probe_id]
        assert within == [pytest.approx(0.0), pytest.approx(0.0)]
        assert between == [pytest.approx(1.0)] * 4
        assert tool.probe_separation_score(within, between) == pytest.approx(1.0)


def test_probe_separation_score_returns_none_without_data() -> None:
    assert tool.probe_separation_score([], [1.0]) is None
    assert tool.probe_separation_score([0.0], []) is None


# ---------------------------------------------------------------------------
# Task 29 checklist: deterministic ordering
# ---------------------------------------------------------------------------


def test_deterministic_ordering(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    for name in ("snap-c", "snap-a", "snap-b"):
        repository.snapshots.save(make_snapshot(snapshot_id=name, model_id="model-a", choice_index=0))

    first = tool.build_analysis(repository, campaign_id="ordering")
    second = tool.build_analysis(repository, campaign_id="ordering")

    assert first["included_snapshot_ids"] == ["snap-a", "snap-b", "snap-c"]
    assert first["included_snapshot_ids"] == second["included_snapshot_ids"]
    assert first["probes"] == second["probes"]
    assert first["pairwise"] == second["pairwise"]
    assert first["repetitions_groups"] == second["repetitions_groups"]


# ---------------------------------------------------------------------------
# Task 29 checklist: tampered snapshot rejected
# ---------------------------------------------------------------------------


def test_tampered_snapshot_rejected(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    repository.snapshots.save(make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0))

    snapshot_path = tmp_path / "data" / "fingerprints" / "snapshots" / "a-01.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["model_id"] = "tampered-model"
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fresh = FingerprintRepository.load(data_root=tmp_path / "data")
    with pytest.raises(FingerprintReferenceError):
        tool.build_analysis(fresh, campaign_id="tamper")


# ---------------------------------------------------------------------------
# Task 29 checklist: fixture data cannot be mistaken for a real campaign
# ---------------------------------------------------------------------------


def test_fixture_and_candidate_roles_are_excluded(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    repository.snapshots.save(
        make_snapshot(
            snapshot_id="fixture-01",
            model_id="fixture-model",
            choice_index=0,
            role=FingerprintSourceRole.TEST_FIXTURE,
        )
    )
    repository.snapshots.save(
        make_snapshot(
            snapshot_id="candidate-01",
            model_id="candidate-model",
            choice_index=0,
            role=FingerprintSourceRole.CANDIDATE_CAPTURE,
        )
    )
    repository.snapshots.save(make_snapshot(snapshot_id="real-01", model_id="real-model", choice_index=0))

    analysis = tool.build_analysis(repository, campaign_id="roles")

    assert analysis["included_snapshot_ids"] == ["real-01"]
    excluded = {entry["snapshot_id"]: entry["reason"] for entry in analysis["excluded_snapshots"]}
    assert set(excluded) == {"fixture-01", "candidate-01"}
    assert excluded["fixture-01"] == "non_reference_source_role:test_fixture"
    assert excluded["candidate-01"] == "non_reference_source_role:candidate_capture"


def test_suite_mismatch_snapshots_are_excluded(tmp_path: Path) -> None:
    variant_path = tmp_path / "fingerprint_v1_variant.json"
    payload = json.loads(default_fingerprint_suite_path().read_text(encoding="utf-8"))
    payload["probes"][0]["prompt"] += " (variant)"
    variant_path.write_text(json.dumps(payload), encoding="utf-8")
    variant_suite = load_fingerprint_suite(variant_path)

    repository = loaded_repository(tmp_path)
    repository.snapshots.save(
        make_snapshot(snapshot_id="variant-01", model_id="model-a", choice_index=0, suite=variant_suite)
    )
    repository.snapshots.save(make_snapshot(snapshot_id="real-01", model_id="model-a", choice_index=0))

    analysis = tool.build_analysis(repository, campaign_id="suite-mismatch")

    assert analysis["included_snapshot_ids"] == ["real-01"]
    excluded = {entry["snapshot_id"]: entry["reason"] for entry in analysis["excluded_snapshots"]}
    assert excluded == {"variant-01": "suite_mismatch"}


# ---------------------------------------------------------------------------
# Campaign-level behaviour (Task 21 / Task 31)
# ---------------------------------------------------------------------------


def test_build_analysis_reports_partial_below_minimum(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    for model, choice in (("model-a", 0), ("model-b", 1)):
        for capture in (1, 2):
            repository.snapshots.save(
                make_snapshot(
                    snapshot_id=f"fp-{model}-capture-{capture:02d}",
                    model_id=model,
                    choice_index=choice,
                )
            )

    analysis = tool.build_analysis(repository, campaign_id="partial-campaign")

    assert analysis["identity_count"] == 2
    assert analysis["capture_count"] == 4
    assert analysis["campaign_completeness"]["meets_minimum"] is False
    assert analysis["pairwise"]["within_mean"] == pytest.approx(0.0)
    assert analysis["pairwise"]["between_mean"] == pytest.approx(1.0)
    assert analysis["pairwise"]["conservative_margin"] == pytest.approx(1.0)
    assert analysis["validation"]["validated"] is None
    for probe in analysis["probes"]:
        assert probe["classification"] == "PROMISING"


def test_write_reports_round_trip(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    repository.snapshots.save(make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0))

    analysis = tool.build_analysis(repository, campaign_id="round-trip")
    json_path, markdown_path = tool.write_reports(analysis, tmp_path / "reports")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "round-trip"
    assert payload["suite"]["id"] == SUITE.suite_id
    assert payload["capture_count"] == 1
    assert payload["notes"][0] == (
        "These thresholds are exploratory labels only and are not production identity decision thresholds."
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Fingerprint Reference Analysis — round-trip" in markdown
    assert "### 10. Next step: KEEP / TUNE / REDESIGN?" in markdown
    assert "PARTIAL campaign per Task 31" in markdown


def test_main_writes_reports_and_exits_zero(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    repository.snapshots.save(make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0))
    output_dir = tmp_path / "reports"

    exit_code = tool.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--campaign-id",
            "cli-campaign",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "analysis.json").exists()
    assert (output_dir / "analysis.md").exists()


def test_main_rejects_policy_without_version(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tool.main(["--policy-id", "some-policy"])


def test_validation_summary_reads_existing_policy(tmp_path: Path) -> None:
    repository = loaded_repository(tmp_path)
    repository.snapshots.save(make_snapshot(snapshot_id="a-01", model_id="model-a", choice_index=0))
    repository.policies.save(
        build_fingerprint_policy(
            policy_id="research-policy",
            policy_version="1.0.0",
            fingerprint_set_id="research-set",
            fingerprint_set_content_sha256="b" * 64,
            suite_content_sha256=SUITE.content_sha256,
            validated=False,
            minimum_comparable_probes=len(SUITE.probes),
            max_far_target=0.01,
            identity_count=1,
            held_out_capture_count=1,
        )
    )

    analysis = tool.build_analysis(
        repository,
        campaign_id="with-policy",
        policy_id="research-policy",
        policy_version="1.0.0",
    )

    assert analysis["validation"]["policy_id"] == "research-policy"
    assert analysis["validation"]["validated"] is False
    assert analysis["validation"]["threshold"] is None
