#!/usr/bin/env python3
"""Real fingerprint reference discrimination analysis (v0.6-R1, Task 14-22).

A read-only research tool over an existing fingerprint snapshot repository.
It exists to answer one question: does ``llmtrace-fingerprint-categorical``
separate real model identities far better than it separates repeated captures
of the same identity?

What it computes (Task 15-20):

* pairwise weighted JSD distances between snapshots, split into within-identity
  and between-identity groups (reusing the production
  ``compute_fingerprint_distance`` — never a second JSD implementation);
* a global separation summary including the conservative margin
  ``min(between) - max(within)``;
* per-probe within/between JSD, a separation score, invalid rates and an
  exploratory research label (PROMISING / WEAK / UNSTABLE / INVALID_HEAVY /
  INSUFFICIENT_DATA).

What it will never do:

* modify snapshots, reference sets or policies;
* re-select or suggest production decision thresholds;
* make HTTP requests.

Reports are written to ``<output-dir>/analysis.json`` and ``analysis.md``.
The markdown answers the ten research questions of Task 22 directly from the
data — including an explicit "insufficient data" whenever that is the honest
answer.

NOTE: reports may embed real endpoint identifiers (provider/model ids).
Committing them to a public repository is an Owner decision (Task 28).

Usage:
    python tools/analyze_fingerprint_references.py \
        [--data-dir ~/.llmtrace] \
        [--campaign-id real-fingerprint-v1] \
        [--output-dir reports/research/real-fingerprint-v1] \
        [--policy-id <id> --policy-version <version>]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from llmtrace.fingerprint.distance import (
    FingerprintDistance,
    FingerprintDistanceError,
    compute_fingerprint_distance,
)
from llmtrace.fingerprint.models import FingerprintSourceRole, FingerprintSuite
from llmtrace.fingerprint.reference import FingerprintReferenceError, FingerprintReferenceSnapshot
from llmtrace.fingerprint.repository import FingerprintRepository, FingerprintRepositoryError
from llmtrace.fingerprint.suite import compute_generation_config_sha256, load_fingerprint_suite

#: Minimum real Reference Campaign scale (Task 4): 5 identities x 2 captures.
MIN_CAMPAIGN_IDENTITIES = 5
MIN_CAMPAIGN_CAPTURES_PER_IDENTITY = 2

#: Only these source roles count as real reference campaign data (Task 5/29):
#: test fixtures and candidate captures must never be mistaken for a real
#: campaign, and vice versa.
_REFERENCE_ROLES = frozenset(
    {
        FingerprintSourceRole.TRUSTED_REFERENCE,
        FingerprintSourceRole.OFFICIAL_BASELINE,
    }
)

_PROBE_LABELS_NOTE = "These thresholds are exploratory labels only and are not production identity decision thresholds."


@dataclass
class ProbeResearchThresholds:
    """Exploratory thresholds behind the research labels (Task 20).

    These are research-report labels only. They MUST NOT leak into
    ``FingerprintDecisionPolicy`` or any production identity decision.
    """

    max_invalid_rate: float = 0.20
    max_within_mean: float = 0.15
    min_separation: float = 0.05


@dataclass
class PairwiseSplit:
    """Result of splitting all snapshot pairs into within/between groups."""

    within: list[float] = field(default_factory=list)
    between: list[float] = field(default_factory=list)
    within_records: list[dict[str, object]] = field(default_factory=list)
    between_records: list[dict[str, object]] = field(default_factory=list)
    skipped: list[dict[str, object]] = field(default_factory=list)
    probe_within: dict[str, list[float]] = field(default_factory=dict)
    probe_between: dict[str, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core analysis primitives (Task 15-19)
# ---------------------------------------------------------------------------


def identity_key(snapshot: FingerprintReferenceSnapshot) -> tuple[str, str]:
    """The identity a capture belongs to: ``(provider_id, model_id)`` (Task 6)."""
    return snapshot.provider_id, snapshot.model_id


def identity_label(key: tuple[str, str]) -> str:
    """Stable single-string rendering of an identity key."""
    return f"{key[0]}/{key[1]}"


def pair_distance(
    left: FingerprintReferenceSnapshot,
    right: FingerprintReferenceSnapshot,
    *,
    suite: FingerprintSuite,
) -> float:
    """Weighted JSD distance between two snapshots (production distance, Task 15)."""
    return _pair_distance_details(left, right, suite=suite).distance


def _pair_distance_details(
    left: FingerprintReferenceSnapshot,
    right: FingerprintReferenceSnapshot,
    *,
    suite: FingerprintSuite,
) -> FingerprintDistance:
    """Full per-probe distance result between two snapshots (Task 15/17)."""
    return compute_fingerprint_distance(
        probes=suite.probes,
        candidate=left.distributions,
        reference=right.distributions,
        minimum_comparable_probes=len(suite.probes),
    )


def split_pairwise_distances(
    snapshots: Sequence[FingerprintReferenceSnapshot],
    *,
    suite: FingerprintSuite,
) -> PairwiseSplit:
    """Split every snapshot pair into within-identity / between-identity JSD (Task 15)."""
    split = PairwiseSplit(
        probe_within={probe.probe_id: [] for probe in suite.probes},
        probe_between={probe.probe_id: [] for probe in suite.probes},
    )

    ordered = sorted(snapshots, key=lambda snapshot: snapshot.snapshot_id)
    for left, right in combinations(ordered, 2):
        try:
            result = _pair_distance_details(left, right, suite=suite)
        except FingerprintDistanceError as error:
            split.skipped.append({"left": left.snapshot_id, "right": right.snapshot_id, "reason": str(error)})
            continue

        record: dict[str, object] = {
            "left": left.snapshot_id,
            "right": right.snapshot_id,
            "left_identity": identity_label(identity_key(left)),
            "right_identity": identity_label(identity_key(right)),
            "distance": result.distance,
        }
        same_identity = identity_key(left) == identity_key(right)
        if same_identity:
            split.within.append(result.distance)
            split.within_records.append(record)
        else:
            split.between.append(result.distance)
            split.between_records.append(record)

        for probe_result in result.per_probe:
            if not probe_result.comparable or probe_result.distance is None:
                continue
            target = split.probe_within if same_identity else split.probe_between
            if probe_result.probe_id in target:
                target[probe_result.probe_id].append(probe_result.distance)

    return split


def separation_summary(within: list[float], between: list[float]) -> dict[str, float | None]:
    """Aggregate within/between distances plus the conservative margin (Task 16)."""
    if not within or not between:
        return {
            "within_mean": None,
            "within_median": None,
            "within_max": None,
            "between_mean": None,
            "between_median": None,
            "between_min": None,
            "conservative_margin": None,
        }

    return {
        "within_mean": mean(within),
        "within_median": median(within),
        "within_max": max(within),
        "between_mean": mean(between),
        "between_median": median(between),
        "between_min": min(between),
        "conservative_margin": min(between) - max(within),
    }


def probe_separation_score(within: list[float], between: list[float]) -> float | None:
    """First-version research discrimination metric: ``mean_between - mean_within`` (Task 18).

    This is a research discrimination metric only — it is NOT an accuracy and
    NOT a confidence, and it must never be named either.
    """
    if not within or not between:
        return None

    return mean(between) - mean(within)


def invalid_rate(snapshot: FingerprintReferenceSnapshot, probe_id: str) -> float | None:
    """INVALID share of one probe's samples in one snapshot (Task 19)."""
    for distribution in snapshot.distributions:
        if distribution.probe_id != probe_id:
            continue

        if distribution.sample_count == 0:
            return None

        return distribution.invalid_count / distribution.sample_count

    return None


def aggregate_invalid_rate(
    snapshots: Sequence[FingerprintReferenceSnapshot],
    probe_id: str,
) -> float | None:
    """INVALID share of one probe's samples across a group of snapshots (Task 19)."""
    total_samples = 0
    total_invalid = 0
    for snapshot in snapshots:
        for distribution in snapshot.distributions:
            if distribution.probe_id != probe_id:
                continue
            total_samples += distribution.sample_count
            total_invalid += distribution.invalid_count

    if total_samples == 0:
        return None

    return total_invalid / total_samples


def per_identity_invalid_rates(
    snapshots: Sequence[FingerprintReferenceSnapshot],
    probe_id: str,
) -> dict[str, float]:
    """Per-identity INVALID rate for one probe, keyed by identity label."""
    rates: dict[str, float] = {}
    identities = sorted({identity_key(snapshot) for snapshot in snapshots})
    for key in identities:
        group = [snapshot for snapshot in snapshots if identity_key(snapshot) == key]
        rate = aggregate_invalid_rate(group, probe_id)
        if rate is not None:
            rates[identity_label(key)] = rate
    return rates


def classify_probe(
    *,
    within: list[float],
    between: list[float],
    invalid: float | None,
    thresholds: ProbeResearchThresholds,
) -> str:
    """Exploratory research label for one probe (Task 20).

    Research-report label only — never a production identity verdict.
    """
    if not within or not between:
        return "INSUFFICIENT_DATA"
    if invalid is not None and invalid > thresholds.max_invalid_rate:
        return "INVALID_HEAVY"
    if mean(within) > thresholds.max_within_mean:
        return "UNSTABLE"
    if mean(between) - mean(within) < thresholds.min_separation:
        return "WEAK"
    return "PROMISING"


def identity_confusability(between_records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Nearest-impostor distance per identity, most confusable first (Task 22 Q2)."""
    nearest: dict[str, float] = {}
    for record in between_records:
        distance = float(record["distance"])
        for identity in (record["left_identity"], record["right_identity"]):
            current = nearest.get(identity)
            if current is None or distance < current:
                nearest[identity] = distance

    ranked = sorted(nearest.items(), key=lambda item: (item[1], item[0]))
    return [{"identity": identity, "nearest_impostor_distance": distance} for identity, distance in ranked]


# ---------------------------------------------------------------------------
# Snapshot selection (Task 10 integrity gate + Task 5 role gate)
# ---------------------------------------------------------------------------


def _select_snapshots(
    repository: FingerprintRepository,
    *,
    suite: FingerprintSuite,
) -> tuple[list[FingerprintReferenceSnapshot], list[dict[str, str]]]:
    """Pick analysable snapshots; verify every snapshot's self-hash first (Task 10).

    A tampered snapshot fails closed: ``verify_content_hash`` raises and the
    whole analysis aborts rather than quietly comparing corrupted evidence.
    """
    selected: list[FingerprintReferenceSnapshot] = []
    excluded: list[dict[str, str]] = []

    for snapshot in sorted(repository.snapshots.list(), key=lambda item: item.snapshot_id):
        snapshot.verify_content_hash()

        if snapshot.source_role not in _REFERENCE_ROLES:
            excluded.append(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "reason": f"non_reference_source_role:{snapshot.source_role}",
                }
            )
            continue

        declared = (snapshot.suite_id, snapshot.suite_version, snapshot.suite_content_sha256)
        expected = (suite.suite_id, suite.suite_version, suite.content_sha256)
        if declared != expected:
            excluded.append({"snapshot_id": snapshot.snapshot_id, "reason": "suite_mismatch"})
            continue

        selected.append(snapshot)

    return selected, excluded


def _analyze_group(
    snapshots: Sequence[FingerprintReferenceSnapshot],
    *,
    suite: FingerprintSuite,
    thresholds: ProbeResearchThresholds,
) -> dict[str, object]:
    """Full within/between analysis for one repetitions group."""
    split = split_pairwise_distances(snapshots, suite=suite)

    pairwise: dict[str, object] = separation_summary(split.within, split.between)
    pairwise["within_pair_count"] = len(split.within)
    pairwise["between_pair_count"] = len(split.between)

    probes: list[dict[str, object]] = []
    for probe in suite.probes:
        within = split.probe_within[probe.probe_id]
        between = split.probe_between[probe.probe_id]
        invalid = aggregate_invalid_rate(snapshots, probe.probe_id)
        probes.append(
            {
                "probe_id": probe.probe_id,
                "within_mean": mean(within) if within else None,
                "within_max": max(within) if within else None,
                "between_mean": mean(between) if between else None,
                "between_min": min(between) if between else None,
                "separation": probe_separation_score(within, between),
                "invalid_rate": invalid,
                "classification": classify_probe(
                    within=within, between=between, invalid=invalid, thresholds=thresholds
                ),
                "within_distances": list(within),
                "between_distances": list(between),
                "per_identity_invalid_rates": per_identity_invalid_rates(snapshots, probe.probe_id),
            }
        )

    identities = sorted({identity_key(snapshot) for snapshot in snapshots})
    captures_per_identity = {
        identity_label(key): sum(1 for snapshot in snapshots if identity_key(snapshot) == key) for key in identities
    }

    return {
        "repetitions": snapshots[0].repetitions if snapshots else None,
        "capture_count": len(snapshots),
        "identity_count": len(identities),
        "identities": [identity_label(key) for key in identities],
        "captures_per_identity": captures_per_identity,
        "pairwise": pairwise,
        "probes": probes,
        "identity_confusability": identity_confusability(split.between_records),
        "within_distances": split.within_records,
        "between_distances": split.between_records,
        "skipped_pairs": split.skipped,
    }


def load_validation_summary(
    repository: FingerprintRepository,
    *,
    policy_id: str,
    policy_version: str,
) -> dict[str, object]:
    """Read an existing held-out validation policy (read-only, Task 12/13)."""
    repository.policies.verify(policy_id, policy_version)
    policy = repository.policies.get(policy_id, policy_version)
    return {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "fingerprint_set_id": policy.fingerprint_set_id,
        "validated": policy.validated,
        "top1_accuracy": policy.top1_accuracy,
        "top3_accuracy": policy.top3_accuracy,
        "true_positive_rate": policy.true_positive_rate,
        "false_accept_rate": policy.false_accept_rate,
        "threshold": policy.distance_threshold,
        "max_far_target": policy.max_far_target,
        "identity_count": policy.identity_count,
        "held_out_capture_count": policy.held_out_capture_count,
    }


def _select_primary_group(analyzed_groups: Sequence[dict[str, object]]) -> dict[str, object] | None:
    """Primary group = most captures; ties broken toward fewer repetitions."""
    primary: dict[str, object] | None = None
    for group in analyzed_groups:
        if primary is None or (
            group["capture_count"],
            -(group["repetitions"] or 0),
        ) > (primary["capture_count"], -(primary["repetitions"] or 0)):
            primary = group
    return primary


# ---------------------------------------------------------------------------
# Campaign-level report assembly (Task 21)
# ---------------------------------------------------------------------------


def build_analysis(
    repository: FingerprintRepository,
    *,
    campaign_id: str,
    policy_id: str | None = None,
    policy_version: str | None = None,
    thresholds: ProbeResearchThresholds | None = None,
) -> dict[str, object]:
    """Assemble the full analysis payload for one campaign."""
    thresholds = thresholds or ProbeResearchThresholds()
    suite = load_fingerprint_suite()

    selected, excluded = _select_snapshots(repository, suite=suite)

    groups: dict[int, list[FingerprintReferenceSnapshot]] = {}
    for snapshot in selected:
        groups.setdefault(snapshot.repetitions, []).append(snapshot)

    analyzed_groups = [
        _analyze_group(groups[repetitions], suite=suite, thresholds=thresholds) for repetitions in sorted(groups)
    ]

    # Primary group = most captures; ties broken toward fewer repetitions.
    primary = _select_primary_group(analyzed_groups)

    identity_keys = {identity_key(snapshot) for snapshot in selected}
    captures_per_identity = [sum(1 for snapshot in selected if identity_key(snapshot) == key) for key in identity_keys]
    min_captures = min(captures_per_identity) if captures_per_identity else 0

    validation: dict[str, object] | None = None
    if policy_id is not None and policy_version is not None:
        validation = load_validation_summary(repository, policy_id=policy_id, policy_version=policy_version)

    return {
        "campaign_id": campaign_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": {
            "id": suite.suite_id,
            "version": suite.suite_version,
            "content_sha256": suite.content_sha256,
            "normalization_policy_id": suite.normalization_policy_id,
            "normalization_policy_version": suite.normalization_policy_version,
            "generation_config_sha256": compute_generation_config_sha256(suite),
        },
        "identity_count": len(identity_keys),
        "capture_count": len(selected),
        "included_snapshot_ids": [snapshot.snapshot_id for snapshot in selected],
        "excluded_snapshots": excluded,
        "campaign_completeness": {
            "required_min_identities": MIN_CAMPAIGN_IDENTITIES,
            "required_min_captures_per_identity": MIN_CAMPAIGN_CAPTURES_PER_IDENTITY,
            "identity_count": len(identity_keys),
            "min_captures_per_identity": min_captures,
            "meets_minimum": (
                len(identity_keys) >= MIN_CAMPAIGN_IDENTITIES and min_captures >= MIN_CAMPAIGN_CAPTURES_PER_IDENTITY
            ),
        },
        "repetitions_groups": analyzed_groups,
        "pairwise_source_repetitions": primary["repetitions"] if primary else None,
        "pairwise": primary["pairwise"] if primary else separation_summary([], []),
        "probes": primary["probes"] if primary else _empty_probe_summary(suite),
        "validation": validation
        or {
            "policy_id": None,
            "policy_version": None,
            "fingerprint_set_id": None,
            "validated": None,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "true_positive_rate": None,
            "false_accept_rate": None,
            "threshold": None,
            "max_far_target": None,
            "identity_count": None,
            "held_out_capture_count": None,
        },
        "notes": [
            _PROBE_LABELS_NOTE,
            "The separation score is a research discrimination metric, not an accuracy or confidence.",
            "This report may embed real endpoint identifiers; committing it publicly is an Owner decision (Task 28).",
        ],
    }


def _empty_probe_summary(suite: FingerprintSuite) -> list[dict[str, object]]:
    """Probe skeleton used when no analysable snapshot exists (no invented values)."""
    return [
        {
            "probe_id": probe.probe_id,
            "within_mean": None,
            "within_max": None,
            "between_mean": None,
            "between_min": None,
            "separation": None,
            "invalid_rate": None,
            "classification": "INSUFFICIENT_DATA",
            "within_distances": [],
            "between_distances": [],
            "per_identity_invalid_rates": {},
        }
        for probe in suite.probes
    ]


# ---------------------------------------------------------------------------
# Markdown rendering (Task 22)
# ---------------------------------------------------------------------------


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(analysis: dict[str, object]) -> str:
    suite = analysis["suite"]
    assert isinstance(suite, dict)
    completeness = analysis["campaign_completeness"]
    assert isinstance(completeness, dict)
    pairwise = analysis["pairwise"]
    assert isinstance(pairwise, dict)
    probes = analysis["probes"]
    assert isinstance(probes, list)
    validation = analysis["validation"]
    assert isinstance(validation, dict)
    groups = analysis["repetitions_groups"]
    assert isinstance(groups, list)
    primary_group = _select_primary_group(groups) or {}

    lines: list[str] = []
    lines.append(f"# Fingerprint Reference Analysis — {analysis['campaign_id']}")
    lines.append("")
    lines.append(f"Generated {analysis['generated_at']} by `tools/analyze_fingerprint_references.py` (read-only).")
    lines.append("")
    lines.append("> " + _PROBE_LABELS_NOTE)
    lines.append(">")
    lines.append(
        "> This report may embed real endpoint identifiers (provider/model ids); "
        "committing it to a public repository is an Owner decision (Task 28)."
    )
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| campaign_id | {analysis['campaign_id']} |")
    lines.append(f"| suite | {suite['id']} {suite['version']} |")
    lines.append(f"| suite_content_sha256 | `{suite['content_sha256']}` |")
    lines.append(
        f"| normalization_policy | {suite['normalization_policy_id']} {suite['normalization_policy_version']} |"
    )
    lines.append(f"| identities | {analysis['identity_count']} |")
    lines.append(f"| captures | {analysis['capture_count']} |")
    group_summary = ", ".join(
        f"repetitions={group['repetitions']}: {group['capture_count']} captures" for group in groups
    )
    lines.append(f"| repetitions groups | {group_summary or 'none'} |")
    lines.append(f"| excluded snapshots | {len(analysis['excluded_snapshots'])} |")
    lines.append(f"| primary group repetitions | {analysis['pairwise_source_repetitions']} |")
    lines.append("")

    if not completeness["meets_minimum"]:
        lines.append(
            f"> WARNING: this dataset is below the real Reference Campaign minimum "
            f"({MIN_CAMPAIGN_IDENTITIES} identities x {MIN_CAMPAIGN_CAPTURES_PER_IDENTITY} captures): "
            f"{completeness['identity_count']} identities, min {completeness['min_captures_per_identity']} "
            f"captures/identity. Treat every conclusion below as PARTIAL (Task 31)."
        )
        lines.append("")

    lines.append("## Pairwise separation (primary repetitions group)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| within mean | {_fmt(pairwise['within_mean'])} |")
    lines.append(f"| within median | {_fmt(pairwise['within_median'])} |")
    lines.append(f"| within max | {_fmt(pairwise['within_max'])} |")
    lines.append(f"| between mean | {_fmt(pairwise['between_mean'])} |")
    lines.append(f"| between median | {_fmt(pairwise['between_median'])} |")
    lines.append(f"| between min | {_fmt(pairwise['between_min'])} |")
    lines.append(f"| conservative separation margin | {_fmt(pairwise['conservative_margin'])} |")
    lines.append(f"| within pairs | {pairwise.get('within_pair_count', 0)} |")
    lines.append(f"| between pairs | {pairwise.get('between_pair_count', 0)} |")
    lines.append("")
    lines.append(
        "conservative separation margin = min(between) - max(within): the closest different-identity "
        "pair versus the least stable same-identity pair. A positive margin on this sample is a good "
        "sign, but it is NOT proof of generalization."
    )
    lines.append("")

    lines.append("## Per-probe summary (primary repetitions group)")
    lines.append("")
    lines.append("| probe | within mean | between mean | separation | invalid rate | label |")
    lines.append("|---|---|---|---|---|---|")
    for probe in probes:
        assert isinstance(probe, dict)
        lines.append(
            f"| {probe['probe_id']} | {_fmt(probe['within_mean'])} | {_fmt(probe['between_mean'])} "
            f"| {_fmt(probe['separation'])} | {_fmt(probe['invalid_rate'])} | {probe['classification']} |"
        )
    lines.append("")

    lines.append("## Research questions (Task 22)")
    lines.append("")
    lines.extend(_render_questions(analysis, pairwise, probes, validation, primary_group))
    lines.append("")
    return "\n".join(lines)


def _render_questions(
    analysis: dict[str, object],
    pairwise: dict[str, object],
    probes: list[dict[str, object]],
    validation: dict[str, object],
    primary_group: dict[str, object],
) -> list[str]:
    completeness = analysis["campaign_completeness"]
    assert isinstance(completeness, dict)
    capture_count = analysis["capture_count"]
    assert isinstance(capture_count, int)
    identity_count = analysis["identity_count"]
    assert isinstance(identity_count, int)

    out: list[str] = []

    # Q1 — can the suite distinguish these real models?
    out.append("### 1. Can the current probe suite distinguish these real models?")
    margin = pairwise["conservative_margin"]
    if capture_count == 0:
        out.append("- No analysable reference captures are present in the repository; no conclusion.")
    elif pairwise["within_mean"] is None or pairwise["between_mean"] is None:
        out.append(
            "- Insufficient grouping to answer: the repository needs at least 2 captures of one "
            "identity (within) and at least 2 distinct identities (between)."
        )
    elif margin is not None and margin > 0:
        out.append(
            f"- On this sample, yes: the closest different-identity pair is still "
            f"{_fmt(margin)} further than the least stable same-identity pair. "
            "This is sample-level separation, not proof of generalization."
        )
    else:
        out.append(
            "- No: same-identity and different-identity distances overlap on this sample (conservative margin <= 0)."
        )
    out.append("")

    # Q2 — which identities are most easily confused?
    out.append("### 2. Which identities are most easily confused?")
    confusability = primary_group.get("identity_confusability", [])
    assert isinstance(confusability, list)
    confusable = list(confusability[:3])
    if not confusable:
        out.append("- Not answerable: no between-identity pairs exist in the data.")
    else:
        for entry in confusable:
            assert isinstance(entry, dict)
            out.append(
                f"- {_fmt(entry['nearest_impostor_distance'])}: nearest impostor distance for {entry['identity']}"
            )
        out.append("- Lower nearest-impostor distance means more easily confused.")
    out.append("")

    # Q3 — which probes discriminate best?
    out.append("### 3. Which probes discriminate best?")
    ranked = sorted(probes, key=lambda probe: (probe["separation"] is None, -(probe["separation"] or 0.0)))
    best = [probe for probe in ranked if probe["separation"] is not None][:3]
    if not best:
        out.append("- Not answerable: no probe has both within and between data.")
    else:
        for probe in best:
            out.append(f"- {probe['probe_id']}: separation {_fmt(probe['separation'])}")
    out.append("")

    # Q4 — which probes are unstable?
    out.append("### 4. Which probes are unstable?")
    unstable = [probe for probe in probes if probe["classification"] == "UNSTABLE"]
    if not unstable:
        out.append("- No probe exceeds the exploratory within-mean threshold in this data.")
    else:
        for probe in unstable:
            out.append(f"- {probe['probe_id']}: within mean {_fmt(probe['within_mean'])}")
    out.append("")

    # Q5 — which probes have too many INVALID?
    out.append("### 5. Which probes have too many INVALID?")
    invalid_heavy = [probe for probe in probes if probe["classification"] == "INVALID_HEAVY"]
    if not invalid_heavy:
        worst = sorted(probes, key=lambda probe: (probe["invalid_rate"] is None, -(probe["invalid_rate"] or 0.0)))
        top = worst[0] if worst else None
        if top is not None and top["invalid_rate"] is not None:
            out.append(
                f"- No probe exceeds the exploratory invalid-rate threshold; highest is "
                f"{top['probe_id']} at {_fmt(top['invalid_rate'])}."
            )
        else:
            out.append("- Not answerable: no invalid data recorded.")
    else:
        for probe in invalid_heavy:
            out.append(f"- {probe['probe_id']}: invalid rate {_fmt(probe['invalid_rate'])}")
    out.append("")

    # Q6 — do same/different distributions overlap?
    out.append("### 6. Do same-identity and different-identity distances overlap?")
    within_max = pairwise["within_max"]
    between_min = pairwise["between_min"]
    if within_max is None or between_min is None:
        out.append("- Not answerable: within or between distances are missing.")
    elif within_max >= between_min:
        out.append(
            f"- YES: max(within) = {_fmt(within_max)} >= min(between) = {_fmt(between_min)}. "
            "The two distributions overlap on this sample."
        )
    else:
        out.append(f"- NO: max(within) = {_fmt(within_max)} < min(between) = {_fmt(between_min)} on this sample.")
    out.append("")

    # Q7 — is STANDARD=8 sufficient?
    out.append("### 7. Does STANDARD=8 look sufficient?")
    captures_per_identity = primary_group.get("captures_per_identity", {})
    assert isinstance(captures_per_identity, dict)
    if not captures_per_identity:
        out.append("- Not answerable: no captures available.")
    else:
        pair_counts = [count * (count - 1) // 2 for count in captures_per_identity.values()]
        out.append(
            f"- Within-identity evidence is thin: each identity contributes only "
            f"{min(pair_counts) if pair_counts else 0}–{max(pair_counts) if pair_counts else 0} "
            "within pairs at the current capture counts, so any within- variance estimate is "
            "statistically fragile."
        )
        repetitions_groups = analysis["repetitions_groups"]
        assert isinstance(repetitions_groups, list)
        if len(repetitions_groups) > 1:
            out.append(
                "- Multiple repetitions groups exist; compare their within means in analysis.json "
                "for the QUICK/STANDARD/RESEARCH cost question (Task 25)."
            )
        else:
            out.append(
                "- Only one repetitions group exists; the QUICK vs STANDARD vs RESEARCH comparison "
                "(Task 25) needs separate captures per profile."
            )
    out.append("")

    # Q8 — is the held-out policy validated?
    out.append("### 8. Is the held-out policy validated?")
    if validation["policy_id"] is None:
        out.append(
            "- No policy was supplied to this analysis (pass --policy-id/--policy-version to "
            "include held-out validation results)."
        )
    elif validation["validated"]:
        out.append(
            f"- YES: policy {validation['policy_id']} v{validation['policy_version']} is validated "
            f"(Top-1 {_fmt(validation['top1_accuracy'])}, TPR {_fmt(validation['true_positive_rate'])}, "
            f"FAR {_fmt(validation['false_accept_rate'])} <= target {_fmt(validation['max_far_target'])}, "
            f"threshold {_fmt(validation['threshold'])})."
        )
    else:
        out.append(
            f"- NO: policy {validation['policy_id']} v{validation['policy_version']} exists but is "
            "not validated; it must not be used for claim-consistency verdicts."
        )
    out.append("")

    # Q9 — do current results support real claim consistency?
    out.append("### 9. Do current results support using Fingerprint v1 for real claim consistency?")
    meets = completeness["meets_minimum"]
    assert isinstance(meets, bool)
    if capture_count == 0:
        out.append("- No conclusion: no real reference data has been analysed yet.")
    elif not meets:
        out.append(
            "- No conclusion: the campaign is below the minimum scale "
            f"({identity_count} identities observed, {MIN_CAMPAIGN_IDENTITIES} required); "
            "claim-consistency support cannot be claimed from a PARTIAL campaign (Task 31)."
        )
    elif validation["policy_id"] is not None and validation["validated"] and margin is not None and margin > 0:
        out.append(
            "- Promising: separation exists on this sample and a validated policy exists. This "
            "supports cautious use for claim-consistency, subject to the stated limitations."
        )
    else:
        out.append(
            "- Not yet: either separation is insufficient or no validated policy exists. "
            "Claim-consistency verdicts require a validated policy (Rule 2)."
        )
    out.append("")

    # Q10 — KEEP / TUNE / REDESIGN draft.
    out.append("### 10. Next step: KEEP / TUNE / REDESIGN?")
    if capture_count == 0 or not meets:
        out.append(
            "- The decision gate cannot be concluded from this dataset (below the "
            f"{MIN_CAMPAIGN_IDENTITIES}x{MIN_CAMPAIGN_CAPTURES_PER_IDENTITY} minimum). "
            "This is a PARTIAL campaign per Task 31; no production validation is claimed."
        )
    elif margin is not None and margin > 0 and validation["validated"]:
        out.append(
            "- Draft suggestion: KEEP — separation holds and the policy validates. Final gate is "
            "the Owner's decision per Task 23."
        )
    elif margin is not None and margin > 0:
        weak = [probe for probe in probes if probe["classification"] in {"WEAK", "UNSTABLE", "INVALID_HEAVY"}]
        if weak:
            out.append(
                f"- Draft suggestion: TUNE — overall separation exists but {len(weak)} probe(s) "
                "are weak/unstable/invalid-heavy; future suite 0.2.0 must use fresh independent "
                "captures (Task 24)."
            )
        else:
            out.append(
                "- Draft suggestion: KEEP — separation holds but no validated policy exists yet; "
                "run held-out validation before any claim-consistency use."
            )
    else:
        out.append(
            "- Draft suggestion: REDESIGN — within/between distances overlap on this sample; "
            "categorical probes alone do not separate identities well enough. This is a legitimate "
            "result (Task 23 Outcome C)."
        )
    out.append("")

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def write_reports(analysis: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    """Write analysis.json and analysis.md (Task 21)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    markdown_path = output_dir / "analysis.md"
    markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")

    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only discrimination analysis of real fingerprint reference snapshots.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="fingerprint data root (default: $LLMTRACE_HOME or ~/.llmtrace)",
    )
    parser.add_argument(
        "--campaign-id",
        default="real-fingerprint-v1",
        help="campaign identifier recorded in the report (default: real-fingerprint-v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report directory (default: reports/research/<campaign-id>)",
    )
    parser.add_argument("--policy-id", default=None, help="held-out validation policy id (optional)")
    parser.add_argument("--policy-version", default=None, help="held-out validation policy version (optional)")
    args = parser.parse_args(argv)

    if bool(args.policy_id) != bool(args.policy_version):
        parser.error("--policy-id and --policy-version must be provided together")

    try:
        repository = FingerprintRepository.load(data_root=args.data_dir if args.data_dir is not None else None)
        analysis = build_analysis(
            repository,
            campaign_id=args.campaign_id,
            policy_id=args.policy_id,
            policy_version=args.policy_version,
        )
        output_dir = args.output_dir or Path("reports/research") / args.campaign_id
        json_path, markdown_path = write_reports(analysis, output_dir)
    except (FingerprintRepositoryError, FingerprintReferenceError) as error:
        print(f"analysis failed: {error}", file=sys.stderr)
        return 1

    completeness = analysis["campaign_completeness"]
    assert isinstance(completeness, dict)
    status = "COMPLETE" if completeness["meets_minimum"] else "PARTIAL"
    print(f"analysis written: {json_path}")
    print(f"report written: {markdown_path}")
    print(f"real data campaign: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
