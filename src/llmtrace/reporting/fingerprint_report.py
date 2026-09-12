"""Fingerprint 报告段（Task 44）—— 身份证据的 machine-readable 序列化.

本模块只做序列化：它不计算距离、不做判定、不引入任何阈值。它把已经产生的
fingerprint artifact（候选 capture、排序结果、Rule 2 门禁记录）与 run 级证据
（routing v2 判定、四组件 confidence bundle）拼成 JSON 与 HTML 共用的一个 dict，
从而使两份报告描述的是同一份数据。

语义纪律（Rule 2 / Rule 3 / Rule 7 / Task 48）：

- ``top_k`` 是"行为最接近的参考"的排序窗口，不是身份结论，也不是 Top-K 判定；
- ``jsd_reference`` 显式写出逐 probe / 聚合 JSD 描述的是哪一次比较
  （claimed reference 还是 top-ranked），读者不会把数字错配到别的参考；
- ``verdict`` 只使用既有标签，``claim_verdict_produced`` 为 ``False`` 时报告里
  不存在任何"身份已确认"的说法。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmtrace.fingerprint.matcher import (
    MATCH_DISCLAIMER,
    FingerprintMatchEntry,
    FingerprintMatchResult,
    FingerprintVerificationResult,
)
from llmtrace.fingerprint.models import FingerprintSuite
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot
from llmtrace.fingerprint.reference_set import FingerprintReferenceSet

if TYPE_CHECKING:  # pragma: no cover - 只用于类型标注，运行时避免 reporting → analysis 的环
    from llmtrace.analysis.confidence_v2 import ConfidenceBundle
    from llmtrace.fingerprint.routing import RoutingAssessmentV2

#: 报告里的 Top-K 排序窗口。这是**序列化窗口**，不是判定阈值、不是 Top-K 结论（Rule 3）。
TOP_K = 5

#: claimed reference 距离与 entry 距离配对时的浮点容差。
_DISTANCE_TOLERANCE = 1e-9

#: 逐 probe / 聚合 JSD 描述的那次比较来自哪里。
BASIS_CLAIMED_REFERENCE = "claimed_reference"
BASIS_TOP_RANKED = "top_ranked"


def build_fingerprint_section(
    *,
    match: FingerprintMatchResult | None,
    verification: FingerprintVerificationResult | None,
    snapshot: FingerprintReferenceSnapshot | None,
    suite: FingerprintSuite | None = None,
    reference_set: FingerprintReferenceSet | None = None,
    routing: RoutingAssessmentV2 | None = None,
    confidence: ConfidenceBundle | None = None,
) -> dict[str, object] | None:
    """拼出 ``fingerprint`` 报告段；本次 run 没有任何身份证据时返回 ``None``.

    Args:
        match: 排序结果（含逐 probe JSD 与结论等级）.
        verification: Rule 2 门禁记录；``None`` 表示没有可用 policy 记录.
        snapshot: 本次 run 的候选 capture（probe coverage / invalid outcome 的来源）.
        suite: 本次执行的指纹套件（suite id/version/hash 的来源）.
        reference_set: 本次使用的参考集（fingerprint set id/version/hash 的来源）.
        routing: routing v2 判定；``None`` 如实写 ``None``（未评估），不伪造标签.
        confidence: 四组件 confidence bundle；``None`` 如实写 ``None``.

    Returns:
        JSON 可序列化的 dict；所有缺失项写 ``None`` 而不是省略，读方可直接判断"没有"。
    """
    if match is None and verification is None and snapshot is None:
        return None

    entry, basis = _selected_entry(match)

    return {
        "available": True,
        "experimental": True,
        "disclaimer": MATCH_DISCLAIMER,
        "suite": _suite_block(suite, snapshot),
        "reference_set": _reference_set_block(reference_set),
        "decision_policy": _policy_block(verification),
        "repetitions": _repetitions(reference_set, snapshot),
        "probe_coverage": _probe_coverage(suite, snapshot, entry),
        "invalid_outcome_count": _invalid_outcome_count(snapshot),
        "candidate_sample_count": _candidate_sample_count(snapshot),
        "jsd_reference": _jsd_reference(entry, basis),
        "per_probe_jsd": _per_probe_jsd(entry, snapshot),
        "aggregate_jsd": entry.distance if entry is not None else None,
        "top_k": _top_k(match),
        "claimed_model_id": match.claimed_model_id if match is not None else None,
        "claimed_reference_distance": match.claimed_reference_distance if match is not None else None,
        "verdict": _verdict_block(verification, match),
        "routing": routing.model_dump(mode="json") if routing is not None else None,
        "confidence": confidence.model_dump(mode="json") if confidence is not None else None,
    }


def _selected_entry(match: FingerprintMatchResult | None) -> tuple[FingerprintMatchEntry | None, str | None]:
    """选出逐 probe JSD 所描述的那次比较：优先 claimed reference，否则 top-ranked."""
    if match is None or not match.entries:
        return None, None

    if match.claimed_model_id is not None and match.claimed_reference_distance is not None:
        claimed_entries = [entry for entry in match.entries if entry.model_id == match.claimed_model_id]
        if claimed_entries:
            closest = min(claimed_entries, key=lambda entry: entry.distance)
            if abs(closest.distance - match.claimed_reference_distance) <= _DISTANCE_TOLERANCE:
                return closest, BASIS_CLAIMED_REFERENCE

    return match.entries[0], BASIS_TOP_RANKED


def _suite_block(
    suite: FingerprintSuite | None,
    snapshot: FingerprintReferenceSnapshot | None,
) -> dict[str, object] | None:
    """suite id / version / hash；没有套件对象时退回快照自述的身份，hash 写 None."""
    if suite is not None:
        return {
            "suite_id": suite.suite_id,
            "suite_version": suite.suite_version,
            "content_sha256": suite.content_sha256,
            "normalization_policy_id": suite.normalization_policy_id,
            "normalization_policy_version": suite.normalization_policy_version,
        }
    if snapshot is None:
        return None
    return {
        "suite_id": snapshot.suite_id,
        "suite_version": snapshot.suite_version,
        "content_sha256": None,
        "normalization_policy_id": snapshot.normalization_policy_id,
        "normalization_policy_version": snapshot.normalization_policy_version,
    }


def _reference_set_block(reference_set: FingerprintReferenceSet | None) -> dict[str, object] | None:
    """fingerprint set id / version / hash."""
    if reference_set is None:
        return None
    return {
        "fingerprint_set_id": reference_set.fingerprint_set_id,
        "fingerprint_set_version": reference_set.fingerprint_set_version,
        "content_sha256": reference_set.content_sha256,
        "member_count": len(reference_set.members),
    }


def _policy_block(verification: FingerprintVerificationResult | None) -> dict[str, object] | None:
    """decision policy id / version；``None`` 表示本次没有可用 policy（Rule 2）."""
    policy = verification.policy if verification is not None else None
    if policy is None:
        return None
    return {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "validated": policy.validated,
        "minimum_comparable_probes": policy.minimum_comparable_probes,
        "distance_threshold": policy.distance_threshold,
    }


def _repetitions(
    reference_set: FingerprintReferenceSet | None,
    snapshot: FingerprintReferenceSnapshot | None,
) -> int | None:
    if reference_set is not None:
        return reference_set.repetitions
    return snapshot.repetitions if snapshot is not None else None


def _probe_coverage(
    suite: FingerprintSuite | None,
    snapshot: FingerprintReferenceSnapshot | None,
    entry: FingerprintMatchEntry | None,
) -> dict[str, object] | None:
    """probe coverage：参与比较的 probe 数 / 该套件的 probe 总数."""
    if entry is not None and entry.per_probe:
        total = len(entry.per_probe)
    elif suite is not None:
        total = len(suite.probes)
    elif snapshot is not None:
        total = len(snapshot.distributions)
    else:
        return None

    compared = entry.comparable_probes if entry is not None else 0
    return {
        "total_probes": total,
        "compared_probes": compared,
        "ratio": compared / total if total else 0.0,
    }


def _invalid_outcome_count(snapshot: FingerprintReferenceSnapshot | None) -> int | None:
    """候选 capture 中归一化失败（保留为 __INVALID__）的样本总数，分母不缩小."""
    if snapshot is None:
        return None
    return sum(distribution.invalid_count for distribution in snapshot.distributions)


def _candidate_sample_count(snapshot: FingerprintReferenceSnapshot | None) -> int | None:
    if snapshot is None:
        return None
    return sum(distribution.sample_count for distribution in snapshot.distributions)


def _jsd_reference(entry: FingerprintMatchEntry | None, basis: str | None) -> dict[str, object] | None:
    """逐 probe / 聚合 JSD 对应的参考；``basis`` 说明它是 claimed 还是 top-ranked."""
    if entry is None:
        return None
    return {
        "basis": basis,
        "model_id": entry.model_id,
        "provider_id": entry.provider_id,
        "comparable_probes": entry.comparable_probes,
    }


def _per_probe_jsd(
    entry: FingerprintMatchEntry | None,
    snapshot: FingerprintReferenceSnapshot | None,
) -> list[dict[str, object]]:
    """逐 probe JSD；候选侧样本数/失败数一并给出，使 coverage 可复核."""
    if entry is None:
        return []
    candidate_by_probe = (
        {distribution.probe_id: distribution for distribution in snapshot.distributions} if snapshot is not None else {}
    )
    return [
        {
            "probe_id": probe.probe_id,
            "weight": probe.weight,
            "comparable": probe.comparable,
            "distance": probe.distance,
            "note": probe.note,
            "candidate_sample_count": candidate.sample_count if candidate is not None else None,
            "candidate_invalid_count": candidate.invalid_count if candidate is not None else None,
        }
        for probe in entry.per_probe
        for candidate in (candidate_by_probe.get(probe.probe_id),)
    ]


def _top_k(match: FingerprintMatchResult | None) -> list[dict[str, object]]:
    """Top-K 排序窗口；标签只是"行为最接近的参考"（Rule 3）."""
    entries: Sequence[FingerprintMatchEntry] = match.entries if match is not None else ()
    return [
        {
            "rank": rank,
            "model_id": entry.model_id,
            "provider_id": entry.provider_id,
            "distance": entry.distance,
            "similarity": entry.similarity,
            "comparable_probes": entry.comparable_probes,
        }
        for rank, entry in enumerate(entries[:TOP_K], start=1)
    ]


def _verdict_block(
    verification: FingerprintVerificationResult | None,
    match: FingerprintMatchResult | None,
) -> dict[str, object]:
    """结论等级与 Rule 2 门禁记录；没有 verdict 时如实写 False."""
    return {
        "match_status": match.status.value if match is not None else None,
        "claim_verdict_produced": verification.claim_verdict_produced if verification is not None else False,
        "minimum_comparable_probes": verification.minimum_comparable_probes if verification is not None else None,
        "reference_identity_count": (
            verification.reference_identity_count
            if verification is not None
            else (len(match.entries) if match is not None else None)
        ),
        "notes": list(verification.notes) if verification is not None else [],
    }


__all__: list[str] = [
    "BASIS_CLAIMED_REFERENCE",
    "BASIS_TOP_RANKED",
    "TOP_K",
    "build_fingerprint_section",
]
