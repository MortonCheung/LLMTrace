"""Fingerprint policy 验证（Task 18）.

目标：用 LLMTrace **自己**的数据生成 ``FingerprintDecisionPolicy``，不抄任何外部
研究的准确率数字（Task 17 / Rule 6）。

流程：

1. Step 18.1 最低验证条件 —— ``>= 5`` 个不同 identity、每个 identity ``>= 2`` 次
   独立 capture、同一套件/配置。不满足则 ``validated = False``，**不生成**
   production threshold（也不发布那些数据集上的准确率数字）。
2. Step 18.2 Leave-One-Capture-Out —— 每次留出一个 capture 作为 held-out，参考库
   为"集合减去该 capture"，记录：与自身 identity 的距离、最近 impostor 距离、
   top-1 / top-3 是否正确。
3. Step 18.3 threshold —— 在 ``false_accept_rate <= max_far`` 的候选里最大化 TPR；
   找不到满足约束的 threshold 时同样 ``validated = False``。

``false accept`` 语义：把"别的 identity"判成与声称一致。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from llmtrace.fingerprint.aggregation import aggregate_fingerprint_reference
from llmtrace.fingerprint.matcher import FingerprintMatcher
from llmtrace.fingerprint.models import FingerprintSuite
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy, build_fingerprint_policy
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot
from llmtrace.fingerprint.reference_set import FingerprintReferenceSet
from llmtrace.fingerprint.suite import verify_fingerprint_suite
from llmtrace.fingerprint.temporal import (
    collect_within_reference_temporal_distances,
    temporal_divergence_baseline,
)

#: Step 18.1 的保守下限：第一版不放宽。
MINIMUM_DISTINCT_IDENTITIES = 5
MINIMUM_CAPTURES_PER_IDENTITY = 2


class FingerprintValidationError(Exception):
    """验证输入自相矛盾（集合与快照不匹配、套件不一致等）."""

    error_code = "FINGERPRINT_VALIDATION_ERROR"


class FingerprintHoldoutOutcome(BaseModel):
    """一次 leave-one-capture-out 的记录（Step 18.2）."""

    snapshot_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    provider_id: str = Field(..., min_length=1)

    own_distance: float = Field(..., ge=0.0, le=1.0)
    nearest_impostor_distance: float | None = Field(default=None, ge=0.0, le=1.0)

    top1_correct: bool
    top3_correct: bool

    model_config = {"frozen": True, "extra": "forbid"}


class FingerprintValidationReport(BaseModel):
    """验证过程与结果的可审计记录（不包含任何 threshold 语义的判断）."""

    fingerprint_set_id: str = Field(..., min_length=1)
    fingerprint_set_content_sha256: str

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    suite_content_sha256: str

    identity_count: int = Field(..., ge=0)
    capture_count: int = Field(..., ge=0)

    minimum_distinct_identities: int = Field(..., ge=1)
    minimum_captures_per_identity: int = Field(..., ge=1)
    minimum_comparable_probes: int = Field(..., ge=1)
    max_far_target: float = Field(..., ge=0.0, le=1.0)

    meets_minimum_conditions: bool
    insufficient_reason: str | None = None

    outcomes: tuple[FingerprintHoldoutOutcome, ...]

    top1_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    top3_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Task 30：每个 reference capture 自身两个时间窗之间的距离（按 snapshot_id 排序）。
    #: 它是"同一 identity 本来就会漂多少"的经验基线，供 routing 判定使用；为空表示
    #: 参考采集的轮数不足以形成两个时间窗，此时不给出 baseline（不伪造 0）。
    within_reference_temporal_distances: tuple[float, ...] = ()

    model_config = {"frozen": True, "extra": "forbid"}


def select_distance_threshold(
    same: Sequence[float],
    impostor: Sequence[float],
    *,
    max_far: float,
) -> tuple[float, float, float] | None:
    """在 ``FAR <= max_far`` 的候选阈值中最大化 TPR（Step 18.3）.

    Returns:
        ``(threshold, true_positive_rate, false_accept_rate)``；输入为空或没有任何
        候选满足 FAR 约束时返回 ``None``（不生成 production threshold）。

    Raises:
        FingerprintValidationError: ``max_far`` 越界，或距离取值不在 ``[0, 1]``。
    """
    if not 0.0 <= max_far <= 1.0:
        raise FingerprintValidationError(f"max_far must be in [0, 1], got {max_far!r}")

    for value in (*same, *impostor):
        if not 0.0 <= value <= 1.0:
            raise FingerprintValidationError(f"distances must be in [0, 1], got {value!r}")

    if not same or not impostor:
        return None

    candidates = sorted({*same, *impostor})

    best: tuple[float, float, float] | None = None

    for threshold in candidates:
        true_positive_rate = sum(value <= threshold for value in same) / len(same)

        false_accept_rate = sum(value <= threshold for value in impostor) / len(impostor)

        if false_accept_rate > max_far:
            continue

        candidate = (
            threshold,
            true_positive_rate,
            false_accept_rate,
        )

        if best is None:
            best = candidate
            continue

        # 优先最大化 TPR；同 TPR 时取更小的 FAR（更保守的判定边界）。
        improves_tpr = true_positive_rate > best[1]
        ties_with_lower_far = true_positive_rate == best[1] and false_accept_rate < best[2]
        if improves_tpr or ties_with_lower_far:
            best = candidate

    return best


def validate_fingerprint_policy(
    *,
    reference_set: FingerprintReferenceSet,
    snapshots: Sequence[FingerprintReferenceSnapshot],
    suite: FingerprintSuite,
    policy_id: str,
    policy_version: str,
    minimum_comparable_probes: int,
    max_far_target: float,
    minimum_distinct_identities: int = MINIMUM_DISTINCT_IDENTITIES,
    minimum_captures_per_identity: int = MINIMUM_CAPTURES_PER_IDENTITY,
) -> tuple[FingerprintDecisionPolicy, FingerprintValidationReport]:
    """对参考集合执行 held-out 验证并生成 policy（Task 18 / Task 19）.

    Returns:
        ``(policy, report)``；数据集不满足最低条件时 policy 为
        ``validated = False`` 且不携带 threshold。

    Raises:
        FingerprintValidationError: 集合与快照不一一对应、set/suite 身份不一致、
            或最低条件参数非法。
        FingerprintMatchError: 参考数据本身不可比（fail closed，不静默跳过）。
    """
    if minimum_comparable_probes < 1:
        raise FingerprintValidationError(f"minimum_comparable_probes must be >= 1, got {minimum_comparable_probes}")
    if minimum_distinct_identities < 1 or minimum_captures_per_identity < 1:
        raise FingerprintValidationError("minimum validation conditions must be >= 1")

    verify_fingerprint_suite(suite)
    reference_set.verify_content_hash()
    if reference_set.suite_content_sha256 != suite.content_sha256:
        raise FingerprintValidationError(
            f"fingerprint set '{reference_set.fingerprint_set_id}' was captured with suite hash "
            f"{reference_set.suite_content_sha256!r}, not the suite being validated ({suite.content_sha256!r})"
        )

    ordered_snapshots = sorted(snapshots, key=lambda snapshot: snapshot.snapshot_id)
    _assert_set_matches_snapshots(reference_set, ordered_snapshots)

    groups: dict[tuple[str, str], list[FingerprintReferenceSnapshot]] = {}
    for snapshot in ordered_snapshots:
        groups.setdefault((snapshot.provider_id, snapshot.model_id), []).append(snapshot)

    identity_count = len(groups)
    capture_count = len(ordered_snapshots)
    insufficient_reason = _minimum_conditions_failure(
        groups,
        identity_count=identity_count,
        minimum_distinct_identities=minimum_distinct_identities,
        minimum_captures_per_identity=minimum_captures_per_identity,
    )

    if insufficient_reason is not None:
        return _unvalidated_result(
            reference_set=reference_set,
            suite=suite,
            policy_id=policy_id,
            policy_version=policy_version,
            minimum_comparable_probes=minimum_comparable_probes,
            max_far_target=max_far_target,
            minimum_distinct_identities=minimum_distinct_identities,
            minimum_captures_per_identity=minimum_captures_per_identity,
            identity_count=identity_count,
            capture_count=capture_count,
            insufficient_reason=insufficient_reason,
        )

    outcomes = _leave_one_capture_out(
        snapshots=ordered_snapshots,
        groups=groups,
        suite=suite,
        minimum_comparable_probes=minimum_comparable_probes,
    )
    same = [outcome.own_distance for outcome in outcomes]
    impostor = [
        outcome.nearest_impostor_distance for outcome in outcomes if outcome.nearest_impostor_distance is not None
    ]

    top1_accuracy = sum(outcome.top1_correct for outcome in outcomes) / len(outcomes)
    top3_accuracy = sum(outcome.top3_correct for outcome in outcomes) / len(outcomes)

    selection = select_distance_threshold(same, impostor, max_far=max_far_target)
    threshold, true_positive_rate, false_accept_rate = selection if selection is not None else (None, None, None)

    # Task 30：对 reference capture 分半，收出"同一 identity 自身的时间漂移"基线。
    # 只有拿到 threshold（即通过验证）的 policy 才允许携带它。
    temporal_distances = collect_within_reference_temporal_distances(
        snapshots=ordered_snapshots,
        suite=suite,
        minimum_comparable_probes=minimum_comparable_probes,
    )
    temporal_baseline = temporal_divergence_baseline(temporal_distances) if selection is not None else None

    report = FingerprintValidationReport(
        fingerprint_set_id=reference_set.fingerprint_set_id,
        fingerprint_set_content_sha256=reference_set.content_sha256,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_content_sha256=suite.content_sha256,
        identity_count=identity_count,
        capture_count=capture_count,
        minimum_distinct_identities=minimum_distinct_identities,
        minimum_captures_per_identity=minimum_captures_per_identity,
        minimum_comparable_probes=minimum_comparable_probes,
        max_far_target=max_far_target,
        meets_minimum_conditions=True,
        insufficient_reason=(
            None if selection is not None else f"no candidate threshold satisfies false_accept_rate <= {max_far_target}"
        ),
        outcomes=outcomes,
        top1_accuracy=top1_accuracy,
        top3_accuracy=top3_accuracy,
        within_reference_temporal_distances=temporal_distances,
    )

    policy = build_fingerprint_policy(
        policy_id=policy_id,
        policy_version=policy_version,
        fingerprint_set_id=reference_set.fingerprint_set_id,
        fingerprint_set_content_sha256=reference_set.content_sha256,
        suite_content_sha256=suite.content_sha256,
        validated=selection is not None,
        distance_threshold=threshold,
        temporal_divergence_baseline=temporal_baseline,
        minimum_comparable_probes=minimum_comparable_probes,
        max_far_target=max_far_target,
        identity_count=identity_count,
        held_out_capture_count=len(outcomes),
        top1_accuracy=top1_accuracy,
        top3_accuracy=top3_accuracy,
        true_positive_rate=true_positive_rate,
        false_accept_rate=false_accept_rate,
    )
    return policy, report


def _minimum_conditions_failure(
    groups: dict[tuple[str, str], list[FingerprintReferenceSnapshot]],
    *,
    identity_count: int,
    minimum_distinct_identities: int,
    minimum_captures_per_identity: int,
) -> str | None:
    """Step 18.1 的 fail-closed 最低条件；返回不满足的原因（满足则 ``None``）."""
    if identity_count < minimum_distinct_identities:
        return (
            f"only {identity_count} distinct model identities in the reference set; "
            f"at least {minimum_distinct_identities} are required"
        )
    thin = sorted(key for key, items in groups.items() if len(items) < minimum_captures_per_identity)
    if thin:
        return (
            f"identities without {minimum_captures_per_identity} independent captures: {thin!r}; "
            f"held-out validation needs at least two captures per identity"
        )
    return None


def _unvalidated_result(
    *,
    reference_set: FingerprintReferenceSet,
    suite: FingerprintSuite,
    policy_id: str,
    policy_version: str,
    minimum_comparable_probes: int,
    max_far_target: float,
    minimum_distinct_identities: int,
    minimum_captures_per_identity: int,
    identity_count: int,
    capture_count: int,
    insufficient_reason: str,
) -> tuple[FingerprintDecisionPolicy, FingerprintValidationReport]:
    """最低条件不满足：不跑 held-out、不发布数字、不生成 threshold（Step 18.1）."""
    report = FingerprintValidationReport(
        fingerprint_set_id=reference_set.fingerprint_set_id,
        fingerprint_set_content_sha256=reference_set.content_sha256,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_content_sha256=suite.content_sha256,
        identity_count=identity_count,
        capture_count=capture_count,
        minimum_distinct_identities=minimum_distinct_identities,
        minimum_captures_per_identity=minimum_captures_per_identity,
        minimum_comparable_probes=minimum_comparable_probes,
        max_far_target=max_far_target,
        meets_minimum_conditions=False,
        insufficient_reason=insufficient_reason,
        outcomes=(),
    )
    policy = build_fingerprint_policy(
        policy_id=policy_id,
        policy_version=policy_version,
        fingerprint_set_id=reference_set.fingerprint_set_id,
        fingerprint_set_content_sha256=reference_set.content_sha256,
        suite_content_sha256=suite.content_sha256,
        validated=False,
        minimum_comparable_probes=minimum_comparable_probes,
        max_far_target=max_far_target,
        identity_count=identity_count,
        held_out_capture_count=0,
    )
    return policy, report


def _leave_one_capture_out(
    *,
    snapshots: Sequence[FingerprintReferenceSnapshot],
    groups: dict[tuple[str, str], list[FingerprintReferenceSnapshot]],
    suite: FingerprintSuite,
    minimum_comparable_probes: int,
) -> tuple[FingerprintHoldoutOutcome, ...]:
    """Step 18.2：逐个 capture 留出，参考库为集合减去该 capture."""
    matcher = FingerprintMatcher(suite=suite)
    outcomes: list[FingerprintHoldoutOutcome] = []

    for held_out in snapshots:
        held_out_key = (held_out.provider_id, held_out.model_id)
        references = [
            aggregate_fingerprint_reference(
                snapshots=[snapshot for snapshot in groups[key] if snapshot.snapshot_id != held_out.snapshot_id],
                suite=suite,
            )
            for key in sorted(groups)
        ]
        result = matcher.match(
            candidate=held_out.distributions,
            references=references,
            minimum_comparable_probes=minimum_comparable_probes,
        )

        own_distance: float | None = None
        impostor_distances: list[float] = []
        ranked_keys: list[tuple[str, str]] = []
        for entry in result.entries:
            key = (entry.provider_id, entry.model_id)
            ranked_keys.append(key)
            if key == held_out_key:
                own_distance = entry.distance
            else:
                impostor_distances.append(entry.distance)

        if own_distance is None:  # pragma: no cover - 最低条件保证自身 identity 仍在库中
            raise FingerprintValidationError(
                f"held-out capture '{held_out.snapshot_id}' has no reference of its own identity; "
                f"reference library is incomplete"
            )

        outcomes.append(
            FingerprintHoldoutOutcome(
                snapshot_id=held_out.snapshot_id,
                model_id=held_out.model_id,
                provider_id=held_out.provider_id,
                own_distance=own_distance,
                nearest_impostor_distance=min(impostor_distances) if impostor_distances else None,
                top1_correct=ranked_keys[0] == held_out_key,
                top3_correct=held_out_key in ranked_keys[:3],
            )
        )

    return tuple(outcomes)


def _assert_set_matches_snapshots(
    reference_set: FingerprintReferenceSet,
    snapshots: Sequence[FingerprintReferenceSnapshot],
) -> None:
    """集合成员与传入快照必须一一对应（身份与 id 都要一致），否则 fail closed."""
    members = {member.snapshot_id: member for member in reference_set.members}
    by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}

    missing = sorted(set(members) - set(by_id))
    extra = sorted(set(by_id) - set(members))
    if missing or extra:
        raise FingerprintValidationError(
            f"fingerprint set '{reference_set.fingerprint_set_id}' and the supplied snapshots disagree: "
            f"missing snapshots {missing!r}, unexpected snapshots {extra!r}"
        )

    for snapshot_id, member in members.items():
        snapshot = by_id[snapshot_id]
        if (snapshot.provider_id, snapshot.model_id) != (member.provider_id, member.model_id):
            raise FingerprintValidationError(
                f"fingerprint snapshot '{snapshot_id}' is recorded as "
                f"({member.provider_id!r}, {member.model_id!r}) in the set but declares "
                f"({snapshot.provider_id!r}, {snapshot.model_id!r})"
            )


__all__: list[str] = [
    "MINIMUM_DISTINCT_IDENTITIES",
    "MINIMUM_CAPTURES_PER_IDENTITY",
    "FingerprintValidationError",
    "FingerprintHoldoutOutcome",
    "FingerprintValidationReport",
    "select_distance_threshold",
    "validate_fingerprint_policy",
]
