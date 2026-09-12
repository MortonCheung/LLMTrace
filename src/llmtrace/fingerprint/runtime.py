"""Fingerprint runtime context —— 把参考数据解析成可匹配的输入（Task 21 / Task 36）.

本模块只做**脱机**解析与校验：不发送 HTTP、不创建 provider、不消费 budget。
它把磁盘上的 ``FingerprintReferenceSet`` + 成员快照 + 可选 decision policy 解析成
``FingerprintRuntimeContext``，供 runner 在第一个真实请求之前完成 fail-closed 预检。

歧义一律 fail closed：多个兼容 policy 不会随机挑一个，而是返回 None（只能 RANKED_ONLY）。
没有已验证 policy 时，可比 probe 下限取套件的 probe 总数 —— 即要求套件内每个 probe
都比可，这是最保守的下限（Task 11 不提供隐式默认值，这里的默认值显式且写死在套件维度）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llmtrace.fingerprint.aggregation import (
    AggregatedFingerprintReference,
    FingerprintAggregationError,
    aggregate_fingerprint_reference,
)
from llmtrace.fingerprint.models import FingerprintSuite
from llmtrace.fingerprint.policy import (
    FingerprintDecisionPolicy,
    FingerprintPolicyIntegrityError,
)
from llmtrace.fingerprint.reference import (
    FingerprintReferenceError,
    FingerprintReferenceSnapshot,
)
from llmtrace.fingerprint.reference_set import (
    FingerprintReferenceSet,
    FingerprintReferenceSetIntegrityError,
    FingerprintReferenceSetMember,
)
from llmtrace.fingerprint.repository import (
    FingerprintRepository,
    FingerprintSnapshotIntegrityError,
    FingerprintSnapshotNotFoundError,
)
from llmtrace.fingerprint.suite import (
    compute_generation_config_sha256,
    load_fingerprint_suite,
    verify_fingerprint_suite,
)


class FingerprintRuntimeError(Exception):
    """参考数据无法被诚实地解析成可匹配的输入."""

    error_code = "FINGERPRINT_RUNTIME_ERROR"


@dataclass(frozen=True)
class FingerprintRuntimeContext:
    """一次 run 的指纹参考上下文（预检期解析，执行期只读）."""

    suite: FingerprintSuite
    reference_set: FingerprintReferenceSet
    references: tuple[AggregatedFingerprintReference, ...]

    #: 已验证策略；None 表示本次只能给出 RANKED_ONLY / INCONCLUSIVE（Rule 2）。
    policy: FingerprintDecisionPolicy | None

    minimum_comparable_probes: int

    #: 为什么（没有）使用某个 policy —— 直接进入 run warnings，供审计。
    notes: tuple[str, ...]


def resolve_fingerprint_context(
    *,
    set_path: Path,
    repository: FingerprintRepository,
    repetitions: int,
    suite: FingerprintSuite | None = None,
) -> FingerprintRuntimeContext:
    """解析并校验指纹参考数据（脱机、fail closed）.

    ``repetitions`` 是本次运行将要执行的轮数；参考集必须是用同样轮数采集的，
    否则分布的可比性无法成立。

    Raises:
        FingerprintRuntimeError: set 不可读/不可解析、与套件或重复轮数不兼容、成员快照
            缺失或身份不一致、成员无法聚合，或 policy 内容身份自校验失败。
        ValueError: repetitions < 1。
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")

    resolved_suite = suite if suite is not None else load_fingerprint_suite()
    verify_fingerprint_suite(resolved_suite)

    try:
        raw = set_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FingerprintRuntimeError(f"fingerprint reference set unreadable: {set_path}") from exc
    try:
        reference_set = FingerprintReferenceSet.model_validate_json(raw)
    except ValueError as exc:
        raise FingerprintRuntimeError(f"fingerprint reference set is not a valid set: {set_path}") from exc

    try:
        reference_set.verify_content_hash()
    except FingerprintReferenceSetIntegrityError as exc:
        raise FingerprintRuntimeError(f"fingerprint reference set integrity check failed: {exc}") from exc

    _assert_suite_compatible(reference_set, resolved_suite)
    if reference_set.repetitions != repetitions:
        raise FingerprintRuntimeError(
            f"fingerprint reference set '{reference_set.fingerprint_set_id}' "
            f"v{reference_set.fingerprint_set_version} was captured with {reference_set.repetitions} repetitions, "
            f"but this run uses {repetitions}; distributions would not be comparable"
        )

    snapshots = [_load_member(reference_set, member, repository) for member in reference_set.members]
    references = _aggregate_references(snapshots, resolved_suite)
    policy, notes = _resolve_policy(reference_set, resolved_suite, repository)
    minimum_comparable_probes = policy.minimum_comparable_probes if policy is not None else len(resolved_suite.probes)

    return FingerprintRuntimeContext(
        suite=resolved_suite,
        reference_set=reference_set,
        references=references,
        policy=policy,
        minimum_comparable_probes=minimum_comparable_probes,
        notes=notes,
    )


def _assert_suite_compatible(reference_set: FingerprintReferenceSet, suite: FingerprintSuite) -> None:
    """集合声明的套件身份必须与将要执行的套件逐项一致（fail closed）.

    套件身份不含重复轮数（那是集合属性，由调用方另行校验），这里只比对
    套件 / 归一化策略 / 生成配置三项内容身份。
    """
    expected = (
        suite.suite_id,
        suite.suite_version,
        suite.content_sha256,
        suite.normalization_policy_id,
        suite.normalization_policy_version,
        compute_generation_config_sha256(suite),
    )
    declared = (
        reference_set.suite_id,
        reference_set.suite_version,
        reference_set.suite_content_sha256,
        reference_set.normalization_policy_id,
        reference_set.normalization_policy_version,
        reference_set.generation_config_sha256,
    )
    if declared != expected:
        raise FingerprintRuntimeError(
            f"fingerprint reference set '{reference_set.fingerprint_set_id}' "
            f"v{reference_set.fingerprint_set_version} was built for a different suite / normalization policy / "
            f"generation config than the suite being executed"
        )


def _load_member(
    reference_set: FingerprintReferenceSet,
    member: FingerprintReferenceSetMember,
    repository: FingerprintRepository,
) -> FingerprintReferenceSnapshot:
    """从快照仓库取出成员并校验其磁盘字节身份."""
    try:
        snapshot = repository.snapshots.get(member.snapshot_id)
    except FingerprintSnapshotNotFoundError as exc:
        raise FingerprintRuntimeError(
            f"fingerprint reference set member '{member.snapshot_id}' is not present in the snapshot repository"
        ) from exc

    if snapshot.model_id != member.model_id or snapshot.provider_id != member.provider_id:
        raise FingerprintRuntimeError(
            f"fingerprint reference set member '{member.snapshot_id}' declares "
            f"({member.provider_id!r}, {member.model_id!r}) but the stored snapshot declares "
            f"({snapshot.provider_id!r}, {snapshot.model_id!r})"
        )

    try:
        repository.snapshots.verify(member.snapshot_id, member.snapshot_sha256)
    except (FingerprintSnapshotIntegrityError, FingerprintReferenceError) as exc:
        raise FingerprintRuntimeError(
            f"fingerprint reference set member '{member.snapshot_id}' failed integrity verification: {exc}"
        ) from exc
    return snapshot


def _aggregate_references(
    snapshots: Sequence[FingerprintReferenceSnapshot],
    suite: FingerprintSuite,
) -> tuple[AggregatedFingerprintReference, ...]:
    """按 identity 分组聚合（每次 capture 等权），顺序确定."""
    groups: dict[tuple[str, str], list[FingerprintReferenceSnapshot]] = {}
    for snapshot in snapshots:
        groups.setdefault((snapshot.provider_id, snapshot.model_id), []).append(snapshot)

    references: list[AggregatedFingerprintReference] = []
    for key in sorted(groups):
        try:
            references.append(aggregate_fingerprint_reference(snapshots=groups[key], suite=suite))
        except (FingerprintAggregationError, FingerprintReferenceError) as exc:
            raise FingerprintRuntimeError(
                f"fingerprint reference identity {key!r} cannot be aggregated: {exc}"
            ) from exc
    return tuple(references)


def _resolve_policy(
    reference_set: FingerprintReferenceSet,
    suite: FingerprintSuite,
    repository: FingerprintRepository,
) -> tuple[FingerprintDecisionPolicy | None, tuple[str, ...]]:
    """按 (set id, set content hash, suite content hash) 发现唯一兼容 policy.

    多个兼容 policy 一律不选（不随机挑）；没有兼容 policy 时返回 None 并给出 note。
    未验证的唯一 policy 仍会被使用（它的存在本身就是审计信息），但 matcher 只会给
    RANKED_ONLY（Rule 2）。
    """
    compatible = [
        policy
        for policy in repository.policies.list()
        if policy.fingerprint_set_id == reference_set.fingerprint_set_id
        and policy.fingerprint_set_content_sha256 == reference_set.content_sha256
        and policy.suite_content_sha256 == suite.content_sha256
    ]

    if len(compatible) > 1:
        return None, (
            f"{len(compatible)} compatible fingerprint policies found; refusing to choose one — ranking only (Rule 2)",
        )
    if not compatible:
        return None, ("no compatible fingerprint policy found; no claim verdict will be produced (Rule 2)",)

    policy = compatible[0]
    try:
        policy.verify_content_hash()
    except FingerprintPolicyIntegrityError as exc:
        raise FingerprintRuntimeError(
            f"fingerprint policy '{policy.policy_id}' v{policy.policy_version} failed integrity verification: {exc}"
        ) from exc

    if not policy.validated:
        return policy, (
            f"fingerprint policy '{policy.policy_id}' v{policy.policy_version} is not validated; "
            f"no claim verdict will be produced (Rule 2)",
        )
    return policy, ()


__all__: list[str] = [
    "FingerprintRuntimeContext",
    "FingerprintRuntimeError",
    "resolve_fingerprint_context",
]
