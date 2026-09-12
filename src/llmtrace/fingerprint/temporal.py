"""Temporal fingerprint distribution（Task 29 / Task 30）.

Task 29 —— 同一次 capture 内部按轮次分半，形成两个 temporal window：

    round 0–3  vs  round 4–7        （STANDARD，8 repetitions）

两个 window 各自按 probe 形成分布，再用**已有的** JSD 距离机制（Task 10 / Task 11）
计算 aggregate divergence。轮数不足最低要求时，temporal fingerprint 明确记
``UNAVAILABLE`` —— 不用 2 个样本去算"漂移"。

Task 30 —— 验证阶段对每个 reference capture 做同样的分半，收集
``within_reference_temporal_distances``，得到"同一 identity 自身在时间窗之间
本来就会漂多少"的 baseline。该 baseline 只在 policy ``validated == True`` 时
才允许作为 threshold 使用（Rule 2）：未验证的 policy 不得携带它。

边界（Rule 3）：本模块只度量"同一 capture 内部两个窗口之间的距离"，不比较两个
identity，因此不产生任何 identity 结论。
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from llmtrace.fingerprint.distance import (
    FingerprintDistance,
    FingerprintDistanceError,
    compute_fingerprint_distance,
)
from llmtrace.fingerprint.distribution import build_probe_distribution
from llmtrace.fingerprint.models import FingerprintSuite, ProbeDistribution
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot
from llmtrace.fingerprint.suite import compute_generation_config_sha256, verify_fingerprint_suite

#: 每个时间窗至少要有的轮数；"2 个样本算漂移"被明确排除（Task 29）。
MINIMUM_ROUNDS_PER_WINDOW = 2

#: 能形成两个时间窗所需的最低 repetitions。
MINIMUM_TEMPORAL_REPETITIONS = MINIMUM_ROUNDS_PER_WINDOW * 2


class FingerprintTemporalError(Exception):
    """temporal 输入自相矛盾（snapshot 与套件不属于同一次声明）."""

    error_code = "FINGERPRINT_TEMPORAL_ERROR"


class TemporalFingerprintStatus(StrEnum):
    """temporal fingerprint 是否可用；不可用时必须给出原因，不做静默降级."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class TemporalWindow(BaseModel):
    """一个轮次区间内的行为分布；``round_end`` 为开区间上界."""

    window_id: str = Field(..., min_length=1)

    round_start: int = Field(..., ge=0, description="Inclusive first round index")
    round_end: int = Field(..., ge=0, description="Exclusive last round index")

    distributions: tuple[ProbeDistribution, ...]

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_window(self) -> TemporalWindow:
        span = self.round_end - self.round_start
        if span < MINIMUM_ROUNDS_PER_WINDOW:
            raise ValueError(
                f"temporal window must span at least {MINIMUM_ROUNDS_PER_WINDOW} rounds, got {span} "
                f"({self.round_start}..{self.round_end})"
            )
        if not self.distributions:
            raise ValueError("a temporal window must carry at least one probe distribution")
        probe_ids = [distribution.probe_id for distribution in self.distributions]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError(f"duplicate probe_id in temporal window distributions: {probe_ids}")
        return self

    @property
    def round_count(self) -> int:
        """该窗口覆盖的轮数."""
        return self.round_end - self.round_start


class TemporalFingerprint(BaseModel):
    """同一次 capture 的两个时间窗，以及它们之间的聚合 JSD（Task 29）."""

    status: TemporalFingerprintStatus

    repetitions: int = Field(..., ge=1)
    minimum_repetitions: int = Field(default=MINIMUM_TEMPORAL_REPETITIONS, ge=1)

    windows: tuple[TemporalWindow, ...] = ()

    divergence: FingerprintDistance | None = None

    reason: str | None = None

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_status(self) -> TemporalFingerprint:
        if self.status is TemporalFingerprintStatus.AVAILABLE:
            if len(self.windows) != 2:
                raise ValueError(f"an available temporal fingerprint needs exactly 2 windows, got {len(self.windows)}")
            if self.divergence is None:
                raise ValueError("an available temporal fingerprint must carry its aggregate divergence")
            if self.reason is not None:
                raise ValueError("an available temporal fingerprint must not carry an unavailability reason")
        else:
            if self.windows:
                raise ValueError("an unavailable temporal fingerprint must not carry windows")
            if self.divergence is not None:
                raise ValueError("an unavailable temporal fingerprint must not carry a divergence")
            if not self.reason:
                raise ValueError("an unavailable temporal fingerprint must explain itself via reason")
        return self


def split_temporal_windows(repetitions: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """把 ``repetitions`` 轮分成前后两个区间；轮数不足时返回 ``None``.

    ``8`` → ``((0, 4), (4, 8))``（即报告里的 round 0–3 / round 4–7）。
    奇数轮时余数归入第二个窗口，两个窗口都至少 ``MINIMUM_ROUNDS_PER_WINDOW`` 轮。
    """
    if repetitions < MINIMUM_TEMPORAL_REPETITIONS:
        return None
    split = repetitions // 2
    return ((0, split), (split, repetitions))


def build_temporal_fingerprint(
    *,
    snapshot: FingerprintReferenceSnapshot,
    suite: FingerprintSuite,
    minimum_comparable_probes: int,
) -> TemporalFingerprint:
    """把一个 capture 的轮次分半，计算两个时间窗之间的聚合 JSD（Task 29）.

    Raises:
        FingerprintTemporalError: snapshot 与套件的 (suite, policy, generation
            config, repetitions) 声明不一致。
        FingerprintReferenceError: snapshot 内容身份自校验失败。
        FingerprintSuiteIntegrityError: 套件内容身份自校验失败。
    """
    verify_fingerprint_suite(suite)
    snapshot.verify_content_hash()
    _assert_snapshot_compatible(snapshot, suite)
    if minimum_comparable_probes < 1:
        raise FingerprintTemporalError(f"minimum_comparable_probes must be >= 1, got {minimum_comparable_probes}")

    split = split_temporal_windows(snapshot.repetitions)
    if split is None:
        return TemporalFingerprint(
            status=TemporalFingerprintStatus.UNAVAILABLE,
            repetitions=snapshot.repetitions,
            reason=(
                f"{snapshot.repetitions} repetitions cannot support two temporal windows of "
                f"{MINIMUM_ROUNDS_PER_WINDOW} rounds; at least {MINIMUM_TEMPORAL_REPETITIONS} are required "
                f"(no drift estimate from two samples)"
            ),
        )

    windows = tuple(
        _build_window(
            snapshot=snapshot,
            suite=suite,
            window_id=f"rounds_{round_start}_{round_end - 1}",
            round_start=round_start,
            round_end=round_end,
        )
        for round_start, round_end in split
    )

    try:
        divergence = compute_fingerprint_distance(
            probes=suite.probes,
            candidate=windows[0].distributions,
            reference=windows[1].distributions,
            minimum_comparable_probes=minimum_comparable_probes,
        )
    except FingerprintDistanceError as exc:
        # 窗口不可比（缺样本 / 空样本 / support 不一致）时不得给出漂移数字。
        return TemporalFingerprint(
            status=TemporalFingerprintStatus.UNAVAILABLE,
            repetitions=snapshot.repetitions,
            reason=f"temporal windows of '{snapshot.snapshot_id}' are not comparable: {exc}",
        )

    return TemporalFingerprint(
        status=TemporalFingerprintStatus.AVAILABLE,
        repetitions=snapshot.repetitions,
        windows=windows,
        divergence=divergence,
    )


def collect_within_reference_temporal_distances(
    *,
    snapshots: Sequence[FingerprintReferenceSnapshot],
    suite: FingerprintSuite,
    minimum_comparable_probes: int,
) -> tuple[float, ...]:
    """验证阶段收集每个 reference capture 的窗口间距离（Task 30）.

    顺序按 ``snapshot_id`` 排序，保证相同输入得到逐字节相同的元组。轮数不足以形成
    两个窗口的 capture 不会贡献数字（其 temporal fingerprint 是 ``UNAVAILABLE``），
    这一点由调用方在报告里以"baseline 缺失"体现，而不是被当成 0。
    """
    distances: list[float] = []
    for snapshot in sorted(snapshots, key=lambda item: item.snapshot_id):
        temporal = build_temporal_fingerprint(
            snapshot=snapshot,
            suite=suite,
            minimum_comparable_probes=minimum_comparable_probes,
        )
        if temporal.status is TemporalFingerprintStatus.AVAILABLE and temporal.divergence is not None:
            distances.append(temporal.divergence.distance)
    return tuple(distances)


def temporal_divergence_baseline(distances: Sequence[float]) -> float | None:
    """把 ``within_reference_temporal_distances`` 收成一个保守 baseline（Task 30）.

    取**最大值**：既然同一 identity 在自身窗口之间最大也曾漂到这个程度，只有在候选
    超过它时才算"超出已验证的参考基线"，这是六个统计量里最不容易误报的选择。
    没有可用距离时返回 ``None``（不伪造 0）。
    """
    if not distances:
        return None
    return max(distances)


def _build_window(
    *,
    snapshot: FingerprintReferenceSnapshot,
    suite: FingerprintSuite,
    window_id: str,
    round_start: int,
    round_end: int,
) -> TemporalWindow:
    """按套件声明顺序为窗口内的轮次构建每个 probe 的分布."""
    selected = [
        observation for observation in snapshot.observations if round_start <= observation.round_index < round_end
    ]
    return TemporalWindow(
        window_id=window_id,
        round_start=round_start,
        round_end=round_end,
        distributions=tuple(build_probe_distribution(probe, selected) for probe in suite.probes),
    )


def _assert_snapshot_compatible(snapshot: FingerprintReferenceSnapshot, suite: FingerprintSuite) -> None:
    """snapshot 必须声明与套件完全相同的 (suite, policy, generation config)."""
    declared = (
        snapshot.suite_id,
        snapshot.suite_version,
        snapshot.suite_content_sha256,
        snapshot.normalization_policy_id,
        snapshot.normalization_policy_version,
        snapshot.generation_config_sha256,
    )
    expected = (
        suite.suite_id,
        suite.suite_version,
        suite.content_sha256,
        suite.normalization_policy_id,
        suite.normalization_policy_version,
        compute_generation_config_sha256(suite),
    )
    if declared != expected:
        raise FingerprintTemporalError(
            f"fingerprint snapshot '{snapshot.snapshot_id}' is not compatible with suite "
            f"'{suite.suite_id}' v{suite.suite_version}: declared (suite, policy, generation config) "
            f"identity differs from the suite being measured"
        )


__all__: list[str] = [
    "MINIMUM_ROUNDS_PER_WINDOW",
    "MINIMUM_TEMPORAL_REPETITIONS",
    "FingerprintTemporalError",
    "TemporalFingerprintStatus",
    "TemporalWindow",
    "TemporalFingerprint",
    "split_temporal_windows",
    "build_temporal_fingerprint",
    "collect_within_reference_temporal_distances",
    "temporal_divergence_baseline",
]
