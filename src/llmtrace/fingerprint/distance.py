"""Fingerprint 距离（Task 10 / Task 11）.

纯 Python 实现，只依赖 stdlib ``math``：不引入 numpy / scipy / sklearn /
sentence-transformers / chromadb（Rule 5 技术梯级）。

语义边界：

- 这里只产出"行为分布距离"，不产出任何 identity 结论；
- 距离只在 support 完全一致且两侧均有样本时计算，否则该 probe 以显式 note
  记为 not comparable —— 不静默跳过、不缩小分母（fail closed with coverage）。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

from pydantic import BaseModel, Field, model_validator

from llmtrace.fingerprint.models import (
    AggregatedProbeDistribution,
    FingerprintProbe,
    ProbeDistribution,
)

#: JSD 只依赖 ``probe_id`` / ``support`` / ``sample_count`` / ``ordered_probabilities()``：
#: 单次 capture 的 ``ProbeDistribution`` 与多次 capture 聚合出的
#: ``AggregatedProbeDistribution``（Task 15）都满足这一形状。
BehavioralDistribution = ProbeDistribution | AggregatedProbeDistribution

#: 概率归一化容差：仅用于识别"概率与计数不一致"的构造错误。
_NORMALIZATION_TOLERANCE = 1e-6


class FingerprintDistanceError(Exception):
    """距离计算无法进行（comparable probe 不足或输入自相矛盾）."""

    error_code = "FINGERPRINT_DISTANCE_ERROR"


def _kl(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    """KL 散度 ``D(p || q)``；``p`` 中为 0 的项按定义贡献 0."""
    value = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi == 0.0:
            continue
        value += pi * log2(pi / qi)
    return value


def js_divergence(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    """Jensen–Shannon 散度（以 2 为底，取值落在 ``[0, 1]``）."""
    if len(p) != len(q):
        raise ValueError(f"distribution lengths differ: {len(p)} != {len(q)}")
    midpoint = tuple((pi + qi) / 2.0 for pi, qi in zip(p, q, strict=True))
    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


class ProbeDistance(BaseModel):
    """单个 probe 的距离；不可比时以显式 note 说明原因（Task 11）."""

    probe_id: str = Field(..., min_length=1)

    weight: float = Field(..., gt=0.0)

    comparable: bool
    distance: float | None = Field(default=None, ge=0.0, le=1.0)

    note: str | None = None

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_comparability(self) -> ProbeDistance:
        if self.comparable:
            if self.distance is None:
                raise ValueError("a comparable probe distance must carry a distance value")
            if self.note is not None:
                raise ValueError("a comparable probe distance must not carry a note")
        else:
            if self.distance is not None:
                raise ValueError("a non-comparable probe must not carry a distance value")
            if not self.note:
                raise ValueError("a non-comparable probe must explain itself via note")
        return self


class FingerprintDistance(BaseModel):
    """候选与参考之间的加权聚合距离（Task 11）."""

    distance: float = Field(..., ge=0.0, le=1.0)

    comparable_probes: int = Field(..., ge=0)

    sample_count_candidate: int = Field(..., ge=0)
    sample_count_reference: int = Field(..., ge=0)

    per_probe: tuple[ProbeDistance, ...]

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_coverage(self) -> FingerprintDistance:
        actual = sum(1 for probe in self.per_probe if probe.comparable)
        if actual != self.comparable_probes:
            raise ValueError(
                f"comparable_probes ({self.comparable_probes}) must equal the number of "
                f"comparable per_probe entries ({actual})"
            )
        if self.comparable_probes == 0:
            raise ValueError("a fingerprint distance requires at least one comparable probe")
        return self


def _comparability_gate(
    candidate: BehavioralDistribution | None,
    reference: BehavioralDistribution | None,
) -> str | None:
    """返回不可比原因；``None`` 表示可比."""
    if candidate is None:
        return "missing_candidate_distribution"
    if reference is None:
        return "missing_reference_distribution"
    if candidate.sample_count == 0:
        return "empty_candidate_samples"
    if reference.sample_count == 0:
        return "empty_reference_samples"
    if candidate.support != reference.support:
        # support 有序比较：集合相同但顺序不同也会破坏 zip 对齐，故一并拒绝。
        return "support_mismatch"
    for distribution, label in ((candidate, "candidate"), (reference, "reference")):
        total = sum(distribution.probabilities[item] for item in distribution.support)
        if abs(total - 1.0) > _NORMALIZATION_TOLERANCE:
            return f"unnormalized_{label}_distribution"
    return None


def compute_fingerprint_distance(
    *,
    probes: Sequence[FingerprintProbe],
    candidate: Sequence[BehavioralDistribution],
    reference: Sequence[BehavioralDistribution],
    minimum_comparable_probes: int,
) -> FingerprintDistance:
    """按 ``sum(weight * distance) / sum(weight)`` 聚合候选与参考的距离.

    ``reference`` 既可以是单次 capture 的 ``ProbeDistribution``，也可以是同一
    identity 多次 capture 聚合出的 ``AggregatedProbeDistribution``（Task 15）。

    ``minimum_comparable_probes`` 必须由调用方显式给出（来自 decision policy），
    本函数不提供隐式默认值（Task 11）。

    Raises:
        FingerprintDistanceError: 输入含重复 probe_id、``minimum_comparable_probes < 1``，
            或可比 probe 数低于该下限（fail closed）。
    """
    if minimum_comparable_probes < 1:
        raise FingerprintDistanceError(f"minimum_comparable_probes must be >= 1, got {minimum_comparable_probes}")

    candidate_by_id = _index_by_probe_id(candidate, "candidate")
    reference_by_id = _index_by_probe_id(reference, "reference")

    per_probe: list[ProbeDistance] = []
    weighted_sum = 0.0
    active_weight = 0.0
    sample_count_candidate = 0
    sample_count_reference = 0

    for probe in probes:
        candidate_distribution = candidate_by_id.get(probe.probe_id)
        reference_distribution = reference_by_id.get(probe.probe_id)

        reason = _comparability_gate(candidate_distribution, reference_distribution)
        if reason is not None:
            per_probe.append(ProbeDistance(probe_id=probe.probe_id, weight=probe.weight, comparable=False, note=reason))
            continue

        assert candidate_distribution is not None and reference_distribution is not None
        raw = js_divergence(
            candidate_distribution.ordered_probabilities(),
            reference_distribution.ordered_probabilities(),
        )
        # 浮点误差可能让 JSD 略微越过 1.0；夹取而不是拒绝。
        distance = min(1.0, max(0.0, raw))

        per_probe.append(
            ProbeDistance(probe_id=probe.probe_id, weight=probe.weight, comparable=True, distance=distance)
        )
        weighted_sum += probe.weight * distance
        active_weight += probe.weight
        sample_count_candidate += candidate_distribution.sample_count
        sample_count_reference += reference_distribution.sample_count

    comparable_probes = sum(1 for entry in per_probe if entry.comparable)
    if comparable_probes < minimum_comparable_probes:
        raise FingerprintDistanceError(
            f"comparable probes ({comparable_probes}) below the policy minimum "
            f"({minimum_comparable_probes}); refusing to aggregate"
        )

    return FingerprintDistance(
        distance=min(1.0, max(0.0, weighted_sum / active_weight)),
        comparable_probes=comparable_probes,
        sample_count_candidate=sample_count_candidate,
        sample_count_reference=sample_count_reference,
        per_probe=tuple(per_probe),
    )


def _index_by_probe_id(
    distributions: Sequence[BehavioralDistribution],
    label: str,
) -> dict[str, BehavioralDistribution]:
    """按 probe_id 建索引；重复 id 直接失败，避免静默覆盖."""
    indexed: dict[str, BehavioralDistribution] = {}
    for distribution in distributions:
        if distribution.probe_id in indexed:
            raise FingerprintDistanceError(f"duplicate probe_id in {label} distributions: {distribution.probe_id!r}")
        indexed[distribution.probe_id] = distribution
    return indexed


__all__: list[str] = [
    "BehavioralDistribution",
    "FingerprintDistanceError",
    "ProbeDistance",
    "FingerprintDistance",
    "js_divergence",
    "compute_fingerprint_distance",
]
