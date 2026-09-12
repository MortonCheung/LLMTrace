"""Routing v2（Task 28 / Task 31）.

v1（``analysis/routing.py``，保留不动）只看服务端自报的 model identifier 与失败率。
v2 沿用同一组标签，但把证据分成两层强度：

    强证据（strong）       足以单独支持"路由不一致"的判断
    支持证据（supporting） 只能与其它观察一起支持判断，单独不构成强主张

并接入 fingerprint 的时间维证据（Task 29 / Task 30）：候选 capture 自身的窗口间
散度只有超过"已验证 policy"携带的参考基线、或已验证参考匹配在不同时间窗之间切换
时，才算强证据。

文本纪律（Task 31 / Task 48）：只输出既有四个标签，禁止任何 "70% GPT / 30% Claude"
式的伪造比例；措辞只允许 "routing inconsistency observed" / "mixed routing is
possible" 这一类行为层面的描述。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from statistics import pstdev

from pydantic import BaseModel, Field, model_validator

from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.analysis.routing import RoutingStabilityLevel
from llmtrace.benchmarks.models import ItemStatus
from llmtrace.fingerprint.matcher import FingerprintMatchResult
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy
from llmtrace.fingerprint.temporal import TemporalFingerprint, TemporalFingerprintStatus

#: 强证据：服务端在样本间给出多个稳定出现、没有单一主导的 model identifier。
SIGNAL_MULTIPLE_RESPONSE_MODELS = "multiple_stable_response_model_identifiers"
#: 强证据：已验证参考匹配在两个时间窗之间切换（Task 29 的窗口切分）。
SIGNAL_REFERENCE_MATCH_SWITCH = "reference_match_switches_between_temporal_windows"
#: 强证据：候选的时间散度超过已验证 policy 携带的参考基线（Task 30）。
SIGNAL_TEMPORAL_DIVERGENCE = "temporal_divergence_exceeds_validated_reference_baseline"
#: 支持证据：延迟离散度。
SIGNAL_LATENCY_DISPERSION = "latency_dispersion"
#: 支持证据：输出 token 离散度。
SIGNAL_TOKEN_DISPERSION = "token_dispersion"
#: 支持证据：provider 失败率。
SIGNAL_PROVIDER_FAILURE = "provider_failure"


class RoutingEvidenceStrength(StrEnum):
    """路由证据强度（Task 28 Step 28.2）."""

    STRONG = "strong"
    SUPPORTING = "supporting"


class RoutingSignal(BaseModel):
    """一条具名路由信号：是否观察到，以及为什么（Task 28 Step 28.1）."""

    signal_id: str = Field(..., min_length=1)
    strength: RoutingEvidenceStrength
    observed: bool
    reason: str = Field(..., min_length=1)

    model_config = {"frozen": True, "extra": "forbid"}


class RoutingV2Policy(BaseModel):
    """v2 的版本化阈值；改阈值必须换版本（与 v1 / ConfidencePolicy 同纪律）."""

    policy_id: str = "llmtrace-routing-v2"
    policy_version: str = "2.0.0"

    minimum_samples: int = Field(default=8, ge=1)
    minimum_dominant_ratio: float = Field(default=0.9, gt=0.0, le=1.0)
    degraded_failure_ratio_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    dispersion_coefficient_threshold: float = Field(default=0.5, ge=0.0)

    model_config = {"frozen": True, "extra": "forbid"}

    @classmethod
    def create_v2(cls) -> RoutingV2Policy:
        return cls()


class RoutingAssessmentV2(BaseModel):
    """一次 routing v2 判定：四个既有标签之一 + 全部信号与限制."""

    level: RoutingStabilityLevel

    policy_id: str = Field(..., min_length=1)
    policy_version: str = Field(..., min_length=1)
    experimental: bool = Field(default=True, description="Experimental label, not statistical proof")

    item_count: int = Field(..., ge=0)
    reported_model_count: int = Field(..., ge=0)
    response_model_coverage: float = Field(..., ge=0.0, le=1.0)
    distinct_response_models: int = Field(..., ge=0)
    dominant_model_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_ratio: float = Field(..., ge=0.0, le=1.0)

    temporal_status: TemporalFingerprintStatus

    signals: tuple[RoutingSignal, ...]
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_signals(self) -> RoutingAssessmentV2:
        signal_ids = [signal.signal_id for signal in self.signals]
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError(f"routing signals must be unique, got {signal_ids!r}")
        return self


def assess_routing_v2(
    *,
    snapshot: BehaviorRunSnapshot,
    temporal: TemporalFingerprint | None = None,
    window_matches: Sequence[FingerprintMatchResult] = (),
    fingerprint_policy: FingerprintDecisionPolicy | None = None,
    policy: RoutingV2Policy | None = None,
) -> RoutingAssessmentV2:
    """按强/支持两层证据给出 routing 标签（Task 31）.

    判定顺序（fail closed）：

    1. ``Insufficient Data`` —— 样本数不足，或服务端从未在任一 item 上报告 model
       identifier。此时候选连"服务端自称是谁"都无法确认。
    2. ``Suspicious`` —— 至少一条强证据被观察到。
    3. ``Mostly Stable`` —— 无强证据，但存在支持证据、存在游离的第二个 identifier，
       或没有可用的 temporal fingerprint（无法正面确认时间稳定性）。
    4. ``Stable`` —— 无强证据、无支持证据、单一 model identifier，且 temporal
       fingerprint 可用。

    Args:
        snapshot: 本次 run 的行为观测（复用 v1 的输入模型）。
        temporal: 候选 capture 的时间窗散度（Task 29）；``None`` 表示未采集。
        window_matches: 每个时间窗各自的匹配结果（调用方用 ``FingerprintMatcher``
            逐窗计算）。少于两个窗口时无法比较，不会产生切换强证据。
        fingerprint_policy: 本次使用的判定规则；未提供或未验证时，所有 temporal
            强证据一律不成立，并记为 limitation（Rule 2）。
        policy: v2 阈值；缺省用 ``RoutingV2Policy``。

    Raises:
        ValueError: ``window_matches`` 与 ``temporal`` 的窗口数不一致（调用方错误）。
    """
    active = policy if policy is not None else RoutingV2Policy()

    items = snapshot.items
    total = len(items)
    reported = [item.response_model for item in items if item.response_model is not None]
    counts = Counter(reported)
    distinct = len(counts)
    dominant_ratio = counts.most_common(1)[0][1] / len(reported) if reported else None
    coverage = len(reported) / total if total else 0.0
    failures = sum(1 for item in items if item.status is ItemStatus.FAILURE)
    failure_ratio = failures / total if total else 0.0

    if temporal is not None and temporal.windows and len(window_matches) not in (0, len(temporal.windows)):
        raise ValueError(
            f"window_matches ({len(window_matches)}) must be empty or match the temporal windows "
            f"({len(temporal.windows)})"
        )

    reasons: list[str] = [f"{total} behavior samples", f"{distinct} distinct reported model identifier(s)"]
    limitations: list[str] = []
    temporal_available = temporal is not None and temporal.status is TemporalFingerprintStatus.AVAILABLE
    if temporal is None:
        limitations.append("no temporal fingerprint was captured; temporal identity stability was not assessed")
    elif not temporal_available:
        limitations.append(f"temporal fingerprint unavailable: {temporal.reason}")
    if fingerprint_policy is None or not fingerprint_policy.validated:
        limitations.append(
            "no validated decision policy is available; temporal evidence cannot support an identity verdict"
        )

    latency_values: list[float] = []
    token_values: list[float] = []
    for item in items:
        if item.latency_ms is not None:
            latency_values.append(item.latency_ms)
        if item.output_tokens is not None:
            token_values.append(float(item.output_tokens))

    signals = (
        _response_model_signal(distinct, dominant_ratio, active),
        _reference_switch_signal(window_matches, fingerprint_policy),
        _temporal_divergence_signal(temporal, temporal_available, fingerprint_policy, limitations),
        _dispersion_signal(
            signal_id=SIGNAL_LATENCY_DISPERSION,
            label="latency",
            values=latency_values,
            threshold=active.dispersion_coefficient_threshold,
        ),
        _dispersion_signal(
            signal_id=SIGNAL_TOKEN_DISPERSION,
            label="output token",
            values=token_values,
            threshold=active.dispersion_coefficient_threshold,
        ),
        _provider_failure_signal(
            failures=failures,
            total=total,
            failure_ratio=failure_ratio,
            threshold=active.degraded_failure_ratio_threshold,
        ),
    )

    strong = [
        signal.signal_id for signal in signals if signal.strength is RoutingEvidenceStrength.STRONG and signal.observed
    ]
    supporting = [
        signal.signal_id
        for signal in signals
        if signal.strength is RoutingEvidenceStrength.SUPPORTING and signal.observed
    ]

    if total < active.minimum_samples:
        level = RoutingStabilityLevel.INSUFFICIENT_DATA
        reasons.append(f"only {total} behavior samples; at least {active.minimum_samples} are required")
    elif not reported:
        level = RoutingStabilityLevel.INSUFFICIENT_DATA
        reasons.append("the server reported no model identifier on any item")
    elif strong:
        level = RoutingStabilityLevel.SUSPICIOUS
        reasons.append(f"strong routing evidence observed: {', '.join(strong)}")
        reasons.append("routing inconsistency observed; mixed routing is possible")
    elif distinct > 1 or supporting or not temporal_available:
        level = RoutingStabilityLevel.MOSTLY_STABLE
        if distinct > 1 and dominant_ratio is not None:
            reasons.append(f"one dominant model ({dominant_ratio:.0%}) with stray identifiers")
        if supporting:
            reasons.append(f"supporting routing evidence observed: {', '.join(supporting)}")
        if not temporal_available:
            reasons.append("identity stability was not positively confirmed by temporal fingerprint evidence")
    else:
        level = RoutingStabilityLevel.STABLE
        reasons.append("single reported model identifier with a stable temporal fingerprint")

    return RoutingAssessmentV2(
        level=level,
        policy_id=active.policy_id,
        policy_version=active.policy_version,
        item_count=total,
        reported_model_count=len(reported),
        response_model_coverage=coverage,
        distinct_response_models=distinct,
        dominant_model_ratio=dominant_ratio,
        failure_ratio=failure_ratio,
        temporal_status=(temporal.status if temporal is not None else TemporalFingerprintStatus.UNAVAILABLE),
        signals=signals,
        reasons=tuple(reasons),
        limitations=tuple(limitations),
    )


def _response_model_signal(
    distinct: int,
    dominant_ratio: float | None,
    policy: RoutingV2Policy,
) -> RoutingSignal:
    """多个稳定 identifier 且无单一主导 → 强证据；单个游离 identifier 不算."""
    split = distinct >= 2 and dominant_ratio is not None and dominant_ratio < policy.minimum_dominant_ratio
    if distinct < 2 or dominant_ratio is None:
        reason = "a single response_model identifier was reported"
    elif split:
        reason = (
            f"{distinct} response_model identifiers reported with no dominant one "
            f"(largest share {dominant_ratio:.0%} < {policy.minimum_dominant_ratio:.0%})"
        )
    else:
        reason = (
            f"{distinct} response_model identifiers reported, largest share {dominant_ratio:.0%} "
            f">= {policy.minimum_dominant_ratio:.0%} (stray identifier, not a model split)"
        )
    return RoutingSignal(
        signal_id=SIGNAL_MULTIPLE_RESPONSE_MODELS,
        strength=RoutingEvidenceStrength.STRONG,
        observed=split,
        reason=reason,
    )


def _reference_switch_signal(
    window_matches: Sequence[FingerprintMatchResult],
    policy: FingerprintDecisionPolicy | None,
) -> RoutingSignal:
    """已验证参考匹配在两个时间窗之间切换 → 强证据（Rule 2 门禁）."""
    observed = False
    if policy is None or not policy.validated:
        reason = "no validated decision policy; temporal window matches cannot support an identity verdict"
    elif len(window_matches) < 2:
        reason = f"{len(window_matches)} temporal window match(es) available; at least 2 are required"
    else:
        top_identities = {
            (match.entries[0].provider_id, match.entries[0].model_id) if match.entries else None
            for match in window_matches
        }
        statuses = {match.status for match in window_matches}
        observed = len(top_identities) > 1 or len(statuses) > 1
        reason = (
            f"the closest behavioral reference changes across {len(window_matches)} temporal windows"
            if observed
            else f"the closest behavioral reference is stable across {len(window_matches)} temporal windows"
        )
    return RoutingSignal(
        signal_id=SIGNAL_REFERENCE_MATCH_SWITCH,
        strength=RoutingEvidenceStrength.STRONG,
        observed=observed,
        reason=reason,
    )


def _temporal_divergence_signal(
    temporal: TemporalFingerprint | None,
    temporal_available: bool,
    policy: FingerprintDecisionPolicy | None,
    limitations: list[str],
) -> RoutingSignal:
    """候选散度超过已验证参考基线 → 强证据（Task 30 threshold 门禁）."""
    observed = False
    if policy is None or not policy.validated:
        reason = "no validated decision policy; there is no reference temporal baseline"
    elif policy.temporal_divergence_baseline is None:
        reason = "the validated decision policy carries no reference temporal baseline"
        limitations.append(
            "the validated policy carries no temporal baseline; temporal divergence could not be compared"
        )
    elif not temporal_available or temporal is None or temporal.divergence is None:
        reason = "temporal fingerprint is unavailable for this capture"
    else:
        baseline = policy.temporal_divergence_baseline
        observed = temporal.divergence.distance > baseline
        reason = (
            f"temporal divergence {temporal.divergence.distance:.4f} exceeds reference baseline {baseline:.4f}"
            if observed
            else f"temporal divergence {temporal.divergence.distance:.4f} within reference baseline {baseline:.4f}"
        )
    return RoutingSignal(
        signal_id=SIGNAL_TEMPORAL_DIVERGENCE,
        strength=RoutingEvidenceStrength.STRONG,
        observed=observed,
        reason=reason,
    )


def _dispersion_signal(
    *,
    signal_id: str,
    label: str,
    values: Sequence[float],
    threshold: float,
) -> RoutingSignal:
    """离散度只作支持证据：单独出现时不得升级为强主张（Task 54）."""
    coefficient = _coefficient_of_variation(values)
    if coefficient is None:
        reason = f"{label} dispersion unavailable (fewer than 2 samples or a zero mean)"
        observed = False
    else:
        observed = coefficient > threshold
        reason = f"{label} dispersion coefficient {coefficient:.2f} {'>' if observed else '<='} {threshold:.2f}"
    return RoutingSignal(
        signal_id=signal_id,
        strength=RoutingEvidenceStrength.SUPPORTING,
        observed=observed,
        reason=reason,
    )


def _provider_failure_signal(
    *,
    failures: int,
    total: int,
    failure_ratio: float,
    threshold: float,
) -> RoutingSignal:
    """失败率只作支持证据：v1 里的"高失败率即 Suspicious"在 v2 不再单独成立."""
    observed = failure_ratio >= threshold
    return RoutingSignal(
        signal_id=SIGNAL_PROVIDER_FAILURE,
        strength=RoutingEvidenceStrength.SUPPORTING,
        observed=observed,
        reason=f"{failures}/{total} provider failures (ratio {failure_ratio:.2f}, threshold {threshold:.2f})",
    )


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    """变异系数（population stdev / mean）；样本不足或均值为 0 时返回 ``None``."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean <= 0.0:
        return None
    return pstdev(values) / mean


__all__: list[str] = [
    "SIGNAL_MULTIPLE_RESPONSE_MODELS",
    "SIGNAL_REFERENCE_MATCH_SWITCH",
    "SIGNAL_TEMPORAL_DIVERGENCE",
    "SIGNAL_LATENCY_DISPERSION",
    "SIGNAL_TOKEN_DISPERSION",
    "SIGNAL_PROVIDER_FAILURE",
    "RoutingEvidenceStrength",
    "RoutingSignal",
    "RoutingV2Policy",
    "RoutingAssessmentV2",
    "assess_routing_v2",
]
