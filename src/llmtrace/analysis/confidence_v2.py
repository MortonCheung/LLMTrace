"""Confidence v2（Task 32）—— 把一次 run 的置信度拆成四个独立组件.

v1 的 ``assess_confidence()``（``analysis/confidence.py``）保留不动：它给出的是
**一个**混装 label；v2 关心的是"哪个结论有支撑、缺什么"，所以拆成

    measurement  测量健康（coverage / failures / items）
    calibration  参考锚定（calibrated? / reference identity count / compatibility）
    fingerprint  行为指纹证据（policy validated? / comparable probes /
                 valid sample coverage / claimed reference exists?）
    routing      路由稳定性证据（sample count / response_model coverage /
                 validated temporal fingerprint available?）

每个组件只输出既有的 ``ConfidenceLevel`` 标签与文字理由，**不输出任何伪造概率**
（Task 32 Step 32.4 / Task 48）。阈值复用 v1 的版本化 ``ConfidencePolicy`` 与
routing v2 的 ``minimum_samples``，不新增"看起来很合理"的数字（Rule 5 / Rule 6）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from llmtrace.analysis.confidence import ConfidenceLevel, ConfidencePolicy
from llmtrace.analysis.routing import RoutingStabilityLevel
from llmtrace.fingerprint.matcher import FingerprintMatchResult, FingerprintVerificationResult
from llmtrace.fingerprint.models import ProbeDistribution
from llmtrace.fingerprint.routing import RoutingAssessmentV2, RoutingV2Policy
from llmtrace.fingerprint.temporal import TemporalFingerprintStatus
from llmtrace.scoring.models import CapabilityProfile

if TYPE_CHECKING:
    # Annotation-only import: the execution package imports this module, so a
    # runtime import here would create a cycle (execution → runner → this).
    from llmtrace.execution.models import BenchmarkMeasurementSummary

#: v2 组件复用的版本化阈值（v1 policy 的数字，不重复定义）。
DEFAULT_CONFIDENCE_POLICY = ConfidencePolicy()

#: routing 组件复用的样本下限（v2 routing policy 的数字，不重复定义）。
DEFAULT_ROUTING_POLICY = RoutingV2Policy()


class ConfidenceComponent(BaseModel):
    """一个独立置信度组件：标签 + 理由 + 缺什么（Task 32 Step 32）."""

    level: ConfidenceLevel
    reasons: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    model_config = {"frozen": True, "extra": "forbid"}


class ConfidenceBundle(BaseModel):
    """一次 run 的四个置信度组件；彼此不合并成一个数字（Task 32）."""

    measurement: ConfidenceComponent
    calibration: ConfidenceComponent
    fingerprint: ConfidenceComponent
    routing: ConfidenceComponent

    model_config = {"frozen": True, "extra": "forbid"}


def measurement_confidence(
    measurement: BenchmarkMeasurementSummary | None,
    *,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> ConfidenceComponent:
    """测量健康组件：只看 coverage / failures / items（Step 32.1）."""
    if measurement is None or measurement.graded_item_count == 0:
        return ConfidenceComponent(
            level=ConfidenceLevel.UNAVAILABLE,
            reasons=("no graded benchmark measurement available",),
            limitations=("capability scores are unavailable: nothing was graded",),
        )

    total = measurement.total_item_count
    graded = measurement.graded_item_count
    failures = measurement.failure_item_count
    coverage = measurement.execution_coverage
    graded_ratio = graded / total if total > 0 else 0.0
    failure_ratio = failures / total if total > 0 else 0.0

    reasons = [f"{graded}/{total} items graded, {failures} provider failures, execution coverage {coverage:.0%}"]

    if coverage < policy.low_execution_coverage_threshold or failure_ratio >= policy.degraded_failure_ratio_threshold:
        reasons.append(f"measurement degraded (coverage {coverage:.0%})")
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons))

    if graded == total and failures == 0:
        reasons.append("measurement complete: every item graded with zero provider failures")
        return ConfidenceComponent(level=ConfidenceLevel.HIGH, reasons=tuple(reasons))

    if graded_ratio >= policy.mostly_complete_threshold:
        reasons.append("measurement largely complete but not fully clean")
        return ConfidenceComponent(level=ConfidenceLevel.MEDIUM, reasons=tuple(reasons))

    reasons.append("mostly-incomplete measurement")
    return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons))


def calibration_confidence(
    capability_profile: CapabilityProfile | None,
    *,
    expected_reference_set_id: str | None = None,
    expected_reference_set_content_sha256: str | None = None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> ConfidenceComponent:
    """校准组件：calibrated? / reference identity count / reference compatibility（Step 32.2）.

    ``expected_reference_set_*`` 是调用方声明的"本次 run 应该用哪个参考集"；未提供时
    只能把"参考兼容性未检查"写成 limitation —— 不假装检查过。
    """
    calibration = capability_profile.calibration if capability_profile is not None else None
    if calibration is None:
        return ConfidenceComponent(
            level=ConfidenceLevel.UNAVAILABLE,
            reasons=("raw capability only (no reference calibration)",),
            limitations=("no formal reference calibration; capability scores are unanchored",),
        )

    limitations: list[str] = []
    reasons = [
        f"calibrated against {calibration.reference_identity_count} reference identities "
        f"via '{calibration.reference_set_id}' v{calibration.reference_set_version}"
    ]

    compatibility: bool | None = None
    if expected_reference_set_id is None and expected_reference_set_content_sha256 is None:
        limitations.append("reference compatibility was not checked (no expected reference set identity supplied)")
    else:
        compatibility = (
            expected_reference_set_id is None or calibration.reference_set_id == expected_reference_set_id
        ) and (
            expected_reference_set_content_sha256 is None
            or calibration.reference_set_content_sha256 == expected_reference_set_content_sha256
        )
        if not compatibility:
            reasons.append("the calibration reference set does not match the expected reference set")

    if compatibility is False:
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    if calibration.coverage_weight < policy.low_execution_coverage_threshold:
        reasons.append(f"weak calibration coverage weight ({calibration.coverage_weight:.2f})")
        return ConfidenceComponent(level=ConfidenceLevel.MEDIUM, reasons=tuple(reasons), limitations=tuple(limitations))

    if calibration.reference_identity_count >= policy.minimum_identities_for_high:
        reasons.append("reference universe meets the high bar")
        return ConfidenceComponent(level=ConfidenceLevel.HIGH, reasons=tuple(reasons), limitations=tuple(limitations))

    reasons.append(
        f"reference universe below the high bar ({calibration.reference_identity_count} < "
        f"{policy.minimum_identities_for_high} identities)"
    )
    return ConfidenceComponent(level=ConfidenceLevel.MEDIUM, reasons=tuple(reasons), limitations=tuple(limitations))


def fingerprint_confidence(
    *,
    verification: FingerprintVerificationResult | None,
    match: FingerprintMatchResult | None = None,
    candidate: Sequence[ProbeDistribution] = (),
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> ConfidenceComponent:
    """指纹组件：policy validated? / comparable probes / valid sample coverage / claimed reference（Step 32.3）.

    ``verification`` 是 Rule 2 门禁记录，``match`` 是排序结果，``candidate`` 是本次
    capture 的分布（用于 valid sample coverage：归一化成功的样本占比，分母不缩小）。
    """
    if verification is None and match is None:
        return ConfidenceComponent(
            level=ConfidenceLevel.UNAVAILABLE,
            reasons=("no fingerprint evidence was recorded for this run",),
            limitations=("behavioral identity evidence was not collected",),
        )

    limitations: list[str] = []
    reasons: list[str] = []

    decision_policy = verification.policy if verification is not None else None
    validated = decision_policy is not None and decision_policy.validated
    if decision_policy is None:
        limitations.append("no decision policy was available; only a behavioral ranking is reported")
    elif not validated:
        limitations.append("the decision policy is UNVALIDATED; only a behavioral ranking is reported")

    entry = match.entries[0] if match is not None and match.entries else None
    comparable_probes = entry.comparable_probes if entry is not None else None
    if comparable_probes is not None:
        reasons.append(f"closest behavioral reference computed from {comparable_probes} comparable probe(s)")

    valid_sample_coverage = _valid_sample_coverage(candidate)
    if valid_sample_coverage is not None:
        reasons.append(f"valid sample coverage {valid_sample_coverage:.0%}")
    else:
        limitations.append("valid sample coverage was not measurable (no candidate distributions supplied)")

    claimed_reference_exists: bool | None = None
    if match is not None:
        claimed_reference_exists = match.claimed_reference_distance is not None
        if match.claimed_model_id is None:
            limitations.append("no claimed model was supplied, so no claim could be compared")
        elif not claimed_reference_exists:
            reasons.append(f"the claimed model '{match.claimed_model_id}' has no capture in the fingerprint set")
        else:
            reasons.append(f"the claimed model '{match.claimed_model_id}' exists in the fingerprint set")

    if verification is not None and verification.claim_verdict_produced:
        reasons.append(f"claim consistency: {verification.match_status.value}")

    if not validated:
        if not reasons:
            reasons.append("no validated decision policy is available")
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    if claimed_reference_exists is False:
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    if valid_sample_coverage is not None and valid_sample_coverage < policy.low_execution_coverage_threshold:
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    if valid_sample_coverage is not None and valid_sample_coverage < policy.mostly_complete_threshold:
        return ConfidenceComponent(level=ConfidenceLevel.MEDIUM, reasons=tuple(reasons), limitations=tuple(limitations))

    return ConfidenceComponent(level=ConfidenceLevel.HIGH, reasons=tuple(reasons), limitations=tuple(limitations))


def routing_confidence(
    assessment: RoutingAssessmentV2 | None,
    *,
    validated_policy_available: bool | None = None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
    routing_policy: RoutingV2Policy = DEFAULT_ROUTING_POLICY,
) -> ConfidenceComponent:
    """路由组件：sample count / response_model coverage / validated temporal fingerprint（Step 32.4）.

    ``validated_policy_available`` 说明本次有没有"已验证 policy"支撑 temporal 证据；
    为 ``None`` 时按"没有"处理（fail closed）。
    """
    if assessment is None:
        return ConfidenceComponent(
            level=ConfidenceLevel.UNAVAILABLE,
            reasons=("no routing assessment was performed for this run",),
            limitations=("routing consistency was not assessed",),
        )

    temporal_available = assessment.temporal_status is TemporalFingerprintStatus.AVAILABLE
    policy_backed_temporal = temporal_available and validated_policy_available is True

    reasons = [
        f"{assessment.item_count} behavior samples, response_model coverage "
        f"{assessment.response_model_coverage:.0%}, routing level {assessment.level.value}"
    ]
    limitations = list(assessment.limitations)

    if assessment.item_count < routing_policy.minimum_samples:
        reasons.append(f"fewer samples than the routing minimum ({routing_policy.minimum_samples})")
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    if assessment.response_model_coverage < policy.mostly_complete_threshold:
        reasons.append(
            f"the server reported a model identifier on only {assessment.response_model_coverage:.0%} of items"
        )
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    if not policy_backed_temporal:
        if temporal_available:
            limitations.append("temporal evidence was not backed by a validated policy")
        reasons.append("routing stability was not positively confirmed by validated temporal evidence")
        return ConfidenceComponent(level=ConfidenceLevel.MEDIUM, reasons=tuple(reasons), limitations=tuple(limitations))

    if assessment.level is RoutingStabilityLevel.SUSPICIOUS:
        reasons.append("routing inconsistency observed; mixed routing is possible")
        return ConfidenceComponent(level=ConfidenceLevel.MEDIUM, reasons=tuple(reasons), limitations=tuple(limitations))

    if assessment.level is RoutingStabilityLevel.INSUFFICIENT_DATA:
        reasons.append("routing evidence was insufficient for a stability label")
        return ConfidenceComponent(level=ConfidenceLevel.LOW, reasons=tuple(reasons), limitations=tuple(limitations))

    reasons.append("validated temporal fingerprint available and routing looked consistent")
    return ConfidenceComponent(level=ConfidenceLevel.HIGH, reasons=tuple(reasons), limitations=tuple(limitations))


def build_confidence_bundle(
    *,
    measurement: BenchmarkMeasurementSummary | None = None,
    capability_profile: CapabilityProfile | None = None,
    verification: FingerprintVerificationResult | None = None,
    match: FingerprintMatchResult | None = None,
    candidate: Sequence[ProbeDistribution] = (),
    routing: RoutingAssessmentV2 | None = None,
    expected_reference_set_id: str | None = None,
    expected_reference_set_content_sha256: str | None = None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
    routing_policy: RoutingV2Policy = DEFAULT_ROUTING_POLICY,
) -> ConfidenceBundle:
    """组装四个组件；缺哪一块就把那一块标成 Unavailable，不猜."""
    decision_policy = verification.policy if verification is not None else None
    return ConfidenceBundle(
        measurement=measurement_confidence(measurement, policy=policy),
        calibration=calibration_confidence(
            capability_profile,
            expected_reference_set_id=expected_reference_set_id,
            expected_reference_set_content_sha256=expected_reference_set_content_sha256,
            policy=policy,
        ),
        fingerprint=fingerprint_confidence(
            verification=verification,
            match=match,
            candidate=candidate,
            policy=policy,
        ),
        routing=routing_confidence(
            routing,
            validated_policy_available=decision_policy.validated if decision_policy is not None else None,
            policy=policy,
            routing_policy=routing_policy,
        ),
    )


def _valid_sample_coverage(candidate: Sequence[ProbeDistribution]) -> float | None:
    """归一化成功的样本占比；没有样本时返回 ``None``（不伪造 0%）."""
    total = sum(distribution.sample_count for distribution in candidate)
    if total == 0:
        return None
    invalid = sum(distribution.invalid_count for distribution in candidate)
    return (total - invalid) / total


__all__: list[str] = [
    "ConfidenceComponent",
    "ConfidenceBundle",
    "measurement_confidence",
    "calibration_confidence",
    "fingerprint_confidence",
    "routing_confidence",
    "build_confidence_bundle",
]
