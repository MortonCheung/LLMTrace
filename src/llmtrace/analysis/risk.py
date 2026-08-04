"""风险分析."""

from __future__ import annotations

from llmtrace.models.audit import RiskLevel
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity


def analyze_risk(findings: list[FindingResult]) -> RiskLevel:
    """基于探针发现计算风险等级."""
    if not findings:
        return RiskLevel.INCONCLUSIVE

    high_failures = [f for f in findings if f.status == ProbeStatus.FAIL and f.severity == Severity.HIGH]
    medium_warnings = [
        f for f in findings if f.status in (ProbeStatus.FAIL, ProbeStatus.WARN) and f.severity == Severity.MEDIUM
    ]

    # 检查是否有高风险证据
    for f in high_failures:
        if "无效模型名称仍成功生成内容" in " ".join(f.inferences):
            return RiskLevel.HIGH
        if "同一会话中返回了多个不同的模型标识" in " ".join(f.inferences):
            return RiskLevel.HIGH

    # 检查是否有连接失败
    if any("连接失败" in " ".join(f.inferences) for f in high_failures):
        return RiskLevel.INCONCLUSIVE
    if any("鉴权失败" in " ".join(f.inferences) for f in high_failures):
        return RiskLevel.INCONCLUSIVE

    if high_failures:
        return RiskLevel.HIGH

    if medium_warnings:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def risk_explanation(level: RiskLevel) -> str:
    """获取风险等级的解释."""
    explanations = {
        RiskLevel.LOW: "本次有限测试未发现明显异常。这不表示已证明服务商真实调用了其声明模型。",
        RiskLevel.MEDIUM: "检测到中等级别的异常证据，建议进一步调查。可能包括模型列表不一致、元数据不稳定等。",
        RiskLevel.HIGH: "检测到高风险证据，接口行为存在重大异常。可能包括无效模型成功返回、模型标识不一致等。",
        RiskLevel.INCONCLUSIVE: "无法得出有效结论，可能因为鉴权失败、网络不可达或样本不足。",
    }
    return explanations[level]
