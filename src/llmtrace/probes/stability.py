"""会话稳定性探针."""

from __future__ import annotations

import statistics

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe
from llmtrace.providers.base import BaseProvider


class StabilityProbe(BaseProbe):
    """会话稳定性探针."""

    rule_id = "LLMTRACE-STAB-001"
    probe_name = "会话稳定性"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> FindingResult:
        return self._result(
            ProbeStatus.SKIPPED,
            Severity.INFO,
            facts=["会话稳定性分析在审计流程中基于累积证据执行"],
            inferences=["见此探针的 analyze 方法"],
        )

    def analyze(self, evidence_list: list[HTTPEvidence]) -> FindingResult:
        """基于证据列表分析会话稳定性."""
        facts: list[str] = []
        inferences: list[str] = []

        successes = [e for e in evidence_list if e.success]
        total = len(evidence_list)

        if total == 0:
            return self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=["无证据数据"],
                inferences=["无法分析会话稳定性"],
            )

        facts.append(f"总请求数: {total}, 成功: {len(successes)}")

        # 检查返回模型一致性
        response_models = {e.response_model for e in successes if e.response_model}
        if len(response_models) > 1:
            inferences.append(f"返回模型不一致: {sorted(response_models)}")
        elif len(response_models) == 1:
            facts.append(f"返回模型一致: {list(response_models)[0]}")
        else:
            inferences.append("所有成功请求均未返回模型字段")

        # 检查状态码一致性
        status_codes = {e.http_status for e in evidence_list if e.http_status is not None}
        if len(status_codes) > 1:
            inferences.append(f"状态码不一致: {sorted(status_codes)}")

        # 检查 Token 字段稳定性
        if successes:
            input_token_present = sum(1 for e in successes if e.input_tokens is not None)
            output_token_present = sum(1 for e in successes if e.output_tokens is not None)
            if input_token_present not in (0, len(successes)):
                inferences.append(f"input_tokens 时有时无 ({input_token_present}/{len(successes)})")
            if output_token_present not in (0, len(successes)):
                inferences.append(f"output_tokens 时有时无 ({output_token_present}/{len(successes)})")

        # 检查延迟分组
        if len(successes) >= 2:
            latencies = [e.total_latency_ms for e in successes if e.total_latency_ms is not None]
            if len(latencies) >= 2:
                median = statistics.median(latencies)
                mad = statistics.median(abs(x - median) for x in latencies)
                facts.append(f"延迟中位数: {median:.0f}ms, MAD: {mad:.0f}ms")

                # 检查是否有异常分组（延迟差异超过 3x MAD 且 MAD > 100ms）
                if mad > 100:
                    max_lat = max(latencies)
                    min_lat = min(latencies)
                    if max_lat > min_lat * 3:
                        inferences.append(f"延迟存在明显异常分组: min={min_lat:.0f}ms, max={max_lat:.0f}ms")

        # 检查部分成功部分失败
        if 0 < len(successes) < total:
            inferences.append(f"部分请求失败 ({len(successes)}/{total} 成功)")

        if not inferences:
            return self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["会话表现稳定，未发现明显异常"],
            )
        else:
            return self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=inferences,
                limitations=["首版只做可解释的规则统计，不实现机器学习分类"],
            )
