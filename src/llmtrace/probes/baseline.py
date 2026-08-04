"""正常基线探针."""

from __future__ import annotations

import secrets

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe
from llmtrace.providers.base import BaseProvider


class BaselineProbe(BaseProbe):
    """正常基线探针."""

    rule_id = "LLMTRACE-BASE-001"
    probe_name = "正常基线"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> FindingResult:
        facts: list[str] = []
        inferences: list[str] = []
        evidence_list: list[HTTPEvidence] = []
        evidence_refs: list[str] = []

        for idx in range(self.config.repeat_count):
            nonce = secrets.token_hex(4)
            messages = [{"role": "user", "content": f"Reply with only the word: {nonce}"}]
            evidence = await self.provider.complete(self.config.model, messages)
            evidence_list.append(evidence)
            ref = f"baseline_{idx + 1}"
            evidence_refs.append(ref)

            if evidence.success:
                facts.append(
                    f"请求 {idx + 1}: 成功, 延迟 {evidence.total_latency_ms:.0f}ms, "
                    f"返回模型: {evidence.response_model or 'N/A'}"
                )
            else:
                facts.append(
                    f"请求 {idx + 1}: 失败, HTTP {evidence.http_status}, 异常: {evidence.exception_type or 'N/A'}"
                )

        # 统计成功率
        successes = sum(1 for e in evidence_list if e.success)
        success_rate = successes / len(evidence_list) if evidence_list else 0
        facts.append(f"成功率: {successes}/{len(evidence_list)} ({success_rate:.0%})")

        # 统计返回模型
        response_models = {e.response_model for e in evidence_list if e.response_model}
        facts.append(f"返回模型集合: {sorted(response_models)}" if response_models else "返回模型: 无")

        if success_rate == 0:
            return self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=["所有基线请求均失败，无法继续分析"],
                evidence_refs=evidence_refs,
            )

        if len(response_models) > 1:
            inferences.append(f"同一会话中返回了多个不同的模型标识: {sorted(response_models)}")

        if success_rate < 1.0:
            inferences.append(f"部分请求失败 ({successes}/{len(evidence_list)})")

        if len(response_models) <= 1 and success_rate == 1.0:
            return self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["所有基线请求成功，返回模型一致"],
                evidence_refs=evidence_refs,
            )
        elif success_rate >= 0.5:
            return self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=inferences,
                evidence_refs=evidence_refs,
            )
        else:
            return self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=inferences,
                evidence_refs=evidence_refs,
            )
