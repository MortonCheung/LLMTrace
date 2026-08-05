"""无效模型探针."""

from __future__ import annotations

import uuid

from llmtrace.config import AuditConfig
from llmtrace.models.findings import ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe, ProbeOutcome
from llmtrace.providers.base import BaseProvider


class InvalidModelProbe(BaseProbe):
    """无效模型探针."""

    rule_id = "LLMTRACE-INV-001"
    probe_name = "无效模型"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> ProbeOutcome:
        facts: list[str] = []

        invalid_model = f"llmtrace-invalid-{uuid.uuid4().hex[:12]}"
        messages = [{"role": "user", "content": "Reply with only the word: hello"}]
        evidence = await self.provider.complete(invalid_model, messages)
        evidence.evidence_type = "invalid_model"
        evidence_ref = str(evidence.evidence_id)

        facts.append(f"配置声明模型: {self.config.model}")
        facts.append(f"本次随机无效模型: {invalid_model}")
        facts.append(f"HTTP 状态码: {evidence.http_status}")
        facts.append(f"服务端返回模型: {evidence.response_model or 'N/A'}")
        facts.append(f"返回文本: {evidence.response_text[:100]}")

        if evidence.exception_type:
            facts.append(f"异常: {evidence.exception_type}: {evidence.exception_message}")
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=["无法确定无效模型处理方式，网络异常"],
                limitations=["网络异常导致无法判断"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        if evidence.http_status in (400, 404):
            result = self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=[f"无效模型被正确拒绝 (HTTP {evidence.http_status})"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        if evidence.success and evidence.response_text:
            result = self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=["无效模型名称仍成功生成内容，接口可能忽略 model 参数，或执行了未披露的默认模型回退"],
                limitations=["不能单凭此结果认定中转站造假，但属于高风险证据"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        if evidence.http_status in (401, 403):
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=["鉴权问题导致无法判断无效模型处理"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        if evidence.http_status == 429:
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=["限流导致无法判断无效模型处理"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        result = self._result(
            ProbeStatus.WARN,
            Severity.INFO,
            facts=facts,
            inferences=[f"无效模型返回非预期状态码 {evidence.http_status}，结论不充分"],
            limitations=["非标准错误码，无法确定是模型拒绝还是其他错误"],
            evidence_refs=[evidence_ref],
        )
        return ProbeOutcome(findings=[result], evidence=[evidence])
