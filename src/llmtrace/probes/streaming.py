"""流式一致性探针."""

from __future__ import annotations

import secrets

from llmtrace.config import AuditConfig
from llmtrace.models.findings import ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe, ProbeOutcome
from llmtrace.providers.base import BaseProvider


class StreamingProbe(BaseProbe):
    """流式一致性探针."""

    rule_id = "LLMTRACE-STR-001"
    probe_name = "流式一致性"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> ProbeOutcome:
        if not self.config.check_streaming:
            result = self._result(
                ProbeStatus.SKIPPED,
                Severity.INFO,
                facts=["流式检查已关闭"],
                inferences=["用户禁用了流式探针"],
            )
            return ProbeOutcome(findings=[result], evidence=[])

        facts: list[str] = []
        inferences: list[str] = []

        nonce = secrets.token_hex(4)
        messages = [{"role": "user", "content": f"Reply with only the word: {nonce}"}]

        # 非流式对照请求（不属于 baseline，独立证据类型）
        non_stream = await self.provider.complete(self.config.model, messages)
        non_stream.evidence_type = "streaming_comparison"
        # 流式请求
        stream = await self.provider.stream_complete(self.config.model, messages)
        stream.evidence_type = "streaming_baseline"

        evidence_refs = [str(non_stream.evidence_id), str(stream.evidence_id)]

        facts.append(f"非流式 - HTTP {non_stream.http_status}, 模型: {non_stream.response_model or 'N/A'}")
        facts.append(f"流式 - HTTP {stream.http_status}, 模型: {stream.response_model or 'N/A'}")

        if stream.exception_type:
            facts.append(f"流式异常: {stream.exception_type}: {stream.exception_message}")
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=["流式接口不可用，只作为协议能力证据"],
                evidence_refs=evidence_refs,
            )
            return ProbeOutcome(findings=[result], evidence=[non_stream, stream])

        if stream.http_status and stream.http_status >= 400:
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=[f"流式接口返回错误 {stream.http_status}，只作为协议能力证据"],
                evidence_refs=evidence_refs,
            )
            return ProbeOutcome(findings=[result], evidence=[non_stream, stream])

        # 比较模型
        if non_stream.response_model and stream.response_model and non_stream.response_model != stream.response_model:
            inferences.append(
                f"流式与非流式返回模型不一致: 非流式={non_stream.response_model}, 流式={stream.response_model}"
            )

        # 比较文本
        if (
            non_stream.response_text
            and stream.response_text
            and non_stream.response_text.strip() != stream.response_text.strip()
        ):
            inferences.append("流式与非流式响应文本存在差异")

        # 比较响应结构
        has_stream_text = bool(stream.response_text)
        facts.append(f"流式响应文本: {'有' if has_stream_text else '无'}")

        if not stream.response_text:
            inferences.append("流式响应未能提取文本内容")

        if not inferences:
            result = self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["流式与非流式接口表现一致"],
                evidence_refs=evidence_refs,
            )
        else:
            result = self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=inferences,
                limitations=["流式与非流式差异不能自动判定为模型造假"],
                evidence_refs=evidence_refs,
            )

        return ProbeOutcome(findings=[result], evidence=[non_stream, stream])
