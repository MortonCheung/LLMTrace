"""配置预检与连接鉴权探针."""

from __future__ import annotations

from llmtrace.config import AuditConfig, Protocol
from llmtrace.models.findings import ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe, ProbeOutcome
from llmtrace.providers.base import BaseProvider
from llmtrace.security.redaction import check_api_key


class ConfigPrecheckProbe(BaseProbe):
    """配置预检探针."""

    rule_id = "LLMTRACE-PRE-001"
    probe_name = "配置预检"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> ProbeOutcome:
        facts: list[str] = []
        errors: list[str] = []

        if self.config.protocol not in (Protocol.OPENAI, Protocol.ANTHROPIC):
            errors.append(f"不支持的协议: {self.config.protocol}")

        if not self.config.base_url.startswith(("http://", "https://")):
            errors.append(f"无效的 base_url: {self.config.base_url}")

        if not self.config.model:
            errors.append("模型名称为空")

        api_key = check_api_key(self.config.api_key_env)
        if api_key is None:
            errors.append(f"环境变量 {self.config.api_key_env} 不存在或为空")

        if self.config.repeat_count < 1:
            errors.append(f"重复次数无效: {self.config.repeat_count}")

        if self.config.timeout < 1:
            errors.append(f"超时设置无效: {self.config.timeout}")

        if self.config.max_output_tokens < 1:
            errors.append(f"max_output_tokens 无效: {self.config.max_output_tokens}")

        facts.append(f"协议: {self.config.protocol.value}")
        facts.append(f"Base URL: {self.config.base_url}")
        facts.append(f"模型: {self.config.model}")
        facts.append(f"密钥环境变量: {self.config.api_key_env}")
        facts.append(f"重复次数: {self.config.repeat_count}")
        facts.append(f"超时: {self.config.timeout}s")
        facts.append(f"最大输出 Token: {self.config.max_output_tokens}")

        if errors:
            result = self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=errors,
                limitations=["配置错误将阻止后续探针执行"],
            )
            return ProbeOutcome(findings=[result], evidence=[])

        result = self._result(
            ProbeStatus.PASS,
            Severity.INFO,
            facts=facts,
            inferences=["所有配置项检查通过"],
        )
        return ProbeOutcome(findings=[result], evidence=[])


class ConnectivityProbe(BaseProbe):
    """连接与鉴权探针."""

    rule_id = "LLMTRACE-CONN-001"
    probe_name = "连接与鉴权"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> ProbeOutcome:
        facts: list[str] = []
        inferences: list[str] = []

        # 发送一个最小请求来测试连接
        messages = [{"role": "user", "content": "hi"}]
        evidence = await self.provider.complete(self.config.model, messages)
        evidence.evidence_type = "connectivity"
        evidence_ref = str(evidence.evidence_id)

        facts.append(f"HTTP 状态码: {evidence.http_status}")
        facts.append(f"Content-Type: {evidence.response_headers.get('content-type', 'N/A')}")

        if evidence.exception_type:
            facts.append(f"异常类型: {evidence.exception_type}")
            facts.append(f"异常信息: {evidence.exception_message}")
            result = self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=[f"连接失败: {evidence.exception_message}"],
                limitations=["网络问题可能导致后续探针无法执行"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        if evidence.http_status == 401:
            result = self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=["鉴权失败 (401)，请检查 API Key"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])
        elif evidence.http_status == 403:
            result = self._result(
                ProbeStatus.FAIL,
                Severity.HIGH,
                facts=facts,
                inferences=["访问被拒绝 (403)"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])
        elif evidence.http_status == 429:
            result = self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=["遇到限流 (429)，后续探针可能受影响"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])
        elif evidence.http_status is not None and evidence.http_status >= 500:
            result = self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=[f"服务端错误 ({evidence.http_status})"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        # 检查 Content-Type
        content_type = evidence.response_headers.get("content-type", "")
        if "json" not in content_type.lower():
            facts.append(f"Content-Type 不是 JSON: {content_type}")
            inferences.append("响应可能不是标准 JSON，但可能仍可解析")

        if evidence.success:
            result = self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["连接正常，鉴权通过"],
                evidence_refs=[evidence_ref],
            )
        else:
            result = self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=[f"请求返回非 2xx 状态码: {evidence.http_status}"],
                evidence_refs=[evidence_ref],
            )

        return ProbeOutcome(findings=[result], evidence=[evidence])
