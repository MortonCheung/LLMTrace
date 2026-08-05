"""模型列表探针."""

from __future__ import annotations

from llmtrace.config import AuditConfig
from llmtrace.models.findings import ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe, ProbeOutcome
from llmtrace.providers.base import BaseProvider


class ModelCatalogProbe(BaseProbe):
    """模型列表探针."""

    rule_id = "LLMTRACE-CAT-001"
    probe_name = "模型列表"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> ProbeOutcome:
        facts: list[str] = []
        inferences: list[str] = []

        evidence, models = await self.provider.list_models()
        evidence_ref = str(evidence.evidence_id)

        facts.append(f"HTTP 状态码: {evidence.http_status}")

        if evidence.exception_type:
            facts.append(f"异常: {evidence.exception_type}: {evidence.exception_message}")
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=["模型列表接口不可用，不影响后续探针"],
                limitations=["模型列表不可用不代表接口不可用"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        if evidence.http_status == 404:
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=["模型列表接口不存在 (404)，只记为警告"],
                limitations=["模型列表不可用不代表接口不可用"],
                evidence_refs=[evidence_ref],
            )
            return ProbeOutcome(findings=[result], evidence=[evidence])

        facts.append(f"模型列表数量: {len(models)}")
        if models:
            facts.append(f"模型列表前5个: {', '.join(models[:5])}")

        target_in_list = self.config.model in models
        facts.append(f"目标模型 '{self.config.model}' {'在' if target_in_list else '不在'}列表中")

        if not target_in_list:
            inferences.append(f"目标模型 '{self.config.model}' 不在模型列表中，但可能仍可调用")

        if evidence.success and target_in_list:
            result = self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["模型列表可用，目标模型在列表中"],
                evidence_refs=[evidence_ref],
            )
        elif evidence.success:
            result = self._result(
                ProbeStatus.WARN,
                Severity.LOW,
                facts=facts,
                inferences=inferences,
                limitations=["模型列表不一致不能单凭模型列表认定造假"],
                evidence_refs=[evidence_ref],
            )
        else:
            result = self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=facts,
                inferences=inferences,
                limitations=["模型列表接口异常"],
                evidence_refs=[evidence_ref],
            )

        return ProbeOutcome(findings=[result], evidence=[evidence])
