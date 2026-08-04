"""元数据完整性探针."""

from __future__ import annotations

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe
from llmtrace.providers.base import BaseProvider


class MetadataProbe(BaseProbe):
    """元数据完整性探针."""

    rule_id = "LLMTRACE-META-001"
    probe_name = "元数据完整性"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> FindingResult:
        """基于已有的基线证据分析元数据完整性."""
        # 此探针需要外部提供证据列表
        # 在审计流程中，由 AuditRunner 传入累积的证据
        return self._result(
            ProbeStatus.SKIPPED,
            Severity.INFO,
            facts=["元数据完整性分析在审计流程中基于累积证据执行"],
            inferences=["见此探针的 analyze 方法"],
        )

    def analyze(self, evidence_list: list[HTTPEvidence]) -> FindingResult:
        """基于证据列表分析元数据完整性."""
        facts: list[str] = []
        inferences: list[str] = []
        missing_fields: dict[str, int] = {}

        for _i, ev in enumerate(evidence_list):
            if ev.request_model is None:
                missing_fields.setdefault("request_model", 0)
                missing_fields["request_model"] += 1
            if ev.response_model is None:
                missing_fields.setdefault("response_model", 0)
                missing_fields["response_model"] += 1
            if ev.response_id is None:
                missing_fields.setdefault("response_id", 0)
                missing_fields["response_id"] += 1
            if ev.input_tokens is None:
                missing_fields.setdefault("input_tokens", 0)
                missing_fields["input_tokens"] += 1
            if ev.output_tokens is None:
                missing_fields.setdefault("output_tokens", 0)
                missing_fields["output_tokens"] += 1
            if ev.finish_reason is None:
                missing_fields.setdefault("finish_reason", 0)
                missing_fields["finish_reason"] += 1

        total = len(evidence_list)
        if total == 0:
            return self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=["无证据数据"],
                inferences=["无法分析元数据完整性"],
            )

        facts.append(f"分析证据数: {total}")

        if missing_fields:
            for field, count in sorted(missing_fields.items()):
                facts.append(f"{field} 缺失: {count}/{total}")
            inferences.append(f"共 {len(missing_fields)} 个字段存在缺失，不要把单个字段缺失直接定为高风险")
            return self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=inferences,
                limitations=["字段缺失不代表模型造假，可能是服务商设计选择"],
            )
        else:
            facts.append("所有元数据字段完整")
            return self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["所有元数据字段均存在"],
            )
