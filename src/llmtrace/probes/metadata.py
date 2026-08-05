"""元数据完整性探针."""

from __future__ import annotations

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.probes.base import BaseProbe, ProbeOutcome
from llmtrace.providers.base import BaseProvider


class MetadataProbe(BaseProbe):
    """元数据完整性探针 — 按证据类型分别检查."""

    rule_id = "LLMTRACE-META-001"
    probe_name = "元数据完整性"

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        super().__init__(config, provider)

    async def run(self) -> ProbeOutcome:
        return ProbeOutcome(
            findings=[
                self._result(
                    ProbeStatus.SKIPPED,
                    Severity.INFO,
                    facts=["元数据完整性分析在审计流程中基于累积证据执行"],
                    inferences=["见此探针的 analyze 方法"],
                )
            ],
            evidence=[],
        )

    def analyze(self, evidence_list: list[HTTPEvidence]) -> FindingResult:
        """基于证据列表分析元数据完整性 — 按证据类型分别检查."""
        facts: list[str] = []
        inferences: list[str] = []
        missing_fields: dict[str, int] = {}

        # 分类证据
        baseline_ev = [e for e in evidence_list if e.evidence_type == "baseline"]
        model_catalog_ev = [e for e in evidence_list if e.evidence_type == "model_catalog"]
        invalid_model_ev = [e for e in evidence_list if e.evidence_type == "invalid_model"]
        streaming_ev = [e for e in evidence_list if e.evidence_type == "streaming_baseline"]
        connectivity_ev = [e for e in evidence_list if e.evidence_type == "connectivity"]

        total = len(evidence_list)
        if total == 0:
            return self._result(
                ProbeStatus.WARN,
                Severity.INFO,
                facts=["无证据数据"],
                inferences=["无法分析元数据完整性"],
            )

        facts.append(f"分析证据总数: {total}")
        facts.append(
            f"baseline: {len(baseline_ev)}, 模型列表: {len(model_catalog_ev)}, "
            f"无效模型: {len(invalid_model_ev)}, 流式: {len(streaming_ev)}, 连接: {len(connectivity_ev)}"
        )

        # 对 baseline 检查 model、usage、response_id、finish_reason 等生成字段
        for _i, ev in enumerate(baseline_ev):
            if ev.success:
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

        # 流式证据按流式协议允许的字段位置检查
        for _i, ev in enumerate(streaming_ev):
            if ev.success and ev.response_model is None:
                missing_fields.setdefault("response_model (streaming)", 0)
                missing_fields["response_model (streaming)"] += 1

        # 模型列表不要求 input_tokens、output_tokens、finish_reason
        for _i, ev in enumerate(model_catalog_ev):
            if ev.exception_type is not None:
                missing_fields.setdefault("model_catalog_error", 0)
                missing_fields["model_catalog_error"] += 1

        # 无效模型被正常拒绝时，不要求模型和 Token 字段
        for _i, ev in enumerate(invalid_model_ev):
            if ev.http_status in (400, 404):
                # 预期被拒绝，不检查生成字段
                pass
            elif ev.success and ev.response_text:
                # 无效模型意外成功，记录异常
                inferences.append("无效模型意外成功生成内容，应检查模型字段")
                if ev.response_model is not None:
                    facts.append(f"无效模型返回模型: {ev.response_model}")

        if missing_fields:
            for field, count in sorted(missing_fields.items()):
                facts.append(
                    f"{field} 缺失: {count}/{len(baseline_ev)}"
                    if "baseline" not in field
                    else f"{field} 缺失: {count}/{len(baseline_ev)}"
                )
            inferences.append(f"共 {len(missing_fields)} 个字段存在缺失，不要把单个字段缺失直接定为高风险")
            return self._result(
                ProbeStatus.WARN,
                Severity.MEDIUM,
                facts=facts,
                inferences=inferences,
                limitations=["字段缺失不代表模型造假，可能是服务商设计选择"],
            )
        else:
            facts.append("所有必要元数据字段完整")
            return self._result(
                ProbeStatus.PASS,
                Severity.INFO,
                facts=facts,
                inferences=["所有元数据字段均存在"],
            )
