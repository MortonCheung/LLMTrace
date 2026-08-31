"""Protocol audit orchestration — extracted from the legacy CLI.

``ProtocolAuditExecutor`` runs the eight protocol probes and returns a
``ProtocolAuditOutcome``.  Both the legacy ``llmtrace audit`` command and the
unified ``llmtrace run`` pipeline use this single implementation, so there is
never a second copy of the probe orchestration to keep in sync.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from llmtrace.analysis.risk import analyze_risk
from llmtrace.analysis.schema_fingerprint import generate_schema_fingerprint
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.models.findings import FindingResult
from llmtrace.probes.baseline import BaselineProbe
from llmtrace.probes.connectivity import ConfigPrecheckProbe, ConnectivityProbe
from llmtrace.probes.invalid_model import InvalidModelProbe
from llmtrace.probes.metadata import MetadataProbe
from llmtrace.probes.model_catalog import ModelCatalogProbe
from llmtrace.probes.stability import StabilityProbe
from llmtrace.probes.streaming import StreamingProbe
from llmtrace.utilities.hashing import short_id
from llmtrace.utilities.time import format_file_time, utc_now
from llmtrace.utilities.version import get_llmtrace_version, get_platform, get_python_version

if TYPE_CHECKING:
    from llmtrace.config import AuditConfig
    from llmtrace.models.evidence import HTTPEvidence
    from llmtrace.providers.base import BaseProvider

# ---------------------------------------------------------------------------
# Probe plan (also the source of protocol request counts)
# ---------------------------------------------------------------------------


def build_audit_plan(config: AuditConfig) -> list[dict[str, object]]:
    """构建协议审计计划，用于 dry-run 和实际执行与请求数统计."""
    plan: list[dict[str, object]] = []

    # 1. 配置预检（0 次请求）
    plan.append({"probe": "配置预检", "request_type": "none", "count": 0, "model_type": "N/A", "streaming": False})

    # 2. 连接与鉴权（1 次请求，声明模型）
    plan.append(
        {
            "probe": "连接与鉴权",
            "request_type": "completion",
            "count": 1,
            "model_type": "声明模型",
            "streaming": False,
        }
    )

    # 3. 模型列表（1 次 GET 请求）
    plan.append({"probe": "模型列表", "request_type": "GET", "count": 1, "model_type": "N/A", "streaming": False})

    # 4. 正常基线（repeat_count 次请求，声明模型）
    plan.append(
        {
            "probe": "正常基线",
            "request_type": "completion",
            "count": config.repeat_count,
            "model_type": "声明模型",
            "streaming": False,
        }
    )

    # 5. 无效模型（1 次请求，随机无效模型）
    plan.append(
        {
            "probe": "无效模型",
            "request_type": "completion",
            "count": 1,
            "model_type": "随机无效模型",
            "streaming": False,
        }
    )

    # 6. 流式一致性（1 次非流式 + 1 次流式）
    if config.check_streaming:
        plan.append(
            {
                "probe": "流式一致性",
                "request_type": "completion+stream",
                "count": 2,
                "model_type": "声明模型",
                "streaming": True,
            }
        )

    # 7-8. 元数据完整性和会话稳定性（0 次额外请求，分析已有证据）
    plan.append(
        {"probe": "元数据完整性", "request_type": "analysis", "count": 0, "model_type": "N/A", "streaming": False}
    )
    plan.append(
        {"probe": "会话稳定性", "request_type": "analysis", "count": 0, "model_type": "N/A", "streaming": False}
    )

    return plan


def protocol_probe_request_count(config: AuditConfig) -> int:
    """Total real HTTP requests the protocol probes will send."""
    total = 0
    for item in build_audit_plan(config):
        total += cast(int, item["count"])
    return total


def protocol_output_token_ceiling(config: AuditConfig) -> int:
    """Upper bound on output tokens the protocol probes may consume.

    Every protocol completion request uses ``config.max_output_tokens``; GET
    and analysis-only probes consume no output tokens.
    """
    completion_requests = sum(
        cast(int, item["count"])
        for item in build_audit_plan(config)
        if item["request_type"] in ("completion", "completion+stream")
    )
    return completion_requests * config.max_output_tokens


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ProtocolAuditOutcome(BaseModel):
    """Outcome of one protocol audit execution."""

    result: AuditResult = Field(..., description="Full audit result (evidence + findings + risk)")
    blocking_failure: bool = Field(default=False, description="True when precheck/connectivity blocked the run")
    blocking_stage: str | None = Field(default=None, description="Stage that blocked the run, if any")


class ProtocolAuditExecutor:
    """Run the protocol probes once and collect evidence + findings.

    A config-precheck failure or connectivity/auth failure is a *blocking*
    failure: the unified runner must skip the benchmark instead of wasting 32
    real requests against an unreachable endpoint.
    """

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        self._config = config
        self._provider = provider

    async def run(self) -> ProtocolAuditOutcome:
        config = self._config
        evidence_list: list[HTTPEvidence] = []
        findings: list[FindingResult] = []

        result = AuditResult(
            config=config,
            llmtrace_version=get_llmtrace_version(),
            python_version=get_python_version(),
            platform=get_platform(),
            report_id=f"llmtrace_{format_file_time(utc_now())}_{short_id(6)}",
            start_time=utc_now(),
        )

        async with self._provider:
            # 1. 配置预检
            precheck = ConfigPrecheckProbe(config, self._provider)
            outcome = await precheck.run()
            findings.extend(outcome.findings)
            if outcome.findings and outcome.findings[0].status.value == "fail":
                result.findings = findings
                result.risk_level = RiskLevel.INCONCLUSIVE
                result.end_time = utc_now()
                return ProtocolAuditOutcome(result=result, blocking_failure=True, blocking_stage="配置预检")

            # 2. 连接与鉴权
            conn = ConnectivityProbe(config, self._provider)
            outcome = await conn.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)
            if outcome.findings and outcome.findings[0].status.value == "fail":
                result.findings = findings
                result.evidence = evidence_list
                result.risk_level = RiskLevel.INCONCLUSIVE
                result.end_time = utc_now()
                return ProtocolAuditOutcome(result=result, blocking_failure=True, blocking_stage="连接与鉴权")

            # 3. 模型列表
            catalog = ModelCatalogProbe(config, self._provider)
            outcome = await catalog.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)
            if outcome.evidence:
                list_ev = outcome.evidence[0]
                result.model_list_available = list_ev.success
                if list_ev.success and list_ev.response_body_summary:
                    try:
                        data = json.loads(list_ev.response_body_summary)
                        models = data.get("data", [])
                        if isinstance(models, list):
                            result.model_list = [m.get("id", "") for m in models if isinstance(m, dict)]
                            result.model_in_list = config.model in result.model_list
                    except (json.JSONDecodeError, AttributeError):
                        pass

            # 4. 正常基线
            baseline = BaselineProbe(config, self._provider)
            outcome = await baseline.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)

            # 5. 无效模型
            invalid = InvalidModelProbe(config, self._provider)
            outcome = await invalid.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)

            # 6. 流式一致性
            if config.check_streaming:
                streaming = StreamingProbe(config, self._provider)
                outcome = await streaming.run()
                findings.extend(outcome.findings)
                evidence_list.extend(outcome.evidence)

            # 7. 元数据完整性
            metadata = MetadataProbe(config, self._provider)
            findings.append(metadata.analyze(evidence_list))

            # 8. 会话稳定性
            stability = StabilityProbe(config, self._provider)
            findings.append(stability.analyze(evidence_list))

            # 生成结构指纹
            for ev in evidence_list:
                if ev.response_body_summary:
                    fp = generate_schema_fingerprint(ev.response_body_summary)
                    if fp:
                        result.schema_fingerprints.append(fp)

            # 完整性校验
            self._validate_evidence_refs(findings, evidence_list)
            self._check_duplicate_evidence_ids(evidence_list)

        result.evidence = evidence_list
        result.findings = findings
        result.risk_level = analyze_risk(findings)
        result.end_time = utc_now()
        return ProtocolAuditOutcome(result=result, blocking_failure=False, blocking_stage=None)

    @staticmethod
    def _validate_evidence_refs(findings: list[FindingResult], evidence_list: list[HTTPEvidence]) -> None:
        evidence_ids = {str(e.evidence_id) for e in evidence_list}
        for f in findings:
            for ref in f.evidence_refs:
                if ref not in evidence_ids:
                    raise ValueError(f"证据引用 '{ref}' (探针: {f.probe_name}) 在证据集合中找不到。")

    @staticmethod
    def _check_duplicate_evidence_ids(evidence_list: list[HTTPEvidence]) -> None:
        seen: set[str] = set()
        for ev in evidence_list:
            eid = str(ev.evidence_id)
            if eid in seen:
                raise ValueError(f"重复的 evidence_id: {eid}")
            seen.add(eid)
