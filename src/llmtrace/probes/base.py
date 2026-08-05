"""探针基类."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.providers.base import BaseProvider


class ProbeOutcome(BaseModel):
    """探针执行结果：包含 Findings 和本次真实 Evidence."""

    findings: list[FindingResult] = Field(default_factory=list)
    evidence: list[HTTPEvidence] = Field(default_factory=list)


class BaseProbe(ABC):
    """探针抽象基类."""

    rule_id: str
    probe_name: str

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        self.config = config
        self.provider = provider

    @abstractmethod
    async def run(self) -> ProbeOutcome:
        """执行探针，返回 Findings + Evidence."""
        ...

    def _result(
        self,
        status: ProbeStatus,
        severity: Severity,
        facts: list[str] | None = None,
        inferences: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> FindingResult:
        """构建探针结果."""
        return FindingResult(
            rule_id=self.rule_id,
            probe_name=self.probe_name,
            status=status,
            severity=severity,
            facts=facts or [],
            inferences=inferences or [],
            evidence_refs=evidence_refs or [],
            limitations=limitations or [],
        )
