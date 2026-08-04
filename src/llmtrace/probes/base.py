"""探针基类."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmtrace.config import AuditConfig
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.providers.base import BaseProvider


class BaseProbe(ABC):
    """探针抽象基类."""

    rule_id: str
    probe_name: str

    def __init__(self, config: AuditConfig, provider: BaseProvider) -> None:
        self.config = config
        self.provider = provider

    @abstractmethod
    async def run(self) -> FindingResult:
        """执行探针."""
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
