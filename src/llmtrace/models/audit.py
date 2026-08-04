"""审计模型."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult


class RiskLevel(StrEnum):
    """风险等级."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    INCONCLUSIVE = "INCONCLUSIVE"


class AuditResult(BaseModel):
    """审计结果."""

    config: AuditConfig
    evidence: list[HTTPEvidence] = Field(default_factory=list)
    findings: list[FindingResult] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.INCONCLUSIVE
    schema_fingerprints: list[str] = Field(default_factory=list)
    model_list: list[str] = Field(default_factory=list)
    model_list_status: int | None = None
    model_list_available: bool = False
    model_in_list: bool | None = None
    probe_summary: dict[str, Any] = Field(default_factory=dict)

    start_time: datetime | None = None
    end_time: datetime | None = None
    llmtrace_version: str = ""
    python_version: str = ""
    platform: str = ""
    report_id: str = ""
    content_hash: str = ""
