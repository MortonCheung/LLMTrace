"""探针结果数据模型."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProbeStatus(StrEnum):
    """探针执行状态."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"
    SKIPPED = "skipped"


class Severity(StrEnum):
    """严重程度."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    INFO = "info"


class FindingResult(BaseModel):
    """单个探针发现结果."""

    rule_id: str
    probe_name: str
    status: ProbeStatus
    severity: Severity
    facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
