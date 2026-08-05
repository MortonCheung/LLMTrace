"""报告数据模型."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DriftLevel(StrEnum):
    """漂移程度."""

    NO_SIGNIFICANT_DRIFT = "NO_SIGNIFICANT_DRIFT"
    POSSIBLE_DRIFT = "POSSIBLE_DRIFT"
    LIKELY_DRIFT = "LIKELY_DRIFT"
    INCONCLUSIVE = "INCONCLUSIVE"


class CompareResult(BaseModel):
    """多报告比较结果."""

    reports: list[str]
    report_count: int
    report_times: list[str]
    endpoints: list[str]
    claimed_models: list[str]
    test_suite_versions: list[str]
    version_mismatch: bool = False

    success_rates: list[float] = Field(default_factory=list)
    latency_medians_ms: list[float] = Field(default_factory=list)
    latency_mads_ms: list[float] = Field(default_factory=list)
    response_model_sets: list[list[str]] = Field(default_factory=list)
    fingerprint_sets: list[list[str]] = Field(default_factory=list)
    token_field_rates: list[float] = Field(default_factory=list)
    request_id_rates: list[float] = Field(default_factory=list)
    error_types: list[list[str]] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)

    drift_level: DriftLevel = DriftLevel.INCONCLUSIVE
    drift_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReportMeta(BaseModel):
    """报告元数据."""

    llmtrace_version: str
    test_suite_version: str
    utc_time: str
    local_timezone: str
    python_version: str
    platform: str
    report_id: str
    config_summary: dict[str, Any]
    probe_list: list[str]
    risk_level: str
    schema_version: str
    content_hash: str
