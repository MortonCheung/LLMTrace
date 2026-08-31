"""JSON 报告生成."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from llmtrace.analysis.behavior_drift import BehaviorDriftResult
from llmtrace.models.audit import AuditResult
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult
from llmtrace.reporting.benchmark_models import BenchmarkReportSection
from llmtrace.reporting.evidence_validation import validate_report_evidence_refs
from llmtrace.scoring.comparison import ComparisonResult
from llmtrace.utilities.hashing import sha256_hash

# Single source of truth for schema version
# 1.1 → 1.2: added the optional `behavior_drift` section (v0.3-D).
SCHEMA_VERSION = "1.2"


def evidence_to_dict(ev: HTTPEvidence) -> dict[str, object]:
    """将证据对象转换为字典."""
    return {
        "evidence_id": str(ev.evidence_id),
        "evidence_type": ev.evidence_type,
        "request_method": ev.request_method,
        "request_url_redacted": ev.request_url_redacted,
        "request_path": ev.request_path,
        "request_headers_redacted": ev.request_headers_redacted,
        "request_body_redacted": ev.request_body_redacted,
        "request_time": ev.request_time.isoformat() if ev.request_time else None,
        "response_time": ev.response_time.isoformat() if ev.response_time else None,
        "http_status": ev.http_status,
        "response_headers": ev.response_headers,
        "response_body_summary": ev.response_body_summary,
        "response_body_sha256": ev.response_body_sha256,
        "response_truncated": ev.response_truncated,
        "response_body_size": ev.response_body_size,
        "first_token_latency_ms": ev.first_token_latency_ms,
        "total_latency_ms": ev.total_latency_ms,
        "request_id": ev.request_id,
        "response_id": ev.response_id,
        "request_model": ev.request_model,
        "response_model": ev.response_model,
        "input_tokens": ev.input_tokens,
        "output_tokens": ev.output_tokens,
        "finish_reason": ev.finish_reason,
        "response_text": ev.response_text[:500],
        "exception_type": ev.exception_type,
        "exception_message": ev.exception_message,
        "success": ev.success,
    }


def finding_to_dict(f: FindingResult) -> dict[str, object]:
    """将发现结果转换为字典."""
    return {
        "rule_id": f.rule_id,
        "probe_name": f.probe_name,
        "status": f.status.value,
        "severity": f.severity.value,
        "facts": f.facts,
        "inferences": f.inferences,
        "evidence_refs": f.evidence_refs,
        "limitations": f.limitations,
    }


def _comparison_to_dict(comparison: ComparisonResult) -> dict[str, object]:
    """将 ComparisonResult 序列化为 reference_comparison 段."""
    return {
        "reference_snapshot": comparison.reference_snapshot_id,
        "suite_id": comparison.suite_id,
        "suite_version": comparison.suite_version,
        "model_a": comparison.model_a,
        "model_b": comparison.model_b,
        "coverage_diff": comparison.coverage_diff,
        "dimension_delta": comparison.dimension_delta_dict(),
    }


def _behavior_drift_to_dict(drift: BehaviorDriftResult) -> dict[str, object]:
    """将 BehaviorDriftResult 序列化为 behavior_drift 段（不含完整输出文本）."""
    return {
        "baseline_run_id": drift.baseline_run_id,
        "current_run_id": drift.current_run_id,
        "target_id": drift.target_id,
        "candidate_model_id": drift.candidate_model_id,
        "suite_id": drift.suite_id,
        "suite_version": drift.suite_version,
        "policy_id": drift.policy_id,
        "policy_version": drift.policy_version,
        "drift_level": drift.drift_level.value,
        "summary": {
            "total_items": drift.total_items,
            "graded_overlap_count": drift.graded_overlap_count,
            "graded_overlap_ratio": drift.graded_overlap_ratio,
            "outcome_changed_count": drift.outcome_changed_count,
            "outcome_changed_ratio": drift.outcome_changed_ratio,
            "status_changed_count": drift.status_changed_count,
            "status_changed_ratio": drift.status_changed_ratio,
            "output_changed_count": drift.output_changed_count,
            "output_changed_ratio": drift.output_changed_ratio,
            "response_model_change_count": drift.response_model_change_count,
            "finish_reason_change_count": drift.finish_reason_change_count,
        },
        "dimension_drift": [
            {
                "dimension": d.dimension.value,
                "baseline_score": d.baseline_score,
                "current_score": d.current_score,
                "delta": d.delta,
                "absolute_delta": d.absolute_delta,
            }
            for d in drift.dimension_diffs
        ],
        "item_drift": [
            {
                "task_id": item.key.task_id,
                "source_sample_id": item.key.source_sample_id,
                "input_sha256": item.key.input_sha256,
                "baseline_status": item.baseline_status.value,
                "current_status": item.current_status.value,
                "baseline_score": item.baseline_score,
                "current_score": item.current_score,
                "score_delta": item.score_delta,
                "outcome_changed": item.outcome_changed,
                "status_changed": item.status_changed,
                "output_changed": item.output_changed,
                "operational_changed": item.operational_changed,
                "baseline_evidence_refs": list(item.baseline_evidence_refs),
                "current_evidence_refs": list(item.current_evidence_refs),
            }
            for item in drift.item_diffs
        ],
        "warnings": list(drift.warnings),
    }


def generate_json_report(
    result: AuditResult,
    output_path: Path,
    benchmark_sections: Sequence[BenchmarkReportSection] | None = None,
    reference_comparison: ComparisonResult | None = None,
    behavior_drift: BehaviorDriftResult | None = None,
) -> Path:
    """生成 JSON 报告（v1.2 — 支持 benchmark 与 behavior_drift 段).

    Args:
        result: 审计结果.
        output_path: 输出文件路径.
        benchmark_sections: 可选 benchmark 报告段列表.
        reference_comparison: 可选 reference comparison 段（Reference Snapshot vs Candidate）.
        behavior_drift: 可选 behavior drift 段（BehaviorRunSnapshot vs BehaviorRunSnapshot）.

    Returns:
        写入后的输出文件路径.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Enforce evidence reference integrity before generating report
    if benchmark_sections:
        validate_report_evidence_refs(benchmark_sections, result.evidence)

    # Build benchmark list
    benchmarks: list[dict[str, object]] = []
    if benchmark_sections:
        for bs in benchmark_sections:
            benchmarks.append(bs.model_dump(mode="json"))

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report_id": result.report_id,
        "meta": {
            "llmtrace_version": result.llmtrace_version,
            "test_suite_version": result.config.test_suite_version,
            "utc_time": result.start_time.isoformat() if result.start_time else "",
            "local_timezone": "Asia/Shanghai",
            "python_version": result.python_version,
            "platform": result.platform,
            "report_id": result.report_id,
            "config_summary": {
                "protocol": result.config.protocol.value,
                "base_url": result.config.base_url,
                "model": result.config.model,
            },
            "probe_list": [f.probe_name for f in result.findings],
            "risk_level": result.risk_level.value,
            "schema_version": SCHEMA_VERSION,
            "content_hash": "",
        },
        "config": {
            "protocol": result.config.protocol.value,
            "base_url": result.config.base_url,
            "model": result.config.model,
            "api_key_env": result.config.api_key_env,
            "auth_style": result.config.auth_style.value,
            "repeat_count": result.config.repeat_count,
            "timeout": result.config.timeout,
            "max_output_tokens": result.config.max_output_tokens,
            "check_streaming": result.config.check_streaming,
        },
        "evidence": [evidence_to_dict(e) for e in result.evidence],
        "findings": [finding_to_dict(f) for f in result.findings],
        "risk_level": result.risk_level.value,
        "schema_fingerprints": result.schema_fingerprints,
        "model_list": result.model_list,
        "model_list_available": result.model_list_available,
        "model_in_list": result.model_in_list,
        "benchmarks": benchmarks,
    }

    if reference_comparison is not None:
        report["reference_comparison"] = _comparison_to_dict(reference_comparison)

    if behavior_drift is not None:
        report["behavior_drift"] = _behavior_drift_to_dict(behavior_drift)

    # 计算内容哈希（包含 benchmarks）
    content_json = json.dumps(report, sort_keys=True, ensure_ascii=False)
    content_hash = sha256_hash(content_json)
    meta = report["meta"]
    assert isinstance(meta, dict)
    meta["content_hash"] = content_hash
    report["content_hash"] = content_hash

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path
