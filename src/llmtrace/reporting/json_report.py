"""JSON 报告生成."""

from __future__ import annotations

import json
from pathlib import Path

from llmtrace.models.audit import AuditResult
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult
from llmtrace.utilities.hashing import sha256_hash


def evidence_to_dict(ev: HTTPEvidence) -> dict[str, object]:
    """将证据对象转换为字典."""
    return {
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


def generate_json_report(result: AuditResult, output_path: Path) -> Path:
    """生成 JSON 报告."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "1.0",
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
            "schema_version": "1.0",
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
    }

    # 计算内容哈希
    content_json = json.dumps(report, sort_keys=True, ensure_ascii=False)
    content_hash = sha256_hash(content_json)
    meta = report["meta"]
    assert isinstance(meta, dict)
    meta["content_hash"] = content_hash
    report["content_hash"] = content_hash

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path
