"""HTML 报告生成."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from llmtrace.analysis.risk import risk_explanation
from llmtrace.models.audit import AuditResult
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.security.redaction import sanitize_for_html


def _escape_evidence(ev: HTTPEvidence) -> dict[str, object]:
    """HTML 转义证据中的文本字段."""
    return {
        "request_method": sanitize_for_html(ev.request_method),
        "request_url_redacted": sanitize_for_html(ev.request_url_redacted),
        "request_path": sanitize_for_html(ev.request_path),
        "http_status": ev.http_status,
        "total_latency_ms": ev.total_latency_ms,
        "request_model": sanitize_for_html(ev.request_model or ""),
        "response_model": sanitize_for_html(ev.response_model or ""),
        "response_id": sanitize_for_html(ev.response_id or ""),
        "input_tokens": ev.input_tokens,
        "output_tokens": ev.output_tokens,
        "finish_reason": sanitize_for_html(ev.finish_reason or ""),
        "response_text": sanitize_for_html(ev.response_text[:200]),
        "response_body_sha256": ev.response_body_sha256,
        "response_body_summary": sanitize_for_html(ev.response_body_summary[:500]),
        "response_truncated": ev.response_truncated,
        "exception_type": sanitize_for_html(ev.exception_type or ""),
        "exception_message": sanitize_for_html(ev.exception_message or ""),
        "success": ev.success,
        "response_headers": {sanitize_for_html(k): sanitize_for_html(v) for k, v in ev.response_headers.items()},
        "request_headers_redacted": {
            sanitize_for_html(k): sanitize_for_html(str(v)) for k, v in ev.request_headers_redacted.items()
        },
    }


def generate_html_report(result: AuditResult, output_path: Path) -> Path:
    """生成 HTML 报告."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=PackageLoader("llmtrace", "reporting/templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    # 计算统计数据
    evidence = [e for e in result.evidence if e.request_model == result.config.model]
    token_evidence = [e for e in evidence if e.input_tokens is not None and e.output_tokens is not None]
    rid_evidence = [e for e in evidence if e.response_id is not None]
    models = {e.response_model for e in evidence if e.response_model}

    html = template.render(
        report_id=result.report_id,
        utc_time=result.start_time.isoformat() if result.start_time else "",
        llmtrace_version=result.llmtrace_version,
        python_version=result.python_version,
        platform=result.platform,
        config={
            "base_url": sanitize_for_html(result.config.base_url),
            "protocol": result.config.protocol.value,
            "model": sanitize_for_html(result.config.model),
            "repeat_count": result.config.repeat_count,
            "timeout": result.config.timeout,
            "max_output_tokens": result.config.max_output_tokens,
        },
        risk_level=result.risk_level.value,
        risk_explanation=risk_explanation(result.risk_level),
        findings=[
            {
                "probe_name": sanitize_for_html(f.probe_name),
                "status": f.status.value,
                "severity": f.severity.value,
                "facts": [sanitize_for_html(x) for x in f.facts],
                "inferences": [sanitize_for_html(x) for x in f.inferences],
                "limitations": [sanitize_for_html(x) for x in f.limitations],
            }
            for f in result.findings
        ],
        evidence=[_escape_evidence(e) for e in result.evidence],
        response_models=sorted(models),
        fingerprints=result.schema_fingerprints,
        token_rate=f"{len(token_evidence)}/{len(evidence)}" if evidence else "N/A",
        rid_rate=f"{len(rid_evidence)}/{len(evidence)}" if evidence else "N/A",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
