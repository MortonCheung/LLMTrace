"""HTML 报告生成."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from llmtrace.analysis.risk import risk_explanation
from llmtrace.models.audit import AuditResult
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.reporting.benchmark_models import BenchmarkReportSection
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


def generate_html_report(
    result: AuditResult,
    output_path: Path,
    benchmark_sections: Sequence[BenchmarkReportSection] | None = None,
) -> Path:
    """生成 HTML 报告."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=PackageLoader("llmtrace", "reporting/templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    # 计算统计数据 — 只统计 baseline 证据
    evidence = [e for e in result.evidence if e.evidence_type == "baseline"]
    token_evidence = [e for e in evidence if e.input_tokens is not None and e.output_tokens is not None]
    rid_evidence = [e for e in evidence if e.response_id is not None]
    models = {e.response_model for e in evidence if e.response_model}

    # 构建 Benchmark 数据
    benchmark_data: list[dict[str, object]] = []
    if benchmark_sections:
        _status_class_map = {
            "success": "status-pass",
            "partial_failure": "status-warn",
            "failure": "status-fail",
            "incomplete": "status-warn",
            "skipped": "status-warn",
        }
        for section in benchmark_sections:
            section_dict = section.model_dump(mode="json")

            # Computed display fields
            section_dict["status_class"] = _status_class_map.get(section.status.value, "status-warn")
            section_dict["started_at_display"] = (
                sanitize_for_html(section.started_at.isoformat()) if section.started_at else "N/A"
            )
            section_dict["finished_at_display"] = (
                sanitize_for_html(section.finished_at.isoformat()) if section.finished_at else "N/A"
            )
            if section.estimated_cost is not None:
                section_dict["estimated_cost_display"] = sanitize_for_html(f"${section.estimated_cost:.6f}")
            else:
                section_dict["estimated_cost_display"] = "未估算"

            # Sanitize text fields on the section
            for _field in (
                "suite_id",
                "suite_version",
                "source_id",
                "source_revision",
                "adapter_id",
                "adapter_version",
            ):
                section_dict[_field] = sanitize_for_html(str(section_dict[_field]))

            # Build per-task display data
            tasks_display: list[dict[str, object]] = []
            for task in section.tasks:
                task_dict: dict[str, object] = task.model_dump(mode="json")

                task_dict["status_class"] = _status_class_map.get(task.status.value, "status-warn")

                if task.raw_score is not None:
                    task_dict["score_display"] = task.raw_score
                else:
                    task_dict["score_display"] = "N/A"

                if task.failure:
                    task_dict["failure_display"] = sanitize_for_html(
                        f"[{task.failure.error_code}] {task.failure.message}"
                    )
                else:
                    task_dict["failure_display"] = None

                if task.evidence_refs:
                    task_dict["evidence_refs_display"] = sanitize_for_html(", ".join(task.evidence_refs))
                else:
                    task_dict["evidence_refs_display"] = "N/A"

                task_dict["is_smoke"] = not task.capability_score_eligible

                # Sanitize task text fields
                task_dict["task_id"] = sanitize_for_html(task.task_id)
                task_dict["grader_id"] = sanitize_for_html(task.grader_id or "")

                tasks_display.append(task_dict)

            section_dict["tasks"] = tasks_display
            benchmark_data.append(section_dict)

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
        benchmarks=benchmark_data,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
