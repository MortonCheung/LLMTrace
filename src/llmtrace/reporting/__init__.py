"""LLMTrace reporting package."""

from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.benchmark_models import (
    BenchmarkReportSection,
    BenchmarkRunSummary,
    FailureReportItem,
    TaskReportItem,
)
from llmtrace.reporting.console import (
    print_audit_summary,
    print_compare_result,
    print_dry_run,
    print_error,
)
from llmtrace.reporting.evidence_validation import validate_report_evidence_refs
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import SCHEMA_VERSION, generate_json_report

__all__ = [
    "print_audit_summary",
    "print_compare_result",
    "print_dry_run",
    "print_error",
    "generate_json_report",
    "generate_html_report",
    "build_benchmark_report_section",
    "BenchmarkReportSection",
    "BenchmarkRunSummary",
    "TaskReportItem",
    "FailureReportItem",
    "SCHEMA_VERSION",
    "validate_report_evidence_refs",
]
