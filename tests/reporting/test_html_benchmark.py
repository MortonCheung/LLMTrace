"""Integration tests for HTML report with benchmark sections."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from uuid import uuid4

from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkRunResult,
    BudgetEstimate,
    FailureCategory,
    RunPlan,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.html_report import generate_html_report
from tests.reporting.test_json_report_integration import (
    _make_benchmark_section,
    _make_minimal_audit_result,
    _provenance,
)


class TestHtmlNoBenchmarks:
    def test_legacy_html_no_benchmarks(self) -> None:
        """HTML report without benchmark sections still renders correctly."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path)
            html = output_path.read_text()

        assert "<h1>LLMTrace 审计报告</h1>" in html
        assert "<h2>1. 被测接口摘要</h2>" in html
        assert "<h2>7. 限制和免责声明</h2>" in html
        assert "能力评测" not in html

    def test_legacy_html_with_none_benchmarks(self) -> None:
        """Passing None for benchmark_sections works as before."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=None)
            html = output_path.read_text()

        assert "<h2>7. 限制和免责声明</h2>" in html
        assert "能力评测" not in html

    def test_legacy_html_with_empty_benchmarks(self) -> None:
        """Passing empty list for benchmark_sections works as before."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[])
            html = output_path.read_text()

        assert "<h2>7. 限制和免责声明</h2>" in html
        assert "能力评测" not in html


class TestHtmlWithBenchmarks:
    def test_success_benchmark_section(self) -> None:
        """HTML contains suite_id, status=success, raw_score, evidence_refs UUID."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        assert "<h2>7. 能力评测</h2>" in html
        assert "<h2>8. 限制和免责声明</h2>" in html
        assert section.suite_id in html
        assert "success" in html
        # Verify raw_score 0.8 appears
        assert "0.8" in html
        # Verify evidence_refs UUID appears in the HTML
        task_ev_refs = section.tasks[0].evidence_refs
        assert len(task_ev_refs) > 0
        assert task_ev_refs[0] in html

    def test_failure_benchmark_section(self) -> None:
        """HTML reflects failure status and error_code."""
        result = _make_minimal_audit_result()
        p = _provenance()
        plan = RunPlan(
            plan_id="fail-plan",
            task_ids=["task_a"],
            total_samples=2,
            budget=BudgetEstimate(planned_requests=2, maximum_requests=2, estimated_cost=None),
            **{k: v for k, v in p.items() if k in RunPlan.model_fields},
        )
        attempt = TaskAttempt(
            attempt_id="att-fail",
            task_id="task_a",
            status=TaskStatus.FAILURE,
            evidence_refs=[str(uuid4())],
            failure=AdapterFailure(
                error_code="TEST_ERROR",
                category=FailureCategory.PROVIDER,
                message="Provider rejected request",
                retryable=True,
            ),
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        rr = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[str(uuid4())],
            **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
        )
        section = build_benchmark_report_section(plan, rr)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        assert "failure" in html
        assert "TEST_ERROR" in html

    def test_ungraded_benchmark(self) -> None:
        """Ungraded task shows N/A, no forged 0.0 score."""
        result = _make_minimal_audit_result()
        p = _provenance()
        ev_uuid = str(uuid4())
        plan = RunPlan(
            plan_id="ungraded-plan",
            task_ids=["task_a"],
            total_samples=1,
            budget=BudgetEstimate(planned_requests=1, maximum_requests=1, estimated_cost=None),
            **{k: v for k, v in p.items() if k in RunPlan.model_fields},
        )
        attempt = TaskAttempt(
            attempt_id="att-no-grade",
            task_id="task_a",
            status=TaskStatus.SUCCESS,
            evidence_refs=[ev_uuid],
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        rr = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[ev_uuid],
            **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
        )
        section = build_benchmark_report_section(plan, rr)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        # Extract the benchmark section HTML only
        bm_match = re.search(r"<h2>7\. 能力评测</h2>(.*?)<h2>8\.", html, re.DOTALL)
        assert bm_match is not None
        bm_html = bm_match.group(1)

        assert "N/A" in bm_html
        # Verify no forged 0.0 score in benchmark section
        assert ">0.0<" not in bm_html

    def test_smoke_note(self) -> None:
        """Smoke tasks show the smoke note text in HTML."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section(smoke=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        assert "不计入正式能力评分" in html

    def test_evidence_uuid_in_html(self) -> None:
        """Evidence UUID string appears in the HTML output."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section()

        # Get one of the evidence refs UUIDs
        ev_uuid = section.tasks[0].evidence_refs[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        assert ev_uuid in html

    def test_html_escaping(self) -> None:
        """HTML-special chars in user text are escaped."""
        result = _make_minimal_audit_result()
        p = _provenance()
        xss_task_id = '<script>alert("xss")</script>'
        xss_message = "<img src=x onerror=alert(1)>"

        plan = RunPlan(
            plan_id="xss-plan",
            task_ids=[xss_task_id],
            total_samples=1,
            budget=BudgetEstimate(planned_requests=1, maximum_requests=1, estimated_cost=None),
            **{k: v for k, v in p.items() if k in RunPlan.model_fields},
        )
        attempt = TaskAttempt(
            attempt_id="att-xss",
            task_id=xss_task_id,
            status=TaskStatus.FAILURE,
            evidence_refs=[str(uuid4())],
            failure=AdapterFailure(
                error_code="ERR",
                category=FailureCategory.PROVIDER,
                message=xss_message,
                retryable=False,
            ),
            **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
        )
        rr = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[str(uuid4())],
            **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
        )
        section = build_benchmark_report_section(plan, rr)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        # Raw script tags should not appear
        assert "<script>" not in html
        assert "<img src=x" not in html
        # Escaped versions should appear
        assert "&lt;script&gt;" in html
        assert "&lt;img" in html

    def test_estimated_cost_none_display(self) -> None:
        """estimated_cost=None displays as 未估算."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        assert "未估算" in html

    def test_no_total_score_text(self) -> None:
        """Strings total_score and capability_score do not appear in HTML."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, benchmark_sections=[section])
            html = output_path.read_text()

        assert "total_score" not in html
        assert "capability_score" not in html
