"""Integration tests for JSON report with benchmark sections (schema 1.1)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkRunResult,
    BudgetEstimate,
    FailureCategory,
    GradeResult,
    GradeStatus,
    RunPlan,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.benchmark_models import BenchmarkReportSection
from llmtrace.reporting.json_report import generate_json_report


def _provenance() -> dict[str, str]:
    return {
        "suite_id": "test-suite",
        "suite_version": "1.0.0",
        "source_id": "test-source",
        "source_revision": "abc123",
        "adapter_id": "lm-eval",
        "adapter_version": "0.4.12",
    }


def _make_minimal_audit_result() -> AuditResult:
    """Build a minimal valid AuditResult for testing."""
    return AuditResult(
        config=AuditConfig(
            protocol=Protocol.OPENAI,
            base_url="https://api.example.com",
            model="test-model",
            api_key_env="TEST_KEY",
            auth_style=AuthStyle.BEARER,
            repeat_count=1,
            timeout=30.0,
            max_output_tokens=100,
            check_streaming=False,
        ),
        evidence=[],
        findings=[],
        risk_level=RiskLevel.INCONCLUSIVE,
        schema_fingerprints=[],
        model_list=[],
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, 1, tzinfo=UTC),
        llmtrace_version="0.2.0",
        python_version="3.12",
        platform="darwin",
        report_id="test-report-id",
        content_hash="",
    )


def _make_benchmark_section(
    run_id: str = "11111111-1111-1111-1111-111111111111",
    smoke: bool = False,
) -> BenchmarkReportSection:
    p = _provenance()
    plan = RunPlan(
        plan_id="test-plan",
        task_ids=["task_a"],
        total_samples=4,
        budget=BudgetEstimate(planned_requests=4, maximum_requests=4, estimated_cost=None),
        **{k: v for k, v in p.items() if k in RunPlan.model_fields},
    )

    meta = {"llmtrace_smoke_task": True} if smoke else {}
    attempt = TaskAttempt(
        attempt_id="att-1",
        task_id="task_a",
        status=TaskStatus.SUCCESS,
        evidence_refs=[str(uuid4())],
        metadata=meta,
        **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
    )

    grade = GradeResult(
        grade_id="grade-1",
        attempt_id="att-1",
        task_id="task_a",
        grader_id="exact_match",
        raw_score=0.8,
        normalized_score=0.8,
        **{k: v for k, v in p.items() if k in GradeResult.model_fields},
    )

    run_result = BenchmarkRunResult(
        run_id=run_id,
        task_attempts=[attempt],
        grade_results=[grade],
        evidence_refs=[str(uuid4())],
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
    )

    return build_benchmark_report_section(plan, run_result)


def _add_evidence_for_section(
    result: AuditResult,
    section: BenchmarkReportSection,
) -> None:
    """Add matching HTTPEvidence for all task evidence_refs in the section."""
    evidence_ids_seen: set[str] = set()
    for task in section.tasks:
        for ref in task.evidence_refs:
            if ref not in evidence_ids_seen:
                evidence_ids_seen.add(ref)
                result.evidence.append(
                    HTTPEvidence(
                        evidence_id=UUID(ref),
                        evidence_type="smoke_test",
                        request_method="POST",
                        request_url_redacted="https://api.example.com/v1/chat/completions",
                        request_path="/v1/chat/completions",
                        request_headers_redacted={"Authorization": "Bearer sk-fake-***"},
                        request_model="test-model",
                        response_model="test-model",
                        total_latency_ms=100.0,
                        http_status=200,
                        response_text="fake response",
                    )
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJsonReportNoBenchmarks:
    def test_legacy_report_no_benchmarks(self) -> None:
        """Audit report without benchmark sections still works."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path)
            data = json.loads(output_path.read_text())

        assert data["schema_version"] == "1.2"
        assert data["benchmarks"] == []
        assert "content_hash" in data
        assert data["meta"]["schema_version"] == "1.2"

    def test_none_benchmark_sections_produces_empty_list(self) -> None:
        """Passing None for benchmark_sections produces empty list."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=None)
            data = json.loads(output_path.read_text())

        assert data["benchmarks"] == []


class TestJsonReportWithBenchmarks:
    def test_single_benchmark_section(self) -> None:
        """Report with one benchmark section."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section()
        _add_evidence_for_section(result, section)
        bm: Sequence[BenchmarkReportSection] = [section]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=bm)
            data = json.loads(output_path.read_text())

        assert len(data["benchmarks"]) == 1
        bm_data = data["benchmarks"][0]
        assert isinstance(bm_data, dict)
        assert bm_data["plan_id"] == "test-plan"
        assert bm_data["status"] == "success"
        assert "tasks" in bm_data

    def test_multiple_benchmark_sections(self) -> None:
        """Report with multiple benchmark sections."""
        result = _make_minimal_audit_result()
        sections: Sequence[BenchmarkReportSection] = [
            _make_benchmark_section(run_id="11111111-1111-1111-1111-111111111111"),
            _make_benchmark_section(run_id="22222222-2222-2222-2222-222222222222"),
        ]
        for s in sections:
            _add_evidence_for_section(result, s)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=sections)
            data = json.loads(output_path.read_text())

        assert len(data["benchmarks"]) == 2

    def test_content_hash_changes_with_benchmarks(self) -> None:
        """Content hash varies when benchmark sections are added."""
        result = _make_minimal_audit_result()

        # Without benchmarks
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "r1.json"
            generate_json_report(result, p1)
            h1 = json.loads(p1.read_text())["content_hash"]

        # With one benchmark
        section = _make_benchmark_section()
        _add_evidence_for_section(result, section)
        with tempfile.TemporaryDirectory() as tmpdir:
            p2 = Path(tmpdir) / "r2.json"
            generate_json_report(result, p2, benchmark_sections=[section])
            h2 = json.loads(p2.read_text())["content_hash"]

        assert h1 != h2, "Content hash should change when benchmarks are included"

    def test_schema_version_is_1_2(self) -> None:
        """Both top-level and meta.schema_version equal 1.2."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path)
            data = json.loads(output_path.read_text())

        assert data["schema_version"] == "1.2"
        assert data["meta"]["schema_version"] == "1.2"

    def test_benchmark_failure_json(self) -> None:
        """Report correctly serializes a benchmark section with failures."""
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
        _add_evidence_for_section(result, section)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=[section])
            data = json.loads(output_path.read_text())

        bm_data = data["benchmarks"][0]
        assert bm_data["status"] == "failure"
        task_data = bm_data["tasks"][0]
        assert task_data["failure"] is not None
        assert task_data["failure"]["error_code"] == "TEST_ERROR"

    def test_smoke_task_capability_score_eligible_false(self) -> None:
        """Smoke tasks are serialized with capability_score_eligible=false."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section(smoke=True)
        _add_evidence_for_section(result, section)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=[section])
            data = json.loads(output_path.read_text())

        task = data["benchmarks"][0]["tasks"][0]
        assert task["capability_score_eligible"] is False

    def test_estimated_cost_null(self) -> None:
        """estimated_cost=None serializes as null in JSON."""
        result = _make_minimal_audit_result()
        section = _make_benchmark_section()
        _add_evidence_for_section(result, section)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=[section])
            data = json.loads(output_path.read_text())

        bm = data["benchmarks"][0]
        assert bm["estimated_cost"] is None
        assert bm["summary"]["estimated_cost"] is None


class TestSchemaVersionConstant:
    def test_schema_version_is_single_constant(self) -> None:
        """SCHEMA_VERSION is a single constant used everywhere."""
        from llmtrace.reporting import json_report

        assert isinstance(json_report.SCHEMA_VERSION, str)
        # Verify both usages use the same constant
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path)
            data = json.loads(output_path.read_text())

        assert data["schema_version"] == json_report.SCHEMA_VERSION
        assert data["meta"]["schema_version"] == json_report.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers for evidence validation tests
# ---------------------------------------------------------------------------


def _make_evidence(evidence_id: UUID | None = None) -> HTTPEvidence:
    """Create a minimal HTTPEvidence for testing."""
    return HTTPEvidence(
        evidence_id=evidence_id if evidence_id is not None else uuid4(),
        evidence_type="smoke_test",
        request_method="POST",
        request_url_redacted="https://api.example.com/v1/chat/completions",
        request_path="/v1/chat/completions",
        request_headers_redacted={"Authorization": "Bearer sk-fake-***"},
        request_model="test-model",
        response_model="test-model",
        total_latency_ms=100.0,
        http_status=200,
        response_text="fake response",
    )


def _make_benchmark_section_with_refs(
    evidence_refs: list[str],
    run_id: str = "11111111-1111-1111-1111-111111111111",
) -> BenchmarkReportSection:
    """Create a BenchmarkReportSection with specific evidence_refs."""
    p = _provenance()
    plan = RunPlan(
        plan_id="test-plan",
        task_ids=["task_a"],
        total_samples=4,
        budget=BudgetEstimate(planned_requests=4, maximum_requests=4, estimated_cost=None),
        **{k: v for k, v in p.items() if k in RunPlan.model_fields},
    )
    attempt = TaskAttempt(
        attempt_id="att-1",
        task_id="task_a",
        status=TaskStatus.SUCCESS,
        evidence_refs=evidence_refs,
        **{k: v for k, v in p.items() if k in TaskAttempt.model_fields},
    )
    grade = GradeResult(
        grade_id="grade-1",
        attempt_id="att-1",
        task_id="task_a",
        grader_id="exact_match",
        raw_score=0.8,
        normalized_score=0.8,
        status=GradeStatus.GRADED,
        evidence_refs=evidence_refs,
        **{k: v for k, v in p.items() if k in GradeResult.model_fields},
    )
    run_result = BenchmarkRunResult(
        run_id=run_id,
        task_attempts=[attempt],
        grade_results=[grade],
        evidence_refs=evidence_refs,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        **{k: v for k, v in p.items() if k in BenchmarkRunResult.model_fields},
    )
    return build_benchmark_report_section(plan, run_result)


# ---------------------------------------------------------------------------
# Evidence validation — production entry tests
# ---------------------------------------------------------------------------


class TestJsonReportEvidenceValidation:
    """Tests that generate_json_report enforces evidence reference integrity."""

    def test_evidence_closure_success(self) -> None:
        """Matching evidence allows report generation."""
        ev = _make_evidence()
        result = _make_minimal_audit_result()
        result.evidence = [ev]
        section = _make_benchmark_section_with_refs([str(ev.evidence_id)])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=[section])
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert len(data["evidence"]) == 1
            assert data["evidence"][0]["evidence_id"] == str(ev.evidence_id)

    def test_missing_evidence_raises_valueerror(self) -> None:
        """Orphan evidence_ref raises ValueError, no file created."""
        result = _make_minimal_audit_result()
        result.evidence = []  # empty — no evidence to resolve
        section = _make_benchmark_section_with_refs([str(uuid4())])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            with pytest.raises(ValueError, match="unresolvable evidence_id"):
                generate_json_report(result, output_path, benchmark_sections=[section])
            assert not output_path.exists()

    def test_duplicate_evidence_id_raises_valueerror(self) -> None:
        """Duplicate evidence_id raises ValueError."""
        dup_id = uuid4()
        ev1 = _make_evidence(evidence_id=dup_id)
        ev2 = _make_evidence(evidence_id=dup_id)
        result = _make_minimal_audit_result()
        result.evidence = [ev1, ev2]
        section = _make_benchmark_section_with_refs([str(dup_id)])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            with pytest.raises(ValueError, match="Duplicate evidence_id"):
                generate_json_report(result, output_path, benchmark_sections=[section])

    def test_no_benchmarks_preserves_old_behavior(self) -> None:
        """benchmark_sections=None does not trigger validation."""
        result = _make_minimal_audit_result()
        result.evidence = []  # empty evidence is fine with no benchmarks

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, benchmark_sections=None)
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert data["benchmarks"] == []
