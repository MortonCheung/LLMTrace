"""Tests for evidence reference validation (validate_report_evidence_refs)."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from llmtrace.models.evidence import HTTPEvidence
from llmtrace.reporting.benchmark_models import (
    BenchmarkReportSection,
    BenchmarkReportStatus,
    TaskReportItem,
    TaskReportStatus,
)
from llmtrace.reporting.evidence_validation import validate_report_evidence_refs


def _section(task_id: str, evidence_refs: list[str]) -> BenchmarkReportSection:
    return BenchmarkReportSection(
        run_id=UUID(uuid4().hex),
        plan_id="test-plan",
        suite_id="test-suite",
        suite_version="1.0",
        source_id="test",
        source_revision="abc123",
        adapter_id="test-adapter",
        adapter_version="1.0",
        status=BenchmarkReportStatus.SUCCESS,
        planned_requests=1,
        maximum_requests=1,
        actual_requests=1,
        tasks=[
            TaskReportItem(
                task_id=task_id,
                attempt_id="att-1",
                status=TaskReportStatus.SUCCESS,
                evidence_refs=evidence_refs,
                capability_score_eligible=True,
            )
        ],
        summary={
            "planned_requests": 1,
            "maximum_requests": 1,
            "actual_requests": 1,
            "success_count": 1,
            "failure_count": 0,
            "skip_count": 0,
            "ungraded_count": 0,
            "ungradable_count": 0,
            "warnings": [],
        },
        warnings=[],
    )


def _evidence(eid: UUID) -> HTTPEvidence:
    return HTTPEvidence(
        evidence_id=eid,
        request_method="POST",
        request_url_redacted="https://test.example.com",
        request_path="/test",
        request_headers_redacted={},
        request_model="test",
        response_model="test",
        response_text="ok",
    )


class TestValidateReportEvidenceRefs:
    """Evidence reference integrity validation."""

    def test_empty_passes(self) -> None:
        validate_report_evidence_refs([], [])

    def test_no_benchmarks_with_evidence_passes(self) -> None:
        validate_report_evidence_refs([], [_evidence(uuid4())])

    def test_resolvable_refs_pass(self) -> None:
        eid = uuid4()
        section = _section("task_a", evidence_refs=[str(eid)])
        validate_report_evidence_refs([section], [_evidence(eid)])

    def test_unresolvable_ref_raises(self) -> None:
        eid = uuid4()
        missing = str(uuid4())
        section = _section("task_a", evidence_refs=[missing])
        with pytest.raises(ValueError, match=f"unresolvable evidence_id '{missing}'"):
            validate_report_evidence_refs([section], [_evidence(eid)])

    def test_duplicate_evidence_id_raises(self) -> None:
        eid = uuid4()
        section = _section("task_a", evidence_refs=[str(eid)])
        with pytest.raises(ValueError, match="Duplicate evidence_id"):
            validate_report_evidence_refs(
                [section],
                [_evidence(eid), _evidence(eid)],
            )

    def test_extra_evidence_no_ref_is_ok(self) -> None:
        """Extra evidence with no benchmark referrer is fine (audit-only)."""
        eid = uuid4()
        section = _section("task_a", evidence_refs=[str(eid)])
        validate_report_evidence_refs(
            [section],
            [_evidence(eid), _evidence(uuid4())],
        )

    def test_multiple_tasks_resolve_correctly(self) -> None:
        e1 = uuid4()
        e2 = uuid4()
        section = BenchmarkReportSection(
            run_id=UUID(uuid4().hex),
            plan_id="test-plan",
            suite_id="test-suite",
            suite_version="1.0",
            source_id="test",
            source_revision="abc123",
            adapter_id="test-adapter",
            adapter_version="1.0",
            status=BenchmarkReportStatus.SUCCESS,
            planned_requests=2,
            maximum_requests=2,
            actual_requests=2,
            tasks=[
                TaskReportItem(
                    task_id="task_a",
                    attempt_id="att-1",
                    status=TaskReportStatus.SUCCESS,
                    evidence_refs=[str(e1)],
                    capability_score_eligible=True,
                ),
                TaskReportItem(
                    task_id="task_b",
                    attempt_id="att-2",
                    status=TaskReportStatus.SUCCESS,
                    evidence_refs=[str(e2)],
                    capability_score_eligible=True,
                ),
            ],
            summary={
                "planned_requests": 2,
                "maximum_requests": 2,
                "actual_requests": 2,
                "success_count": 2,
                "failure_count": 0,
                "skip_count": 0,
                "ungraded_count": 0,
                "ungradable_count": 0,
                "warnings": [],
            },
            warnings=[],
        )
        validate_report_evidence_refs([section], [_evidence(e1), _evidence(e2)])
