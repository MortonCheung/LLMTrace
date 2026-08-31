"""Tests for JSON and HTML report `reference_comparison` sections."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import generate_json_report
from llmtrace.scoring.comparison import CapabilityComparator, ComparisonResult
from llmtrace.scoring.errors import (
    IncompatibleCoverageError,
    ScoringPolicyMismatchError,
    SuiteMismatchError,
)
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.reference import ReferenceRepository, ReferenceSnapshot
from tests.reporting.test_json_report_integration import _make_minimal_audit_result

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "reference_profiles"

_CANDIDATE_SCORES: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: 0.80,
    CapabilityDimension.CODING: 0.90,
    CapabilityDimension.MATH_SCIENCE: 0.70,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 1.0,
}


def _reference_snapshot() -> ReferenceSnapshot:
    return ReferenceRepository.load(_FIXTURE_DIR).get("openai-gpt-x-quick-v1")


def _candidate_profile() -> CapabilityProfile:
    dims = tuple(
        DimensionScoreResult(
            dimension=dim,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=score,
        )
        for dim, score in _CANDIDATE_SCORES.items()
    )
    return CapabilityProfile(
        scoring_policy_id="llmtrace-capability-v1",
        scoring_policy_version="0.1.0",
        dimensions=dims,
        coverage_weight=0.75,
    )


def _comparison() -> ComparisonResult:
    return CapabilityComparator().compare(
        _reference_snapshot(),
        _candidate_profile(),
        candidate_suite_id="llmtrace_quick_v1",
        candidate_suite_version="0.1.0",
        candidate_model_id="candidate-api",
    )


class TestJsonReferenceComparison:
    def test_reference_comparison_section_present(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, reference_comparison=_comparison())
            data = json.loads(output_path.read_text())

        rc = data["reference_comparison"]
        assert rc["reference_snapshot"] == "openai-gpt-x-quick-v1"
        assert rc["suite_id"] == "llmtrace_quick_v1"
        assert rc["suite_version"] == "0.1.0"
        assert rc["model_a"] == "gpt-x"
        assert rc["model_b"] == "candidate-api"
        assert rc["coverage_diff"] == 0.0

    def test_dimension_delta_present(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, reference_comparison=_comparison())
            data = json.loads(output_path.read_text())

        dd = data["reference_comparison"]["dimension_delta"]
        assert set(dd) == {"reasoning", "coding", "math_science", "instruction_following"}
        assert dd["reasoning"]["reference"] == 0.92
        assert dd["reasoning"]["candidate"] == 0.80
        assert round(dd["reasoning"]["delta"], 2) == -0.12

    def test_no_reference_comparison_key_when_none(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path)
            data = json.loads(output_path.read_text())

        assert "reference_comparison" not in data

    def test_content_hash_changes_with_reference_comparison(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "r1.json"
            generate_json_report(result, p1)
            h1 = json.loads(p1.read_text())["content_hash"]

            p2 = Path(tmpdir) / "r2.json"
            generate_json_report(result, p2, reference_comparison=_comparison())
            h2 = json.loads(p2.read_text())["content_hash"]

        assert h1 != h2


class TestHtmlReferenceComparison:
    def test_reference_comparison_section_rendered(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, reference_comparison=_comparison())
            html = output_path.read_text()

        assert "<h2>7. Reference Comparison</h2>" in html
        assert "<h2>8. 限制和免责声明</h2>" in html
        assert "openai-gpt-x-quick-v1" in html
        assert "gpt-x" in html
        assert "candidate-api" in html

    def test_reference_comparison_table_headers(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, reference_comparison=_comparison())
            html = output_path.read_text()

        assert "<th>Dimension</th>" in html
        assert "<th>Reference</th>" in html
        assert "<th>Candidate</th>" in html
        assert "<th>Delta</th>" in html
        assert "reasoning" in html
        assert "math_science" in html

    def test_uncallibrated_warning_present(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, reference_comparison=_comparison())
            html = output_path.read_text()

        assert "UNCALIBRATED" in html
        assert "Reference Comparison 不是 Calibration" in html
        assert "不构成模型身份结论" in html

    def test_no_reference_comparison_when_none(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path)
            html = output_path.read_text()

        assert "Reference Comparison" not in html
        assert "<h2>7. 限制和免责声明</h2>" in html

    def test_no_identity_conclusion_in_html(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, reference_comparison=_comparison())
            html = output_path.read_text()

        assert "detected as" not in html
        assert "candidate is GPT" not in html


class TestIncompatibleComparisonNeverReachesReport:
    """Fail closed: an incompatible pair must never produce a comparison section.

    The reporting layer only accepts an already-validated ``ComparisonResult``,
    so the guarantee is that the comparator refuses *before* such an object
    exists.  These tests exercise the real comparator → report path.
    """

    def _render_without_comparison(self) -> dict[str, object]:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path)
            return dict(json.loads(output_path.read_text()))

    def test_suite_mismatch_produces_no_reference_comparison(self) -> None:
        with pytest.raises(SuiteMismatchError):
            CapabilityComparator().compare(
                _reference_snapshot(),
                _candidate_profile(),
                candidate_suite_id="llmtrace_quick_v2",
                candidate_suite_version="0.1.0",
                candidate_model_id="candidate-api",
            )
        assert "reference_comparison" not in self._render_without_comparison()

    def test_scoring_policy_mismatch_produces_no_reference_comparison(self) -> None:
        mismatched = CapabilityProfile(
            scoring_policy_id="another-policy",
            scoring_policy_version="0.2.0",
            dimensions=_candidate_profile().dimensions,
            coverage_weight=0.75,
        )
        with pytest.raises(ScoringPolicyMismatchError):
            CapabilityComparator().compare(
                _reference_snapshot(),
                mismatched,
                candidate_suite_id="llmtrace_quick_v1",
                candidate_suite_version="0.1.0",
                candidate_model_id="candidate-api",
            )
        assert "reference_comparison" not in self._render_without_comparison()

    def test_unavailable_dimension_produces_no_reference_comparison(self) -> None:
        """An unmeasured dimension must never appear in a report as a score drop."""
        dims = tuple(
            DimensionScoreResult(
                dimension=dim,
                status=(
                    DimensionScoreStatus.UNAVAILABLE
                    if dim is CapabilityDimension.MATH_SCIENCE
                    else DimensionScoreStatus.UNCALIBRATED
                ),
                raw_normalized_score=score,
            )
            for dim, score in _CANDIDATE_SCORES.items()
        )
        partial = CapabilityProfile(
            scoring_policy_id="llmtrace-capability-v1",
            scoring_policy_version="0.1.0",
            dimensions=dims,
            coverage_weight=0.60,
        )
        with pytest.raises(IncompatibleCoverageError):
            CapabilityComparator().compare(
                _reference_snapshot(),
                partial,
                candidate_suite_id="llmtrace_quick_v1",
                candidate_suite_version="0.1.0",
                candidate_model_id="candidate-api",
            )
        assert "reference_comparison" not in self._render_without_comparison()


class TestHtmlEscapingOfComparisonLabels:
    """Labels are user/model supplied and must be Jinja-escaped in the HTML report."""

    _HOSTILE_A = "<script>alert('a')</script>"
    _HOSTILE_B = "<img src=x onerror=alert('b')>"

    def _hostile_comparison(self) -> ComparisonResult:
        snapshot = _reference_snapshot().model_copy(update={"model_id": self._HOSTILE_A})
        return CapabilityComparator().compare(
            snapshot,
            _candidate_profile(),
            candidate_suite_id="llmtrace_quick_v1",
            candidate_suite_version="0.1.0",
            candidate_model_id=self._HOSTILE_B,
        )

    def test_hostile_reference_label_is_escaped(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, reference_comparison=self._hostile_comparison())
            html = output_path.read_text()

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_hostile_candidate_label_is_escaped(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, reference_comparison=self._hostile_comparison())
            html = output_path.read_text()

        assert "<img src=x onerror" not in html
        assert "&lt;img src=x onerror" in html

    def test_hostile_label_is_escaped_in_json_without_execution(self) -> None:
        """JSON must carry the raw label verbatim (escaping is an HTML concern only)."""
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, reference_comparison=self._hostile_comparison())
            data = json.loads(output_path.read_text())

        assert data["reference_comparison"]["model_a"] == self._HOSTILE_A
        assert data["reference_comparison"]["model_b"] == self._HOSTILE_B
