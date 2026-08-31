"""Tests for the `behavior_drift` JSON/HTML report section."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from llmtrace.analysis.behavior_drift import BehaviorDriftEngine, BehaviorDriftResult
from llmtrace.analysis.behavior_models import BehaviorDriftPolicy
from llmtrace.benchmarks.models import ItemStatus
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import generate_json_report
from tests.analysis.conftest import make_snapshot
from tests.reporting.test_json_report_integration import _make_minimal_audit_result

_ENGINE = BehaviorDriftEngine()
_POLICY = BehaviorDriftPolicy.create_v1()


def _drift_result(**kwargs: object) -> BehaviorDriftResult:
    base_items = [
        {"task_id": "gsm8k_quick_v1", "source_sample_id": f"s{i}", "status": ItemStatus.GRADED, "score": 1.0}
        for i in range(4)
    ]
    current_items = [
        {
            "task_id": "gsm8k_quick_v1",
            "source_sample_id": f"s{i}",
            "status": ItemStatus.GRADED,
            "score": 1.0 if i < 3 else 0.0,
        }
        for i in range(4)
    ]
    baseline = make_snapshot(items=base_items, **kwargs)
    current = make_snapshot(items=current_items, **kwargs)
    return _ENGINE.compare(baseline, current, _POLICY)


class TestJsonBehaviorDrift:
    def test_behavior_drift_section_present(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, behavior_drift=_drift_result())
            data = json.loads(output_path.read_text())

        bd = data["behavior_drift"]
        assert bd["drift_level"] == "OBSERVED_DRIFT"
        assert bd["suite_id"] == "llmtrace_quick_v1"
        assert bd["policy_id"] == "llmtrace_behavior_drift_v1"
        assert bd["summary"]["total_items"] == 4
        assert bd["summary"]["outcome_changed_count"] == 1
        assert len(bd["dimension_drift"]) > 0
        assert len(bd["item_drift"]) == 4

    def test_no_behavior_drift_key_when_none(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path)
            data = json.loads(output_path.read_text())

        assert "behavior_drift" not in data

    def test_no_full_response_text_in_json(self) -> None:
        distinctive = "THE-SECRET-MODEL-OUTPUT-SENTENCE"
        base_items = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": distinctive,
            }
        ]
        current_items = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": "other",
            }
        ]
        drift = _ENGINE.compare(make_snapshot(items=base_items), make_snapshot(items=current_items), _POLICY)
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, behavior_drift=drift)
            raw = output_path.read_text()

        assert distinctive not in raw

    def test_schema_version_is_1_3_with_behavior_drift(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(result, output_path, behavior_drift=_drift_result())
            data = json.loads(output_path.read_text())

        assert data["schema_version"] == "1.3"


class TestHtmlBehaviorDrift:
    def test_behavior_drift_section_rendered(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, behavior_drift=_drift_result())
            html = output_path.read_text()

        assert "Behavior Drift" in html
        assert "OBSERVED_DRIFT" in html
        assert "Graded Overlap" in html
        assert "Outcome Changed" in html

    def test_no_behavior_drift_when_none(self) -> None:
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path)
            html = output_path.read_text()

        assert "Behavior Drift" not in html

    def test_hostile_labels_escaped(self) -> None:
        drift = _drift_result(candidate_model_id="<script>alert('x')</script>")
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, behavior_drift=drift)
            html = output_path.read_text()

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_hostile_task_id_escaped(self) -> None:
        base_items = [
            {
                "task_id": "<img src=x onerror=alert(1)>",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
            }
        ]
        drift = _ENGINE.compare(make_snapshot(items=base_items), make_snapshot(items=base_items), _POLICY)
        result = _make_minimal_audit_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generate_html_report(result, output_path, behavior_drift=drift)
            html = output_path.read_text()

        assert "<img src=x onerror" not in html
        assert "&lt;img src=x onerror" in html
