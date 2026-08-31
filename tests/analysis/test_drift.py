"""Regression tests for the legacy protocol/operational drift (analysis/drift.py)."""

from __future__ import annotations

from llmtrace.analysis.drift import compare_reports
from llmtrace.models.report import DriftLevel


def _report(
    *,
    report_id: str,
    risk_level: str,
    evidence: list[dict],
    schema_fingerprints: list[str],
) -> dict:
    return {
        "report_id": report_id,
        "risk_level": risk_level,
        "config": {"base_url": "https://api.example.com", "model": "gpt-4"},
        "meta": {"test_suite_version": "1.0.0", "utc_time": "2026-08-31T00:00:00Z"},
        "evidence": evidence,
        "schema_fingerprints": schema_fingerprints,
    }


class TestFiveSignalsNotInconclusive:
    def test_five_drift_signals_escalate_to_likely(self) -> None:
        """5 simultaneous drift signals must NOT regress to INCONCLUSIVE.

        This is the regression for the old `_compute_drift` bug where
        ``drift_signals > 4 → INCONCLUSIVE`` — the strongest evidence of drift
        was misreported as the weakest conclusion.
        """
        a = _report(
            report_id="r1",
            risk_level="LOW",
            evidence=[
                {"success": True, "response_model": "gpt-4", "total_latency_ms": 100},
                {"success": True, "response_model": "gpt-4", "total_latency_ms": 100},
            ],
            schema_fingerprints=["fp-a"],
        )
        b = _report(
            report_id="r2",
            risk_level="HIGH",
            evidence=[
                {"success": True, "response_model": "gpt-3.5-turbo", "total_latency_ms": 400},
                {"success": False, "http_status": 500, "total_latency_ms": None},
            ],
            schema_fingerprints=["fp-b"],
        )

        result = compare_reports([a, b])

        # All five signals fire: success rate, latency, model set, fingerprint, risk level.
        assert "成功率出现显著变化" in result.drift_notes
        assert "延迟中位数出现显著变化" in result.drift_notes
        assert "返回模型集合发生变化" in result.drift_notes
        assert "响应结构指纹发生变化" in result.drift_notes
        assert "风险等级发生变化" in result.drift_notes

        assert result.drift_level == DriftLevel.LIKELY_DRIFT
        assert result.drift_level != DriftLevel.INCONCLUSIVE

    def test_three_signals_still_likely(self) -> None:
        a = _report(
            report_id="r1",
            risk_level="LOW",
            evidence=[{"success": True, "response_model": "gpt-4", "total_latency_ms": 100}],
            schema_fingerprints=["fp-a"],
        )
        b = _report(
            report_id="r2",
            risk_level="HIGH",
            evidence=[{"success": True, "response_model": "gpt-3.5-turbo", "total_latency_ms": 400}],
            schema_fingerprints=["fp-b"],
        )
        result = compare_reports([a, b])
        # latency + model set + fingerprint + risk = 4 signals → LIKELY
        assert result.drift_level == DriftLevel.LIKELY_DRIFT
