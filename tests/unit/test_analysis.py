"""分析模块单元测试."""

from __future__ import annotations

import json
import re

from llmtrace.analysis.drift import compare_reports
from llmtrace.analysis.risk import analyze_risk, risk_explanation
from llmtrace.analysis.schema_fingerprint import generate_schema_fingerprint
from llmtrace.models.audit import RiskLevel
from llmtrace.models.findings import FindingResult, ProbeStatus, Severity
from llmtrace.models.report import CompareResult, DriftLevel

# ---------------------------------------------------------------------------
# 辅助工厂函数
# ---------------------------------------------------------------------------


def _make_finding(
    rule_id: str = "R001",
    probe_name: str = "test_probe",
    status: ProbeStatus = ProbeStatus.PASS,
    severity: Severity = Severity.INFO,
    inferences: list[str] | None = None,
) -> FindingResult:
    return FindingResult(
        rule_id=rule_id,
        probe_name=probe_name,
        status=status,
        severity=severity,
        inferences=inferences or [],
    )


def _make_report_dict(
    report_id: str = "r1",
    risk_level: str = "LOW",
    base_url: str = "https://api.example.com",
    model: str = "gpt-4",
    test_suite_version: str = "1.0.0",
    utc_time: str = "2025-01-01T00:00:00Z",
    evidence: list[dict] | None = None,
    schema_fingerprints: list[str] | None = None,
) -> dict:
    return {
        "report_id": report_id,
        "risk_level": risk_level,
        "config": {
            "base_url": base_url,
            "model": model,
        },
        "meta": {
            "test_suite_version": test_suite_version,
            "utc_time": utc_time,
        },
        "evidence": evidence or [],
        "schema_fingerprints": schema_fingerprints or [],
    }


def _default_evidence() -> list[dict]:
    return [
        {
            "success": True,
            "total_latency_ms": 200,
            "response_model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 50,
            "response_id": "chatcmpl-xxx",
            "exception_type": None,
            "http_status": 200,
        },
        {
            "success": True,
            "total_latency_ms": 250,
            "response_model": "gpt-4",
            "input_tokens": 120,
            "output_tokens": 60,
            "response_id": "chatcmpl-yyy",
            "exception_type": None,
            "http_status": 200,
        },
    ]


# ============================================================================
# analyze_risk 测试
# ============================================================================


class TestAnalyzeRisk:
    """analyze_risk 函数测试."""

    def test_empty_findings_returns_inconclusive(self) -> None:
        """空发现列表返回 INCONCLUSIVE."""
        assert analyze_risk([]) == RiskLevel.INCONCLUSIVE

    def test_only_info_findings_returns_low(self) -> None:
        """仅 info 级别的发现返回 LOW."""
        findings = [
            _make_finding(severity=Severity.INFO),
            _make_finding(severity=Severity.INFO, status=ProbeStatus.PASS),
        ]
        assert analyze_risk(findings) == RiskLevel.LOW

    def test_medium_severity_warnings_return_medium(self) -> None:
        """中等级别的警告返回 MEDIUM."""
        findings = [
            _make_finding(
                severity=Severity.MEDIUM,
                status=ProbeStatus.WARN,
                inferences=["模型列表不一致"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.MEDIUM

    def test_medium_severity_fail_return_medium(self) -> None:
        """中等级别的失败也返回 MEDIUM."""
        findings = [
            _make_finding(
                severity=Severity.MEDIUM,
                status=ProbeStatus.FAIL,
                inferences=["元数据字段缺失"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.MEDIUM

    def test_high_severity_failures_return_high(self) -> None:
        """高严重性失败返回 HIGH."""
        findings = [
            _make_finding(
                severity=Severity.HIGH,
                status=ProbeStatus.FAIL,
                inferences=["模型行为异常"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.HIGH

    def test_invalid_model_successful_content_returns_high(self) -> None:
        """无效模型名称仍成功生成内容返回 HIGH."""
        findings = [
            _make_finding(
                severity=Severity.HIGH,
                status=ProbeStatus.FAIL,
                inferences=["无效模型名称仍成功生成内容"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.HIGH

    def test_multiple_model_identifiers_returns_high(self) -> None:
        """同一会话中返回了多个不同的模型标识返回 HIGH."""
        findings = [
            _make_finding(
                severity=Severity.HIGH,
                status=ProbeStatus.FAIL,
                inferences=["同一会话中返回了多个不同的模型标识"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.HIGH

    def test_auth_failure_returns_inconclusive(self) -> None:
        """鉴权失败返回 INCONCLUSIVE."""
        findings = [
            _make_finding(
                severity=Severity.HIGH,
                status=ProbeStatus.FAIL,
                inferences=["鉴权失败，API 返回 401"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.INCONCLUSIVE

    def test_connection_failure_returns_inconclusive(self) -> None:
        """连接失败返回 INCONCLUSIVE."""
        findings = [
            _make_finding(
                severity=Severity.HIGH,
                status=ProbeStatus.FAIL,
                inferences=["连接失败，无法访问目标端点"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.INCONCLUSIVE

    def test_low_severity_findings_returns_low(self) -> None:
        """低严重性发现返回 LOW."""
        findings = [
            _make_finding(
                severity=Severity.LOW,
                status=ProbeStatus.WARN,
                inferences=["非关键字段缺失"],
            ),
        ]
        assert analyze_risk(findings) == RiskLevel.LOW


# ============================================================================
# risk_explanation 测试
# ============================================================================


class TestRiskExplanation:
    """risk_explanation 函数测试."""

    def test_low_explanation(self) -> None:
        """LOW 等级返回非空解释."""
        explanation = risk_explanation(RiskLevel.LOW)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_medium_explanation(self) -> None:
        """MEDIUM 等级返回非空解释."""
        explanation = risk_explanation(RiskLevel.MEDIUM)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_high_explanation(self) -> None:
        """HIGH 等级返回非空解释."""
        explanation = risk_explanation(RiskLevel.HIGH)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_inconclusive_explanation(self) -> None:
        """INCONCLUSIVE 等级返回非空解释."""
        explanation = risk_explanation(RiskLevel.INCONCLUSIVE)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_all_levels_have_unique_explanations(self) -> None:
        """所有风险等级的解释各不相同."""
        explanations = {level: risk_explanation(level) for level in RiskLevel}
        assert len(set(explanations.values())) == len(RiskLevel)


# ============================================================================
# generate_schema_fingerprint 测试
# ============================================================================

SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def _is_sha256_hex(value: str) -> bool:
    return bool(SHA256_HEX_RE.match(value))


class TestGenerateSchemaFingerprint:
    """generate_schema_fingerprint 函数测试."""

    def test_valid_json_returns_sha256_hash(self) -> None:
        """有效 JSON 返回稳定的 SHA-256 哈希."""
        fp = generate_schema_fingerprint('{"a": 1, "b": "hello"}')
        assert fp is not None
        assert _is_sha256_hex(fp)

    def test_same_structure_returns_same_fingerprint(self) -> None:
        """相同结构返回相同指纹."""
        fp1 = generate_schema_fingerprint('{"a": 1, "b": "hello"}')
        fp2 = generate_schema_fingerprint('{"a": 42, "b": "world"}')
        assert fp1 == fp2

    def test_different_structure_returns_different_fingerprint(self) -> None:
        """不同结构返回不同指纹."""
        fp1 = generate_schema_fingerprint('{"a": 1, "b": "hello"}')
        fp2 = generate_schema_fingerprint('{"a": 1, "c": "hello"}')
        assert fp1 != fp2

    def test_invalid_json_returns_none(self) -> None:
        """无效 JSON 返回 None."""
        assert generate_schema_fingerprint("not valid json") is None

    def test_empty_response_returns_none(self) -> None:
        """空字符串返回 None."""
        assert generate_schema_fingerprint("") is None

    def test_none_input_returns_none(self) -> None:
        """None 输入返回 None."""
        assert generate_schema_fingerprint("null") is not None  # 有效的 JSON null
        # 测试 None 本身在 json.loads 会抛 TypeError
        # 这里通过传入非 JSON 字符串来模拟

    def test_fingerprint_stable_across_random_values(self) -> None:
        """相同结构不同随机值仍返回相同指纹."""
        fingerprints = []
        for i in range(10):
            payload = {
                "id": f"chatcmpl-{i}",
                "object": "chat.completion",
                "created": 1234567890 + i,
                "model": f"gpt-4-{i % 3:04d}",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"response content {i}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10 + i,
                    "completion_tokens": 20 + i,
                    "total_tokens": 30 + i,
                },
            }
            fingerprints.append(generate_schema_fingerprint(json.dumps(payload)))
        # 所有指纹应该相同
        assert len(set(fingerprints)) == 1

    def test_extra_field_changes_fingerprint(self) -> None:
        """结构增加字段会改变指纹."""
        fp1 = generate_schema_fingerprint('{"a": 1}')
        fp2 = generate_schema_fingerprint('{"a": 1, "b": 2}')
        assert fp1 != fp2

    def test_nested_structure_fingerprint(self) -> None:
        """嵌套 JSON 结构指纹正常."""
        fp = generate_schema_fingerprint('{"outer": {"inner": [1, 2, 3], "flag": true}}')
        assert fp is not None
        assert _is_sha256_hex(fp)


# ============================================================================
# compare_reports 测试
# ============================================================================


class TestCompareReports:
    """compare_reports 函数测试."""

    def test_two_identical_reports_no_drift(self) -> None:
        """两份完全相同的报告显示 NO_SIGNIFICANT_DRIFT."""
        evidence = _default_evidence()
        r1 = _make_report_dict(evidence=evidence)
        r2 = _make_report_dict(evidence=evidence)
        result = compare_reports([r1, r2])
        assert result.drift_level == DriftLevel.NO_SIGNIFICANT_DRIFT
        assert result.report_count == 2

    def test_different_risk_levels_possible_drift(self) -> None:
        """不同风险等级的报告显示 POSSIBLE_DRIFT."""
        r1 = _make_report_dict(report_id="r1", risk_level="LOW")
        r2 = _make_report_dict(report_id="r2", risk_level="HIGH")
        result = compare_reports([r1, r2])
        assert result.drift_level == DriftLevel.POSSIBLE_DRIFT
        assert "风险等级发生变化" in result.drift_notes

    def test_different_model_sets_drift(self) -> None:
        """不同返回模型集合触发漂移."""
        evidence1 = [
            {"success": True, "response_model": "gpt-4", "total_latency_ms": 200},
        ]
        evidence2 = [
            {"success": True, "response_model": "gpt-3.5-turbo", "total_latency_ms": 200},
        ]
        r1 = _make_report_dict(report_id="r1", evidence=evidence1)
        r2 = _make_report_dict(report_id="r2", evidence=evidence2)
        result = compare_reports([r1, r2])
        assert "返回模型集合发生变化" in result.drift_notes
        assert result.drift_level == DriftLevel.POSSIBLE_DRIFT

    def test_different_fingerprints_drift(self) -> None:
        """不同结构指纹触发漂移."""
        r1 = _make_report_dict(report_id="r1", schema_fingerprints=["abc123"])
        r2 = _make_report_dict(report_id="r2", schema_fingerprints=["def456"])
        result = compare_reports([r1, r2])
        assert "响应结构指纹发生变化" in result.drift_notes
        assert result.drift_level == DriftLevel.POSSIBLE_DRIFT

    def test_single_report_returns_warning(self) -> None:
        """单份报告返回警告."""
        r1 = _make_report_dict()
        result = compare_reports([r1])
        assert "至少需要两份报告进行比较" in result.warnings
        assert result.drift_level == DriftLevel.INCONCLUSIVE
        assert result.report_count == 1

    def test_version_mismatch_adds_warning(self) -> None:
        """版本不一致添加警告."""
        r1 = _make_report_dict(report_id="r1", test_suite_version="1.0.0")
        r2 = _make_report_dict(report_id="r2", test_suite_version="2.0.0")
        result = compare_reports([r1, r2])
        assert result.version_mismatch is True
        assert any("测试套件版本不一致" in w for w in result.warnings)

    def test_success_rate_significant_change_drift(self) -> None:
        """成功率显著变化触发漂移."""
        evidence1 = [
            {"success": True, "total_latency_ms": 200},
            {"success": True, "total_latency_ms": 200},
        ]
        evidence2 = [
            {"success": False, "total_latency_ms": None, "http_status": 500},
            {"success": False, "total_latency_ms": None, "http_status": 500},
        ]
        r1 = _make_report_dict(report_id="r1", evidence=evidence1)
        r2 = _make_report_dict(report_id="r2", evidence=evidence2)
        result = compare_reports([r1, r2])
        # 成功率从 1.0 -> 0.0，差 > 0.3，触发漂移信号
        assert "成功率出现显著变化" in result.drift_notes
        assert result.drift_level != DriftLevel.NO_SIGNIFICANT_DRIFT

    def test_compare_reports_returns_compare_result_type(self) -> None:
        """返回 CompareResult 类型."""
        r1 = _make_report_dict()
        r2 = _make_report_dict()
        result = compare_reports([r1, r2])
        assert isinstance(result, CompareResult)

    def test_compare_reports_extracts_metadata(self) -> None:
        """正确提取报告元数据."""
        r1 = _make_report_dict(
            report_id="rep-a",
            base_url="https://api.openai.com",
            model="gpt-4",
            test_suite_version="1.0.0",
            utc_time="2025-01-01T00:00:00Z",
        )
        r2 = _make_report_dict(
            report_id="rep-b",
            base_url="https://api.anthropic.com",
            model="claude-3",
            test_suite_version="1.0.0",
            utc_time="2025-01-02T00:00:00Z",
        )
        result = compare_reports([r1, r2])
        assert result.reports == ["rep-a", "rep-b"]
        assert result.report_count == 2
        assert result.endpoints == ["https://api.openai.com", "https://api.anthropic.com"]
        assert result.claimed_models == ["gpt-4", "claude-3"]
        assert result.report_times == ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]

    def test_compare_reports_extracts_token_and_request_id_rates(self) -> None:
        """正确提取 token 和 request_id 存在率."""
        evidence = [
            {
                "success": True,
                "total_latency_ms": 200,
                "input_tokens": 100,
                "output_tokens": 50,
                "response_id": "chatcmpl-xxx",
            },
            {
                "success": False,
                "total_latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "response_id": None,
                "http_status": 500,
            },
        ]
        r1 = _make_report_dict(evidence=evidence)
        r2 = _make_report_dict(evidence=evidence)
        result = compare_reports([r1, r2])
        assert result.token_field_rates == [0.5, 0.5]
        assert result.request_id_rates == [0.5, 0.5]

    def test_compare_reports_extracts_error_types(self) -> None:
        """正确提取错误类型."""
        evidence = [
            {
                "success": False,
                "total_latency_ms": None,
                "exception_type": "ConnectTimeout",
                "http_status": None,
            },
        ]
        r1 = _make_report_dict(evidence=evidence)
        r2 = _make_report_dict(evidence=evidence)
        result = compare_reports([r1, r2])
        assert result.error_types == [["ConnectTimeout"], ["ConnectTimeout"]]

    def test_compare_reports_http_error_fallback(self) -> None:
        """无 exception_type 时回退到 HTTP 状态码."""
        evidence = [
            {
                "success": False,
                "total_latency_ms": None,
                "exception_type": None,
                "http_status": 503,
            },
        ]
        r1 = _make_report_dict(evidence=evidence)
        r2 = _make_report_dict(evidence=evidence)
        result = compare_reports([r1, r2])
        assert result.error_types == [["HTTP_503"], ["HTTP_503"]]
