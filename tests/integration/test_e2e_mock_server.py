"""端到端集成测试：使用 mock server 验证三种模式.

核心验证：
- dry-run 计划次数 == Mock Server 实际收到请求数 == 报告证据数
- 证据类型精确分类（baseline 与 streaming_comparison 分离）
- honest -> LOW / fallback -> HIGH / inconsistent -> HIGH 及其直接原因
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
MOCK_SERVER_SCRIPT = EXAMPLES_DIR / "mock_proxy_server.py"

# 直接访问本地 mock server，绕过系统代理
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

EXPECTED_EVIDENCE_COUNTS = {
    "baseline": 3,
    "streaming_comparison": 1,
    "streaming_baseline": 1,
    "connectivity": 1,
    "model_catalog": 1,
    "invalid_model": 1,
}

# 各模式的风险等级、触发该风险的探针 rule_id 与必须出现的推断文本
MODE_EXPECTATIONS = {
    "honest": {
        "risk": "LOW",
        "high_rule": None,
        "high_inference": None,
    },
    "fallback": {
        "risk": "HIGH",
        "high_rule": "LLMTRACE-INV-001",
        "high_inference": "无效模型名称仍成功生成内容",
    },
    "inconsistent": {
        "risk": "HIGH",
        "high_rule": "LLMTRACE-BASE-001",
        "high_inference": "同一会话中返回了多个不同的模型标识",
    },
}


def _get_free_port() -> int:
    """通过 socket 绑定端口 0 获取空闲端口."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_mock_server(mode: str, port: int) -> subprocess.Popen[str]:
    """以子进程启动 mock server（独立进程，启动时自动清空请求日志）."""
    env = os.environ.copy()
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"
    return subprocess.Popen(
        [sys.executable, str(MOCK_SERVER_SCRIPT), "--mode", mode, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )


def _stop_mock_server(proc: subprocess.Popen[str]) -> None:
    """终止 mock server 子进程并回收端口."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_for_server(port: int, timeout: float = 30.0) -> None:
    """轮询调试端点直到服务可访问，替代固定 sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with _NO_PROXY_OPENER.open(f"http://127.0.0.1:{port}/debug/requests", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"mock server on port {port} did not become ready within {timeout}s")


def _fetch_debug_requests(port: int) -> dict[str, Any]:
    """读取 Mock Server 实际收到的请求日志（不计入审计请求数）."""
    with _NO_PROXY_OPENER.open(f"http://127.0.0.1:{port}/debug/requests", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_cli(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """运行 llmtrace CLI（使用当前 Python 解释器）."""
    env = os.environ.copy()
    env["TEST_API_KEY"] = "sk-test-mock-key"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"
    return subprocess.run(
        [sys.executable, "-m", "llmtrace.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


class TestE2EMockServer:
    """端到端集成测试."""

    @pytest.mark.parametrize("mode", ["honest", "fallback", "inconsistent"])
    def test_e2e_mode(self, mode: str, tmp_path: Path) -> None:
        """测试三种 mock server 模式：风险等级、请求计数、证据分类、证据引用."""
        expected = MODE_EXPECTATIONS[mode]
        port = _get_free_port()

        # 启动 mock server
        proc = _start_mock_server(mode, port)

        try:
            _wait_for_server(port)

            output_dir = str(tmp_path / "reports")
            result = _run_cli(
                [
                    "audit",
                    "--protocol",
                    "openai",
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--model",
                    "mock-model-v1",
                    "--api-key-env",
                    "TEST_API_KEY",
                    "--repeat",
                    "3",
                    "-y",
                    "-o",
                    output_dir,
                ]
            )
            assert result.returncode == 0, f"audit failed: {result.stderr}"

            # 1. 风险等级
            assert f"风险等级           │ {expected['risk']}" in result.stdout, (
                f"Expected risk {expected['risk']} in output:\n{result.stdout}"
            )

            # 2. 读取 Mock Server 实际请求数（/debug/requests 本身不计入）
            debug_result = _fetch_debug_requests(port)
            assert debug_result["count"] == 8, f"Mock server received {debug_result['count']} requests, expected 8"

            # 3. JSON 报告
            report_files = list(Path(output_dir).glob("*.json"))
            assert len(report_files) == 1, f"Expected 1 JSON report, got {len(report_files)}"
            with open(report_files[0]) as f:
                report: dict[str, Any] = json.load(f)

            assert report["risk_level"] == expected["risk"]
            assert len(report["evidence"]) == 8, f"Expected 8 evidence items, got {len(report['evidence'])}"

            # 4. 无 API Key 泄露
            assert "sk-test-mock-key" not in json.dumps(report), "API key leaked in report"

            # 5. 证据类型精确分类
            evidence_types = [e.get("evidence_type") for e in report["evidence"]]
            for etype, expected_count in EXPECTED_EVIDENCE_COUNTS.items():
                actual = evidence_types.count(etype)
                assert actual == expected_count, (
                    f"evidence_type '{etype}' count: expected {expected_count}, got {actual}; all={evidence_types}"
                )

            # 6. evidence_refs 有效性
            findings = report.get("findings", [])
            evidence = report.get("evidence", [])
            evidence_ids = {e.get("evidence_id") for e in evidence}
            assert len(evidence_ids) == len(evidence), "duplicate evidence_id in report"
            for finding in findings:
                for ref in finding.get("evidence_refs", []):
                    assert ref in evidence_ids, (
                        f"evidence_ref '{ref}' not found in evidence (probe: {finding.get('probe_name')})"
                    )

            # 7. 风险直接原因：检查对应探针的 rule_id 与 inference
            if expected["high_rule"]:
                high_findings = [
                    f for f in findings if f.get("rule_id") == expected["high_rule"] and f.get("status") == "fail"
                ]
                assert high_findings, (
                    f"mode={mode}: no fail finding with rule_id {expected['high_rule']}; findings={findings}"
                )
                assert any(
                    expected["high_inference"] in inf for f in high_findings for inf in f.get("inferences", [])
                ), (
                    f"mode={mode}: finding {expected['high_rule']} lacks inference "
                    f"'{expected['high_inference']}': {high_findings}"
                )
            else:
                assert not any(f.get("status") == "fail" and f.get("severity") == "high" for f in findings), (
                    f"mode={mode}: expected no high finding, got {findings}"
                )

            # 8. request_model 与实际传入模型一致
            for ev in evidence:
                etype = ev.get("evidence_type")
                req_model = ev.get("request_model")
                if etype == "invalid_model":
                    assert req_model != "mock-model-v1", (
                        f"invalid_model evidence should NOT have config model as request_model: {req_model}"
                    )
                    assert req_model and req_model.startswith("llmtrace-invalid-"), (
                        f"invalid_model evidence should have random invalid model: {req_model}"
                    )
                elif etype in ("baseline", "connectivity", "streaming_baseline", "streaming_comparison"):
                    assert req_model == "mock-model-v1", (
                        f"{etype} evidence request_model should be config model: {req_model}"
                    )

            # 9. 流式证据有时间字段
            streaming_ev = [e for e in evidence if e.get("evidence_type") == "streaming_baseline"]
            if streaming_ev:
                sev = streaming_ev[0]
                assert sev.get("total_latency_ms") is not None, "Streaming evidence missing total_latency_ms"
                assert sev.get("response_time") is not None, "Streaming evidence missing response_time"

            # 10. 报告必须生成 HTML
            html_files = list(Path(output_dir).glob("*.html"))
            assert len(html_files) == 1, f"Expected 1 HTML report, got {len(html_files)}"

        finally:
            _stop_mock_server(proc)

    def test_dry_run_matches_actual(self, tmp_path: Path) -> None:
        """验证 dry-run 计划次数与 Mock Server 实际收到请求数一致."""
        port = _get_free_port()
        proc = _start_mock_server("honest", port)

        try:
            _wait_for_server(port)

            # dry-run
            result = _run_cli(
                [
                    "audit",
                    "--protocol",
                    "openai",
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--model",
                    "mock-model-v1",
                    "--api-key-env",
                    "TEST_API_KEY",
                    "--dry-run",
                ],
                timeout=10,
            )
            assert result.returncode == 0
            assert "预计调用次数   │ 8" in result.stdout, f"Expected 8 in dry-run:\n{result.stdout}"
            dry_run_count = 8

            # 真实执行
            output_dir = str(tmp_path / "reports")
            result = _run_cli(
                [
                    "audit",
                    "--protocol",
                    "openai",
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--model",
                    "mock-model-v1",
                    "--api-key-env",
                    "TEST_API_KEY",
                    "--repeat",
                    "3",
                    "-y",
                    "-o",
                    output_dir,
                ]
            )
            assert result.returncode == 0, f"audit failed: {result.stderr}"

            debug_result = _fetch_debug_requests(port)
            assert debug_result["count"] == dry_run_count, f"dry-run={dry_run_count} != actual={debug_result['count']}"

            report_files = list(Path(output_dir).glob("*.json"))
            assert len(report_files) == 1
            with open(report_files[0]) as f:
                report: dict[str, Any] = json.load(f)
            assert len(report["evidence"]) == dry_run_count, (
                f"dry-run={dry_run_count} != evidence={len(report['evidence'])}"
            )

            # 三个值必须一致
            assert dry_run_count == debug_result["count"] == len(report["evidence"]) == 8
        finally:
            _stop_mock_server(proc)
