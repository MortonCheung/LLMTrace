"""端到端集成测试：使用 mock server 验证三种模式."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import pytest


def _run_mock_server(mode: str, port: int) -> None:
    """在子进程中运行 mock server."""
    import sys
    from http.server import HTTPServer

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))

    import mock_proxy_server as mps

    mps.MODE = mode
    server = HTTPServer(("127.0.0.1", port), mps.MockProxyHandler)
    server.serve_forever()


def _run_audit(base_url: str, output_dir: str) -> Any:
    """运行 audit 命令."""
    import subprocess

    env = os.environ.copy()
    env["TEST_API_KEY"] = "sk-test-mock-key"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"

    result = subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "llmtrace.cli",
            "audit",
            "--protocol",
            "openai",
            "--base-url",
            base_url,
            "--model",
            "mock-model-v1",
            "--api-key-env",
            "TEST_API_KEY",
            "--repeat",
            "3",
            "-y",
            "-o",
            output_dir,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return result


class TestE2EMockServer:
    """端到端集成测试."""

    @pytest.mark.parametrize(
        "mode,expected_risk",
        [("honest", "LOW"), ("fallback", "HIGH"), ("inconsistent", "HIGH")],
    )
    def test_e2e_mode(self, mode: str, expected_risk: str, tmp_path: Path) -> None:
        """测试三种 mock server 模式."""
        port = 8120 + hash(mode) % 100

        # 启动 mock server
        proc = multiprocessing.Process(target=_run_mock_server, args=(mode, port))
        proc.daemon = True
        proc.start()
        time.sleep(2)

        try:
            output_dir = str(tmp_path / "reports")
            result = _run_audit(f"http://localhost:{port}", output_dir)

            assert result.returncode == 0, f"audit failed: {result.stderr}"

            # 验证风险等级
            assert f"风险等级           │ {expected_risk}" in result.stdout, (
                f"Expected risk {expected_risk} in output:\n{result.stdout}"
            )

            # 验证 JSON 报告
            report_files = list(Path(output_dir).glob("*.json"))
            assert len(report_files) == 1, f"Expected 1 JSON report, got {len(report_files)}"

            with open(report_files[0]) as f:
                report: dict[str, Any] = json.load(f)

            # 验证风险等级
            assert report["risk_level"] == expected_risk, f"Expected risk {expected_risk}, got {report['risk_level']}"

            # 验证无 API Key 泄露
            report_str = json.dumps(report)
            assert "sk-test-mock-key" not in report_str, "API key leaked in report"

            # 验证 evidence_refs 有效性
            findings = report.get("findings", [])
            evidence = report.get("evidence", [])
            evidence_ids = {e.get("evidence_id") for e in evidence}
            for finding in findings:
                for ref in finding.get("evidence_refs", []):
                    assert ref in evidence_ids, (
                        f"evidence_ref '{ref}' not found in evidence (probe: {finding.get('probe_name')})"
                    )

            # 验证证据数量与 dry-run 一致
            # dry-run 预计 8 次请求（1 conn + 1 models + 3 baseline + 1 invalid + 2 streaming）
            assert len(evidence) == 8, f"Expected 8 evidence items, got {len(evidence)}"

            # 验证证据类型
            evidence_types = [e.get("evidence_type") for e in evidence]
            assert "connectivity" in evidence_types
            assert "model_catalog" in evidence_types
            assert evidence_types.count("baseline") == 4  # 3 baseline + 1 streaming non-stream
            assert "invalid_model" in evidence_types
            assert "streaming_baseline" in evidence_types

            # 验证 request_model 与实际传入模型一致
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
                elif etype in ("baseline", "connectivity", "streaming_baseline"):
                    assert req_model == "mock-model-v1", (
                        f"{etype} evidence request_model should be config model: {req_model}"
                    )

            # 验证流式证据有 response_time 和 total_latency_ms
            streaming_ev = [e for e in evidence if e.get("evidence_type") == "streaming_baseline"]
            if streaming_ev:
                sev = streaming_ev[0]
                assert sev.get("total_latency_ms") is not None, "Streaming evidence missing total_latency_ms"
                assert sev.get("response_time") is not None, "Streaming evidence missing response_time"

        finally:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()

    def test_dry_run_matches_actual(self) -> None:
        """验证 dry-run 计划次数与实际请求次数一致."""
        import subprocess

        env = os.environ.copy()
        env["NO_PROXY"] = "localhost,127.0.0.1"
        env["no_proxy"] = "localhost,127.0.0.1"

        # dry-run
        result = subprocess.run(
            [
                ".venv/bin/python",
                "-m",
                "llmtrace.cli",
                "audit",
                "--protocol",
                "openai",
                "--base-url",
                "http://localhost:9999",
                "--model",
                "mock-model-v1",
                "--api-key-env",
                "TEST_API_KEY",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0
        assert "预计调用次数   │ 8" in result.stdout, f"Expected 8 in dry-run:\n{result.stdout}"
