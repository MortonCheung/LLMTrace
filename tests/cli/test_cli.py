"""CLI 测试 - LLMTrace typer 命令."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from typer.testing import CliRunner

from llmtrace.cli import app

runner = CliRunner()

# 新版 Typer/Rich 的帮助输出可能包含 ANSI 控制符（尤其在 CI 环境），
# 断言前需剥离（等效 click.utils.strip_ansi；click 已不再是 typer 的依赖）。
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


# ---------------------------------------------------------------------------
# --help 测试
# ---------------------------------------------------------------------------


def test_main_help() -> None:
    """测试主应用 --help 输出."""
    result = runner.invoke(app, ["--help"])
    stdout = _strip_ansi(result.stdout)
    assert result.exit_code == 0
    assert "LLMTrace" in stdout
    assert "audit" in stdout
    assert "compare" in stdout
    assert "inspect" in stdout


def test_audit_help() -> None:
    """测试 audit 子命令 --help 输出."""
    result = runner.invoke(app, ["audit", "--help"])
    stdout = _strip_ansi(result.stdout)
    assert result.exit_code == 0
    assert "audit" in stdout.lower() or "审计" in stdout
    assert "--protocol" in stdout
    assert "--base-url" in stdout
    assert "--model" in stdout
    assert "--api-key-env" in stdout
    assert "--dry-run" in stdout


def test_compare_help() -> None:
    """测试 compare 子命令 --help 输出."""
    result = runner.invoke(app, ["compare", "--help"])
    stdout = _strip_ansi(result.stdout)
    assert result.exit_code == 0
    assert "compare" in stdout.lower() or "比较" in stdout
    assert "report_a" in stdout


def test_inspect_help() -> None:
    """测试 inspect 子命令 --help 输出."""
    result = runner.invoke(app, ["inspect", "--help"])
    stdout = _strip_ansi(result.stdout)
    assert result.exit_code == 0
    assert "inspect" in stdout.lower() or "查看" in stdout
    assert "report_path" in stdout


# ---------------------------------------------------------------------------
# audit --dry-run 测试
# ---------------------------------------------------------------------------


def test_audit_dry_run() -> None:
    """测试 audit --dry-run 输出执行计划且不发送请求."""
    result = runner.invoke(
        app,
        [
            "audit",
            "--protocol",
            "openai",
            "--base-url",
            "http://test.example.com",
            "--model",
            "test-model",
            "--api-key-env",
            "TEST_API_KEY",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry Run" in result.stdout
    assert "执行计划" in result.stdout
    assert "未发送任何请求" in result.stdout
    assert "openai" in result.stdout
    assert "http://test.example.com" in result.stdout
    assert "test-model" in result.stdout


# ---------------------------------------------------------------------------
# audit 错误场景测试
# ---------------------------------------------------------------------------


def test_audit_missing_env_var() -> None:
    """测试 audit 缺少必需环境变量时以错误退出."""
    # 确保环境变量不存在
    env_var_name = "LLMTRACE_TEST_NONEXISTENT_KEY_12345"
    if env_var_name in os.environ:
        del os.environ[env_var_name]

    result = runner.invoke(
        app,
        [
            "audit",
            "--protocol",
            "openai",
            "--base-url",
            "http://test.example.com",
            "--model",
            "test-model",
            "--api-key-env",
            env_var_name,
        ],
    )
    assert result.exit_code == 1
    assert env_var_name in result.stdout


def test_audit_invalid_protocol() -> None:
    """测试 audit 使用无效协议时报错."""
    result = runner.invoke(
        app,
        [
            "audit",
            "--protocol",
            "invalid_protocol_xyz",
            "--base-url",
            "http://test.example.com",
            "--model",
            "test-model",
            "--api-key-env",
            "TEST_API_KEY",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# inspect 测试
# ---------------------------------------------------------------------------


def test_inspect_nonexistent_file() -> None:
    """测试 inspect 指定不存在的文件时报错."""
    nonexistent = "/tmp/llmtrace_nonexistent_report_12345.json"
    result = runner.invoke(app, ["inspect", nonexistent])
    assert result.exit_code == 1
    assert "报告文件不存在" in result.stdout or "报告文件不存在" in result.stderr


def test_inspect_valid_report(tmp_path: Path) -> None:
    """测试 inspect 查看有效的 JSON 报告."""
    report_file = tmp_path / "test_report.json"
    report_data = {
        "report_id": "llmtrace_test_001",
        "meta": {
            "llmtrace_version": "0.1.0",
            "test_suite_version": "1.0",
            "utc_time": "2025-01-01T00:00:00Z",
            "python_version": "3.12.0",
            "platform": "macOS-14.0",
            "content_hash": "abcdef1234567890",
        },
        "config": {
            "base_url": "http://test.example.com",
            "protocol": "openai",
            "model": "test-model",
        },
        "risk_level": "low",
        "findings": [
            {"probe_name": "test_probe", "status": "pass", "severity": "info"},
        ],
        "evidence": [
            {
                "request_model": "test-model",
                "response_model": "test-model",
                "http_status": 200,
                "success": True,
            },
        ],
    }
    report_file.write_text(json.dumps(report_data), encoding="utf-8")

    result = runner.invoke(app, ["inspect", str(report_file)])
    assert result.exit_code == 0
    assert "llmtrace_test_001" in result.stdout
    assert "test-model" in result.stdout
    assert "openai" in result.stdout


# ---------------------------------------------------------------------------
# compare 测试
# ---------------------------------------------------------------------------


def test_compare_nonexistent_file() -> None:
    """测试 compare 指定不存在的文件时报错."""
    nonexistent = "/tmp/llmtrace_nonexistent_report_12345.json"
    result = runner.invoke(app, ["compare", nonexistent, "/tmp/llmtrace_another_nonexistent.json"])
    assert result.exit_code == 1
    assert "报告文件不存在" in result.stdout or "报告文件不存在" in result.stderr


def test_compare_two_reports(tmp_path: Path) -> None:
    """测试 compare 比较两份有效报告."""
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"

    base_report = {
        "report_id": "llmtrace_001",
        "meta": {
            "llmtrace_version": "0.1.0",
            "test_suite_version": "1.0",
            "utc_time": "2025-01-01T00:00:00Z",
            "python_version": "3.12.0",
            "platform": "macOS-14.0",
            "content_hash": "aaa",
        },
        "config": {
            "base_url": "http://test.example.com",
            "protocol": "openai",
            "model": "test-model",
        },
        "risk_level": "low",
        "findings": [
            {"probe_name": "test_probe", "status": "pass", "severity": "info", "rule_id": "LLMTRACE-001"},
        ],
        "evidence": [
            {
                "request_model": "test-model",
                "response_model": "test-model",
                "http_status": 200,
                "success": True,
                "response_body_summary": {"object": "chat.completion"},
                "input_tokens": 10,
                "output_tokens": 20,
                "response_id": "resp-001",
                "latency_ms": 100.0,
            },
        ],
        "schema_fingerprints": [],
        "model_list": ["test-model"],
        "model_list_available": True,
        "model_in_list": True,
    }

    report_a.write_text(json.dumps(base_report), encoding="utf-8")

    report_b_data = dict(base_report)
    report_b_data["report_id"] = "llmtrace_002"
    report_b.write_text(json.dumps(report_b_data), encoding="utf-8")

    result = runner.invoke(app, ["compare", str(report_a), str(report_b)])
    assert result.exit_code == 0
    assert "报告比较" in result.stdout


def test_compare_invalid_json(tmp_path: Path) -> None:
    """测试 compare 比较无效 JSON 文件时报错."""
    bad_file = tmp_path / "bad_report.json"
    bad_file.write_text("this is not json", encoding="utf-8")

    valid_file = tmp_path / "valid_report.json"
    valid_file.write_text(json.dumps({"report_id": "test"}), encoding="utf-8")

    result = runner.invoke(app, ["compare", str(valid_file), str(bad_file)])
    assert result.exit_code == 1
    assert "无法解析报告" in result.stdout or "无法解析报告" in result.stderr
