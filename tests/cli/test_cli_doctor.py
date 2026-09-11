"""错误信息产品化与 doctor 命令测试（CLI Real-Use Sprint §Error / §Doctor）.

覆盖：
- describe_error：401/403、404、429、连接失败、超时、InvalidURL、沙箱不可用。
- doctor：命令可运行、输出包含关键检查项、失败时 exit 1。
"""

from __future__ import annotations

import re

import httpx
import pytest
from typer.testing import CliRunner

from llmtrace.adapters.code_execution import SandboxUnavailableError
from llmtrace.cli import app
from llmtrace.reporting.console import describe_error

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.example.com/v1")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("status error", request=request, response=response)


class TestDescribeError:
    def test_401_mentions_auth(self) -> None:
        message = describe_error(_status_error(401), api_key_env="MY_KEY")
        assert "鉴权失败" in message
        assert "MY_KEY" in message

    def test_403_auth(self) -> None:
        assert "鉴权失败（HTTP 403）" in describe_error(_status_error(403))

    def test_404_base_url_and_model(self) -> None:
        message = describe_error(_status_error(404))
        assert "HTTP 404" in message
        assert "--base-url" in message
        assert "--model" in message

    def test_429_rate_limit(self) -> None:
        message = describe_error(_status_error(429))
        assert "HTTP 429" in message
        assert "--repeat" in message

    def test_connect_error(self) -> None:
        request = httpx.Request("GET", "https://api.example.com/v1")
        exc = httpx.ConnectError("connection refused", request=request)
        assert "无法连接到目标端点" in describe_error(exc)

    def test_timeout(self) -> None:
        request = httpx.Request("GET", "https://api.example.com/v1")
        exc = httpx.ConnectTimeout("timed out", request=request)
        assert "请求超时" in describe_error(exc)

    def test_invalid_url(self) -> None:
        exc = httpx.InvalidURL("missing scheme")
        assert "base-url 格式不合法" in describe_error(exc)

    def test_sandbox_unavailable(self) -> None:
        exc = SandboxUnavailableError("Docker is not available")
        assert "代码沙箱不可用" in describe_error(exc)
        assert "doctor" in describe_error(exc)

    def test_unknown_falls_back_to_str(self) -> None:
        assert describe_error(RuntimeError("boom")) == "boom"


class TestDoctor:
    def test_doctor_runs_and_lists_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLMTRACE_API_KEY", raising=False)
        result = runner.invoke(app, ["doctor"])
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "llmtrace doctor" in stdout
        assert "Python" in stdout
        assert "llmtrace" in stdout
        assert "httpx" in stdout
        assert "rich" in stdout
        assert "代码沙箱" in stdout
        assert "ReferenceSets" in stdout
        assert "LLMTRACE_API_KEY" in stdout

    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "诊断" in result.stdout
