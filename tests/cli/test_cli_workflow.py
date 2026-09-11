"""CLI 真实使用工作流测试（CLI Real-Use Sprint §Workflow）.

覆盖 run 命令的主要用户路径：
- 无参考集 dry-run → Reference Calibration UNAVAILABLE + 说明
- 显式 --reference-set dry-run → 校准信息可见（含 ReferenceSet ID / Calibration Policy）
- 非交互模式缺少 --yes → 退出 1 并提示
- --yes 但 API Key 环境变量缺失 → 退出 1 并提示
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from llmtrace.cli import app
from tests.execution.test_runner_calibration import _build_reference_fixture

runner = CliRunner()

# 新版 Typer/Rich 的帮助输出可能包含 ANSI 控制符（尤其在 CI 环境），
# 断言前需剥离（等效 click.utils.strip_ansi；click 已不再是 typer 的依赖）。
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


def _run_args(*, output_dir: Path, extra: list[str] | None = None) -> list[str]:
    """公共 run 参数（dry-run 场景共用）。"""
    args = [
        "run",
        "--protocol",
        "openai",
        "--base-url",
        "http://test.example.com",
        "--model",
        "test-model",
        "--api-key-env",
        "LLMTRACE_WORKFLOW_KEY",
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    if extra:
        args.extend(extra)
    return args


def _non_interactive_args(output_dir: Path) -> list[str]:
    return [
        "run",
        "--base-url",
        "http://localhost:9999/v1",
        "--model",
        "test-model",
        "--api-key-env",
        "LLMTRACE_WORKFLOW_KEY",
        "--output-dir",
        str(output_dir),
    ]


class TestDryRunCalibration:
    def test_without_reference_set_shows_unavailable(self, tmp_path: Path) -> None:
        """无可信 ReferenceSet → dry-run 明确标注校准不可用及原因."""
        result = runner.invoke(app, _run_args(output_dir=tmp_path))
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "Dry Run" in stdout
        assert "Reference Calibration" in stdout
        assert "UNAVAILABLE" in stdout
        assert "未找到兼容的可信 ReferenceSet" in stdout

    def test_with_explicit_reference_set_shows_calibration(self, tmp_path: Path) -> None:
        """显式 --reference-set → dry-run 展示 ReferenceSet ID / 版本 / 校准策略."""
        set_path = _build_reference_fixture(
            tmp_path,
            tmp_path / "references",
            models=(("ref-model-low", 0.30), ("ref-model-mid", 0.60)),
            set_id="wf-set-v1",
            set_version="1.0.0",
        )
        result = runner.invoke(
            app,
            _run_args(output_dir=tmp_path, extra=["--reference-set", str(set_path)]),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "Reference Calibration" in stdout
        assert "是 (auto)" in stdout or "是" in stdout
        assert "wf-set-v1" in stdout
        assert "1.0.0" in stdout
        assert "Calibration Policy" in stdout


class TestRunGates:
    def test_non_interactive_without_yes_exits_1(self, monkeypatch, tmp_path: Path) -> None:
        """非 TTY 且未传 --yes → 拒绝执行，提示加 --yes."""
        # 先通过 API Key 检查，才能到达确认门。
        monkeypatch.setenv("LLMTRACE_WORKFLOW_KEY", "sk-workflow-test")
        result = runner.invoke(app, _non_interactive_args(tmp_path))
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1, stdout
        assert "需要 --yes" in stdout

    def test_yes_with_missing_api_key_exits_1(self, monkeypatch, tmp_path: Path) -> None:
        """--yes 下 API Key 环境变量缺失 → 明确提示变量名."""
        monkeypatch.delenv("LLMTRACE_WORKFLOW_KEY", raising=False)
        result = runner.invoke(app, _non_interactive_args(tmp_path) + ["--yes"])
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1, stdout
        assert "LLMTRACE_WORKFLOW_KEY" in stdout
