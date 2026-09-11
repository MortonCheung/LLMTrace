"""CliProgress 测试（CLI Real-Use Sprint §Live / §Cancel）.

覆盖：非 TTY 阶段日志、benchmark 进度行、完成/失败/取消终态行、
取消后汇总（已完成项数 + 请求数）、内部状态机（stage/completed/total/requests）。
"""

from __future__ import annotations

import re

from llmtrace.execution.progress import ProgressEvent
from llmtrace.reporting.console import CliProgress

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


class TestNonTtyLog:
    def test_stage_and_progress_lines(self, capsys) -> None:
        """非 TTY：阶段行 + benchmark 进度行按事件打印."""
        progress = CliProgress(tty=False)
        progress.start()
        progress.on_event(ProgressEvent(type="stage", stage="protocol", message="protocol audit started"))
        progress.on_event(
            ProgressEvent(type="stage", stage="benchmark", message="quick suite benchmark started", total=32)
        )
        progress.on_event(ProgressEvent(type="progress", stage="benchmark", completed=3, total=32, requests=9))
        progress.on_event(ProgressEvent(type="done", stage="completed", message="run completed", requests=95))
        progress.stop()
        out = _strip_ansi(capsys.readouterr().out)
        assert "开始执行" in out
        assert "协议审计" in out
        assert "benchmark 进度 3/32" in out
        assert "请求 9" in out
        assert "执行完成" in out
        # 未取消 → 无取消汇总行
        assert "已取消" not in out

    def test_failure_line(self, capsys) -> None:
        """非 TTY：失败事件打印失败终态行."""
        progress = CliProgress(tty=False)
        progress.start()
        progress.on_event(ProgressEvent(type="failed", stage="failed", message="run failed"))
        progress.stop()
        out = _strip_ansi(capsys.readouterr().out)
        assert "执行失败" in out

    def test_state_transition(self) -> None:
        """内部状态机随事件更新（不依赖渲染分支）."""
        progress = CliProgress(tty=False)
        progress.on_event(ProgressEvent(type="stage", stage="benchmark", message="started", completed=0, total=8))
        assert progress.stage == "benchmark"
        assert progress.total == 8
        progress.on_event(ProgressEvent(type="progress", stage="benchmark", completed=5, total=8, requests=11))
        assert progress.completed == 5
        assert progress.requests == 11


class TestCancel:
    def test_cancel_line_reports_completed_and_requests(self, capsys) -> None:
        """取消后汇总行包含已完成项数与请求数."""
        progress = CliProgress(tty=False)
        progress.start()
        progress.on_event(ProgressEvent(type="stage", stage="benchmark", total=32))
        progress.on_event(ProgressEvent(type="progress", stage="benchmark", completed=17, total=32, requests=44))
        progress.announce_cancel()
        assert progress.cancelled is True
        progress.stop()
        out = _strip_ansi(capsys.readouterr().out)
        assert "已取消" in out
        assert "17/32" in out
        assert "44 个请求" in out

    def test_cancel_before_any_item(self, capsys) -> None:
        """无基准项时取消汇总显示 /?."""
        progress = CliProgress(tty=False)
        progress.start()
        progress.announce_cancel()
        progress.stop()
        out = _strip_ansi(capsys.readouterr().out)
        assert "已取消" in out

    def test_description_maps_stage_labels(self) -> None:
        """阶段 token → 中文标签映射."""
        progress = CliProgress(tty=False)
        progress.on_event(ProgressEvent(type="stage", stage="calibration", message="reference calibration"))
        assert "参考校准" in progress._description()
