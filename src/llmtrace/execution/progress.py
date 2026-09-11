"""执行进度与协作取消 —— UnifiedAuditRunner 的最小增强接口。

- ``ProgressEvent``：由 runner 在既有阶段边界与 benchmark item 边界发送；
  语义只做"通报"，不参与执行流（执行流本身由 runner 决定）。
- ``CancellationToken``：cooperative cancellation。Web 用户点击 Cancel 后
  置位；runner 在阶段切换与每个 benchmark item 前检查，抛
  ``RunCancelledError``。已发出的单个 HTTP 请求不要求瞬间强杀（v0.5 边界）。

CLI 默认不传 sink / token，行为与以前完全一致。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# 阶段 token（供 UI 映射与 SSE stage 字段使用）
STAGE_PREFLIGHT = "preflight"
STAGE_PROTOCOL = "protocol"
STAGE_BENCHMARK = "benchmark"
STAGE_SCORING = "scoring"
STAGE_CALIBRATION = "calibration"
STAGE_COMPARISON = "comparison"
STAGE_REPORTING = "reporting"
STAGE_DONE = "completed"
STAGE_FAILED = "failed"
STAGE_CANCELLED = "cancelled"

# 事件类型
EVENT_STAGE = "stage"
EVENT_PROGRESS = "progress"
EVENT_DONE = "done"
EVENT_FAILED = "failed"
EVENT_CANCELLED = "cancelled"


class RunCancelledError(Exception):
    """用户主动取消；区别于失败——状态映射为 CANCELLED。"""

    error_code = "RUN_CANCELLED"


class CancellationToken:
    """线程安全的一次性取消标志（幂等 cancel）。"""

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelledError("run cancelled by user")


@dataclass(frozen=True)
class ProgressEvent:
    """一次进度通报；序列化到 SSE 的 JSON 结构。"""

    type: str
    stage: str
    message: str | None = None
    completed: int | None = None
    total: int | None = None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # 运行结束 / 失败 / 取消时附加终态（status / error）
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "stage": self.stage,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.completed is not None:
            payload["completed"] = self.completed
        if self.total is not None:
            payload["total"] = self.total
        if self.extra:
            payload.update(self.extra)
        return payload


# 同步回调：runner 运行在 async 线程，sink 只做轻量推送，不做 IO。
ProgressSink = Callable[[ProgressEvent], None]
