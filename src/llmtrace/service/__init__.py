"""薄 Service Layer（v0.5 Usable MVP，§十三）。

只封装 :class:`~llmtrace.execution.runner.UnifiedAuditRunner` 的生命周期，
不重新实现 audit。目录按真实结构最小化：``models.py`` 放输入/估算数据模型，
``runs.py`` 放 :class:`~llmtrace.service.runs.RunService`。
"""

from llmtrace.service.models import CreateRunInput, RunEstimate
from llmtrace.service.runs import (
    RunCreateError,
    RunEstimateError,
    RunKeyUnavailableError,
    RunNotFoundError,
    RunPreflightError,
    RunService,
    RunServiceError,
    RunStateConflictError,
)

__all__ = [
    "CreateRunInput",
    "RunCreateError",
    "RunEstimate",
    "RunEstimateError",
    "RunKeyUnavailableError",
    "RunNotFoundError",
    "RunPreflightError",
    "RunService",
    "RunServiceError",
    "RunStateConflictError",
]
