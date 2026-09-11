"""Service 层错误 → HTTP 语义的集中映射（§五十八 / §五十九 / §六十）。

API 路由与页面路由中的 :class:`RunServiceError` 都会落到这里转成
用户可读 JSON：``{"error": {"code": ..., "message": ...}}``。

消息已在 service 内部做过 secret scrub，绝不会泄漏 API Key。
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from llmtrace.service import (
    RunCreateError,
    RunEstimateError,
    RunKeyUnavailableError,
    RunNotFoundError,
    RunPreflightError,
    RunServiceError,
    RunStateConflictError,
)

_ERROR_STATUS: dict[type[Exception], int] = {
    RunNotFoundError: status.HTTP_404_NOT_FOUND,
    RunStateConflictError: status.HTTP_409_CONFLICT,
    RunCreateError: status.HTTP_400_BAD_REQUEST,
    RunEstimateError: status.HTTP_400_BAD_REQUEST,
    RunKeyUnavailableError: status.HTTP_400_BAD_REQUEST,
    RunPreflightError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def service_error_status(exc: Exception) -> int:
    return _ERROR_STATUS.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)


def register_service_error_handler(app: FastAPI) -> None:
    """把 RunServiceError 家族统一渲染为 ``{"error": {...}}`` JSON。"""

    @app.exception_handler(RunServiceError)
    async def _handle_service_error(request: Request, exc: RunServiceError) -> JSONResponse:
        code = getattr(exc, "error_code", "INTERNAL_ERROR")
        return JSONResponse(
            status_code=service_error_status(exc),
            content={"error": {"code": code, "message": str(exc)}},
        )
