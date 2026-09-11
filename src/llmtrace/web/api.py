"""§十七 Web API + §十八 SSE 实时进度。

薄映射层：路由只负责 HTTP 语义（body 解析 / 状态码 / SSE 传输），
一切运行逻辑委托给 :class:`llmtrace.service.runs.RunService`。

Service 层错误不在这里 try/except 包装，而是自然抛给 app 级 handler
（见 :mod:`llmtrace.web.errors`），统一映射为用户可读消息（§五十八）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from llmtrace.service.runs import RunService


class _CreateRunBody(BaseModel):
    base_url: str = Field(min_length=1, description="模型端点 base URL（可含凭据，仅进程内存使用）")
    model: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    protocol: str = "openai"
    auth_style: str = "auto"
    repeat: int = Field(default=3, ge=1, le=20)
    timeout: float = Field(default=30.0, gt=0, le=600)
    check_streaming: bool = True
    reference_set_path: str | None = None


def _get_service(request: Request) -> RunService:
    return cast(RunService, request.app.state.service)


def _require(request: Request, run_id: str) -> None:
    """Streaming 之前先确认 run 存在，让 404 及时返回而不是进到 SSE。"""
    _get_service(request).view(run_id)


def _file_mtime_iso(path: Any) -> str:
    try:
        from datetime import UTC, datetime

        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return ""


def _list_reference_sets(request: Request) -> list[dict[str, Any]]:
    """列出 data root references/sets 下的可用 ReferenceSet（供表单选择）。"""
    sets_dir = request.app.state.layout.reference_sets_dir
    items: list[dict[str, Any]] = []
    if not sets_dir.is_dir():
        return items
    for path in sorted(sets_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        root = data.get("reference_set", data) if isinstance(data, dict) else {}
        members = root.get("members")
        items.append(
            {
                "name": root.get("reference_set_id", path.stem),
                "reference_set_id": root.get("reference_set_id", path.stem),
                "reference_set_version": root.get("reference_set_version"),
                "members": len(members) if isinstance(members, list) else None,
                "path": str(path),
                "modified_at": _file_mtime_iso(path),
            }
        )
    return items


router = APIRouter(prefix="/api", tags=["runs"])


@router.post("/runs", status_code=201)
async def create_run(body: _CreateRunBody, request: Request) -> dict[str, Any]:
    """POST /api/runs —— 校验输入并建立 run 索引（绝不把 key 落盘）。"""
    from llmtrace.service import CreateRunInput

    record = _get_service(request).create(
        CreateRunInput(
            base_url=body.base_url,
            model=body.model,
            api_key=body.api_key,
            protocol=body.protocol,
            auth_style=body.auth_style,
            repeat=body.repeat,
            timeout=body.timeout,
            check_streaming=body.check_streaming,
            reference_set_path=body.reference_set_path,
        )
    )
    return {"run": _get_service(request).view(record.run_id)}


@router.post("/runs/{run_id}/estimate")
async def estimate_run(run_id: str, request: Request) -> dict[str, Any]:
    """POST /api/runs/{id}/estimate —— 脱机估算，不发送任何 HTTP。"""
    estimate = _get_service(request).estimate(run_id)
    return {"estimate": jsonable_encoder(estimate)}


@router.post("/runs/{run_id}/start")
async def start_run(run_id: str, request: Request) -> dict[str, Any]:
    """POST /api/runs/{id}/start —— 后台 asyncio 任务启动完整 audit。"""
    _get_service(request).start(run_id)
    return {"run_id": run_id, "status": "RUNNING"}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    """GET /api/runs/{id} —— 最新索引状态（无 report 内容）。"""
    return {"run": _get_service(request).view(run_id)}


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> Any:
    """GET /api/runs/{id}/events —— Server-Sent Events 实时进度。

    先重放既有事件缓冲，再以 ~250ms 轮询等待新事件；终态事件后关闭。
    """
    _require(request, run_id)
    service = _get_service(request)

    async def _stream() -> AsyncIterator[str]:
        sent = 0
        while True:
            snap = service.progress(run_id)
            events = snap["events"]
            for event in events[sent:]:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent = len(events)
            if snap["terminal"]:
                yield 'event: end\ndata: {"done": true}\n\n'
                return
            await asyncio.sleep(0.25)

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
    """POST /api/runs/{id}/cancel —— cooperative cancellation（幂等）。"""
    return _get_service(request).cancel(run_id)


@router.get("/runs")
async def list_runs(request: Request, limit: int = 50) -> dict[str, Any]:
    """GET /api/runs —— 历史 run 列表（创建时间倒序）。"""
    return {"runs": _get_service(request).history(limit=max(1, min(limit, 500)))}


@router.get("/runs/{run_id}/result")
async def run_result(run_id: str, request: Request) -> dict[str, Any]:
    """GET /api/runs/{id}/result —— run 摘要 + report.json。"""
    return _get_service(request).result(run_id)


@router.get("/reference-sets")
async def reference_sets(request: Request) -> dict[str, Any]:
    """GET /api/reference-sets —— data root 下可选的 ReferenceSet 清单。"""
    return {"reference_sets": _list_reference_sets(request)}
