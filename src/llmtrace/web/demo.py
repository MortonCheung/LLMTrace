"""``--demo`` 模式的进程内 Mock upstream（§六十一）。

Demo 模式不需要任何 API Key：把模型的 OpenAI 兼容端点挂在本进程的
``/__mock__/v1`` 路径下，RunService 创建的 run 把 ``base_url`` 指向这里
即可完整跑通 create → progress → capability → report → history。

安全语义（§六十二）：Demo 永远只展示 raw capability，不冒充正式的
Reference Calibration；页面必须带醒目的 DEMO 标识。
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request

# 挂载路径前缀（与真实 upstream 的 /v1 对齐，避免 URL 拼装差异）。
MOCK_PREFIX = "/__mock__/v1"

# Demo 预设模型名；页面在 demo 模式下预填 model 与 base_url。
DEMO_MODEL = "demo-model"

# 固定的确定性应答：选择题可解析出 (A)，算术/文本任务得到 42，
# 代码任务会失败 —— 目的是完整走通流程而不是展示高能力分。
_DEMO_CONTENT = "The answer is (A). The answer is 42."


def _chat_payload(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-demo-001",
        "object": "chat.completion",
        "created": 1_677_652_288,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _DEMO_CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 32, "completion_tokens": 8, "total_tokens": 40},
    }


def build_demo_router(*, latency_ms: float = 30.0) -> APIRouter:
    """构造 OpenAI 兼容的 Mock upstream router。"""
    router = APIRouter(prefix=MOCK_PREFIX)

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)
        return {
            "object": "list",
            "data": [{"id": DEMO_MODEL, "object": "model", "created": 1_677_652_288}],
        }

    @router.post("/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)
        body = await request.json()
        model = (body or {}).get("model") or DEMO_MODEL
        return _chat_payload(model)

    return router
