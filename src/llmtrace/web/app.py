"""FastAPI 应用组装（local-first，v0.5 Usable MVP）。

- 持有全局唯一 :class:`RunService`（app.state.service）与数据布局
  （app.state.layout）；
- 挂载 §十七 API 路由 / §十八 SSE、页面模板与静态资源；
- ``demo=True`` 时额外挂载进程内 OpenAI 兼容 Mock upstream（§六十一），
  页面预填 ``/__mock__/v1`` 端点。

不引入账号 / 部署平台 / 动态网络更新（§五十三 – §五十五）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llmtrace.adapters.code_execution import CodeExecutionBackend
from llmtrace.appdir import AppLayout, ensure_app_layout
from llmtrace.service.runs import RunService
from llmtrace.web.api import router as api_router
from llmtrace.web.demo import DEMO_MODEL, MOCK_PREFIX, build_demo_router
from llmtrace.web.errors import register_service_error_handler

_WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = _WEB_DIR / "templates"
STATIC_DIR = _WEB_DIR / "static"

_TERMINAL_STATUSES = {"COMPLETED", "COMPLETED_WITH_WARNINGS", "PARTIAL", "FAILED", "CANCELLED"}


def create_app(
    layout: AppLayout | None = None,
    *,
    code_backend: CodeExecutionBackend | None = None,
    demo: bool = False,
) -> FastAPI:
    """构造 Web 应用；service 存活于 app.state，关闭随 lifespan 执行。"""
    app_layout = layout or ensure_app_layout()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        app.state.service.close()

    app = FastAPI(title="LLMTrace Web", version="0.5.0", lifespan=lifespan)
    app.state.layout = app_layout
    # Demo 模式自带 in-process sandbox 注入（进程内 Mock 演示专用）。
    if demo and code_backend is None:
        from llmtrace.adapters.code_execution import create_code_execution_backend

        code_backend = create_code_execution_backend(allow_unsafe_in_process=True)
    # None => runner 使用生产默认（Docker fail-closed）；--demo 注入进程内后端。
    app.state.demo = demo
    app.state.service = RunService(app_layout, code_backend=code_backend)
    register_service_error_handler(app)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.include_router(api_router)
    if demo:
        app.include_router(build_demo_router())

    # ------------------------------------------------------------------ pages

    def _demo_meta(request: Request) -> dict[str, str] | None:
        if not demo:
            return None
        origin = str(request.base_url).rstrip("/")
        return {
            "base_url": f"{origin}{MOCK_PREFIX}",
            "model": DEMO_MODEL,
            "api_key": "demo-key",
        }

    def _page_context(request: Request, **extra: Any) -> dict[str, Any]:
        context: dict[str, Any] = {"request": request, "demo": _demo_meta(request)}
        context.update(extra)
        return context

    @app.get("/")
    async def index_page(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", _page_context(request))

    @app.get("/history")
    async def history_page(request: Request, limit: int = 100) -> Any:
        runs = app.state.service.history(limit=limit)
        return templates.TemplateResponse(request, "history.html", _page_context(request, runs=runs))

    @app.get("/run/{run_id}")
    async def run_page(request: Request, run_id: str) -> Any:
        run = app.state.service.view(run_id)  # 不存在 -> RunNotFoundError -> 404
        terminal = run["status"] in _TERMINAL_STATUSES
        return templates.TemplateResponse(
            request,
            "run.html",
            _page_context(request, run=run, run_id=run_id, terminal=terminal),
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # 静态资源在页面路由之后挂载，避免覆盖 API。
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
