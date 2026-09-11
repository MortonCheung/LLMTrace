"""v0.5 Usable MVP 的本地 Web 应用（FastAPI，local-first）。

入口：:func:`llmtrace.web.app.create_app`；CLI ``llmtrace web`` 直接调用它。
页面模板与静态资源随包分发（pyproject package-data）。

- ``api.py``     —— §十七 Web API 路由 + §十八 SSE 实时进度
- ``app.py``     —— FastAPI 应用组装（service 生命周期、模板/静态、页面路由）
- ``demo.py``    —— ``--demo`` 模式的进程内 OpenAI 兼容 Mock upstream
"""
