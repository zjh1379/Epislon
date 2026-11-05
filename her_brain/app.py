from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api import admin, openai_compat
from .api.auth import record_problem
from .config import Settings
from .services import Services

log = logging.getLogger(__name__)


def create_app(settings: Settings, services: Services | None = None, *, run_worker: bool = True) -> FastAPI:
    """services 为空时在启动时创建；测试可传入预先装配（带假上游）的 Services。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services or Services(settings)
        app.state.services = svc
        await svc.start(run_worker=run_worker)
        log.info(
            "大脑代理已启动：模型名 %s，上游 %s，记忆引擎 %s",
            settings.server.model_name, settings.upstream.base_url, svc.index.name,
        )
        try:
            yield
        finally:
            await svc.close()

    app = FastAPI(title="Her Brain", version=__version__, lifespan=lifespan)
    app.include_router(openai_compat.router)
    app.include_router(admin.router)

    @app.exception_handler(StarletteHTTPException)
    async def _log_not_found(request: Request, exc: StarletteHTTPException) -> Any:
        if exc.status_code == 404:
            hint = ""
            if request.url.path.endswith("/chat/completions"):
                hint = "。入口里的 API 地址应填 http://<代理地址>:8000/v1，不要包含 /chat/completions"
            record_problem(request, "not_found", f"收到了未知地址的请求{hint}")
        return await http_exception_handler(request, exc)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        svc: Services = app.state.services
        return {"ok": True, "persona": svc.persona.name, "engine": svc.index.name, "active_turns": svc.activity.active}

    return app
