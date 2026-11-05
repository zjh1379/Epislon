"""OpenAI 兼容接口：/v1/chat/completions（流式与非流式）、/v1/models。"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import Receive, Scope, Send

from ..llm.client import UpstreamError
from ..proxy import BadRequest
from .auth import check_api_key, record_problem

log = logging.getLogger(__name__)
router = APIRouter()


def openai_error(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": err_type, "code": status}}, status_code=status)


class GuardedStreamingResponse(StreamingResponse):
    """无论流是正常结束、被入口断开，还是根本没开始，都保证执行一次清理。"""

    def __init__(self, content: Any, cleanup: Callable[[], Awaitable[None]], **kwargs: Any):
        super().__init__(content, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


# 入口里的地址少写了 /v1 是最常见的配置错误，这里同时接受两种写法
@router.get("/v1/models")
@router.get("/models", include_in_schema=False)
async def list_models(request: Request) -> dict[str, Any]:
    svc = request.app.state.services
    check_api_key(svc.settings, request)
    name = svc.settings.server.model_name
    return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "her-brain"}]}


@router.post("/v1/chat/completions")
@router.post("/chat/completions", include_in_schema=False)
async def chat_completions(request: Request) -> Any:
    svc = request.app.state.services
    key_cfg = check_api_key(svc.settings, request)
    try:
        body = await request.json()
    except ValueError:
        record_problem(request, "bad_requests", "拒绝了请求：请求体不是合法 JSON")
        return openai_error(400, "请求体不是合法 JSON")
    if not isinstance(body, dict):
        record_problem(request, "bad_requests", "拒绝了请求：请求体必须是 JSON 对象")
        return openai_error(400, "请求体必须是 JSON 对象")

    svc.activity.begin()
    try:
        turn = await svc.chat.prepare(body, key_cfg)
    except BadRequest as exc:
        svc.activity.end()
        record_problem(request, "bad_requests", f"拒绝了请求：{exc}")
        return openai_error(400, str(exc))
    except BaseException:
        svc.activity.end()
        raise

    stream = svc.chat.open_upstream(turn)
    try:
        await stream.__aenter__()
    except UpstreamError as exc:
        svc.chat.finish(turn, "", interrupted=True, error=exc.message)
        record_problem(request, "upstream_failures", f"模型服务出错，没能回复：{exc.message}")
        status = exc.status if 400 <= exc.status < 600 else 502
        return openai_error(status, exc.message, "upstream_error")
    except BaseException:
        svc.chat.finish(turn, "", interrupted=True, error="cancelled")
        raise

    headers = {"X-Her-Prepare-Ms": f"{(turn.t_prepared - turn.t_start) * 1000:.1f}"}
    if turn.ctx is not None:
        headers["X-Her-Retrieval-Ms"] = f"{turn.ctx.retrieval_ms:.1f}"

    if turn.stream:
        body_iter = svc.chat.sse(turn, stream)

        async def cleanup() -> None:
            try:
                await body_iter.aclose()
            except Exception:  # noqa: BLE001
                pass
            await stream.__aexit__(None, None, None)
            svc.chat.finish(turn, "".join(turn.reply_parts), interrupted=True, error=None)

        headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return GuardedStreamingResponse(body_iter, cleanup, media_type="text/event-stream", headers=headers)

    try:
        payload, finalize = await svc.chat.complete(turn, stream)
    except UpstreamError as exc:
        record_problem(request, "upstream_failures", f"模型服务在回复途中出错：{exc.message}")
        return openai_error(502, exc.message, "upstream_error")
    return JSONResponse(payload, headers=headers, background=BackgroundTask(finalize))
