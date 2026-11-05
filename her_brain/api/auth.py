from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Request

from ..config import ApiKeyConfig, Settings

log = logging.getLogger(__name__)
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def client_desc(request: Request) -> str:
    return request.client.host if request.client else "未知"


def record_problem(request: Request, kind: str, detail: str) -> None:
    """在代理窗口打印被拒绝/失败的请求，并记入 /admin/stats，便于排查“发了消息没回复”。"""
    log.warning("%s（来自 %s %s %s）", detail, client_desc(request), request.method, request.url.path)
    services = getattr(request.app.state, "services", None)
    if services is not None:
        services.metrics.event(kind, f"{detail}（{request.method} {request.url.path}，来自 {client_desc(request)}）")


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def check_api_key(settings: Settings, request: Request) -> ApiKeyConfig | None:
    """返回该 Key 对应的入口配置；未配置任何 Key 时不鉴权。"""
    if not settings.api_keys:
        return None
    token = bearer_token(request)
    if token:
        for cfg in settings.api_keys:
            if hmac.compare_digest(token, cfg.key):
                return cfg
    if settings.server.allow_unknown_keys:
        return None
    shown = f"{token[:8]}…" if token else "（没有带 Key）"
    record_problem(
        request, "auth_rejected",
        f"拒绝了请求：API Key 不匹配，收到 {shown}。入口里填的 Key 要和 config/brain.yaml 的 api_keys 之一一致",
    )
    raise HTTPException(
        status_code=401,
        detail={"error": {"message": "无效的 API Key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


def require_admin(request: Request) -> None:
    settings: Settings = request.app.state.services.settings
    token = settings.server.admin_token
    if token:
        provided = request.headers.get("x-admin-token") or bearer_token(request) or ""
        if not hmac.compare_digest(provided, token):
            raise HTTPException(status_code=401, detail="管理令牌无效")
        return
    host = request.client.host if request.client else ""
    if host not in LOOPBACK:
        raise HTTPException(status_code=403, detail="未设置 admin_token 时只允许本机访问管理接口")
