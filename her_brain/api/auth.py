from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from ..config import ApiKeyConfig, Settings

LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


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
