"""llama-server（或任意 OpenAI 兼容服务）的流式客户端。

无论入口要不要流式，代理到上游一律走流式：这样能测到首 token 时间，
入口断开时也能立即中止上游生成，把显卡让出来。
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import httpx

from ..config import LLMEndpoint
from .think import strip_think

log = logging.getLogger(__name__)


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, body: Any = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.body = body


def _deep_setdefault(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_setdefault(target[k], v)
        elif k not in target:
            target[k] = copy.deepcopy(v)


class ChatStream:
    def __init__(self, client: httpx.AsyncClient, body: dict[str, Any]):
        self._client = client
        self._body = body
        self._resp: httpx.Response | None = None

    async def __aenter__(self) -> ChatStream:
        req = self._client.build_request("POST", "/chat/completions", json=self._body)
        try:
            self._resp = await self._client.send(req, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, f"上游模型连接超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(502, f"无法连接上游模型：{exc}") from exc
        if self._resp.status_code != 200:
            raw = await self._resp.aread()
            await self._resp.aclose()
            text = raw.decode("utf-8", "replace")
            try:
                body = json.loads(text)
            except ValueError:
                body = None
            raise UpstreamError(self._resp.status_code, f"上游模型返回 {self._resp.status_code}：{text[:300]}", body)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._resp is not None:
            await self._resp.aclose()

    async def chunks(self) -> AsyncIterator[dict[str, Any]]:
        assert self._resp is not None
        try:
            async for line in self._resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and obj.get("error") and not obj.get("choices"):
                        err = obj["error"]
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        raise UpstreamError(502, f"上游模型生成出错：{msg}", obj)
                    yield obj
                elif line.startswith("{"):
                    # 上游忽略了 stream=true，直接返回了完整 JSON：转换成一个分块
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    yield _completion_to_chunk(obj)
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, f"上游模型生成超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(502, f"上游连接中断：{exc}") from exc


def _completion_to_chunk(obj: dict[str, Any]) -> dict[str, Any]:
    choices = []
    for c in obj.get("choices", []):
        msg = c.get("message") or {}
        choices.append({"index": c.get("index", 0), "delta": msg, "finish_reason": c.get("finish_reason")})
    out = {k: v for k, v in obj.items() if k != "choices"}
    out["object"] = "chat.completion.chunk"
    out["choices"] = choices
    return out


class LLMClient:
    def __init__(self, endpoint: LLMEndpoint, transport: httpx.AsyncBaseTransport | None = None):
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(
            base_url=endpoint.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
            timeout=httpx.Timeout(endpoint.timeout, connect=endpoint.connect_timeout),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            transport=transport,
        )

    def prepare_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """补上模型名、槽位、关闭思考模式等上游参数（入口显式给出的参数优先）。"""
        out = dict(body)
        out["model"] = self.endpoint.model
        out["stream"] = True
        if self.endpoint.slot_id is not None:
            out.setdefault("id_slot", self.endpoint.slot_id)
        out.setdefault("cache_prompt", True)
        if self.endpoint.disable_thinking:
            ctk = dict(out.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            out["chat_template_kwargs"] = ctk
        _deep_setdefault(out, self.endpoint.extra_body)
        return out

    def stream(self, body: dict[str, Any]) -> ChatStream:
        return ChatStream(self._client, self.prepare_body(body))

    async def complete(self, messages: list[dict[str, Any]], **params: Any) -> str:
        """内部使用（后台提取）：收集完整文本。被取消时会中止上游生成。"""
        parts: list[str] = []
        async with self.stream({"messages": messages, **params}) as s:
            async for chunk in s.chunks():
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)
        return strip_think("".join(parts))

    async def warmup(self, messages: list[dict[str, Any]]) -> bool:
        """让上游把静态前缀算进 KV 缓存，下一次真实对话直接复用。"""
        try:
            async with self.stream({"messages": messages + [{"role": "user", "content": "嗯"}], "max_tokens": 1,
                                    "temperature": 0}) as s:
                async for _ in s.chunks():
                    pass
            return True
        except Exception as exc:  # noqa: BLE001 - 预热失败不影响服务
            log.debug("预热失败：%s", exc)
            return False

    async def models(self) -> list[dict[str, Any]]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def aclose(self) -> None:
        await self._client.aclose()
