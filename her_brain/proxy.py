"""大脑代理的对话主流程。

读路径（回复前，要快）：解析元数据 → 身份 → 档案/时效记忆/语义检索（超时保护）→ 组装提示词 → 转发。
写路径（回复后，不阻塞）：原文落库 + 近期动态，记忆提取交给空闲时的后台任务。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import ApiKeyConfig
from .extensions import TurnInfo
from .identity.metadata import parse_request_meta
from .identity.models import Identity, RequestMeta, Scene
from .llm.client import ChatStream, UpstreamError
from .llm.think import ThinkStripper
from .memory.rawlog import SKIPPED
from .memory.recent import RecentItem
from .memory.retrieval import MemoryContext
from .prompt.builder import last_user_text, split_entry_messages
from .timeutil import now

if TYPE_CHECKING:
    from .services import Services

log = logging.getLogger(__name__)

# 不转发给上游的字段：由代理自己决定，或只对代理有意义
DROP_KEYS = {
    "model", "messages", "metadata", "her", "stream", "stream_options", "user", "n",
    "enable_thinking", "thinking",
}


class BadRequest(Exception):
    pass


@dataclass
class Turn:
    turn_id: str
    meta: RequestMeta
    client_model: str
    stream: bool
    include_usage: bool
    upstream_body: dict[str, Any]
    t_start: float
    created: int
    raw: bool = False
    identity: Identity | None = None
    scene: Scene | None = None
    user_text: str = ""
    ctx: MemoryContext | None = None
    t_prepared: float = 0.0
    t_first_token: float | None = None
    tool_calls_seen: bool = False
    finished: bool = False
    # 已发给入口的回复片段（入口中途断开时，用它记录“她实际说出口的部分”）
    reply_parts: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class ChatService:
    def __init__(self, services: Services):
        self.s = services
        self.settings = services.settings

    # ------------------------------------------------------------------ 准备（读路径）
    async def prepare(self, body: dict[str, Any], key_cfg: ApiKeyConfig | None) -> Turn:
        t0 = time.perf_counter()
        meta, messages = parse_request_meta(body, key_cfg, self.settings.platform_aliases)
        if not messages:
            raise BadRequest("messages 不能为空")
        turn = Turn(
            turn_id="chatcmpl-her-" + uuid.uuid4().hex[:20],
            meta=meta,
            client_model=str(body.get("model") or self.settings.server.model_name),
            stream=bool(body.get("stream")),
            include_usage=bool((body.get("stream_options") or {}).get("include_usage")),
            upstream_body={},
            t_start=t0,
            created=int(now()),
        )
        passthrough = {k: v for k, v in body.items() if k not in DROP_KEYS}

        if meta.mode == "raw":
            # 原样透传：工具类调用（意图识别、绘画提示词等）不注入人设、不记录
            turn.raw = True
            turn.upstream_body = {**passthrough, "messages": messages}
            turn.t_prepared = time.perf_counter()
            return turn

        identity = self.s.resolver.resolve(meta)
        scene = self.s.resolver.build_scene(meta, identity)
        convo, entry_system = split_entry_messages(messages)
        user_text = last_user_text(convo)
        now_ts = now()
        ctx = await self.s.reader.gather(identity, scene, user_text, now_ts)
        info = TurnInfo(meta=meta, identity=identity, scene=scene, user_text=user_text)
        built = self.s.builder.build(
            identity=identity,
            scene=scene,
            ctx=ctx,
            entry_messages=convo,
            entry_system=entry_system,
            now_ts=now_ts,
            extra_sections=self.s.extensions.prompt_sections(info),
        )
        up = dict(passthrough)
        for k, v in self.settings.upstream.default_params.items():
            up.setdefault(k, v)
        for k, v in self.settings.upstream.scene_params.get(scene.scene, {}).items():
            up.setdefault(k, v)
        up["messages"] = built.messages

        turn.identity, turn.scene, turn.user_text, turn.ctx = identity, scene, user_text, ctx
        turn.upstream_body = up
        turn.debug = {"sections": built.sections, "dynamic": built.dynamic_text}
        turn.t_prepared = time.perf_counter()
        return turn

    def open_upstream(self, turn: Turn) -> ChatStream:
        return self.s.upstream.stream(turn.upstream_body)

    # ------------------------------------------------------------------ 流式
    async def sse(self, turn: Turn, stream: ChatStream) -> AsyncIterator[str]:
        stripper = ThinkStripper() if self.settings.upstream.strip_think else None
        acc = turn.reply_parts
        interrupted, error = True, None
        try:
            async for chunk in stream.chunks():
                out = self._transform(chunk, turn, stripper, acc)
                if out is not None:
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            if stripper is not None:
                rest = stripper.flush()
                if rest:
                    acc.append(rest)
                    yield "data: " + json.dumps(self._content_chunk(turn, rest), ensure_ascii=False) + "\n\n"
            interrupted = False
            yield "data: [DONE]\n\n"
        except UpstreamError as exc:
            error = exc.message
            payload = {"error": {"message": exc.message, "type": "upstream_error", "code": exc.status}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await stream.__aexit__(None, None, None)
            self.finish(turn, "".join(acc), interrupted=interrupted, error=error)

    def _content_chunk(self, turn: Turn, text: str) -> dict[str, Any]:
        return {
            "id": turn.turn_id, "object": "chat.completion.chunk", "created": turn.created,
            "model": turn.client_model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }

    def _transform(
        self, chunk: dict[str, Any], turn: Turn, stripper: ThinkStripper | None, acc: list[str]
    ) -> dict[str, Any] | None:
        chunk = dict(chunk)
        chunk["model"] = turn.client_model
        chunk.pop("timings", None)
        if not turn.include_usage:
            chunk.pop("usage", None)
        choices = chunk.get("choices") or []
        if not choices:
            return chunk if chunk.get("usage") else None
        kept = []
        for choice in choices:
            choice = dict(choice)
            delta = dict(choice.get("delta") or {})
            if self.settings.upstream.strip_think:
                delta.pop("reasoning_content", None)
                delta.pop("reasoning", None)
            content = delta.get("content")
            if content is not None and stripper is not None:
                content = stripper.feed(content)
            if choice.get("finish_reason") and stripper is not None:
                content = (content or "") + stripper.flush()
            if content:
                acc.append(content)
                delta["content"] = content
                if turn.t_first_token is None:
                    turn.t_first_token = time.perf_counter()
            elif "content" in delta or content == "":
                if "role" in delta:
                    delta["content"] = ""
                else:
                    delta.pop("content", None)
            if delta.get("tool_calls"):
                turn.tool_calls_seen = True
            if delta.get("content") or "role" in delta or delta.get("tool_calls") or choice.get("finish_reason"):
                choice["delta"] = delta
                kept.append(choice)
        if not kept:
            return None
        chunk["choices"] = kept
        return chunk

    # ------------------------------------------------------------------ 非流式
    async def complete(
        self, turn: Turn, stream: ChatStream
    ) -> tuple[dict[str, Any], Callable[[], Awaitable[None]]]:
        """返回（完整响应, 收尾函数）。收尾函数在响应发出后执行，保证落库不在回复路径上。"""
        stripper = ThinkStripper() if self.settings.upstream.strip_think else None
        parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason, usage, resp_id = None, None, None
        try:
            async for chunk in stream.chunks():
                resp_id = resp_id or chunk.get("id")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or choice.get("message") or {}
                    content = delta.get("content")
                    if content:
                        if turn.t_first_token is None:
                            turn.t_first_token = time.perf_counter()
                        parts.append(stripper.feed(content) if stripper else content)
                    for tc in delta.get("tool_calls") or []:
                        _merge_tool_call(tool_calls, tc)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
            if stripper is not None:
                parts.append(stripper.flush())
        except BaseException as exc:
            await stream.__aexit__(None, None, None)
            self.finish(turn, "".join(parts), interrupted=True,
                        error=exc.message if isinstance(exc, UpstreamError) else type(exc).__name__)
            raise
        await stream.__aexit__(None, None, None)

        content = "".join(parts)
        message: dict[str, Any] = {"role": "assistant", "content": content if (content or not tool_calls) else None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
            turn.tool_calls_seen = True
        payload = {
            "id": turn.turn_id,
            "object": "chat.completion",
            "created": turn.created,
            "model": turn.client_model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
        }
        if usage:
            payload["usage"] = usage

        async def _finalize() -> None:
            self.finish(turn, content, interrupted=False, error=None)

        return payload, _finalize

    # ------------------------------------------------------------------ 收尾（写路径）
    def finish(self, turn: Turn, reply: str, *, interrupted: bool, error: str | None) -> None:
        if turn.finished:
            return
        turn.finished = True
        try:
            self._record(turn, reply, interrupted, error)
        except Exception:  # noqa: BLE001 - 落库失败不能影响对话
            log.exception("记录对话失败")
        finally:
            self.s.activity.end()

    def _record(self, turn: Turn, reply: str, interrupted: bool, error: str | None) -> None:
        t_end = time.perf_counter()
        m = self.s.metrics
        prep_ms = (turn.t_prepared - turn.t_start) * 1000
        m.observe("proxy_prepare_ms", prep_ms)
        m.incr("turns_raw" if turn.raw else "turns")
        if turn.t_first_token is not None:
            m.observe("first_token_ms", (turn.t_first_token - turn.t_start) * 1000)
        m.observe("total_ms", (t_end - turn.t_start) * 1000)
        if error:
            m.incr("upstream_errors")
        if interrupted:
            m.incr("interrupted")
        if turn.ctx is not None:
            m.observe("retrieval_ms", turn.ctx.retrieval_ms)
            if turn.ctx.timed_out:
                m.incr("retrieval_timeouts")

        if turn.raw or turn.identity is None or turn.scene is None:
            return
        ident, scene = turn.identity, turn.scene
        log.info(
            "对话 %s|%s|%s 准备 %.0fms（检索 %.0fms%s）首字 %s 总计 %.0fms%s",
            scene.channel, scene.scene, ident.user_id, prep_ms, turn.ctx.retrieval_ms if turn.ctx else 0,
            "，超时" if turn.ctx and turn.ctx.timed_out else "",
            f"{(turn.t_first_token - turn.t_start) * 1000:.0f}ms" if turn.t_first_token else "-",
            (t_end - turn.t_start) * 1000, "（中断）" if interrupted else "",
        )
        reply = reply.strip()
        if turn.meta.no_log or not turn.user_text or not reply:
            return
        skip_memory = (
            ident.anonymous
            or turn.meta.no_memory
            or (ident.role == "guest" and not self.settings.memory.remember_guests)
        )
        entry = self.s.rawlog.new(
            user_id=ident.user_id,
            channel=scene.channel,
            scene=scene.scene,
            group_id=scene.group_id,
            privacy=scene.write_privacy,
            speaker_name=ident.display_name,
            user_text=turn.user_text,
            reply_text=reply,
            processed=SKIPPED if skip_memory else 0,
            meta={
                "turn_id": turn.turn_id,
                "platform": turn.meta.platform,
                "group_name": scene.group_name,
                "interrupted": interrupted,
                "speaker_verified": scene.speaker_verified,
            },
        )
        if not ident.anonymous:
            # 先加近期动态再写日志：若该用户的近期动态尚未从日志加载，先写日志会导致同一条被加载两次
            self.s.recent.add(
                ident.user_id,
                RecentItem(entry.created_at, scene.channel, scene.scene, scene.group_id, scene.write_privacy,
                           turn.user_text, reply),
            )
        self.s.rawlog.insert(entry)
        if self.s.extensions.extensions:
            info = TurnInfo(meta=turn.meta, identity=ident, scene=scene, user_text=turn.user_text, reply_text=reply)
            asyncio.get_running_loop().create_task(self.s.extensions.on_reply(info))


def _merge_tool_call(acc: dict[int, dict[str, Any]], tc: dict[str, Any]) -> None:
    idx = int(tc.get("index", len(acc)))
    cur = acc.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
    if tc.get("id"):
        cur["id"] = tc["id"]
    if tc.get("type"):
        cur["type"] = tc["type"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        cur["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        cur["function"]["arguments"] += fn["arguments"]
