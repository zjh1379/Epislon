"""假的 OpenAI 兼容服务：模拟 llama-server 的流式对话、JSON 提取与嵌入。

用途：
- 单元测试（作为 httpx.MockTransport 注入）；
- 没有显卡时联调 AstrBot / 枫云 与大脑代理：
  ``python -m her_brain.devtools.fake_llm --port 8080``，然后把 upstream.base_url 指向它。

提取规则只是几条正则，足以演示“明天要去面试 → 第二天被问起”这类流程，不代表真实模型效果。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..textutil import content_to_text, tokens

ReplyFn = Callable[[list[dict[str, Any]]], str]
ExtractFn = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

_STAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}")


def default_reply(messages: list[dict[str, Any]]) -> str:
    system = "\n".join(content_to_text(m.get("content")) for m in messages if m.get("role") == "system")
    last = next((content_to_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    if "面试" in system and any(w in last for w in ("回来", "到家", "结束")):
        return "欢迎回来！面试怎么样？"
    return f"嗯嗯，我听到你说「{last[:20]}」啦。"


def default_extract(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从提取提示词里的对话记录中，用几条正则模拟模型的提取结果。"""
    transcript = content_to_text(messages[-1].get("content"))
    system = content_to_text(messages[0].get("content"))
    m = re.search(r"称呼对方为“([^”]+)”", system)
    user = m.group(1) if m else "对方"
    facts: list[dict[str, Any]] = []
    day: date | None = None
    for line in transcript.splitlines():
        stamp = _STAMP_RE.match(line)
        if stamp:
            day = datetime.strptime(stamp.group(1), "%Y-%m-%d").date()
            continue
        if not line.startswith(f"{user}："):
            continue
        said = line[len(user) + 1 :]
        explicit = "记住" in said or "别忘了" in said
        base = {"about": "user", "slot": "", "value": "", "date": "", "importance": 3, "explicit": explicit}
        if "明天" in said and day is not None:
            when = (day + timedelta(days=1)).isoformat()
            what = said.split("明天", 1)[1].strip("，。！ ")
            facts.append({**base, "text": f"{user}在 {when} {what}", "kind": "plan", "date": when, "importance": 4})
        if mm := re.search(r"我叫([一-鿿A-Za-z0-9]{1,8})", said):
            facts.append({**base, "text": f"{user}的名字是{mm.group(1)}", "kind": "attribute", "slot": "名字",
                          "value": mm.group(1)})
        if mm := re.search(r"我(?:现在)?(?:在|住在)([一-鿿]{2,4}?)(?:工作|上班|生活|住)", said):
            facts.append({**base, "text": f"{user}住在{mm.group(1)}", "kind": "attribute", "slot": "所在城市",
                          "value": mm.group(1)})
        if mm := re.search(r"我(?:最)?喜欢([一-鿿A-Za-z0-9]{1,10})", said):
            facts.append({**base, "text": f"{user}喜欢{mm.group(1)}", "kind": "preference"})
        if mm := re.search(r"我(?:对)?([一-鿿]{1,6})过敏", said):
            facts.append({**base, "text": f"{user}对{mm.group(1)}过敏", "kind": "attribute", "importance": 5})
    return facts


def hash_embedding(text: str, dims: int = 256) -> list[float]:
    vec = [0.0] * dims
    for tok in tokens(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dims] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeLLM:
    def __init__(
        self,
        reply_fn: ReplyFn = default_reply,
        extract_fn: ExtractFn = default_extract,
        *,
        emit_think: bool = False,
        chunk_delay: float = 0.0,
        dims: int = 256,
    ):
        self.reply_fn = reply_fn
        self.extract_fn = extract_fn
        self.emit_think = emit_think
        self.chunk_delay = chunk_delay
        self.dims = dims
        self.requests: list[dict[str, Any]] = []
        self.fail_next: int | None = None  # 设为状态码时，下一次对话请求返回该错误

    # ------------------------------------------------------------------ 响应生成
    def _is_extraction(self, body: dict[str, Any]) -> bool:
        return "response_format" in body

    def chat_text(self, body: dict[str, Any]) -> str:
        messages = body.get("messages", [])
        if self._is_extraction(body):
            return json.dumps({"facts": self.extract_fn(messages)}, ensure_ascii=False)
        if body.get("max_tokens") == 1:
            return "嗯"
        text = self.reply_fn(messages)
        if self.emit_think:
            text = "<think>\n让我想想……\n</think>\n\n" + text
        return text

    async def sse(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        text = self.chat_text(body)
        cid = "chatcmpl-fake"
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": body.get("model"),
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

        yield chunk({"role": "assistant", "content": None})
        step = max(1, len(text) // 4)
        for i in range(0, len(text), step):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield chunk({"content": text[i : i + step]})
        yield chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    def embed(self, body: dict[str, Any]) -> dict[str, Any]:
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        return {"object": "list", "model": body.get("model"),
                "data": [{"object": "embedding", "index": i, "embedding": hash_embedding(t, self.dims)}
                         for i, t in enumerate(inputs or [])]}

    # ------------------------------------------------------------------ httpx 传输层
    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"{}") if request.method == "POST" else {}
        self.requests.append({"path": path, "body": body})
        if path.endswith("/chat/completions"):
            if self.fail_next:
                status, self.fail_next = self.fail_next, None
                return httpx.Response(status, json={"error": {"message": "模拟的上游错误"}})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self.sse(body))
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=self.embed(body))
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "fake", "object": "model"}]})
        return httpx.Response(404, json={"error": "not found"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @property
    def chat_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["path"].endswith("/chat/completions")]


def create_app(fake: FakeLLM | None = None) -> FastAPI:
    fake = fake or FakeLLM()
    app = FastAPI(title="Fake LLM")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):  # type: ignore[no-untyped-def]
        body = await request.json()
        fake.requests.append({"path": "/v1/chat/completions", "body": body})
        if body.get("stream"):
            return StreamingResponse(fake.sse(body), media_type="text/event-stream")
        text = fake.chat_text(body)
        return JSONResponse({"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
                             "model": body.get("model"),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop"}]})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):  # type: ignore[no-untyped-def]
        return fake.embed(await request.json())

    @app.get("/v1/models")
    async def models():  # type: ignore[no-untyped-def]
        return {"object": "list", "data": [{"id": "fake", "object": "model"}]}

    @app.get("/health")
    async def health():  # type: ignore[no-untyped-def]
        return {"status": "ok"}

    return app


if __name__ == "__main__":  # pragma: no cover
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="假的 OpenAI 兼容服务（联调用）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--delay", type=float, default=0.02, help="每个分块之间的延迟（秒）")
    args = parser.parse_args()
    uvicorn.run(create_app(FakeLLM(chunk_delay=args.delay)), host=args.host, port=args.port)
