from __future__ import annotations

import json
from typing import Any

import httpx

from her_brain.devtools.fake_llm import FakeLLM
from her_brain.memory.rawlog import PENDING, SKIPPED

from .conftest import QQ_GROUP, QQ_PRIVATE, VOICE_OWNER, chat_body


def parse_sse(text: str) -> tuple[list[dict[str, Any]], bool]:
    chunks, done = [], False
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                done = True
            else:
                chunks.append(json.loads(data))
    return chunks, done


def sse_text(chunks: list[dict[str, Any]]) -> str:
    return "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))


async def test_models(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/models")
    assert r.status_code == 200 and r.json()["data"][0]["id"] == "her"


async def test_non_stream_completion_and_raw_log(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    r = await client.post("/v1/chat/completions", json=chat_body("今天好累", metadata=QQ_PRIVATE))
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion" and data["model"] == "her"
    assert "今天好累" in data["choices"][0]["message"]["content"]
    assert float(r.headers["x-her-prepare-ms"]) < 50  # 代理增加的延迟预算

    up = fake.chat_requests[-1]
    assert up["model"] == "fake-model" and up["stream"] is True
    assert up["chat_template_kwargs"] == {"enable_thinking": False}
    assert "metadata" not in up and up["temperature"] == 0.7

    logs = services.rawlog.list(user_id="owner")
    assert len(logs) == 1 and logs[0]["privacy"] == "private" and logs[0]["processed"] == PENDING
    assert logs[0]["user_text"] == "今天好累"


async def test_stream_completion(client: httpx.AsyncClient, fake: FakeLLM) -> None:
    fake.emit_think = True
    r = await client.post("/v1/chat/completions", json=chat_body("你好", metadata=QQ_PRIVATE, stream=True))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    chunks, done = parse_sse(r.text)
    assert done
    text = sse_text(chunks)
    assert "think" not in text and "让我想想" not in text and text.startswith("嗯嗯")
    assert all(c["model"] == "her" for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_prompt_order_and_static_prefix_identical(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    await client.post("/v1/chat/completions", json=chat_body("早上好", metadata=QQ_PRIVATE))
    await client.post("/v1/chat/completions", json=chat_body("大家好", metadata=QQ_GROUP))
    await client.post("/v1/chat/completions", json=chat_body("我回来啦", metadata=VOICE_OWNER))
    static = services.persona.static_text
    systems = [req["messages"][0]["content"] for req in fake.chat_requests]
    assert all(s.startswith(static + "\n\n# 本轮动态信息") for s in systems)
    assert "QQ私聊" in systems[0] and "QQ群聊「测试群」" in systems[1] and "桌面语音" in systems[2]
    assert "场合：公开" in systems[1] and "场合：公开" not in systems[0]
    # [3]→[6] 的顺序
    s = systems[2]
    assert s.index("## 当前场景") < s.index("## 其他入口的近期对话")
    # 人设在最前面
    assert systems[0].startswith("# 你是谁\n你是小E。")


async def test_entry_system_prompt_handling(client: httpx.AsyncClient, fake: FakeLLM) -> None:
    body = chat_body("在吗", system="你是一个助手/no_think", metadata=QQ_PRIVATE)
    await client.post("/v1/chat/completions", json=body)
    system = fake.chat_requests[-1]["messages"][0]["content"]
    assert "## 入口附加说明\n- 你是一个助手" in system and "/no_think" not in system
    assert [m["role"] for m in fake.chat_requests[-1]["messages"]] == ["system", "user"]


async def test_marker_is_never_forwarded(client: httpx.AsyncClient, fake: FakeLLM) -> None:
    marker = '<her-meta>{"platform": "aiocqhttp", "scene": "group", "external_user_id": "10001", "group_id": "9"}</her-meta>'
    await client.post("/v1/chat/completions", json=chat_body("hi", system=marker))
    up = json.dumps(fake.chat_requests[-1], ensure_ascii=False)
    assert "her-meta" not in up and "10001" not in up


async def test_raw_mode_passthrough(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    body = chat_body("判断意图：播放音乐", system="你是意图分类器", metadata={"mode": "raw"})
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    up = fake.chat_requests[-1]
    assert up["messages"][0]["content"] == "你是意图分类器"
    assert services.rawlog.list() == []


async def test_upstream_error_returns_openai_error(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    fake.fail_next = 503
    r = await client.post("/v1/chat/completions", json=chat_body("hi", metadata=QQ_PRIVATE))
    assert r.status_code == 503 and "error" in r.json()
    assert services.activity.active == 0
    assert services.metrics.summary()["counters"]["upstream_errors"] == 1


async def test_bad_requests(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = await client.post("/v1/chat/completions", json={"model": "her", "messages": []})
    assert r.status_code == 400


async def test_anonymous_voice_is_logged_but_not_memorized(client: httpx.AsyncClient, services) -> None:  # type: ignore[no-untyped-def]
    meta = {"channel": "voice", "scene": "voice", "speaker_verified": False}
    await client.post("/v1/chat/completions", json=chat_body("你是谁呀", metadata=meta))
    logs = services.rawlog.list()
    assert logs[0]["user_id"] == "anon:voice" and logs[0]["processed"] == SKIPPED


async def test_group_log_is_public(client: httpx.AsyncClient, services) -> None:  # type: ignore[no-untyped-def]
    await client.post("/v1/chat/completions", json=chat_body("大家晚上好", metadata=QQ_GROUP))
    log = services.rawlog.list()[0]
    assert log["privacy"] == "public" and log["group_id"] == "777"


async def test_tool_calls_are_aggregated(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    async def sse(body: dict[str, Any]):  # type: ignore[no-untyped-def]
        def c(delta: dict[str, Any], finish: str | None = None) -> bytes:
            obj = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(obj)}\n\n".encode()

        yield c({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                       "function": {"name": "get_weather", "arguments": ""}}]})
        yield c({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]})
        yield c({"tool_calls": [{"index": 0, "function": {"arguments": '"杭州"}'}}]})
        yield c({}, "tool_calls")
        yield b"data: [DONE]\n\n"

    fake.sse = sse  # type: ignore[method-assign]
    body = chat_body("杭州天气怎么样", metadata=QQ_PRIVATE)
    body["tools"] = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    r = await client.post("/v1/chat/completions", json=body)
    msg = r.json()["choices"][0]
    assert msg["finish_reason"] == "tool_calls" and msg["message"]["content"] is None
    call = msg["message"]["tool_calls"][0]
    assert call["id"] == "call_1" and json.loads(call["function"]["arguments"]) == {"city": "杭州"}
    assert fake.chat_requests[-1]["tools"]  # 工具定义原样透传
    assert services.rawlog.list() == []  # 只有工具调用、没有文字回复时不落库


async def test_api_keys(make_settings, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    from her_brain.app import create_app
    from her_brain.services import Services, Transports

    settings = make_settings(api_keys=[{"key": "sk-voice", "channel": "voice", "scene": "voice"}],
                             memory={"trust_unverified_voice": True})
    svc = Services(settings, transports=Transports(upstream=fake.transport()))
    app = create_app(settings, svc, run_worker=False)
    app.state.services = svc
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b") as c:
            r = await c.post("/v1/chat/completions", json=chat_body("hi"))
            assert r.status_code == 401
            r = await c.post("/v1/chat/completions", json=chat_body("hi"), headers={"Authorization": "Bearer sk-voice"})
            assert r.status_code == 200
            log = svc.rawlog.list()[0]
            assert (log["channel"], log["scene"], log["user_id"]) == ("voice", "voice", "owner")
    finally:
        await svc.close()


async def test_rejected_requests_are_logged(make_settings, fake: FakeLLM, caplog) -> None:  # type: ignore[no-untyped-def]
    """排查“发了消息没回复”：被拒绝的请求要在代理窗口里打印原因，并能在 /admin/stats 看到。"""
    from her_brain.app import create_app
    from her_brain.services import Services, Transports

    settings = make_settings(api_keys=[{"key": "sk-her-astrbot", "channel": "qq"}])
    svc = Services(settings, transports=Transports(upstream=fake.transport()))
    app = create_app(settings, svc, run_worker=False)
    app.state.services = svc
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b") as c:
            with caplog.at_level("WARNING"):
                r = await c.post("/v1/chat/completions", json=chat_body("hi"), headers={"Authorization": "Bearer sk-wrong-key"})
            assert r.status_code == 401
            assert "API Key 不匹配" in caplog.text and "sk-wrong" in caplog.text
            # 地址少写了 /v1：照样可用
            ok = {"Authorization": "Bearer sk-her-astrbot"}
            assert (await c.post("/chat/completions", json=chat_body("hi", metadata=QQ_PRIVATE), headers=ok)).status_code == 200
            assert (await c.get("/models", headers=ok)).json()["data"][0]["id"] == "her"
            # 地址里多写了 /chat/completions：404 并给出提示
            r = await c.post("/v1/chat/completions/chat/completions", json=chat_body("hi"), headers=ok)
            assert r.status_code == 404
            stats = (await c.get("/admin/stats")).json()["metrics"]
            assert stats["counters"]["auth_rejected"] == 1 and stats["counters"]["not_found"] == 1
            assert any("不要包含 /chat/completions" in e["detail"] for e in stats["recent_events"])
    finally:
        await svc.close()


async def test_non_stream_upstream_error_mid_response(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    async def broken(body: dict[str, Any]):  # type: ignore[no-untyped-def]
        yield b'data: {"choices": [{"index": 0, "delta": {"content": "\xe4\xbd\xa0"}}]}\n\n'
        yield b'data: {"error": {"message": "CUDA out of memory"}}\n\n'

    fake.sse = broken  # type: ignore[method-assign]
    r = await client.post("/v1/chat/completions", json=chat_body("hi", metadata=QQ_PRIVATE))
    assert r.status_code == 502 and "out of memory" in r.json()["error"]["message"]
    assert services.activity.active == 0
    assert services.metrics.summary()["counters"]["upstream_failures"] == 1


async def test_admin_debug_prompt_and_stats(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=chat_body("你好", metadata=QQ_PRIVATE))
    r = await client.post("/admin/debug/prompt", json=chat_body("你好", metadata=QQ_GROUP))
    data = r.json()
    assert data["user_id"] == "owner" and data["scene"]["access"] == "public"
    stats = (await client.get("/admin/stats")).json()
    assert stats["metrics"]["counters"]["turns"] == 1
    assert "proxy_prepare_ms" in stats["metrics"]["latency_ms"]


async def test_scene_params_applied(make_settings, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    from her_brain.services import Services, Transports

    svc = Services(make_settings(upstream={"scene_params": {"voice": {"max_tokens": 123}}}),
                   transports=Transports(upstream=fake.transport()))
    try:
        turn = await svc.chat.prepare(chat_body("嗨", metadata=VOICE_OWNER), None)
        assert turn.upstream_body["max_tokens"] == 123
        turn = await svc.chat.prepare({**chat_body("嗨", metadata=VOICE_OWNER), "max_tokens": 50}, None)
        assert turn.upstream_body["max_tokens"] == 50  # 入口显式给出的参数优先
    finally:
        await svc.close()
