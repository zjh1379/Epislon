"""端到端：复现架构文档图 6 的跨入口记忆连续性与场合过滤。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from her_brain.devtools.fake_llm import FakeLLM
from her_brain.memory.rawlog import DONE, FAILED, PENDING
from her_brain.worker.activity import ActivityTracker, Preempted

from .conftest import QQ_GROUP, QQ_PRIVATE, VOICE_OWNER, chat_body
from .test_proxy import parse_sse, sse_text


def shift_logs_back(services, seconds: float) -> None:  # type: ignore[no-untyped-def]
    services.db.execute("UPDATE raw_log SET created_at = created_at - ?", (seconds,))
    services.recent.forget()


async def test_figure6_cross_entry_memory(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    # 第一天：QQ 私聊说明天要去面试
    r = await client.post("/v1/chat/completions",
                          json=chat_body("我明天要去面试，有点紧张", metadata=QQ_PRIVATE, stream=True))
    assert parse_sse(r.text)[1]
    log = services.rawlog.list(user_id="owner")[0]
    assert log["scene"] == "private" and log["privacy"] == "private"

    # 模拟时间过去一天，空闲时后台提取
    shift_logs_back(services, 86400)
    summary = await services.worker.run_until_idle()
    assert summary["batches"] == 1
    assert services.rawlog.list(user_id="owner")[0]["processed"] == DONE
    mems = services.episodic.list_for_user("owner")
    assert len(mems) == 1 and mems[0].kind == "plan" and mems[0].privacy == "private"
    assert mems[0].source_log_ids == [log["log_id"]]  # 来源可追溯

    # 第二天：桌面语音（声纹确认为主人）说“我回来了”
    r = await client.post("/v1/chat/completions", json=chat_body("我回来了", metadata=VOICE_OWNER, stream=True))
    chunks, done = parse_sse(r.text)
    assert done and sse_text(chunks) == "欢迎回来！面试怎么样？"
    system = fake.chat_requests[-1]["messages"][0]["content"]
    assert "面试" in system and "事情在今天" in system

    # 在 QQ 群里被问起：私密记忆被过滤，不会提及面试
    r = await client.post("/v1/chat/completions", json=chat_body("最近怎么样", metadata=QQ_GROUP))
    system = fake.chat_requests[-1]["messages"][0]["content"]
    assert "面试" not in system
    assert "面试" not in r.json()["choices"][0]["message"]["content"]


async def test_recent_activity_crosses_entries(client: httpx.AsyncClient, fake: FakeLLM) -> None:
    await client.post("/v1/chat/completions", json=chat_body("我刚吃了火锅", metadata=QQ_PRIVATE))
    await client.post("/v1/chat/completions", json=chat_body("在吗", metadata=VOICE_OWNER))
    system = fake.chat_requests[-1]["messages"][0]["content"]
    assert "## 其他入口的近期对话" in system and "火锅" in system and "QQ私聊" in system
    # 同一个对话窗口（QQ 私聊）内不重复带上
    await client.post("/v1/chat/completions", json=chat_body("还在吗", metadata=QQ_PRIVATE))
    system = fake.chat_requests[-1]["messages"][0]["content"]
    assert "其他入口的近期对话" not in system or "火锅" not in system.split("其他入口的近期对话")[1]


async def test_profile_promotion_through_worker(client: httpx.AsyncClient, services) -> None:  # type: ignore[no-untyped-def]
    await client.post("/v1/chat/completions", json=chat_body("我最喜欢草莓", metadata=QQ_PRIVATE))
    await services.worker.run_until_idle()
    assert services.profiles.get("owner").entries == []  # 只提到一次：仍是候选
    await client.post("/v1/chat/completions", json=chat_body("说起来我最喜欢草莓", metadata=VOICE_OWNER))
    await services.worker.run_until_idle()
    entries = services.profiles.get("owner").entries
    assert [e.text for e in entries] == ["小明喜欢草莓"]
    # 档案在之后的每次请求中都被注入
    turn = await services.chat.prepare(chat_body("在干嘛", metadata=QQ_PRIVATE), None)
    assert "小明喜欢草莓" in turn.upstream_body["messages"][0]["content"]


async def test_explicit_remember_goes_to_profile(client: httpx.AsyncClient, services) -> None:  # type: ignore[no-untyped-def]
    await client.post("/v1/chat/completions", json=chat_body("记住，我对花生过敏", metadata=QQ_PRIVATE))
    await services.worker.run_until_idle()
    entries = services.profiles.get("owner").entries
    assert len(entries) == 1 and "花生过敏" in entries[0].text


async def test_extraction_failure_retries_then_gives_up(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    fake.extract_fn = lambda messages: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]

    def bad_text(body: dict[str, Any]) -> str:
        return "这不是 JSON" if "response_format" in body else "好的"

    fake.chat_text = bad_text  # type: ignore[method-assign]
    await client.post("/v1/chat/completions", json=chat_body("随便聊聊", metadata=QQ_PRIVATE))
    for _ in range(3):
        await services.worker.process_batch()
    log = services.rawlog.list()[0]
    assert log["processed"] == FAILED and log["attempts"] == 3 and "JSON" in log["error"]
    # 管理接口可以把失败的日志重新排队
    assert services.rawlog.reprocess() == 1
    assert services.rawlog.list()[0]["processed"] == PENDING


async def test_worker_is_preempted_by_chat(make_settings) -> None:  # type: ignore[no-untyped-def]
    from her_brain.services import Services, Transports

    slow = FakeLLM(chunk_delay=0.2)
    svc = Services(make_settings(), transports=Transports(upstream=slow.transport()))
    try:
        svc.rawlog.insert(svc.rawlog.new(user_id="owner", channel="qq", scene="private", privacy="private",
                                         user_text="我明天要去面试", reply_text="加油"))
        job = asyncio.create_task(svc.worker.process_batch())
        await asyncio.sleep(0.05)
        svc.activity.begin()  # 来了一个对话
        assert await job == "preempted"
        svc.activity.end()
        assert svc.rawlog.list()[0]["processed"] == PENDING  # 稍后重试
        assert svc.activity.preemptions == 1
        # 空闲时间不够时不运行
        assert await svc.worker.tick() == "busy"
    finally:
        await svc.close()


async def test_activity_tracker_basics() -> None:
    tracker = ActivityTracker()

    async def work() -> str:
        await asyncio.sleep(10)
        return "done"

    task = asyncio.create_task(tracker.run_preemptible(work()))
    await asyncio.sleep(0.01)
    tracker.begin()
    with pytest.raises(Preempted):
        await task
    with pytest.raises(Preempted):  # 有对话进行中时直接拒绝启动
        await tracker.run_preemptible(work())
    tracker.end()
    assert tracker.idle_for() >= 0


async def test_tidy_expires_old_plans(services) -> None:  # type: ignore[no-untyped-def]
    services.profiles.add_entry("owner", {"text": "上个月的考试", "kind": "plan", "event_date": "2020-01-01"})
    services.profiles.add_entry("owner", {"text": "喜欢猫"})
    result = await services.worker.tidy()
    assert result["expired_profile_entries"] == 1
    assert [e.text for e in services.profiles.get("owner").entries] == ["喜欢猫"]


async def test_rebuild_from_raw_logs(client: httpx.AsyncClient, services) -> None:  # type: ignore[no-untyped-def]
    await client.post("/v1/chat/completions", json=chat_body("我最喜欢草莓", metadata=QQ_PRIVATE))
    await services.worker.run_until_idle()
    first = services.episodic.list_for_user("owner")
    r = await client.post("/admin/users/owner/rebuild", json={"run_now": True})
    assert r.json()["soft_deleted"] == 1 and r.json()["requeued_logs"] == 1
    again = services.episodic.list_for_user("owner")
    assert [m.text for m in again] == [m.text for m in first] and again[0].memory_id != first[0].memory_id


async def test_admin_memory_endpoints(client: httpx.AsyncClient, services) -> None:  # type: ignore[no-untyped-def]
    r = await client.post("/admin/users/owner/memories", json={"text": "小明的生日是三月一日", "privacy": "private"})
    mid = r.json()["memory_id"]
    r = await client.patch(f"/admin/memories/{mid}", json={"text": "小明的生日是三月二日"})
    assert r.json()["text"] == "小明的生日是三月二日"
    events = (await client.get("/admin/memory-events", params={"memory_id": mid})).json()
    upd = next(e for e in events if e["action"] == "UPDATE")
    r = await client.post(f"/admin/memory-events/{upd['event_id']}/revert")
    assert r.json()["memory"]["text"] == "小明的生日是三月一日"
    assert (await client.post(f"/admin/memory-events/{upd['event_id']}/revert")).status_code == 409
    r = await client.get("/admin/users/owner/memories", params={"q": "生日", "semantic": True})
    assert r.json()[0]["memory_id"] == mid
    since = time.time()
    await client.delete(f"/admin/memories/{mid}")
    r = await client.post("/admin/users/owner/memories/rollback", json={"since": since})
    assert r.json()["restored"] == [mid] and not services.episodic.get(mid).deleted


async def test_admin_profile_and_identity_endpoints(client: httpx.AsyncClient) -> None:
    r = await client.post("/admin/users/owner/profile/entries", json={"text": "是程序员", "slot": "职业"})
    assert r.json()["version"] == 1
    await client.post("/admin/users/owner/profile/entries", json={"text": "是老师", "slot": "职业"})
    r = await client.post("/admin/users/owner/profile/rollback", json={"version": 1})
    assert [e["text"] for e in r.json()["entries"]] == ["是程序员"]
    assert len((await client.get("/admin/users/owner/profile/versions")).json()) == 3
    r = await client.post("/admin/identities", json={"platform": "discord", "external_id": "42", "user_id": "owner"})
    assert r.json()["user_id"] == "owner"
    users = (await client.get("/admin/users")).json()
    owner = next(u for u in users if u["user_id"] == "owner")
    assert "discord:42" in owner["bindings"]
    r = await client.post("/admin/persona/reload")
    assert r.json()["ok"]


async def test_worker_marks_guests_skipped_when_disabled(make_settings, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    from her_brain.services import Services, Transports

    svc = Services(make_settings(memory={"remember_guests": False}), transports=Transports(upstream=fake.transport()))
    try:
        turn = await svc.chat.prepare(chat_body("我叫阿强", metadata={"channel": "qq", "external_user_id": "999"}), None)
        svc.activity.begin()
        svc.chat.finish(turn, "你好阿强", interrupted=False, error=None)
        assert svc.rawlog.list()[0]["processed"] == 2
        assert await svc.worker.process_batch() == "empty"
    finally:
        await svc.close()


def test_attribute_sources_and_explicit_guard() -> None:
    from her_brain.memory.models import Fact
    from her_brain.memory.rawlog import RawLog
    from her_brain.worker.extractor import attribute_sources

    logs = [
        RawLog("l1", "u", "qq", "private", "private", "我明天要去面试，有点紧张", "加油呀", 1.0),
        RawLog("l2", "u", "qq", "private", "private", "记住，我对花生过敏", "好的我记住了", 2.0),
        RawLog("l3", "u", "qq", "private", "private", "你喜欢什么", "我最近在学画画", 3.0),
    ]
    facts = [
        Fact(text="小明在 2026-11-05 有一场面试", kind="plan", explicit=True),  # 模型误标
        Fact(text="小E知道小明对花生过敏", kind="preference", explicit=True),
        Fact(text="小E最近在学画画", about="self"),
        Fact(text="完全无关的一句话XYZ"),
    ]
    out = attribute_sources(facts, logs)
    assert out[0][1] == ["l1"] and out[0][0].explicit is False
    assert out[1][1] == ["l2"] and out[1][0].explicit is True
    assert out[2][1] == ["l3"]  # 关于她自己的事实按她的回复归属
    assert out[3][1] == ["l1", "l2", "l3"]  # 找不到归属时退回整批


async def test_sloppy_explicit_labels_do_not_bypass_confirmation(client: httpx.AsyncClient, services, fake: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    """回归：真实模型把同一段对话里的所有事实都标成 explicit，曾让面试计划直接进入核心档案。"""
    from her_brain.devtools.fake_llm import default_extract

    def sloppy(messages):  # type: ignore[no-untyped-def]
        facts = default_extract(messages)
        for f in facts:
            f["explicit"] = True
            if "过敏" in f["text"]:
                f["text"] = "小E知道" + f["text"]
        return facts

    fake.extract_fn = sloppy
    await client.post("/v1/chat/completions", json=chat_body("我明天要去面试，有点紧张", metadata=QQ_PRIVATE))
    await client.post("/v1/chat/completions", json=chat_body("记住，我对花生过敏", metadata=QQ_PRIVATE))
    await services.worker.run_until_idle()
    texts = [e.text for e in services.profiles.get("owner").entries]
    assert any("花生过敏" in t for t in texts)
    assert not any("面试" in t for t in texts)
    interview = next(m for m in services.episodic.list_for_user("owner") if "面试" in m.text)
    log_ids = {log["user_text"]: log["log_id"] for log in services.rawlog.list(user_id="owner")}
    assert interview.source_log_ids == [log_ids["我明天要去面试，有点紧张"]]  # 来源精确到那一句


def test_extractor_parsing() -> None:
    from her_brain.worker.extractor import FACT_SCHEMA, parse_facts

    text = '```json\n{"facts": [{"text": "小明喜欢猫", "about": "user", "kind": "preference", "importance": "9", "explicit": "true"}, {"text": "短"}, "小明住在杭州"]}\n```'
    facts = parse_facts(text)
    assert [f.text for f in facts] == ["小明喜欢猫", "小明住在杭州"]
    assert facts[0].importance == 5 and facts[0].explicit and facts[1].kind == "other"
    assert parse_facts("") == [] and parse_facts('前缀 {"facts": []} 后缀') == []
    assert json.dumps(FACT_SCHEMA)  # 可序列化，可直接作为 response_format 发给 llama-server
