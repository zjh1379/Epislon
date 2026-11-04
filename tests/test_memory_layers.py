from __future__ import annotations

import time
from datetime import timedelta

import pytest

from her_brain.identity.models import PRIVATE, PUBLIC
from her_brain.llm.think import ThinkStripper, strip_think
from her_brain.memory.models import Fact
from her_brain.timeutil import get_tz, local_date

# ---------------------------------------------------------------------- think 过滤


@pytest.mark.parametrize(
    "pieces, expected",
    [
        (["<think>想想</think>\n\n你好"], "你好"),
        (["<thi", "nk>嗯……", "</th", "ink>\n", "你好呀"], "你好呀"),
        (["你好", "<", "b>不是标签"], "你好<b>不是标签"),
        (["开头<think>中间</think>结尾"], "开头结尾"),
        (["<think>没有结束"], ""),
        (["a < b 并且 c > d"], "a < b 并且 c > d"),
    ],
)
def test_think_stripper(pieces: list[str], expected: str) -> None:
    s = ThinkStripper()
    out = "".join(s.feed(p) for p in pieces) + s.flush()
    assert out == expected


def test_strip_think_helper() -> None:
    assert strip_think("<think>\n</think>\n\n{\"a\": 1}") == '{"a": 1}'


# ---------------------------------------------------------------------- 核心档案


def test_profile_versions_and_rollback(services) -> None:  # type: ignore[no-untyped-def]
    p = services.profiles
    p.add_entry("owner", {"text": "喜欢猫", "privacy": PUBLIC})
    p.add_entry("owner", {"text": "在杭州工作", "slot": "所在城市", "value": "杭州"})
    _, entry = p.add_entry("owner", {"text": "搬到了上海", "slot": "所在城市", "value": "上海"})
    prof = p.get("owner")
    assert prof.version == 3
    assert [e.text for e in prof.entries] == ["喜欢猫", "搬到了上海"]  # 同一 slot 被替换
    p.delete_entry("owner", entry.id)
    assert p.get("owner").version == 4

    rolled = p.rollback("owner", 2)
    assert rolled.version == 5
    assert sorted(e.text for e in rolled.entries) == ["喜欢猫", "在杭州工作"]
    reasons = [v["reason"] for v in p.versions("owner")]
    assert reasons[0] == "回滚到 v2"
    p.invalidate()
    assert p.get("owner").version == 5  # 从数据库读回


def test_profile_render_filters_privacy(services) -> None:  # type: ignore[no-untyped-def]
    p = services.profiles
    p.add_entry("owner", {"text": "公开的爱好：打篮球", "privacy": PUBLIC})
    p.add_entry("owner", {"text": "私密：最近在看心理医生", "privacy": PRIVATE})
    assert p.render("owner", {PUBLIC}) == ["公开的爱好：打篮球"]
    assert len(p.render("owner", {PUBLIC, PRIVATE})) == 2


def _fact(text: str, **kw) -> Fact:  # type: ignore[no-untyped-def]
    return Fact(text=text, **kw)


def test_promotion_needs_independent_confirmations(services) -> None:  # type: ignore[no-untyped-def]
    pr = services.promoter
    f = _fact("小明住在杭州", kind="attribute", slot="城市", value="杭州")
    r1 = pr.observe("owner", f, PRIVATE, ["log1"])
    assert r1 and not r1.promoted and r1.confirmations == 1
    assert services.profiles.get("owner").entries == []
    # 同一批日志重复处理：不计数
    r_dup = pr.observe("owner", f, PRIVATE, ["log1"])
    assert r_dup and not r_dup.counted and r_dup.confirmations == 1
    # 另一次对话再次提到（不同说法）：确认 2 次，晋升
    r2 = pr.observe("owner", _fact("小明现在住杭州", kind="attribute", slot="居住地", value="杭州"), PRIVATE, ["log2"])
    assert r2 and r2.promoted and r2.superseded_slot == "所在城市"
    entries = services.profiles.get("owner").entries
    assert len(entries) == 1 and entries[0].slot == "所在城市" and entries[0].confirmations == 2


def test_single_wrong_extraction_does_not_overwrite(services) -> None:  # type: ignore[no-untyped-def]
    pr = services.promoter
    for i in range(2):
        pr.observe("owner", _fact("小明是学生", kind="attribute", slot="职业", value="学生"), PRIVATE, [f"a{i}"])
    assert services.profiles.get("owner").entries[0].value == "学生"
    # 一次（可能是错误的）提取说他是医生：只成为候选，档案不变
    pr.observe("owner", _fact("小明是医生", kind="attribute", slot="职业", value="医生"), PRIVATE, ["b0"])
    assert services.profiles.get("owner").entries[0].value == "学生"
    # 被再次确认后才替换，旧值保留在历史版本里
    pr.observe("owner", _fact("小明当医生了", kind="attribute", slot="职业", value="医生"), PRIVATE, ["b1"])
    prof = services.profiles.get("owner")
    assert [e.value for e in prof.entries] == ["医生"]
    old = services.profiles.get_version("owner", 1)
    assert old and old[0]["value"] == "学生"
    statuses = {c["value"]: c["status"] for c in pr.list("owner")}
    assert statuses == {"学生": "superseded", "医生": "promoted"}


def test_explicit_remember_promotes_immediately_and_privacy_merges(services) -> None:  # type: ignore[no-untyped-def]
    pr = services.promoter
    r = pr.observe("owner", _fact("小明对花生过敏", kind="attribute", explicit=True), PRIVATE, ["x1"])
    assert r and r.promoted
    entry = services.profiles.get("owner").entries[0]
    assert entry.privacy == PRIVATE
    # 之后在群聊（公开）里又提到：条目改为公开
    pr.observe("owner", _fact("小明对花生过敏", kind="attribute"), PUBLIC, ["x2"])
    assert services.profiles.get("owner").entries[0].privacy == PUBLIC


def test_events_are_not_profile_candidates(services) -> None:  # type: ignore[no-untyped-def]
    assert services.promoter.observe("owner", _fact("小明昨天去爬山了", kind="event"), PRIVATE, ["e"]) is None


# ---------------------------------------------------------------------- 情节记忆


async def test_remember_dedupes_and_tracks_events(services) -> None:  # type: ignore[no-untyped-def]
    ep = services.episodic
    status, rec = await ep.remember("owner", _fact("小明喜欢吃草莓", kind="preference"), PRIVATE, ["l1"])
    assert status == "added"
    status2, rec2 = await ep.remember("owner", _fact("小明喜欢吃草莓。", kind="preference", importance=4), PUBLIC, ["l2"])
    assert status2 == "merged" and rec2.memory_id == rec.memory_id
    assert rec2.mentions == 2 and rec2.privacy == PUBLIC and rec2.importance == 4.5
    status3, _ = await ep.remember("owner", _fact("小明喜欢吃草莓", kind="preference"), PRIVATE, ["l2"])
    assert status3 == "skipped"
    actions = [e["action"] for e in ep.events(user_id="owner")]
    assert actions == ["UPDATE", "ADD"]


async def test_search_respects_privacy_and_deletion(services) -> None:  # type: ignore[no-untyped-def]
    ep = services.episodic
    _, secret = await ep.remember("owner", _fact("小明最近失眠很严重"), PRIVATE, ["p"])
    await ep.remember("owner", _fact("小明周末喜欢去爬山"), PUBLIC, ["q"])
    hits = await ep.search(["owner"], "最近失眠", {PUBLIC, PRIVATE}, 5)
    assert hits and hits[0].record.memory_id == secret.memory_id
    assert not [h for h in await ep.search(["owner"], "最近失眠", {PUBLIC}, 5) if h.record.privacy == PRIVATE]
    await ep.delete(secret.memory_id)
    assert not [h for h in await ep.search(["owner"], "最近失眠", {PUBLIC, PRIVATE}, 5)
                if h.record.memory_id == secret.memory_id]
    await ep.restore(secret.memory_id)
    assert (await ep.search(["owner"], "最近失眠", {PUBLIC, PRIVATE}, 5))[0].record.memory_id == secret.memory_id


async def test_agenda_surfaces_dated_plans(services) -> None:  # type: ignore[no-untyped-def]
    ep = services.episodic
    tz = get_tz("Asia/Shanghai")
    today = local_date(time.time(), tz)
    await ep.remember("owner", _fact("小明有一场面试", kind="plan", date=today.isoformat(), importance=4), PRIVATE, ["a"])
    await ep.remember("owner", _fact("小明下个月去旅游", kind="plan",
                                     date=(today + timedelta(days=30)).isoformat()), PRIVATE, ["b"])
    items = ep.agenda(["owner"], {PUBLIC, PRIVATE}, time.time(), 2, 2, 3)
    assert [i.record.text for i in items] == ["小明有一场面试"]
    assert ep.agenda(["owner"], {PUBLIC}, time.time(), 2, 2, 3) == []


async def test_revert_and_rollback(services) -> None:  # type: ignore[no-untyped-def]
    ep = services.episodic
    _, a = await ep.remember("owner", _fact("小明养了一只叫团子的猫"), PRIVATE, ["1"])
    t_mid = time.time()
    time.sleep(0.01)
    await ep.edit(a.memory_id, {"text": "小明养了一只叫汤圆的狗"})  # 一次错误更新
    _, b = await ep.remember("owner", _fact("小明在学吉他"), PRIVATE, ["2"])

    # 撤销那次错误更新
    events = ep.events(memory_id=a.memory_id)
    update_event = next(e for e in events if e["action"] == "UPDATE")
    res = await ep.revert_event(update_event["event_id"])
    assert res["memory"]["text"] == "小明养了一只叫团子的猫"
    with pytest.raises(ValueError):
        await ep.revert_event(update_event["event_id"])

    # 再做一次改动后，整体回滚到 t_mid：a 恢复原文，b（之后才新增）被软删除
    await ep.edit(a.memory_id, {"text": "又改坏了"})
    result = await ep.rollback_user("owner", t_mid)
    assert set(result["restored"]) == {a.memory_id, b.memory_id}
    assert ep.get(a.memory_id).text == "小明养了一只叫团子的猫"
    assert ep.get(b.memory_id).deleted


async def test_supersede_slot(services) -> None:  # type: ignore[no-untyped-def]
    ep = services.episodic
    _, old = await ep.remember("owner", _fact("小明在杭州工作", slot="所在城市", value="杭州"), PRIVATE, ["1"])
    _, new = await ep.remember("owner", _fact("小明搬到上海了", slot="所在城市", value="上海"), PRIVATE, ["2"])
    assert ep.supersede_slot("owner", "所在城市", "小明搬到上海了", "上海") == 1
    assert ep.get(old.memory_id).superseded and not ep.get(new.memory_id).superseded


async def test_lite_engine_with_embeddings(make_settings, fake) -> None:  # type: ignore[no-untyped-def]
    from her_brain.services import Services, Transports

    svc = Services(make_settings(embedding={"enabled": True, "base_url": "http://fake/v1", "dims": 256}),
                   transports=Transports(upstream=fake.transport(), embedding=fake.transport()))
    try:
        _, rec = await svc.episodic.remember("owner", _fact("小明的猫叫团子"), PRIVATE, ["1"])
        row = svc.db.query_one("SELECT embedding FROM episodic_memory WHERE memory_id = ?", (rec.memory_id,))
        assert row["embedding"]
        hits = await svc.episodic.search(["owner"], "小明的猫叫什么名字", {PRIVATE}, 3)
        assert hits and hits[0].record.memory_id == rec.memory_id
    finally:
        await svc.close()
