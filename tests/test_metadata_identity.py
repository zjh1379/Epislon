from __future__ import annotations

from her_brain.config import ApiKeyConfig, DEFAULT_PLATFORM_ALIASES
from her_brain.identity.metadata import parse_bool, parse_request_meta, strip_soft_switches
from her_brain.identity.models import PRIVATE, PUBLIC, RequestMeta

ALIASES = dict(DEFAULT_PLATFORM_ALIASES)


def test_metadata_field_wins_and_is_not_in_messages() -> None:
    body = {
        "messages": [{"role": "user", "content": "你好"}],
        "metadata": {"channel": "qq", "scene": "group", "external_user_id": 123, "group_id": "789",
                     "speaker_verified": "false"},
    }
    meta, messages = parse_request_meta(body, None, ALIASES)
    assert meta.channel == "qq"
    assert meta.scene == "group"
    assert meta.external_user_id == "123"
    assert meta.group_id == "789"
    assert meta.speaker_verified is False
    assert messages == body["messages"]
    assert meta.sources == ["metadata"]


def test_marker_in_system_prompt_is_parsed_and_stripped() -> None:
    sys_text = '人格留空\n<her-meta>{"platform": "aiocqhttp", "scene": "private", "external_user_id": "42", "user_name": "阿明"}</her-meta>'
    body = {"messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": "在吗"}]}
    meta, messages = parse_request_meta(body, None, ALIASES)
    assert meta.channel == "qq"  # aiocqhttp 映射为 qq
    assert meta.platform == "aiocqhttp"
    assert meta.external_user_id == "42"
    assert meta.user_name == "阿明"
    assert "her-meta" not in messages[0]["content"]
    assert messages[0]["content"] == "人格留空"


def test_short_marker_in_user_message_and_history() -> None:
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "[[her:channel=discord;uid=9;scene=group;gid=5]]早"}]},
            {"role": "assistant", "content": "早呀"},
            {"role": "user", "content": "[[her:channel=discord;uid=9;scene=group;gid=5]]吃了吗"},
        ]
    }
    meta, messages = parse_request_meta(body, None, ALIASES)
    assert (meta.channel, meta.scene, meta.external_user_id, meta.group_id) == ("discord", "group", "9", "5")
    assert messages[0]["content"][0]["text"] == "早"
    assert messages[2]["content"] == "吃了吗"


def test_api_key_defaults_and_override_order() -> None:
    key = ApiKeyConfig(key="k", channel="voice", scene="voice")
    meta, _ = parse_request_meta({"messages": [{"role": "user", "content": "hi"}]}, key, ALIASES)
    assert (meta.channel, meta.scene) == ("voice", "voice")
    meta, _ = parse_request_meta(
        {"messages": [{"role": "user", "content": "hi"}], "metadata": {"speaker_verified": True}}, key, ALIASES
    )
    assert meta.speaker_verified is True and meta.channel == "voice"


def test_scene_inferred_when_missing() -> None:
    meta, _ = parse_request_meta({"messages": [{"role": "user", "content": "x"}], "metadata": {"channel": "qq", "group_id": "1"}}, None, ALIASES)
    assert meta.scene == "group"
    meta, _ = parse_request_meta({"messages": [{"role": "user", "content": "x"}], "metadata": {"channel": "voice"}}, None, ALIASES)
    assert meta.scene == "voice"
    meta, _ = parse_request_meta({"messages": [{"role": "user", "content": "x"}]}, None, ALIASES)
    assert meta.scene == "private"


def test_soft_switch_and_bools() -> None:
    assert strip_soft_switches("你是助手/no_think") == "你是助手"
    assert strip_soft_switches("/no_think") == ""
    assert strip_soft_switches("路径 a/no_thinking 不动") == "路径 a/no_thinking 不动"
    assert parse_bool("开启") is True and parse_bool("关闭") is False and parse_bool(None) is None


ASTRBOT_USER_CONTENT = [
    {"type": "text", "text": "我明天要去面试"},
    {"type": "text", "text": "<system_reminder>User ID: 10001, Nickname: 小明\n"
                             "Current datetime: 2026-11-05 10:00 (CST), Weekday: Thursday</system_reminder>"},
]


def test_astrbot_identifier_fallback_is_conservative() -> None:
    key = ApiKeyConfig(key="k", channel="qq")
    body = {"messages": [{"role": "user", "content": ASTRBOT_USER_CONTENT}]}
    meta, _ = parse_request_meta(body, key, ALIASES)
    assert (meta.external_user_id, meta.user_name) == ("10001", "小明")
    assert meta.sources == ["api_key", "astrbot_identifier"]
    assert meta.scene_uncertain  # 分不清私聊群聊

    group = [dict(ASTRBOT_USER_CONTENT[0]),
             {"type": "text", "text": "<system_reminder>User ID: 10001, Nickname: 小明\nGroup name: 测试群</system_reminder>"}]
    meta, _ = parse_request_meta({"messages": [{"role": "user", "content": group}]}, key, ALIASES)
    assert meta.scene == "group" and meta.group_name == "测试群" and not meta.scene_uncertain

    marker = '<her-meta>{"platform": "aiocqhttp", "scene": "private", "external_user_id": "42"}</her-meta>'
    body = {"messages": [{"role": "system", "content": marker}, {"role": "user", "content": ASTRBOT_USER_CONTENT}]}
    meta, _ = parse_request_meta(body, key, ALIASES)
    assert meta.external_user_id == "42" and not meta.scene_uncertain  # 插件标记优先


def test_strip_reminders_from_user_text() -> None:
    from her_brain.prompt.builder import last_user_text

    assert last_user_text([{"role": "user", "content": ASTRBOT_USER_CONTENT}]) == "我明天要去面试"


def test_identity_resolution(services) -> None:  # type: ignore[no-untyped-def]
    r = services.resolver
    owner = r.resolve(RequestMeta(channel="qq", external_user_id="10001"))
    assert owner.user_id == "owner" and owner.is_owner
    voice = r.resolve(RequestMeta(channel="voice", speaker_verified=True))
    assert voice.user_id == "owner"
    anon = r.resolve(RequestMeta(channel="voice", external_user_id="owner", speaker_verified=False))
    assert anon.anonymous and anon.user_id == "anon:voice"
    guest = r.resolve(RequestMeta(channel="qq", external_user_id="555", user_name="路人甲"))
    assert guest.user_id == "qq:555" and guest.role == "guest" and guest.display_name == "路人甲"
    renamed = r.resolve(RequestMeta(channel="qq", external_user_id="555", user_name="路人乙"))
    assert renamed.display_name == "路人乙"
    assert r.resolve(RequestMeta(channel="qq")).anonymous


def test_scene_access_levels(services) -> None:  # type: ignore[no-untyped-def]
    r = services.resolver
    meta = RequestMeta(channel="qq", scene="group", external_user_id="10001", group_id="1")
    scene = r.build_scene(meta, r.resolve(meta))
    assert scene.access == PUBLIC and scene.write_privacy == PUBLIC
    meta = RequestMeta(channel="qq", scene="private", external_user_id="10001")
    scene = r.build_scene(meta, r.resolve(meta))
    assert scene.access == PRIVATE and scene.write_privacy == PRIVATE
    meta = RequestMeta(channel="voice", scene="voice", speaker_verified=False)
    scene = r.build_scene(meta, r.resolve(meta))
    assert scene.access == PUBLIC


def test_trust_unverified_voice(make_settings, fake) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    from her_brain.services import Services, Transports

    svc = Services(make_settings(memory={"trust_unverified_voice": True}), transports=Transports(upstream=fake.transport()))
    try:
        ident = svc.resolver.resolve(RequestMeta(channel="voice", speaker_verified=None))
        assert ident.user_id == "owner"
    finally:
        asyncio.run(svc.close())


def test_link_identity_moves_logs(services) -> None:  # type: ignore[no-untyped-def]
    r = services.resolver
    guest = r.resolve(RequestMeta(channel="discord", external_user_id="d1"))
    services.rawlog.insert(services.rawlog.new(user_id=guest.user_id, channel="discord", scene="private",
                                               privacy="private", user_text="hi", reply_text="hello"))
    result = r.link("discord", "d1", "owner", move_logs=True)
    assert result["previous_user_id"] == "discord:d1" and result["moved_logs"] == 1
    assert r.resolve(RequestMeta(channel="discord", external_user_id="d1")).user_id == "owner"
    assert services.rawlog.list(user_id="owner")[0]["user_text"] == "hi"
