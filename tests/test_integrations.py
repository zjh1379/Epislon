"""入口集成代码的测试：AstrBot 插件（用桩模块替代 astrbot）、枫云切句辅助函数。"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from her_brain.config import DEFAULT_PLATFORM_ALIASES
from her_brain.identity.metadata import parse_request_meta

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------- AstrBot 插件


@dataclass
class ProviderRequest:
    prompt: str = ""
    system_prompt: str = ""


class FakeEvent:
    def __init__(self, platform: str, private: bool, sender: str, name: str, group: str = "", group_name: str = ""):
        self._p, self._private, self._sender, self._name, self._group = platform, private, sender, name, group
        self.unified_msg_origin = f"{platform}:x"
        self.message_obj = types.SimpleNamespace(group=types.SimpleNamespace(group_name=group_name) if group else None)

    def get_platform_name(self) -> str:
        return self._p

    def get_group_id(self) -> str:
        return self._group

    def is_private_chat(self) -> bool:
        return self._private

    def get_sender_id(self) -> str:
        return self._sender

    def get_sender_name(self) -> str:
        return self._name


def _install_astrbot_stubs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = dict  # type: ignore[attr-defined]
    api.logger = types.SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None)  # type: ignore[attr-defined]
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = FakeEvent  # type: ignore[attr-defined]
    event.filter = types.SimpleNamespace(on_llm_request=lambda **kw: (lambda fn: fn))  # type: ignore[attr-defined]
    provider = types.ModuleType("astrbot.api.provider")
    provider.ProviderRequest = ProviderRequest  # type: ignore[attr-defined]
    star = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context: Any) -> None:
            self.context = context

    star.Star = Star  # type: ignore[attr-defined]
    star.Context = object  # type: ignore[attr-defined]
    star.register = lambda *a, **k: (lambda cls: cls)  # type: ignore[attr-defined]
    for name, mod in {"astrbot": types.ModuleType("astrbot"), "astrbot.api": api, "astrbot.api.event": event,
                      "astrbot.api.provider": provider, "astrbot.api.star": star}.items():
        monkeypatch.setitem(sys.modules, name, mod)


async def test_astrbot_plugin_marker_roundtrip(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _install_astrbot_stubs(monkeypatch)
    plugin_mod = _load(ROOT / "integrations/astrbot_plugin_her_bridge/main.py", "her_bridge_plugin")
    plugin = plugin_mod.HerBridge(context=None, config={"platform_aliases": '{"telegram": "tg"}'})

    req = ProviderRequest(prompt="大家好", system_prompt="")
    await plugin.attach_meta(FakeEvent("aiocqhttp", False, "10001", "小明", "777", "测试群"), req)
    assert req.system_prompt.startswith("<her-meta>")
    meta, messages = parse_request_meta(
        {"messages": [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": req.prompt}]},
        None, dict(DEFAULT_PLATFORM_ALIASES),
    )
    assert (meta.channel, meta.scene, meta.external_user_id, meta.group_id, meta.group_name) == (
        "qq", "group", "10001", "777", "测试群")
    assert messages[0]["content"] == ""

    req = ProviderRequest(prompt="hi", system_prompt="原有提示词")
    await plugin.attach_meta(FakeEvent("telegram", True, "55", "Bob"), req)
    meta, messages = parse_request_meta({"messages": [{"role": "system", "content": req.system_prompt}]}, None, {})
    assert (meta.channel, meta.scene, meta.platform) == ("tg", "private", "telegram")
    assert messages[0]["content"] == "原有提示词"


async def test_astrbot_plugin_respects_provider_filter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _install_astrbot_stubs(monkeypatch)
    plugin_mod = _load(ROOT / "integrations/astrbot_plugin_her_bridge/main.py", "her_bridge_plugin2")
    provider = types.SimpleNamespace(meta=lambda: types.SimpleNamespace(id="cloud-gpt"))
    ctx = types.SimpleNamespace(get_using_provider=lambda umo=None: provider)
    plugin = plugin_mod.HerBridge(context=ctx, config={"enabled_provider_ids": ["her-brain"]})
    req = ProviderRequest(prompt="hi", system_prompt="")
    await plugin.attach_meta(FakeEvent("aiocqhttp", True, "1", "a"), req)
    assert req.system_prompt == ""  # 不是大脑代理的提供商：不附加，避免泄露 QQ 号


# ---------------------------------------------------------------------- 枫云辅助函数


def test_fengyun_split_stream() -> None:
    bridge = _load(ROOT / "integrations/fengyun/her_bridge.py", "her_bridge_fengyun")
    tokens = ["欢迎", "回来！", "面试", "怎么样？", "我一直在想", "你会不会紧张，", "要是有什么不开心的", "可以跟我说说哦"]
    out = list(bridge.split_stream(tokens))
    assert out[0] == "欢迎回来！"
    assert out[1] == "面试怎么样？"
    assert "".join(out) == "".join(tokens)
    # 首句满 6 个字后在逗号处尽早切出（太短的片段如“嗯嗯，”不单独合成）
    assert list(bridge.split_stream(["嗯嗯，我知道啦，", "那我们"]))[0] == "嗯嗯，我知道啦，"
    assert list(bridge.split_stream(["我想想看啊，这个问题", "有点难"]))[0] == "我想想看啊，"
    meta = bridge.her_metadata("小明", "开启")
    assert meta["speaker_verified"] is True and meta["channel"] == "voice"
    assert bridge.her_metadata("小明", "关闭")["speaker_verified"] is False


def test_fengyun_patch_is_well_formed() -> None:
    patch = (ROOT / "integrations/fengyun/fengyun_her_brain.patch").read_bytes()
    assert b'"metadata": {"mode": "raw"}' in patch
    assert b'"speaker_verified": voiceprint_switch' in patch
    assert b"\r\n" in patch  # 枫云源码为 CRLF，补丁需保留
