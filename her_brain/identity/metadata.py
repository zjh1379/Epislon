"""解析入口传来的身份元数据，并从消息中剥离，保证不会传给模型。

来源优先级（后者覆盖前者）：
1. API Key 对应的入口默认值（入口无法发送自定义字段时的退路）
2. 消息中的标记 ``<her-meta>{...json...}</her-meta>``（AstrBot 插件写在系统提示词里）
   兼容简写 ``[[her:channel=qq;scene=group;uid=123]]``
3. 请求体顶层的 ``metadata`` 字段（枫云通过 extra_body 发送）
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from ..config import ApiKeyConfig
from .models import RequestMeta

MARKER_RE = re.compile(r"<her-meta>\s*(\{.*?\})\s*</her-meta>", re.S)
SHORT_MARKER_RE = re.compile(r"\[\[her:([^\]]*)\]\]")
# 入口在系统提示词里附加的思考开关（例如枫云会拼上 /no_think），由代理统一控制
SOFT_SWITCH_RE = re.compile(r"(?<![A-Za-z0-9_/])/(?:no_think|think)(?![A-Za-z0-9_])")

SCENE_ALIASES = {
    "private": "private",
    "friend": "private",
    "dm": "private",
    "direct": "private",
    "c2c": "private",
    "group": "group",
    "guild": "group",
    "channel": "group",
    "room": "group",
    "voice": "voice",
    "speech": "voice",
}

SHORT_KEYS = {"uid": "external_user_id", "gid": "group_id", "name": "user_name", "verified": "speaker_verified"}

TRUE_VALUES = {"1", "true", "yes", "y", "on", "是", "开启", "verified"}
FALSE_VALUES = {"0", "false", "no", "n", "off", "否", "关闭", ""}


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def _parse_short_marker(body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in re.split(r"[;,]", body):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        out[SHORT_KEYS.get(k, k)] = v.strip()
    return out


def _strip_markers_from_text(text: str, found: list[dict[str, Any]]) -> str:
    def _json_repl(m: re.Match[str]) -> str:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                found.append(data)
        except ValueError:
            pass
        return ""

    def _short_repl(m: re.Match[str]) -> str:
        found.append(_parse_short_marker(m.group(1)))
        return ""

    if "<her-meta>" in text:
        text = MARKER_RE.sub(_json_repl, text)
    if "[[her:" in text:
        text = SHORT_MARKER_RE.sub(_short_repl, text)
    return text


def strip_markers(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回（去掉标记后的消息副本，按出现顺序找到的标记内容）。"""
    found: list[dict[str, Any]] = []
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        msg = copy.copy(msg)
        content = msg.get("content")
        if isinstance(content, str):
            new = _strip_markers_from_text(content, found)
            if new is not content and new != content:
                msg["content"] = new.strip() if msg.get("role") == "system" else new.strip("\n")
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part = dict(part)
                    part["text"] = _strip_markers_from_text(part["text"], found)
                parts.append(part)
            msg["content"] = parts
        cleaned.append(msg)
    return cleaned, found


def strip_soft_switches(text: str) -> str:
    return SOFT_SWITCH_RE.sub("", text).strip()


def _apply(meta: RequestMeta, data: dict[str, Any], aliases: dict[str, str], source: str) -> None:
    if not data:
        return
    meta.sources.append(source)
    platform = data.get("platform")
    if platform:
        meta.platform = str(platform)
    channel = data.get("channel") or (aliases.get(str(platform)) if platform else None) or platform
    if channel:
        meta.channel = aliases.get(str(channel), str(channel))
    scene = data.get("scene")
    if scene:
        meta.scene = SCENE_ALIASES.get(str(scene).strip().lower(), meta.scene)
    for key in ("external_user_id", "user_id", "sender_id"):
        if data.get(key) not in (None, ""):
            meta.external_user_id = str(data[key])
            break
    for key in ("user_name", "sender_name", "nickname"):
        if data.get(key):
            meta.user_name = str(data[key])
            break
    if "group_id" in data:
        meta.group_id = str(data["group_id"]) if data["group_id"] not in (None, "") else None
    if data.get("group_name"):
        meta.group_name = str(data["group_name"])
    if "speaker_verified" in data:
        meta.speaker_verified = parse_bool(data["speaker_verified"])
    mode = data.get("mode")
    if mode in ("chat", "raw"):
        meta.mode = mode
    if "no_memory" in data:
        meta.no_memory = bool(parse_bool(data["no_memory"]))
    if "no_log" in data:
        meta.no_log = bool(parse_bool(data["no_log"]))


def parse_request_meta(
    body: dict[str, Any],
    key_cfg: ApiKeyConfig | None,
    aliases: dict[str, str],
) -> tuple[RequestMeta, list[dict[str, Any]]]:
    meta = RequestMeta()
    if key_cfg is not None:
        _apply(meta, {"channel": key_cfg.channel, "scene": key_cfg.scene, "mode": key_cfg.mode}, aliases, "api_key")

    messages, markers = strip_markers(list(body.get("messages") or []))
    for data in markers:
        _apply(meta, data, aliases, "marker")

    # OpenAI 规范里的 user 字段也能当作外部用户 ID 使用
    if body.get("user") and not meta.external_user_id:
        meta.external_user_id = str(body["user"])

    for field_name in ("metadata", "her"):
        data = body.get(field_name)
        if isinstance(data, dict):
            _apply(meta, data, aliases, field_name)

    if not meta.scene:
        if meta.channel == "voice":
            meta.scene = "voice"
        elif meta.group_id:
            meta.scene = "group"
        else:
            meta.scene = "private"
    return meta, messages
