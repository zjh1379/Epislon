"""提示词组装。顺序按“从不变到常变”排列，以最大化前缀缓存命中：

[1] 人设与说话风格            ← 静态，永远相同，可被缓存
[2] 行为规则（场合、隐私）     ← 静态
[3] 当前场景：入口、私聊/群聊、在场的人、时间
[4] 核心档案：当前用户 + 她自己
[5] 相关情节记忆（≤5 条，已按场合过滤）
[6] 跨入口近期动态
[7] 入口传来的近期对话 + 本轮消息
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import PromptConfig
from ..identity.models import PUBLIC, Identity, Scene
from ..memory.recent import channel_label
from ..memory.retrieval import MemoryContext
from ..textutil import content_to_text
from ..timeutil import format_now, get_tz
from .persona import PersonaHolder

DYNAMIC_HEADER = "# 本轮动态信息（每轮都会变化）"


@dataclass
class BuiltPrompt:
    messages: list[dict[str, Any]]
    static_text: str
    dynamic_text: str
    sections: dict[str, list[str]] = field(default_factory=dict)


class PromptBuilder:
    def __init__(self, persona: PersonaHolder, cfg: PromptConfig):
        self.persona = persona
        self.cfg = cfg
        self.tz = get_tz(cfg.timezone)

    def static_messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.persona.static_text}]

    def build(
        self,
        *,
        identity: Identity,
        scene: Scene,
        ctx: MemoryContext,
        entry_messages: list[dict[str, Any]],
        entry_system: str,
        now_ts: float,
        extra_sections: list[tuple[str, list[str]]] | None = None,
    ) -> BuiltPrompt:
        her = self.persona.name
        sections: dict[str, list[str]] = {}

        # [3] 当前场景
        label = channel_label(scene.channel, scene.scene, scene.group_name)
        if scene.uncertain:
            label = f"{channel_label(scene.channel, 'voice')}（无法确认是私聊还是群聊）"
        scene_lines = [f"入口：{label}", f"时间：{format_now(now_ts, self.tz)}"]
        if identity.anonymous:
            scene_lines.append("正在说话的人：身份未确认（可能不是主人）。")
        else:
            scene_lines.append(f"正在和你说话的是：{identity.display_name}（{identity.role_label}）")
        if scene.access == PUBLIC:
            scene_lines.append("场合：公开。不要提任何私密的事。")
        sections["当前场景"] = scene_lines

        # [4] 核心档案
        if ctx.user_profile:
            sections[f"关于{identity.display_name}"] = ctx.user_profile
        if ctx.self_profile:
            sections[f"关于你自己（{her}）"] = ctx.self_profile
        # [5] 相关情节记忆
        if ctx.memory_lines:
            sections["你记得的相关事情"] = ctx.memory_lines
        # [6] 跨入口近期动态
        if ctx.recent:
            sections["其他入口的近期对话"] = ctx.recent
        for title, lines in extra_sections or []:
            if lines:
                sections[title] = lines
        if self.cfg.entry_system_prompt == "append" and entry_system.strip():
            sections["入口附加说明"] = [entry_system.strip()]

        dynamic_lines = [DYNAMIC_HEADER]
        for title, lines in sections.items():
            dynamic_lines.append(f"\n## {title}")
            dynamic_lines.extend(f"- {line}" for line in lines)
        dynamic = "\n".join(dynamic_lines)

        # [7] 入口传来的对话
        convo = [m for m in entry_messages if m.get("role") != "system"]
        if self.cfg.keep_last_messages > 0 and len(convo) > self.cfg.keep_last_messages:
            convo = convo[-self.cfg.keep_last_messages :]
            while convo and convo[0].get("role") in ("tool", "assistant") and len(convo) > 1:
                convo = convo[1:]

        static = self.persona.static_text
        if self.cfg.dynamic_position == "before_last_user":
            last_user = max((i for i, m in enumerate(convo) if m.get("role") == "user"), default=None)
            dyn_msg = {"role": "system", "content": dynamic}
            if last_user is None:
                messages = [{"role": "system", "content": static}, dyn_msg, *convo]
            else:
                messages = [{"role": "system", "content": static}, *convo[:last_user], dyn_msg, *convo[last_user:]]
        else:
            messages = [{"role": "system", "content": f"{static}\n\n{dynamic}"}, *convo]
        return BuiltPrompt(messages=messages, static_text=static, dynamic_text=dynamic, sections=sections)


def split_entry_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """拆出入口自带的系统提示词（会被替换为我们的人设），返回（其余消息, 系统提示词文本）。"""
    from ..identity.metadata import strip_soft_switches

    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            text = strip_soft_switches(content_to_text(m.get("content")))
            if text:
                system_parts.append(text)
        else:
            rest.append(m)
    return rest, "\n".join(system_parts)


def last_user_text(messages: list[dict[str, Any]]) -> str:
    """最后一条用户消息的文字（去掉入口附加的提醒块），用于记录日志、检索记忆和提取事实。"""
    from ..identity.metadata import strip_reminders

    for m in reversed(messages):
        if m.get("role") == "user":
            return strip_reminders(content_to_text(m.get("content")))
    return ""
