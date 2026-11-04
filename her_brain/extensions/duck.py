"""小鸭机器人扩展（占位，下一阶段实现）。

计划中的工作方式：
1. 小鸭作为新入口接入同一个大脑代理，元数据 ``channel="duck"``，复用身份、记忆与人设。
2. 当 channel 为 duck 时，``prompt_sections`` 告诉模型可用的动作列表，并要求在回复末尾输出
   ``<action>{"name": "nod", "params": {}}</action>`` 形式的结构化动作意图。
3. ``on_reply`` 解析动作意图（见 ``parse_actions``），通过 MQTT 发布到 ``her/duck/action`` 主题。
4. 动作标签需要在流式转发时从文本中剥离，避免被 TTS 念出来（下一阶段在 ChatService 中接入）。

本阶段只提供解析函数与空实现，默认不启用。
"""

from __future__ import annotations

import json
import logging
import re

from . import ActionIntent, Extension, TurnInfo

log = logging.getLogger(__name__)

ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.S)

DEFAULT_ACTIONS = ["nod（点头）", "shake（摇头）", "wave（挥手）", "dance（跳舞）", "look_at（看向说话的人）"]


def parse_actions(text: str) -> tuple[str, list[ActionIntent]]:
    """从回复中取出动作意图，返回（去掉动作标签后的文本, 动作列表）。"""
    actions: list[ActionIntent] = []

    def _repl(m: re.Match[str]) -> str:
        try:
            actions.append(ActionIntent.model_validate(json.loads(m.group(1))))
        except Exception:  # noqa: BLE001
            log.warning("无法解析动作意图：%s", m.group(1))
        return ""

    return ACTION_RE.sub(_repl, text).strip(), actions


class DuckActionStub(Extension):
    name = "duck"

    def prompt_sections(self, turn: TurnInfo) -> list[tuple[str, list[str]]]:
        if turn.meta.channel != "duck":
            return []
        return [(
            "你现在的身体（小鸭机器人）",
            [
                "你可以做这些动作：" + "、".join(DEFAULT_ACTIONS),
                '需要做动作时，在回复最后附上 <action>{"name": "动作名", "params": {}}</action>，可以有多个。',
            ],
        )]

    async def on_reply(self, turn: TurnInfo) -> None:
        if turn.meta.channel != "duck":
            return
        _, actions = parse_actions(turn.reply_text)
        for act in actions:
            # 下一阶段：mqtt_client.publish("her/duck/action", act.model_dump_json())
            log.info("[小鸭占位] 动作意图：%s", act.model_dump())
