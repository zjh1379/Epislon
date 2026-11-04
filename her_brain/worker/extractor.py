"""从原始日志中提取原子事实（空闲时运行）。

一批日志只调用一次 LLM，输出受 JSON Schema 约束（llama-server 会转成语法约束），
每条事实带：关于谁、类型、属性槽位、日期、重要度、是否被明确要求记住。
为避免“把已知信息当成新确认”，提示词里不放已有档案，只放这批对话本身。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..llm.client import LLMClient
from ..memory.models import KINDS, Fact
from ..memory.rawlog import RawLog
from ..memory.recent import channel_label
from ..prompt.persona import PersonaHolder
from ..timeutil import format_stamp, get_tz

log = logging.getLogger(__name__)

FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "about": {"type": "string", "enum": ["user", "self"]},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "slot": {"type": "string"},
                    "value": {"type": "string"},
                    "date": {"type": "string"},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "explicit": {"type": "boolean"},
                },
                "required": ["text", "about", "kind", "slot", "value", "date", "importance", "explicit"],
            },
        }
    },
    "required": ["facts"],
}

SYSTEM_TEMPLATE = """你是记忆整理助手。下面是“{her}”和“{user}”的对话记录，请提取值得长期记住的事实。

规则：
1. 只提取对话里明确说出的信息，不推测、不编造、不补全。
2. 每条事实是一句独立、完整的中文陈述，用第三人称：称呼对方为“{user}”，称呼她为“{her}”。
3. 每段对话前标注了发生时间。把“明天”“下周三”等相对时间换算成具体日期写进 date（YYYY-MM-DD），
   并在 text 里写成具体日期，例如“{user}在 2026-11-05 有一场面试”。
4. 忽略寒暄、客套和没有信息量的内容。{her}的一般性回复不要提取；
   只有当{her}说到关于她自己的新设定、经历、承诺时才提取，about 填 "self"。
5. 字段说明：
   - about：关于对方填 "user"，关于{her}自己填 "self"
   - kind：attribute 稳定属性（名字、生日、职业、住在哪、家人、宠物等）| preference 喜好或厌恶 |
     relationship 与他人的关系 | event 已发生的事 | plan 将要发生的事 | other 其他
   - slot：只有当这条事实是“同一时间只有一个值”的属性时才填，尽量使用这些名称：
     名字、昵称、生日、年龄、职业、工作单位、学校、专业、所在城市、家乡、感情状态；否则填空字符串
   - value：slot 不为空时填简短取值（例如“杭州”），否则填空字符串
   - date：事件或计划的日期，没有就填空字符串
   - importance：1~5。5=健康、重大人生事件、对方明确要求记住；4=重要计划或强烈情绪；3=一般个人信息；1~2=琐事
   - explicit：对方是否明确要求你记住（“记住”“别忘了”“帮我记着”）
6. 没有值得记住的内容时返回 {{"facts": []}}。

只输出 JSON：{{"facts": [...]}}"""


class MemoryExtractor:
    def __init__(self, llm: LLMClient, settings: Settings, persona: PersonaHolder):
        self.llm = llm
        self.settings = settings
        self.persona = persona
        self.tz = get_tz(settings.prompt.timezone)

    def build_messages(self, logs: list[RawLog], user_name: str) -> list[dict[str, Any]]:
        her = self.persona.name
        blocks: list[str] = []
        for entry in logs:
            where = channel_label(entry.channel, entry.scene, (entry.meta or {}).get("group_name"))
            blocks.append(
                f"[{format_stamp(entry.created_at, self.tz)} · {where}]\n"
                f"{user_name}：{entry.user_text.strip()}\n"
                f"{her}：{entry.reply_text.strip()}"
            )
        system = SYSTEM_TEMPLATE.format(her=her, user=user_name)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "对话记录：\n\n" + "\n\n".join(blocks)},
        ]

    def request_params(self) -> dict[str, Any]:
        wl = self.settings.worker_llm
        params: dict[str, Any] = {"temperature": wl.temperature, "max_tokens": wl.max_tokens}
        if wl.json_mode == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "facts", "strict": True, "schema": FACT_SCHEMA},
            }
        elif wl.json_mode == "json_object":
            params["response_format"] = {"type": "json_object"}
        return params

    async def extract(self, logs: list[RawLog], user_name: str) -> list[Fact]:
        messages = self.build_messages(logs, user_name)
        text = await self.llm.complete(messages, **self.request_params())
        facts = parse_facts(text)
        log.info("从 %d 条日志中提取到 %d 条事实", len(logs), len(facts))
        return facts


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _load_json(text: str) -> Any:
    text = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue
    raise ValueError(f"提取结果不是合法 JSON：{text[:200]}")


def parse_facts(text: str) -> list[Fact]:
    if not text or not text.strip():
        return []
    data = _load_json(text)
    if isinstance(data, dict):
        items = data.get("facts") or data.get("memory") or data.get("memories") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    facts: list[Fact] = []
    seen: set[str] = set()
    for item in items:
        try:
            fact = Fact(text=item) if isinstance(item, str) else Fact.model_validate(item)
        except ValidationError:
            continue
        if not (4 <= len(fact.text) <= 240) or fact.text in seen:
            continue
        seen.add(fact.text)
        facts.append(fact)
    return facts
