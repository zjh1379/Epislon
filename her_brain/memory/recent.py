"""跨入口近期动态：让她在语音里也知道你刚才在 QQ 说了什么。

只做截断，不在热路径上调用 LLM 做摘要。数据在内存里，首次访问时从原始日志加载。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from ..config import RecentConfig
from ..textutil import truncate
from ..timeutil import get_tz, human_ago, now
from .rawlog import RawLogStore

CHANNEL_LABELS = {"qq": "QQ", "voice": "桌面语音", "discord": "Discord", "telegram": "Telegram", "duck": "小鸭"}
SCENE_LABELS = {"private": "私聊", "group": "群聊", "voice": "语音"}


def channel_label(channel: str, scene: str, group_name: str | None = None) -> str:
    ch = CHANNEL_LABELS.get(channel, channel)
    if scene == "voice" or channel == "voice":
        return ch
    sc = SCENE_LABELS.get(scene, scene)
    if scene == "group" and group_name:
        return f"{ch}{sc}「{group_name}」"
    return f"{ch}{sc}"


@dataclass
class RecentItem:
    ts: float
    channel: str
    scene: str
    group_id: str | None
    privacy: str
    user_text: str
    reply_text: str

    @property
    def conversation_key(self) -> tuple[str, str]:
        return (self.channel, self.group_id or self.scene)


class RecentActivity:
    MAX_PER_USER = 30

    def __init__(self, rawlog: RawLogStore, cfg: RecentConfig, timezone: str = "Asia/Shanghai"):
        self.rawlog = rawlog
        self.cfg = cfg
        self.tz = get_tz(timezone)
        self._items: dict[str, deque[RecentItem]] = {}
        self._lock = threading.Lock()

    def _ensure(self, user_id: str) -> deque[RecentItem]:
        with self._lock:
            dq = self._items.get(user_id)
            if dq is not None:
                return dq
        since = now() - self.cfg.window_hours * 3600
        logs = self.rawlog.recent_for_user(user_id, since, limit=self.MAX_PER_USER)
        dq = deque(
            (RecentItem(l.created_at, l.channel, l.scene, l.group_id, l.privacy, l.user_text, l.reply_text) for l in logs),
            maxlen=self.MAX_PER_USER,
        )
        with self._lock:
            return self._items.setdefault(user_id, dq)

    def add(self, user_id: str, item: RecentItem) -> None:
        self._ensure(user_id).append(item)

    def render(
        self,
        user_id: str,
        user_name: str,
        allowed: set[str],
        exclude: tuple[str, str],
        now_ts: float | None = None,
    ) -> list[str]:
        now_ts = now_ts or now()
        cutoff = now_ts - self.cfg.window_hours * 3600
        items = [
            it
            for it in list(self._ensure(user_id))
            if it.ts >= cutoff and it.privacy in allowed and it.conversation_key != exclude
        ]
        items = items[-self.cfg.max_items :]
        lines = []
        n = self.cfg.max_chars
        for it in items:
            where = channel_label(it.channel, it.scene)
            lines.append(
                f"{human_ago(it.ts, now_ts, self.tz)}（{where}）{user_name}说「{truncate(it.user_text, n)}」，"
                f"你回「{truncate(it.reply_text, n)}」"
            )
        return lines

    def forget(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id is None:
                self._items.clear()
            else:
                self._items.pop(user_id, None)
