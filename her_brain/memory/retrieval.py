"""读路径：只碰第一层（核心档案，内存缓存）和第二层（情节记忆，带超时）。

语义检索超过 retrieval_timeout_ms 就放弃，只带档案、时效记忆与近期动态回复，保证速度底线。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..identity.models import Identity, Scene
from ..textutil import truncate
from ..timeutil import get_tz, human_ago, local_date, now, parse_date, relative_day
from .episodic import EpisodicMemory
from .models import ScoredMemory
from .profile import ProfileStore
from .recent import RecentActivity

log = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    user_profile: list[str] = field(default_factory=list)
    self_profile: list[str] = field(default_factory=list)
    memories: list[ScoredMemory] = field(default_factory=list)
    memory_lines: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    timed_out: bool = False
    search_skipped: bool = False
    error: str | None = None


class MemoryReader:
    def __init__(
        self,
        settings: Settings,
        profiles: ProfileStore,
        episodic: EpisodicMemory,
        recent: RecentActivity,
    ):
        self.settings = settings
        self.cfg = settings.memory
        self.profiles = profiles
        self.episodic = episodic
        self.recent = recent
        self.self_id = settings.self_user_id
        self.tz = get_tz(settings.prompt.timezone)

    async def gather(self, identity: Identity, scene: Scene, query: str, now_ts: float | None = None) -> MemoryContext:
        now_ts = now_ts or now()
        ctx = MemoryContext()
        allowed = scene.allowed_privacy
        pcfg = self.settings.prompt
        user_ids = [self.self_id] if identity.anonymous else [identity.user_id, self.self_id]

        # 第一层：核心档案（内存缓存，微秒级）
        if not identity.anonymous:
            ctx.user_profile = self.profiles.render(identity.user_id, allowed, now_ts, pcfg.max_profile_chars)
        ctx.self_profile = self.profiles.render(self.self_id, allowed, now_ts, pcfg.max_profile_chars // 2)

        # 时效记忆：SQLite 范围查询，亚毫秒
        agenda = self.episodic.agenda(
            user_ids, allowed, now_ts, self.cfg.agenda_days_back, self.cfg.agenda_days_ahead, self.cfg.agenda_max
        )

        # 第二层：语义检索，带超时
        semantic: list[ScoredMemory] = []
        started = time.perf_counter()
        if len(query.strip()) >= self.cfg.min_query_chars:
            try:
                semantic = await asyncio.wait_for(
                    self.episodic.search(user_ids, query, allowed, self.cfg.top_k),
                    timeout=self.cfg.retrieval_timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                ctx.timed_out = True
                log.warning("记忆检索超过 %dms，本轮只带核心档案", self.cfg.retrieval_timeout_ms)
            except Exception as exc:  # noqa: BLE001 - 检索出错不能影响回复
                ctx.error = str(exc)
                log.exception("记忆检索失败")
        else:
            ctx.search_skipped = True
        ctx.retrieval_ms = (time.perf_counter() - started) * 1000

        merged: dict[str, ScoredMemory] = {}
        for item in agenda + semantic:
            if item.record.memory_id not in merged:
                merged[item.record.memory_id] = item
        ranked = sorted(merged.values(), key=lambda s: s.score, reverse=True)[: self.cfg.top_k]
        ctx.memories = ranked
        ctx.memory_lines = self._render_memories(ranked, identity, now_ts, pcfg.max_memory_chars)

        # 跨入口近期动态
        if not identity.anonymous:
            ctx.recent = self.recent.render(
                identity.user_id, identity.display_name, allowed, scene.conversation_key, now_ts
            )
        return ctx

    def _render_memories(self, items: list[ScoredMemory], identity: Identity, now_ts: float, budget: int) -> list[str]:
        today = local_date(now_ts, self.tz)
        lines: list[str] = []
        used = 0
        for item in items:
            rec = item.record
            when = human_ago(rec.created_at, now_ts, self.tz)
            prefix = f"[{when}记下]"
            d = parse_date(rec.event_date)
            if d is not None:
                prefix = f"[{when}记下 · 事情在{relative_day(d, today)}（{d.isoformat()}）]"
            who = "（关于你自己）" if rec.user_id == self.self_id else ""
            stale = "（可能已过时）" if rec.superseded else ""
            text = f"{prefix}{who}{truncate(rec.text, 140)}{stale}"
            if used + len(text) > budget and lines:
                break
            lines.append(text)
            used += len(text)
        return lines
