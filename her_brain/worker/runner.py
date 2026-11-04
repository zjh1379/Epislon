"""后台任务：原始日志 → 情节记忆 → 档案候选/晋升；另有每日备份与空闲整理。

- 只在空闲 idle_seconds 秒后运行；对话一来，正在进行的 LLM 提取会被立即取消。
- 提取（可打断）与写入（不可打断，保证一批日志要么全部生效要么全部重来）分成两段。
- 写入是幂等的：同一批日志重复处理不会重复累计确认次数。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from ..memory.models import Fact
from ..memory.rawlog import DONE, SKIPPED, RawLog
from ..timeutil import get_tz, local_date, now
from .activity import Preempted
from .extractor import attribute_sources

if TYPE_CHECKING:
    from ..services import Services

log = logging.getLogger(__name__)

META_LAST_TIDY = "last_tidy"


class BackgroundWorker:
    def __init__(self, services: Services):
        self.s = services
        self.cfg = services.settings.worker
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._applying: asyncio.Future[Any] | None = None
        self._needs_warmup = False
        self.stats: Counter[str] = Counter()
        self.last_error: str | None = None
        self.last_run: float | None = None

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="her-brain-worker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        if self._applying is not None and not self._applying.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._applying), timeout=30)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                log.warning("退出时仍有记忆写入未完成")

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.poll_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.exception("后台任务出错")

    # ------------------------------------------------------------------ 调度
    async def tick(self, force: bool = False) -> str:
        if not force and self.s.activity.idle_for() < self.cfg.idle_seconds:
            return "busy"
        status = await self.process_batch()
        if status != "empty":
            return status
        # 没有待处理日志：做空闲时的其他事
        if self._needs_warmup and self.s.settings.upstream.warmup:
            self._needs_warmup = False
            await self.s.warmup()
        if self.s.backup.due():
            await asyncio.to_thread(self.s.backup.snapshot, "daily")
            self.stats["backups"] += 1
        if self._tidy_due():
            await self.tidy()
        return "empty"

    async def run_until_idle(self, max_batches: int = 1000) -> dict[str, Any]:
        """立即处理所有待处理日志（管理接口与命令行使用），仍会被新对话打断。"""
        before = dict(self.stats)
        batches = 0
        status = "empty"
        while batches < max_batches:
            status = await self.process_batch()
            if status != "processed":
                break
            batches += 1
        if batches and self.s.settings.upstream.warmup:
            self._needs_warmup = False
            await self.s.warmup()
        delta = {k: v - before.get(k, 0) for k, v in self.stats.items() if v != before.get(k, 0)}
        return {"batches": batches, "stopped_because": status, "delta": delta}

    # ------------------------------------------------------------------ 处理一批
    async def process_batch(self) -> str:
        async with self._lock:
            logs = self.s.rawlog.next_batch(self.cfg.batch_size)
            if not logs:
                return "empty"
            ids = [entry.log_id for entry in logs]
            user_id = logs[0].user_id
            user = self.s.resolver.get_user(user_id)
            if (
                user is None
                or user.get("anonymous")
                or (user.get("role") == "guest" and not self.s.settings.memory.remember_guests)
            ):
                self.s.rawlog.mark(ids, SKIPPED)
                self.stats["logs_skipped"] += len(ids)
                return "processed"

            self.last_run = now()
            started = time.perf_counter()
            try:
                facts = await self.s.activity.run_preemptible(
                    self.s.extractor.extract(logs, user.get("display_name") or user_id)
                )
            except Preempted:
                self.stats["preempted"] += 1
                log.info("有新对话，后台提取让路（%d 条日志稍后重试）", len(ids))
                return "preempted"
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"提取失败：{exc}"
                self.stats["extract_errors"] += 1
                self.s.rawlog.record_failure(ids, str(exc), self.cfg.max_attempts)
                log.warning("提取记忆失败（%s）：%s", user_id, exc)
                return "processed"

            self._applying = asyncio.ensure_future(self._apply(logs, facts, user_id))
            try:
                await asyncio.shield(self._applying)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"写入失败：{exc}"
                self.stats["apply_errors"] += 1
                self.s.rawlog.record_failure(ids, str(exc), self.cfg.max_attempts)
                log.exception("写入记忆失败（%s）", user_id)
                return "processed"
            self.s.metrics.observe("worker_batch_ms", (time.perf_counter() - started) * 1000)
            self._needs_warmup = True
            return "processed"

    async def _apply(self, logs: list[RawLog], facts: list[Fact], user_id: str) -> None:
        ids = [entry.log_id for entry in logs]
        privacy = logs[0].privacy
        by_id = {entry.log_id: entry for entry in logs}
        self_id = self.s.settings.self_user_id
        for fact, sources in attribute_sources(facts, logs):
            target = self_id if fact.about == "self" else user_id
            created_at = max(by_id[s].created_at for s in sources)
            status, _ = await self.s.episodic.remember(target, fact, privacy, sources, created_at=created_at)
            self.stats[f"memory_{status}"] += 1
            result = self.s.promoter.observe(target, fact, privacy, sources)
            if result is not None and result.promoted:
                self.stats["profile_promoted"] += 1
                if result.superseded_slot:
                    self.s.episodic.supersede_slot(target, result.superseded_slot, fact.text, fact.value)
        self.s.rawlog.mark(ids, DONE)
        self.stats["logs_done"] += len(ids)
        self.stats["facts"] += len(facts)

    # ------------------------------------------------------------------ 空闲整理
    def _tidy_due(self) -> bool:
        last = float(self.s.db.get_meta(META_LAST_TIDY, "0") or 0)
        return now() - last >= self.cfg.tidy_interval_hours * 3600

    async def tidy(self) -> dict[str, Any]:
        """借鉴 Letta 的“空闲整理”，但只做确定性的清理，不让模型改写档案。"""
        expired = self.s.promoter.expire_stale()
        tz = get_tz(self.s.settings.prompt.timezone)
        today = local_date(now(), tz)
        cleaned = 0
        for uid in self.s.profiles.users_with_profiles():
            stale = [e.id for e in self.s.profiles.get(uid).entries if self.s.profiles._expired(e, today)]
            if not stale:
                continue

            def _drop(entries: list, stale_ids: set[str] = set(stale)) -> bool:  # noqa: B006
                entries[:] = [e for e in entries if e.id not in stale_ids]
                return True

            self.s.profiles.mutate(uid, _drop, f"清理已过期的计划 {len(stale)} 条")
            cleaned += len(stale)
        self.s.db.set_meta(META_LAST_TIDY, str(now()))
        self.stats["tidy_runs"] += 1
        return {"expired_candidates": expired, "expired_profile_entries": cleaned}

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.cfg.enabled,
            "running": self._task is not None and not self._task.done(),
            "idle_for_s": round(self.s.activity.idle_for(), 1),
            "idle_threshold_s": self.cfg.idle_seconds,
            "active_turns": self.s.activity.active,
            "preemptions": self.s.activity.preemptions,
            "logs": self.s.rawlog.counts(),
            "stats": dict(self.stats),
            "last_run": self.last_run,
            "last_error": self.last_error,
            "backup_due": self.s.backup.due(),
        }
