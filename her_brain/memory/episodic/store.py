"""情节记忆（第二层）：权威副本 + 检索索引 + 变更事件。

- 正文与元数据以 SQLite 为准，索引（Mem0/Qdrant 或轻量引擎）可随时重建。
- 每次新增、修改、删除、撤销都写一条 memory_event，保存前后快照：
  * revert_event：撤销任意一次变更；
  * rollback_user：把某个用户的情节记忆整体恢复到任意时间点。
- 删除是软删除（只打标记），因此恢复不需要重新写索引。
"""

from __future__ import annotations

import logging
from dataclasses import fields, replace
from datetime import timedelta
from typing import Any

from ...identity.models import PRIVATE, PUBLIC
from ...textutil import dice, same_value
from ...timeutil import get_tz, local_date, now
from ..candidates import normalize_slot
from ..db import Database, dumps, loads
from ..models import DATED_KINDS, Fact, MemoryRecord, ScoredMemory
from .index import VectorIndex

log = logging.getLogger(__name__)

TEXT_DUPLICATE_THRESHOLD = 0.8
_RECORD_FIELDS = {f.name for f in fields(MemoryRecord)}


def _merge_privacy(a: str, b: str) -> str:
    return PUBLIC if PUBLIC in (a, b) else PRIVATE


class EpisodicMemory:
    def __init__(
        self,
        db: Database,
        index: VectorIndex,
        *,
        min_score: float | None = None,
        dedupe_score: float | None = None,
        timezone: str = "Asia/Shanghai",
    ):
        self.db = db
        self.index = index
        self.min_score = index.default_min_score if min_score is None else min_score
        self.dedupe_score = index.default_dedupe_score if dedupe_score is None else dedupe_score
        self.tz = get_tz(timezone)

    # ------------------------------------------------------------------ 行读写
    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.db.query_one("SELECT * FROM episodic_memory WHERE memory_id = ?", (memory_id,))
        return MemoryRecord.from_row(row) if row else None

    def get_many(self, memory_ids: list[str]) -> dict[str, MemoryRecord]:
        if not memory_ids:
            return {}
        marks = ",".join("?" * len(memory_ids))
        rows = self.db.query(f"SELECT * FROM episodic_memory WHERE memory_id IN ({marks})", memory_ids)
        return {r["memory_id"]: MemoryRecord.from_row(r) for r in rows}

    def _insert_row(self, rec: MemoryRecord, embedding: list[float] | None) -> None:
        self.db.execute(
            """INSERT INTO episodic_memory(memory_id, user_id, text, privacy, importance, kind, slot, event_date,
                   mentions, superseded, deleted, source_log_ids, embedding, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.memory_id, rec.user_id, rec.text, rec.privacy, rec.importance, rec.kind, rec.slot,
                rec.event_date, rec.mentions, int(rec.superseded), int(rec.deleted), dumps(rec.source_log_ids),
                dumps(embedding) if embedding is not None else None, rec.created_at, rec.updated_at,
            ),
        )

    def _update_row(self, rec: MemoryRecord, embedding: list[float] | None = None) -> None:
        self.db.execute(
            """UPDATE episodic_memory SET text = ?, privacy = ?, importance = ?, kind = ?, slot = ?,
                   event_date = ?, mentions = ?, superseded = ?, deleted = ?, source_log_ids = ?,
                   updated_at = ?, embedding = COALESCE(?, embedding)
               WHERE memory_id = ?""",
            (
                rec.text, rec.privacy, rec.importance, rec.kind, rec.slot, rec.event_date, rec.mentions,
                int(rec.superseded), int(rec.deleted), dumps(rec.source_log_ids), rec.updated_at,
                dumps(embedding) if embedding is not None else None, rec.memory_id,
            ),
        )

    def _log(
        self,
        action: str,
        memory_id: str,
        user_id: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        actor: str,
        note: str = "",
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO memory_event(memory_id, user_id, action, before_json, after_json, actor, note, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id, user_id, action,
                dumps(before) if before is not None else None,
                dumps(after) if after is not None else None,
                actor, note, now(),
            ),
        )
        return int(cur.lastrowid)

    async def _apply_state(self, current: MemoryRecord, target: MemoryRecord, action: str, actor: str, note: str) -> int:
        target = replace(target, memory_id=current.memory_id, user_id=current.user_id, updated_at=now())
        embedding = None
        if target.text != current.text or target.privacy != current.privacy:
            embedding = await self.index.update(target)
        self._update_row(target, embedding)
        self.index.invalidate(target.user_id)
        return self._log(action, target.memory_id, target.user_id, current.snapshot(), target.snapshot(), actor, note)

    # ------------------------------------------------------------------ 写入
    async def remember(
        self,
        user_id: str,
        fact: Fact,
        privacy: str,
        source_log_ids: list[str],
        *,
        created_at: float | None = None,
        actor: str = "worker",
    ) -> tuple[str, MemoryRecord]:
        """写入一条事实。返回（added / merged / skipped, 记录）。重复提及只累计次数并略升重要度。"""
        ts = now()
        dup = await self._find_duplicate(user_id, fact.text)
        if dup is not None:
            if set(source_log_ids) <= set(dup.source_log_ids):
                return "skipped", dup
            before = dup.snapshot()
            dup.mentions += 1
            dup.importance = min(5.0, max(dup.importance, float(fact.effective_importance)) + 0.5)
            dup.privacy = _merge_privacy(dup.privacy, privacy)
            dup.source_log_ids = sorted(set(dup.source_log_ids) | set(source_log_ids))[-30:]
            if fact.date:
                dup.event_date = fact.date
            dup.updated_at = ts
            self._update_row(dup)
            self.index.invalidate(user_id)
            self._log("UPDATE", dup.memory_id, user_id, before, dup.snapshot(), actor, "重复提及")
            return "merged", dup

        rec = MemoryRecord(
            memory_id="",
            user_id=user_id,
            text=fact.text,
            privacy=privacy,
            importance=float(fact.effective_importance),
            kind=fact.kind,
            slot=normalize_slot(fact.slot),
            event_date=fact.date,
            source_log_ids=sorted(set(source_log_ids)),
            created_at=created_at or ts,
            updated_at=ts,
        )
        rec.memory_id, embedding = await self.index.add(rec)
        self._insert_row(rec, embedding)
        self.index.invalidate(user_id)
        self._log("ADD", rec.memory_id, user_id, None, rec.snapshot(), actor)
        return "added", rec

    async def _find_duplicate(self, user_id: str, text: str) -> MemoryRecord | None:
        exact = self.db.query_one(
            "SELECT * FROM episodic_memory WHERE user_id = ? AND text = ? AND deleted = 0 LIMIT 1", (user_id, text)
        )
        if exact:
            return MemoryRecord.from_row(exact)
        try:
            hits = await self.index.search([user_id], text, 5, self.min_score)
        except Exception as exc:  # noqa: BLE001 - 去重失败不应阻止写入
            log.warning("去重检索失败：%s", exc)
            return None
        recs = self.get_many([mid for mid, _ in hits])
        for mid, score in hits:
            rec = recs.get(mid)
            if rec is None or rec.deleted or rec.user_id != user_id:
                continue
            if dice(rec.text, text) >= TEXT_DUPLICATE_THRESHOLD or (
                self.dedupe_score is not None and score >= self.dedupe_score
            ):
                return rec
        return None

    async def add_manual(self, user_id: str, data: dict[str, Any], actor: str = "admin") -> MemoryRecord:
        ts = now()
        rec = MemoryRecord(
            memory_id="",
            user_id=user_id,
            text=str(data["text"]).strip(),
            privacy=data.get("privacy", PRIVATE),
            importance=float(data.get("importance", 3)),
            kind=data.get("kind", "other"),
            slot=normalize_slot(data.get("slot", "")),
            event_date=data.get("event_date"),
            created_at=ts,
            updated_at=ts,
        )
        rec.memory_id, embedding = await self.index.add(rec)
        self._insert_row(rec, embedding)
        self.index.invalidate(user_id)
        self._log("ADD", rec.memory_id, user_id, None, rec.snapshot(), actor, "手动添加")
        return rec

    async def edit(self, memory_id: str, changes: dict[str, Any], actor: str = "admin", note: str = "手动修改") -> MemoryRecord:
        current = self._require(memory_id)
        allowed = {k: v for k, v in changes.items() if k in ("text", "privacy", "importance", "kind", "slot", "event_date")}
        target = replace(current, **allowed)
        await self._apply_state(current, target, "UPDATE", actor, note)
        return self._require(memory_id)

    async def delete(self, memory_id: str, actor: str = "admin", note: str = "删除") -> MemoryRecord:
        current = self._require(memory_id)
        if current.deleted:
            return current
        await self._apply_state(current, replace(current, deleted=True), "DELETE", actor, note)
        return self._require(memory_id)

    async def restore(self, memory_id: str, actor: str = "admin") -> MemoryRecord:
        current = self._require(memory_id)
        if not current.deleted:
            return current
        await self._apply_state(current, replace(current, deleted=False), "RESTORE", actor, "恢复")
        return self._require(memory_id)

    async def purge(self, memory_id: str) -> None:
        """彻底删除（不可撤销）：从索引与权威副本中移除，事件记录保留以便审计。"""
        current = self._require(memory_id)
        try:
            await self.index.remove(memory_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("从索引删除 %s 失败：%s", memory_id, exc)
        self.db.execute("DELETE FROM episodic_memory WHERE memory_id = ?", (memory_id,))
        self.index.invalidate(current.user_id)
        self._log("PURGE", memory_id, current.user_id, current.snapshot(), None, "admin", "彻底删除")

    def _require(self, memory_id: str) -> MemoryRecord:
        rec = self.get(memory_id)
        if rec is None:
            raise KeyError(memory_id)
        return rec

    def supersede_slot(self, user_id: str, slot: str, keep_text: str, keep_value: str, actor: str = "worker") -> int:
        """某个 slot 的新取值晋升进档案后，把描述旧取值的情节记忆标记为“已过时”（降低排序，不删除）。"""
        if not slot:
            return 0
        rows = self.db.query(
            "SELECT * FROM episodic_memory WHERE user_id = ? AND slot = ? AND deleted = 0 AND superseded = 0",
            (user_id, slot),
        )
        count = 0
        for row in rows:
            rec = MemoryRecord.from_row(row)
            if (keep_value and keep_value in rec.text) or same_value(rec.text, keep_text, TEXT_DUPLICATE_THRESHOLD):
                continue
            before = rec.snapshot()
            rec.superseded = True
            rec.updated_at = now()
            self._update_row(rec)
            self._log("UPDATE", rec.memory_id, user_id, before, rec.snapshot(), actor, f"“{slot}”有了新取值，标记为已过时")
            count += 1
        if count:
            self.index.invalidate(user_id)
        return count

    # ------------------------------------------------------------------ 读取
    async def search(self, user_ids: list[str], query: str, allowed: set[str], limit: int) -> list[ScoredMemory]:
        hits = await self.index.search(user_ids, query, max(limit * 3, 10), self.min_score)
        recs = self.get_many([mid for mid, _ in hits])
        out: list[ScoredMemory] = []
        for mid, score in hits:
            rec = recs.get(mid)
            if rec is None or rec.deleted or rec.privacy not in allowed or rec.user_id not in user_ids:
                continue
            adjusted = score + 0.03 * (rec.importance - 3) + min(0.06, 0.02 * (rec.mentions - 1))
            if rec.superseded:
                adjusted -= 0.15
            out.append(ScoredMemory(rec, adjusted, "semantic"))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:limit]

    def agenda(
        self, user_ids: list[str], allowed: set[str], now_ts: float, days_back: int, days_ahead: int, limit: int
    ) -> list[ScoredMemory]:
        """日期临近的计划与事件：不论和当前这句话是否相似，都应该被她“想起来”。"""
        if not user_ids or limit <= 0:
            return []
        today = local_date(now_ts, self.tz)
        lo = (today - timedelta(days=days_back)).isoformat()
        hi = (today + timedelta(days=days_ahead)).isoformat()
        marks = ",".join("?" * len(user_ids))
        kinds = ",".join("?" * len(DATED_KINDS))
        rows = self.db.query(
            f"""SELECT * FROM episodic_memory
                WHERE user_id IN ({marks}) AND deleted = 0 AND superseded = 0
                  AND kind IN ({kinds}) AND event_date BETWEEN ? AND ?
                ORDER BY importance DESC, created_at DESC LIMIT ?""",
            [*user_ids, *DATED_KINDS, lo, hi, limit * 3],
        )
        out: list[ScoredMemory] = []
        for row in rows:
            rec = MemoryRecord.from_row(row)
            if rec.privacy not in allowed:
                continue
            try:
                distance = abs((local_date(now_ts, self.tz) - _to_date(rec.event_date)).days)
            except Exception:  # noqa: BLE001
                distance = days_back
            out.append(ScoredMemory(rec, 2.0 - 0.1 * distance + 0.05 * rec.importance, "agenda"))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:limit]

    def list_for_user(
        self, user_id: str, *, include_deleted: bool = False, q: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM episodic_memory WHERE user_id = ?"
        params: list[Any] = [user_id]
        if not include_deleted:
            sql += " AND deleted = 0"
        if q:
            sql += " AND text LIKE ?"
            params.append(f"%{q}%")
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [MemoryRecord.from_row(r) for r in self.db.query(sql, params)]

    def count(self, user_id: str | None = None) -> int:
        if user_id:
            return int(self.db.scalar("SELECT COUNT(*) FROM episodic_memory WHERE user_id = ? AND deleted = 0", (user_id,)))
        return int(self.db.scalar("SELECT COUNT(*) FROM episodic_memory WHERE deleted = 0"))

    # ------------------------------------------------------------------ 版本
    def events(
        self, *, user_id: str | None = None, memory_id: str | None = None, since: float | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memory_event WHERE 1 = 1"
        params: list[Any] = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if memory_id:
            sql += " AND memory_id = ?"
            params.append(memory_id)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        sql += " ORDER BY event_id DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        for r in rows:
            r["before"] = loads(r.pop("before_json"), None)
            r["after"] = loads(r.pop("after_json"), None)
        return rows

    async def revert_event(self, event_id: int, actor: str = "admin") -> dict[str, Any]:
        ev = self.db.query_one("SELECT * FROM memory_event WHERE event_id = ?", (event_id,))
        if ev is None:
            raise KeyError(f"事件 #{event_id} 不存在")
        if ev["reverted_by"]:
            raise ValueError(f"事件 #{event_id} 已被事件 #{ev['reverted_by']} 撤销")
        current = self.get(ev["memory_id"])
        if current is None:
            raise ValueError("这条记忆已被彻底删除，无法撤销")
        target = self._state_from_snapshot(current, loads(ev["before_json"], None))
        new_event = await self._apply_state(current, target, "REVERT", actor, f"撤销事件 #{event_id}")
        self.db.execute("UPDATE memory_event SET reverted_by = ? WHERE event_id = ?", (new_event, event_id))
        return {"reverted": event_id, "event_id": new_event, "memory": self._require(current.memory_id).snapshot()}

    async def rollback_user(self, user_id: str, since: float, actor: str = "admin") -> dict[str, Any]:
        """把 user_id 的情节记忆恢复到 since 时刻的状态：每条在此之后被改动过的记忆，
        都恢复为它在此之后第一次变更前的快照（之前不存在的记忆会被软删除）。"""
        evs = self.db.query(
            "SELECT * FROM memory_event WHERE user_id = ? AND created_at >= ? ORDER BY event_id ASC",
            (user_id, since),
        )
        first: dict[str, dict[str, Any]] = {}
        for ev in evs:
            first.setdefault(ev["memory_id"], ev)
        restored: list[str] = []
        for memory_id, ev in first.items():
            current = self.get(memory_id)
            if current is None:
                continue
            target = self._state_from_snapshot(current, loads(ev["before_json"], None))
            if _same_state(current, target):
                continue
            await self._apply_state(current, target, "ROLLBACK", actor, f"整体回滚到 {since:.0f}")
            restored.append(memory_id)
        return {"user_id": user_id, "since": since, "restored": restored, "count": len(restored)}

    @staticmethod
    def _state_from_snapshot(current: MemoryRecord, snapshot: dict[str, Any] | None) -> MemoryRecord:
        if snapshot is None:  # 变更前不存在：软删除
            return replace(current, deleted=True)
        data = {k: v for k, v in snapshot.items() if k in _RECORD_FIELDS}
        return replace(current, **{k: v for k, v in data.items() if k not in ("memory_id", "user_id")})

    async def soft_delete_user(self, user_id: str, actor: str, note: str) -> int:
        count = 0
        for rec in self.list_for_user(user_id, limit=100000):
            await self._apply_state(rec, replace(rec, deleted=True), "DELETE", actor, note)
            count += 1
        return count

    async def rebuild_index(self) -> dict[str, Any]:
        records = [MemoryRecord.from_row(r) for r in self.db.query("SELECT * FROM episodic_memory ORDER BY created_at")]
        mapping = await self.index.rebuild(records)
        with self.db.transaction():
            for old, new in mapping.items():
                self.db.execute("UPDATE episodic_memory SET memory_id = ? WHERE memory_id = ?", (new, old))
                self.db.execute("UPDATE memory_event SET memory_id = ? WHERE memory_id = ?", (new, old))
        for uid in {r.user_id for r in records}:
            self.index.invalidate(uid)
        return {"records": len(records), "remapped": len(mapping)}


def _to_date(value: str | None):
    from ...timeutil import parse_date

    d = parse_date(value)
    if d is None:
        raise ValueError(value)
    return d


def _same_state(a: MemoryRecord, b: MemoryRecord) -> bool:
    keys = ("text", "privacy", "importance", "kind", "slot", "event_date", "mentions", "superseded", "deleted")
    return all(getattr(a, k) == getattr(b, k) for k in keys)
