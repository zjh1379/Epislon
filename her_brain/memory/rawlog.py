"""原始日志（第三层）：每轮对话原文全部留底，供后台任务消费，也是重建记忆的依据。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..timeutil import now
from .db import Database, dumps, loads

PENDING, DONE, SKIPPED, FAILED = 0, 1, 2, 3
STATUS_NAMES = {PENDING: "pending", DONE: "done", SKIPPED: "skipped", FAILED: "failed"}


@dataclass
class RawLog:
    log_id: str
    user_id: str
    channel: str
    scene: str
    privacy: str
    user_text: str
    reply_text: str
    created_at: float
    group_id: str | None = None
    speaker_name: str = ""
    processed: int = PENDING
    attempts: int = 0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> RawLog:
        return cls(
            log_id=row["log_id"],
            user_id=row["user_id"],
            channel=row["channel"],
            scene=row["scene"],
            privacy=row["privacy"],
            user_text=row["user_text"],
            reply_text=row["reply_text"],
            created_at=float(row["created_at"]),
            group_id=row.get("group_id"),
            speaker_name=row.get("speaker_name") or "",
            processed=int(row.get("processed") or 0),
            attempts=int(row.get("attempts") or 0),
            error=row.get("error"),
            meta=loads(row.get("meta"), {}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["status"] = STATUS_NAMES.get(self.processed, str(self.processed))
        return d


def new_log_id() -> str:
    return "log_" + uuid.uuid4().hex[:16]


class RawLogStore:
    def __init__(self, db: Database):
        self.db = db

    def insert(self, log: RawLog) -> str:
        self.db.execute(
            """INSERT INTO raw_log(log_id, user_id, channel, scene, group_id, privacy, speaker_name,
                   user_text, reply_text, created_at, processed, attempts, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                log.log_id, log.user_id, log.channel, log.scene, log.group_id, log.privacy, log.speaker_name,
                log.user_text, log.reply_text, log.created_at, log.processed, dumps(log.meta),
            ),
        )
        return log.log_id

    def get(self, log_id: str) -> RawLog | None:
        row = self.db.query_one("SELECT * FROM raw_log WHERE log_id = ?", (log_id,))
        return RawLog.from_row(row) if row else None

    def get_many(self, log_ids: list[str]) -> list[RawLog]:
        if not log_ids:
            return []
        marks = ",".join("?" * len(log_ids))
        rows = self.db.query(f"SELECT * FROM raw_log WHERE log_id IN ({marks}) ORDER BY created_at", log_ids)
        return [RawLog.from_row(r) for r in rows]

    def next_batch(self, limit: int) -> list[RawLog]:
        """最早一条待处理日志所属（用户, 私密度）下的连续若干条，一起交给提取器。"""
        first = self.db.query_one(
            "SELECT user_id, privacy FROM raw_log WHERE processed = 0 ORDER BY created_at LIMIT 1"
        )
        if first is None:
            return []
        rows = self.db.query(
            """SELECT * FROM raw_log WHERE processed = 0 AND user_id = ? AND privacy = ?
               ORDER BY created_at LIMIT ?""",
            (first["user_id"], first["privacy"], limit),
        )
        return [RawLog.from_row(r) for r in rows]

    def mark(self, log_ids: list[str], status: int, error: str | None = None) -> None:
        if not log_ids:
            return
        self.db.executemany(
            "UPDATE raw_log SET processed = ?, error = ? WHERE log_id = ?",
            [(status, error, lid) for lid in log_ids],
        )

    def record_failure(self, log_ids: list[str], error: str, max_attempts: int) -> None:
        for lid in log_ids:
            self.db.execute(
                """UPDATE raw_log SET attempts = attempts + 1, error = ?,
                       processed = CASE WHEN attempts + 1 >= ? THEN 3 ELSE 0 END
                   WHERE log_id = ?""",
                (error[:500], max_attempts, lid),
            )

    def reprocess(self, log_ids: list[str] | None = None, user_id: str | None = None) -> int:
        if log_ids:
            marks = ",".join("?" * len(log_ids))
            cur = self.db.execute(
                f"UPDATE raw_log SET processed = 0, attempts = 0, error = NULL WHERE log_id IN ({marks})",
                log_ids,
            )
        elif user_id:
            cur = self.db.execute(
                "UPDATE raw_log SET processed = 0, attempts = 0, error = NULL WHERE user_id = ?", (user_id,)
            )
        else:
            cur = self.db.execute("UPDATE raw_log SET processed = 0, attempts = 0, error = NULL WHERE processed = 3")
        return cur.rowcount

    def recent_for_user(self, user_id: str, since: float, limit: int = 20) -> list[RawLog]:
        rows = self.db.query(
            """SELECT * FROM raw_log WHERE user_id = ? AND created_at >= ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, since, limit),
        )
        return [RawLog.from_row(r) for r in reversed(rows)]

    def list(
        self, user_id: str | None = None, status: int | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM raw_log WHERE 1 = 1"
        params: list[Any] = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if status is not None:
            sql += " AND processed = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [RawLog.from_row(r).to_dict() for r in self.db.query(sql, params)]

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT processed, COUNT(*) AS n FROM raw_log GROUP BY processed")
        out = {name: 0 for name in STATUS_NAMES.values()}
        for r in rows:
            out[STATUS_NAMES.get(int(r["processed"]), str(r["processed"]))] = int(r["n"])
        return out

    def new(self, **kwargs: Any) -> RawLog:
        kwargs.setdefault("log_id", new_log_id())
        kwargs.setdefault("created_at", now())
        return RawLog(**kwargs)
