"""SQLite 数据层。

单文件、WAL 模式，单用户场景下读写都在毫秒级。所有访问经过一把可重入锁，
既能在事件循环里直接调用（热路径上的读都有内存缓存），也能在线程池里调用。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 用户：owner 主人 / friend 朋友 / guest 访客 / self 她自己
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    role         TEXT NOT NULL DEFAULT 'guest',
    managed      INTEGER NOT NULL DEFAULT 0,   -- 1 表示来自配置文件，显示名不随入口变化
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

-- 身份表：qq:123456、voice:owner 等都映射到同一个 user_id
CREATE TABLE IF NOT EXISTS identities (
    platform    TEXT NOT NULL,
    external_id TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (platform, external_id)
);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities(user_id);

-- 核心档案：每个用户一份（另有一份“她自己”的），content 为条目列表的 JSON
CREATE TABLE IF NOT EXISTS core_profile (
    user_id    TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    version    INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

-- 档案历史：每次改写都完整保存一份，可回滚到任意版本
CREATE TABLE IF NOT EXISTS profile_version (
    user_id    TEXT NOT NULL,
    version    INTEGER NOT NULL,
    content    TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    PRIMARY KEY (user_id, version)
);

-- 档案候选：事实需被独立确认多次（或明确要求记住）才晋升进核心档案
CREATE TABLE IF NOT EXISTS profile_candidate (
    candidate_id   TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    slot           TEXT NOT NULL DEFAULT '',
    value          TEXT NOT NULL DEFAULT '',
    text           TEXT NOT NULL,
    kind           TEXT NOT NULL DEFAULT 'attribute',
    privacy        TEXT NOT NULL,
    confirmations  INTEGER NOT NULL DEFAULT 1,
    explicit       INTEGER NOT NULL DEFAULT 0,
    importance     REAL NOT NULL DEFAULT 3,
    event_date     TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',  -- pending / promoted / superseded / rejected / expired
    profile_entry_id TEXT,
    source_log_ids TEXT NOT NULL DEFAULT '[]',
    first_seen     REAL NOT NULL,
    last_seen      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidate_user ON profile_candidate(user_id, status);

-- 原始日志：全部留底，任何一层出错都能从这里重新生成
CREATE TABLE IF NOT EXISTS raw_log (
    log_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    channel      TEXT NOT NULL,
    scene        TEXT NOT NULL,
    group_id     TEXT,
    privacy      TEXT NOT NULL,
    speaker_name TEXT NOT NULL DEFAULT '',
    user_text    TEXT NOT NULL,
    reply_text   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    processed    INTEGER NOT NULL DEFAULT 0,   -- 0 待处理 / 1 已处理 / 2 跳过 / 3 失败
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_raw_log_pending ON raw_log(processed, created_at);
CREATE INDEX IF NOT EXISTS idx_raw_log_user ON raw_log(user_id, created_at);

-- 情节记忆（权威副本）。Mem0/Qdrant 或轻量引擎只做索引，正文与元数据以这里为准
CREATE TABLE IF NOT EXISTS episodic_memory (
    memory_id      TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    text           TEXT NOT NULL,
    privacy        TEXT NOT NULL,
    importance     REAL NOT NULL DEFAULT 3,
    kind           TEXT NOT NULL DEFAULT 'other',
    slot           TEXT NOT NULL DEFAULT '',
    event_date     TEXT,
    mentions       INTEGER NOT NULL DEFAULT 1,
    superseded     INTEGER NOT NULL DEFAULT 0,
    deleted        INTEGER NOT NULL DEFAULT 0,
    source_log_ids TEXT NOT NULL DEFAULT '[]',
    embedding      TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id, deleted);
CREATE INDEX IF NOT EXISTS idx_episodic_date ON episodic_memory(event_date);

-- 情节记忆变更事件：每次新增/修改/删除都记录前后快照，可撤销任意一次
CREATE TABLE IF NOT EXISTS memory_event (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    action      TEXT NOT NULL,          -- ADD / UPDATE / DELETE / RESTORE
    before_json TEXT,
    after_json  TEXT,
    actor       TEXT NOT NULL DEFAULT 'worker',
    note        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    reverted_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memory_event_user ON memory_event(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_event_mem ON memory_event(memory_id);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    # ------------------------------------------------------------------ 基本操作
    def execute(self, sql: str, params: tuple | list | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq: list) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)

    def query(self, sql: str, params: tuple | list | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple | list | dict = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def scalar(self, sql: str, params: tuple | list | dict = ()) -> Any:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return row[0] if row else None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            if self._tx_depth:
                self._tx_depth += 1
                try:
                    yield
                finally:
                    self._tx_depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._tx_depth = 1
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                self._tx_depth = 0

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        value = self.scalar("SELECT value FROM meta WHERE key = ?", (key,))
        return default if value is None else value

    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ 备份
    def backup_to(self, dest: str | Path) -> None:
        """在线备份（不阻塞写入太久），生成一致的数据库快照。"""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(dest))
            try:
                self._conn.backup(target)
            finally:
                target.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
