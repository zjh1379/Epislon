"""核心档案：每次请求都注入的小档案（第一层记忆）。

档案由一条条带私密度的条目组成，存为 JSON。每次改写都整体存入历史版本，
因此可以回滚到任意版本；回滚本身也会生成一个新版本，不会丢失任何历史。
"""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Callable

from ..config import ProfileConfig
from ..identity.models import PRIVATE, PUBLIC
from ..textutil import same_value, truncate
from ..timeutil import get_tz, local_date, now, parse_date, relative_day
from .db import Database, dumps, loads


@dataclass
class ProfileEntry:
    id: str
    text: str
    slot: str = ""
    value: str = ""
    kind: str = "attribute"
    privacy: str = PRIVATE
    pinned: bool = False
    confirmations: int = 1
    source: str = "worker"  # worker / admin
    source_log_ids: list[str] = field(default_factory=list)
    event_date: str | None = None
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileEntry:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        known.setdefault("id", new_entry_id())
        known.setdefault("text", "")
        return cls(**known)


def new_entry_id() -> str:
    return "p_" + uuid.uuid4().hex[:10]


@dataclass
class Profile:
    user_id: str
    version: int
    entries: list[ProfileEntry]
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "entries": [asdict(e) for e in self.entries],
        }


class ProfileStore:
    def __init__(self, db: Database, cfg: ProfileConfig, timezone: str = "Asia/Shanghai"):
        self.db = db
        self.cfg = cfg
        self.tz = get_tz(timezone)
        self._cache: dict[str, Profile] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 读
    def get(self, user_id: str) -> Profile:
        with self._lock:
            cached = self._cache.get(user_id)
            if cached is not None:
                return cached
            row = self.db.query_one("SELECT * FROM core_profile WHERE user_id = ?", (user_id,))
            if row is None:
                profile = Profile(user_id=user_id, version=0, entries=[])
            else:
                data = loads(row["content"], {}) or {}
                profile = Profile(
                    user_id=user_id,
                    version=int(row["version"]),
                    entries=[ProfileEntry.from_dict(e) for e in data.get("entries", [])],
                    updated_at=float(row["updated_at"]),
                )
            self._cache[user_id] = profile
            return profile

    def render(self, user_id: str, allowed: set[str], now_ts: float | None = None, max_chars: int = 800) -> list[str]:
        """按场合过滤后的档案条目文本。置顶条目优先，其次按确认次数。"""
        profile = self.get(user_id)
        now_ts = now_ts or now()
        today = local_date(now_ts, self.tz)
        entries = [e for e in profile.entries if e.privacy in allowed and not self._expired(e, today)]
        entries.sort(key=lambda e: (not e.pinned, -e.confirmations, -e.updated_at))
        lines: list[str] = []
        used = 0
        for e in entries[: self.cfg.max_render_entries]:
            text = e.text
            d = parse_date(e.event_date)
            if d is not None and e.kind in ("plan", "event"):
                text = f"（{relative_day(d, today)}）{text}"
            text = truncate(text, 120)
            if used + len(text) > max_chars and lines:
                break
            lines.append(text)
            used += len(text)
        return lines

    def _expired(self, entry: ProfileEntry, today) -> bool:
        if entry.kind not in ("plan", "event") or entry.pinned:
            return False
        d = parse_date(entry.event_date)
        return d is not None and today > d + timedelta(days=self.cfg.plan_expire_days)

    # ------------------------------------------------------------------ 写
    def mutate(self, user_id: str, fn: Callable[[list[ProfileEntry]], bool], reason: str) -> tuple[Profile, bool]:
        """在副本上执行 fn；fn 返回 True 时保存为新版本。"""
        with self._lock:
            current = self.get(user_id)
            entries = copy.deepcopy(current.entries)
            changed = fn(entries)
            if not changed:
                return current, False
            return self._save(user_id, entries, reason, current.version), True

    def _save(self, user_id: str, entries: list[ProfileEntry], reason: str, prev_version: int) -> Profile:
        ts = now()
        version = prev_version + 1
        content = dumps({"entries": [asdict(e) for e in entries]})
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO core_profile(user_id, content, version, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET content = excluded.content,
                       version = excluded.version, updated_at = excluded.updated_at""",
                (user_id, content, version, ts),
            )
            self.db.execute(
                "INSERT INTO profile_version(user_id, version, content, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, version, content, reason, ts),
            )
        profile = Profile(user_id=user_id, version=version, entries=entries, updated_at=ts)
        self._cache[user_id] = profile
        return profile

    def replace_all(self, user_id: str, entries: list[dict[str, Any]], reason: str = "管理接口整体替换") -> Profile:
        new_entries = [ProfileEntry.from_dict({**e, "source": e.get("source", "admin")}) for e in entries]

        def fn(current: list[ProfileEntry]) -> bool:
            current[:] = new_entries
            return True

        return self.mutate(user_id, fn, reason)[0]

    def add_entry(self, user_id: str, entry: dict[str, Any], reason: str = "管理接口新增") -> tuple[Profile, ProfileEntry]:
        new = ProfileEntry.from_dict({"source": "admin", "updated_at": now(), **entry})
        new.id = new_entry_id()

        def fn(entries: list[ProfileEntry]) -> bool:
            if new.slot:
                entries[:] = [e for e in entries if e.slot != new.slot]
            entries.append(new)
            return True

        return self.mutate(user_id, fn, reason)[0], new

    def update_entry(self, user_id: str, entry_id: str, changes: dict[str, Any], reason: str = "管理接口修改") -> Profile:
        def fn(entries: list[ProfileEntry]) -> bool:
            for e in entries:
                if e.id == entry_id:
                    for k, v in changes.items():
                        if k in ProfileEntry.__dataclass_fields__ and k != "id":
                            setattr(e, k, v)
                    e.updated_at = now()
                    return True
            raise KeyError(entry_id)

        return self.mutate(user_id, fn, reason)[0]

    def delete_entry(self, user_id: str, entry_id: str, reason: str = "管理接口删除") -> Profile:
        def fn(entries: list[ProfileEntry]) -> bool:
            before = len(entries)
            entries[:] = [e for e in entries if e.id != entry_id]
            if len(entries) == before:
                raise KeyError(entry_id)
            return True

        return self.mutate(user_id, fn, reason)[0]

    def upsert_fact(
        self,
        user_id: str,
        *,
        text: str,
        slot: str,
        value: str,
        kind: str,
        privacy: str,
        confirmations: int,
        source_log_ids: list[str],
        event_date: str | None,
        reason: str,
    ) -> tuple[Profile, ProfileEntry | None]:
        """后台晋升事实：同一 slot 的旧值被替换；无 slot 时与相似条目合并。"""
        result: dict[str, ProfileEntry] = {}

        def fn(entries: list[ProfileEntry]) -> bool:
            ts = now()
            target: ProfileEntry | None = None
            if slot:
                target = next((e for e in entries if e.slot == slot), None)
            else:
                target = next(
                    (e for e in entries if not e.slot and same_value(e.text, text, self.cfg.match_threshold)), None
                )
            if target is None:
                target = ProfileEntry(
                    id=new_entry_id(), text=text, slot=slot, value=value, kind=kind, privacy=privacy,
                    confirmations=confirmations, source_log_ids=list(source_log_ids), event_date=event_date,
                    updated_at=ts,
                )
                entries.append(target)
                result["entry"] = target
                return True
            if target.pinned and target.source == "admin":
                # 手工置顶的条目不被后台覆盖
                return False
            changed = target.text != text or target.privacy != _merge_privacy(target.privacy, privacy)
            target.text = text
            target.value = value
            target.kind = kind
            target.privacy = _merge_privacy(target.privacy, privacy)
            target.confirmations = max(target.confirmations, confirmations)
            target.source_log_ids = sorted(set(target.source_log_ids) | set(source_log_ids))[-20:]
            target.event_date = event_date
            target.updated_at = ts
            result["entry"] = target
            return changed

        profile, _ = self.mutate(user_id, fn, reason)
        return profile, result.get("entry")

    # ------------------------------------------------------------------ 历史
    def versions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            """SELECT version, reason, created_at, length(content) AS size FROM profile_version
               WHERE user_id = ? ORDER BY version DESC LIMIT ?""",
            (user_id, limit),
        )
        return rows

    def get_version(self, user_id: str, version: int) -> list[dict[str, Any]] | None:
        row = self.db.query_one(
            "SELECT content FROM profile_version WHERE user_id = ? AND version = ?", (user_id, version)
        )
        if row is None:
            return None
        return (loads(row["content"], {}) or {}).get("entries", [])

    def rollback(self, user_id: str, version: int) -> Profile:
        entries = self.get_version(user_id, version)
        if entries is None:
            raise KeyError(f"{user_id} 没有版本 v{version}")
        restored = [ProfileEntry.from_dict(e) for e in entries]

        def fn(current: list[ProfileEntry]) -> bool:
            current[:] = restored
            return True

        return self.mutate(user_id, fn, f"回滚到 v{version}")[0]

    def users_with_profiles(self) -> list[str]:
        return [r["user_id"] for r in self.db.query("SELECT user_id FROM core_profile")]

    def invalidate(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id is None:
                self._cache.clear()
            else:
                self._cache.pop(user_id, None)


def _merge_privacy(old: str, new: str) -> str:
    """同一事实只要在公开场合出现过，就视为公开信息。"""
    return PUBLIC if PUBLIC in (old, new) else PRIVATE
