"""身份解析：把（平台, 外部 ID）映射为统一的 user_id。

- 配置文件里登记的用户（例如主人）在启动时写入身份表。
- 未登记的人自动建为访客，user_id 形如 ``qq:123456``。
- 拿不到外部 ID、或声纹未确认的语音，按匿名访客处理：不读私密记忆，也不写入记忆。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..config import Settings
from ..memory.db import Database
from ..timeutil import now
from .models import PRIVATE, PUBLIC, Identity, RequestMeta, Scene

log = logging.getLogger(__name__)

ANONYMOUS_IDS = {"", "unknown", "anonymous", "none", "null", "guest"}


class IdentityResolver:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.self_user_id = settings.self_user_id
        self._cache: dict[tuple[str, str], Identity] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 初始化
    def seed(self) -> None:
        ts = now()
        self._upsert_user(self.self_user_id, "她自己", "self", managed=True, ts=ts)
        for user in self.settings.users:
            self._upsert_user(user.user_id, user.display_name or user.user_id, user.role, managed=True, ts=ts)
            for binding in user.bindings:
                if ":" not in binding:
                    log.warning("忽略格式不正确的身份绑定 %r（应为 平台:外部ID）", binding)
                    continue
                platform, external_id = binding.split(":", 1)
                self.link(platform.strip(), external_id.strip(), user.user_id)

    def _upsert_user(self, user_id: str, display_name: str, role: str, managed: bool, ts: float) -> None:
        self.db.execute(
            """INSERT INTO users(user_id, display_name, role, managed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET display_name = excluded.display_name,
                   role = excluded.role, managed = excluded.managed, updated_at = excluded.updated_at""",
            (user_id, display_name, role, int(managed), ts, ts),
        )
        self._invalidate()

    # ------------------------------------------------------------------ 解析
    def resolve(self, meta: RequestMeta) -> Identity:
        platform = meta.channel or "unknown"
        external = (meta.external_user_id or "").strip()
        if platform == "voice":
            verified = meta.speaker_verified is True or self.settings.memory.trust_unverified_voice
            if not verified:
                external = ""
            elif not external:
                external = "owner"
        if external.lower() in ANONYMOUS_IDS:
            return Identity(
                user_id=f"anon:{platform}",
                display_name=meta.user_name or "陌生人",
                role="guest",
                platform=platform,
                external_id="",
                anonymous=True,
            )

        key = (platform, external)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            if meta.user_name and cached.role == "guest" and cached.display_name != meta.user_name:
                self._rename_guest(cached, meta.user_name)
            return cached

        row = self.db.query_one(
            """SELECT u.user_id, u.display_name, u.role FROM identities i
               JOIN users u ON u.user_id = i.user_id
               WHERE i.platform = ? AND i.external_id = ?""",
            key,
        )
        if row is None:
            row = self._create_guest(platform, external, meta.user_name)
        identity = Identity(
            user_id=row["user_id"],
            display_name=row["display_name"] or meta.user_name or external,
            role=row["role"],
            platform=platform,
            external_id=external,
        )
        with self._lock:
            self._cache[key] = identity
        if meta.user_name and identity.role == "guest" and identity.display_name != meta.user_name:
            self._rename_guest(identity, meta.user_name)
        return identity

    def _create_guest(self, platform: str, external: str, name: str | None) -> dict[str, Any]:
        user_id = f"{platform}:{external}"
        ts = now()
        display = name or external
        with self.db.transaction():
            self.db.execute(
                """INSERT OR IGNORE INTO users(user_id, display_name, role, managed, created_at, updated_at)
                   VALUES (?, ?, 'guest', 0, ?, ?)""",
                (user_id, display, ts, ts),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO identities(platform, external_id, user_id, created_at) VALUES (?, ?, ?, ?)",
                (platform, external, user_id, ts),
            )
        log.info("新访客 %s（%s）", user_id, display)
        row = self.db.query_one("SELECT user_id, display_name, role FROM users WHERE user_id = ?", (user_id,))
        assert row is not None
        return row

    def _rename_guest(self, identity: Identity, name: str) -> None:
        self.db.execute(
            "UPDATE users SET display_name = ?, updated_at = ? WHERE user_id = ? AND managed = 0",
            (name, now(), identity.user_id),
        )
        identity.display_name = name

    def build_scene(self, meta: RequestMeta, identity: Identity) -> Scene:
        scene = meta.scene or "private"
        if identity.anonymous or scene == "group":
            access = PUBLIC
        else:
            access = PRIVATE
        write_privacy = PUBLIC if scene == "group" else PRIVATE
        return Scene(
            channel=meta.channel,
            scene=scene,
            group_id=meta.group_id,
            group_name=meta.group_name,
            speaker_verified=meta.speaker_verified,
            access=access,
            write_privacy=write_privacy,
        )

    # ------------------------------------------------------------------ 管理
    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if row is None:
            if user_id.startswith("anon:"):
                return {"user_id": user_id, "display_name": "陌生人", "role": "guest", "anonymous": True}
            return None
        row["identities"] = self.db.query(
            "SELECT platform, external_id, created_at FROM identities WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        )
        return row

    def display_name(self, user_id: str) -> str:
        row = self.db.query_one("SELECT display_name FROM users WHERE user_id = ?", (user_id,))
        return (row or {}).get("display_name") or user_id

    def list_users(self) -> list[dict[str, Any]]:
        users = self.db.query("SELECT * FROM users ORDER BY role = 'owner' DESC, created_at")
        idents = self.db.query("SELECT platform, external_id, user_id FROM identities")
        by_user: dict[str, list[str]] = {}
        for i in idents:
            by_user.setdefault(i["user_id"], []).append(f"{i['platform']}:{i['external_id']}")
        for u in users:
            u["bindings"] = by_user.get(u["user_id"], [])
        return users

    def create_user(self, user_id: str, display_name: str, role: str = "friend") -> dict[str, Any]:
        self._upsert_user(user_id, display_name, role, managed=False, ts=now())
        return self.get_user(user_id) or {}

    def link(self, platform: str, external_id: str, user_id: str, move_logs: bool = False) -> dict[str, Any]:
        """把一个平台身份绑定到 user_id。move_logs 时把旧访客的原始日志一并迁移并重新提取。"""
        ts = now()
        old = self.db.query_one(
            "SELECT user_id FROM identities WHERE platform = ? AND external_id = ?", (platform, external_id)
        )
        with self.db.transaction():
            self.db.execute(
                """INSERT INTO identities(platform, external_id, user_id, created_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(platform, external_id) DO UPDATE SET user_id = excluded.user_id""",
                (platform, external_id, user_id, ts),
            )
            moved = 0
            if move_logs and old and old["user_id"] != user_id:
                cur = self.db.execute(
                    "UPDATE raw_log SET user_id = ?, processed = 0, attempts = 0 WHERE user_id = ?",
                    (user_id, old["user_id"]),
                )
                moved = cur.rowcount
        self._invalidate()
        return {"platform": platform, "external_id": external_id, "user_id": user_id,
                "previous_user_id": old["user_id"] if old else None, "moved_logs": moved}

    def unlink(self, platform: str, external_id: str) -> bool:
        cur = self.db.execute(
            "DELETE FROM identities WHERE platform = ? AND external_id = ?", (platform, external_id)
        )
        self._invalidate()
        return cur.rowcount > 0

    def _invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
