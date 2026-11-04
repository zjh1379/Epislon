"""每日快照与恢复。

快照内容：
- brain.db：SQLite 在线备份（身份、核心档案及其历史、候选、原始日志、情节记忆权威副本、变更事件）
- persona/：人设 YAML
- 配置文件
- mem0_history.db：Mem0 自己的历史库（若存在）
- qdrant/：Qdrant 集合快照（使用 Qdrant 服务端时，尽力而为）

向量索引始终可以由 brain.db 中的情节记忆权威副本重建（``python -m her_brain reindex``），
因此恢复时默认直接重建索引，而不依赖 Qdrant 快照。
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .memory.db import Database
from .timeutil import get_tz, now, to_local

log = logging.getLogger(__name__)

META_LAST_DAILY = "last_daily_backup"


class BackupManager:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.cfg = settings.backup
        self.db = db
        self.root = Path(self.cfg.dir)
        self.tz = get_tz(settings.prompt.timezone)

    # ------------------------------------------------------------------ 快照
    def snapshot(self, label: str = "manual") -> dict[str, Any]:
        ts = now()
        stamp = to_local(ts, self.tz).strftime("%Y%m%d-%H%M%S")
        target = self.root / f"{stamp}-{label}"
        suffix = 1
        while target.exists():
            target = self.root / f"{stamp}-{label}-{suffix}"
            suffix += 1
        target.mkdir(parents=True)
        manifest: dict[str, Any] = {"created_at": ts, "label": label, "files": [], "warnings": []}

        self.db.backup_to(target / "brain.db")
        manifest["files"].append("brain.db")

        persona = Path(self.settings.persona_path)
        if persona.exists():
            (target / "persona").mkdir()
            shutil.copy2(persona, target / "persona" / persona.name)
            manifest["files"].append(f"persona/{persona.name}")
            manifest["persona_file"] = persona.name

        if self.settings.config_path and Path(self.settings.config_path).exists():
            shutil.copy2(self.settings.config_path, target / "config.yaml")
            manifest["files"].append("config.yaml")

        mem0_hist = self.settings.data_path / "mem0_history.db"
        if mem0_hist.exists():
            try:
                _sqlite_copy(mem0_hist, target / "mem0_history.db")
                manifest["files"].append("mem0_history.db")
            except Exception as exc:  # noqa: BLE001
                manifest["warnings"].append(f"mem0_history.db 备份失败：{exc}")

        if self.settings.memory.engine == "mem0" and self.cfg.qdrant_snapshot and not self.settings.memory.qdrant_path:
            try:
                name = self._qdrant_snapshot(target / "qdrant")
                manifest["files"].append(f"qdrant/{name}")
            except Exception as exc:  # noqa: BLE001 - 向量索引可从 brain.db 重建，失败不影响快照有效性
                manifest["warnings"].append(f"Qdrant 快照失败（可用 reindex 从 brain.db 重建）：{exc}")

        manifest["counts"] = {
            "users": self.db.scalar("SELECT COUNT(*) FROM users"),
            "raw_log": self.db.scalar("SELECT COUNT(*) FROM raw_log"),
            "episodic_memory": self.db.scalar("SELECT COUNT(*) FROM episodic_memory WHERE deleted = 0"),
            "profiles": self.db.scalar("SELECT COUNT(*) FROM core_profile"),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if label == "daily":
            self.db.set_meta(META_LAST_DAILY, to_local(ts, self.tz).date().isoformat())
        removed = self.prune()
        log.info("快照完成：%s（清理旧快照 %d 个）", target, removed)
        return {"path": str(target), **manifest}

    def _qdrant_snapshot(self, dest_dir: Path) -> str:
        base = self.settings.memory.qdrant_url.rstrip("/")
        coll = self.settings.memory.collection
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{base}/collections/{coll}/snapshots")
            resp.raise_for_status()
            name = resp.json()["result"]["name"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            with client.stream("GET", f"{base}/collections/{coll}/snapshots/{name}") as r:
                r.raise_for_status()
                with open(dest_dir / name, "wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
            try:  # 服务端的快照文件已经下载到本地，删掉以免占用空间
                client.delete(f"{base}/collections/{coll}/snapshots/{name}")
            except httpx.HTTPError:
                pass
        return name

    # ------------------------------------------------------------------ 管理
    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for d in sorted(self.root.iterdir(), reverse=True):
            mf = d / "manifest.json"
            if d.is_dir() and mf.exists():
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                out.append({"name": d.name, "path": str(d), **data})
        return out

    def prune(self) -> int:
        if self.cfg.keep_days <= 0 or not self.root.exists():
            return 0
        cutoff = now() - self.cfg.keep_days * 86400
        removed = 0
        for item in self.list():
            if item.get("label") == "daily" and float(item.get("created_at", 0)) < cutoff:
                shutil.rmtree(item["path"], ignore_errors=True)
                removed += 1
        return removed

    def due(self, now_ts: float | None = None) -> bool:
        if not self.cfg.enabled:
            return False
        local = to_local(now_ts or now(), self.tz)
        try:
            hh, mm = (int(x) for x in self.cfg.daily_at.split(":"))
        except ValueError:
            hh, mm = 4, 30
        if (local.hour, local.minute) < (hh, mm):
            return False
        return self.db.get_meta(META_LAST_DAILY) != local.date().isoformat()


def _sqlite_copy(src: Path, dest: Path) -> None:
    source = sqlite3.connect(str(src))
    target = sqlite3.connect(str(dest))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def restore_snapshot(settings: Settings, snapshot: str | Path, *, restore_persona: bool = True) -> dict[str, Any]:
    """把快照恢复到当前数据目录。必须在代理停止时执行。当前数据会先另存一份。"""
    snap = Path(snapshot)
    if not (snap / "brain.db").exists():
        candidates = [d for d in Path(settings.backup.dir).glob(f"{snapshot}*") if (d / "brain.db").exists()]
        if not candidates:
            raise FileNotFoundError(f"找不到快照 {snapshot}")
        snap = sorted(candidates)[-1]

    data = settings.data_path
    data.mkdir(parents=True, exist_ok=True)
    safety = Path(settings.backup.dir) / f"pre-restore-{datetime.now():%Y%m%d-%H%M%S}"
    safety.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []

    db_path = settings.db_path
    if db_path.exists():
        _sqlite_copy(db_path, safety / "brain.db")
    for suffix in ("-wal", "-shm"):
        side = Path(str(db_path) + suffix)
        if side.exists():
            side.unlink()
    shutil.copy2(snap / "brain.db", db_path)
    restored.append("brain.db")

    hist = snap / "mem0_history.db"
    if hist.exists():
        cur = data / "mem0_history.db"
        if cur.exists():
            shutil.copy2(cur, safety / "mem0_history.db")
        shutil.copy2(hist, cur)
        restored.append("mem0_history.db")

    if restore_persona:
        manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8")) if (snap / "manifest.json").exists() else {}
        name = manifest.get("persona_file")
        if name and (snap / "persona" / name).exists():
            persona = Path(settings.persona_path)
            if persona.exists():
                shutil.copy2(persona, safety / persona.name)
            persona.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap / "persona" / name, persona)
            restored.append(f"persona/{name}")

    (safety / "manifest.json").write_text(
        json.dumps({"created_at": now(), "label": "pre-restore", "files": ["brain.db"], "restored_from": str(snap)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"snapshot": str(snap), "restored": restored, "previous_data_saved_to": str(safety)}


def yesterday_snapshot(settings: Settings) -> Path | None:
    """最近一份早于今天零点的快照，用于“回滚到前一天”。"""
    tz = get_tz(settings.prompt.timezone)
    today_start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = (today_start - timedelta(seconds=0)).timestamp()
    mgr_root = Path(settings.backup.dir)
    best: tuple[float, Path] | None = None
    if not mgr_root.exists():
        return None
    for d in mgr_root.iterdir():
        mf = d / "manifest.json"
        if not mf.exists() or not (d / "brain.db").exists():
            continue
        try:
            created = float(json.loads(mf.read_text(encoding="utf-8")).get("created_at", 0))
        except ValueError:
            continue
        if created < cutoff and (best is None or created > best[0]):
            best = (created, d)
    return best[1] if best else None
