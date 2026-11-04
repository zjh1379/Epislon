from __future__ import annotations

import json
import os
import time
from pathlib import Path

from her_brain.backup import restore_snapshot, yesterday_snapshot
from her_brain.identity.models import PRIVATE
from her_brain.memory.models import Fact


async def test_snapshot_and_restore(services) -> None:  # type: ignore[no-untyped-def]
    settings = services.settings
    await services.episodic.remember("owner", Fact(text="小明喜欢猫"), PRIVATE, ["1"])
    services.profiles.add_entry("owner", {"text": "名字叫小明", "slot": "名字"})
    snap = services.backup.snapshot("daily")
    path = Path(snap["path"])
    assert (path / "brain.db").exists() and (path / "persona" / "her.yaml").exists()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["episodic_memory"] == 1
    assert not services.backup.due()  # 今天的每日快照已完成

    # 快照之后又写坏了数据
    await services.episodic.remember("owner", Fact(text="一条错误的记忆"), PRIVATE, ["2"])
    services.profiles.add_entry("owner", {"text": "名字叫小红", "slot": "名字"})
    services.db.close()

    result = restore_snapshot(settings, path)
    assert "brain.db" in result["restored"]
    assert Path(result["previous_data_saved_to"], "brain.db").exists()

    from her_brain.memory.db import Database

    db = Database(settings.db_path)
    try:
        texts = [r["text"] for r in db.query("SELECT text FROM episodic_memory")]
        assert texts == ["小明喜欢猫"]
        assert db.scalar("SELECT version FROM core_profile WHERE user_id = 'owner'") == 1
    finally:
        db.close()
    # 让 fixture 的 close() 不再关闭已关闭的连接
    services.db = Database(settings.db_path)


def test_yesterday_snapshot_selection(make_settings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings()
    root = Path(settings.backup.dir)
    for name, created in (("old", time.time() - 3 * 86400), ("yday", time.time() - 86400 - 3600),
                          ("today", time.time())):
        d = root / name
        d.mkdir(parents=True)
        (d / "brain.db").write_bytes(b"")
        (d / "manifest.json").write_text(json.dumps({"created_at": created}), encoding="utf-8")
    assert yesterday_snapshot(settings) == root / "yday"


def test_prune_keeps_recent(services) -> None:  # type: ignore[no-untyped-def]
    mgr = services.backup
    first = Path(mgr.snapshot("daily")["path"])
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    manifest["created_at"] = time.time() - 30 * 86400
    (first / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    mgr.snapshot("manual")
    assert not first.exists() or os.listdir(first) == []
    assert len(mgr.list()) == 1
