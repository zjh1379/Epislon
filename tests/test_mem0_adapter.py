"""用真实的 Mem0（本地 Qdrant 嵌入模式）验证适配层；未安装 mem0ai 时跳过。

嵌入服务由一个在后台线程中运行的假 OpenAI 兼容服务提供（Mem0 通过 openai SDK 访问它）。
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("mem0")
pytest.importorskip("qdrant_client")

import uvicorn  # noqa: E402

from her_brain.devtools.fake_llm import FakeLLM, create_app  # noqa: E402
from her_brain.identity.models import PRIVATE, PUBLIC  # noqa: E402
from her_brain.memory.models import Fact  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def fake_server() -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(FakeLLM()), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
async def mem0_services(make_settings, fake_server, tmp_path):  # type: ignore[no-untyped-def]
    from her_brain.services import Services

    settings = make_settings(
        upstream={"base_url": fake_server},
        embedding={"enabled": True, "base_url": fake_server, "dims": 256, "model": "fake-embed"},
        memory={"engine": "mem0", "qdrant_path": str(tmp_path / "qdrant"), "retrieval_timeout_ms": 5000},
    )
    svc = Services(settings)
    yield svc
    await svc.close()


async def test_mem0_add_search_privacy_and_versions(mem0_services) -> None:  # type: ignore[no-untyped-def]
    ep = mem0_services.episodic
    assert mem0_services.index.name == "mem0"
    status, rec = await ep.remember("owner", Fact(text="小明最近在准备考研", kind="plan"), PRIVATE, ["l1"])
    assert status == "added" and len(rec.memory_id) > 8  # Mem0 生成的 UUID
    await ep.remember("owner", Fact(text="小明喜欢周末去爬山"), PUBLIC, ["l2"])
    await ep.remember("_self", Fact(text="小E最近在学画画", about="self"), PUBLIC, ["l3"])

    hits = await ep.search(["owner", "_self"], "小明最近在准备考研吗", {PUBLIC, PRIVATE}, 5)
    assert hits and hits[0].record.memory_id == rec.memory_id
    public_hits = await ep.search(["owner", "_self"], "小明最近在准备考研吗", {PUBLIC}, 5)
    assert rec.memory_id not in {h.record.memory_id for h in public_hits}

    # 重复事实：只累计次数，不在 Mem0 中新增
    status, dup = await ep.remember("owner", Fact(text="小明最近在准备考研"), PRIVATE, ["l4"])
    assert status == "merged" and dup.memory_id == rec.memory_id

    # 修改会同步到 Mem0；撤销后恢复
    await ep.edit(rec.memory_id, {"text": "小明最近在准备出国"})
    got = mem0_services.index.memory.get(rec.memory_id)
    assert got["memory"] == "小明最近在准备出国"
    upd = next(e for e in ep.events(memory_id=rec.memory_id) if e["note"] == "手动修改")
    await ep.revert_event(upd["event_id"])
    assert mem0_services.index.memory.get(rec.memory_id)["memory"] == "小明最近在准备考研"


async def test_mem0_close_releases_qdrant_lock(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """回归：Mem0 的 close() 不关 Qdrant 客户端。本地嵌入模式的文件锁会一直占着（同一目录无法再打开），
    并且要等解释器退出时由 __del__ 释放，在 Windows 上打印 “Exception ignored in QdrantClient.__del__”。"""
    from her_brain.config import Settings
    from her_brain.memory.episodic.mem0_index import Mem0Index, build_mem0_config

    (tmp_path / "data").mkdir()
    s = Settings.model_validate({"data_dir": str(tmp_path / "data"), "embedding": {"enabled": True},
                                 "memory": {"engine": "mem0", "qdrant_path": str(tmp_path / "q")}})
    first = Mem0Index(build_mem0_config(s))
    with pytest.raises(RuntimeError, match="already accessed"):
        Mem0Index(build_mem0_config(s))  # 未关闭时目录被锁
    await first.close()
    second = Mem0Index(build_mem0_config(s))  # 关闭后可以重新打开
    await second.close()


async def test_qdrant_locked_by_another_process_has_clear_message(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from her_brain.config import Settings
    from her_brain.memory.db import Database
    from her_brain.memory.episodic import create_index

    (tmp_path / "data").mkdir()
    s = Settings.model_validate({"data_dir": str(tmp_path / "data"), "embedding": {"enabled": True},
                                 "memory": {"engine": "mem0", "qdrant_path": str(tmp_path / "q")}})
    db = Database(s.db_path)
    first = create_index(s, db, None)
    try:
        with pytest.raises(RuntimeError, match="python -m her_brain stop"):
            create_index(s, db, None)
    finally:
        await first.close()
        db.close()


async def test_mem0_reindex_from_sqlite(mem0_services) -> None:  # type: ignore[no-untyped-def]
    ep = mem0_services.episodic
    _, rec = await ep.remember("owner", Fact(text="小明养了一只猫叫团子"), PRIVATE, ["l1"])
    result = await ep.rebuild_index()
    assert result["records"] == 1
    new_id = mem0_services.db.scalar("SELECT memory_id FROM episodic_memory")
    assert mem0_services.db.scalar("SELECT COUNT(*) FROM memory_event WHERE memory_id = ?", (new_id,)) == 1
    # 假嵌入只是字符二元组哈希，没有真实语义，查询需要与原文字面接近
    hits = await ep.search(["owner"], "小明养了一只猫叫什么", {PRIVATE}, 3)
    assert hits and hits[0].record.memory_id == new_id
