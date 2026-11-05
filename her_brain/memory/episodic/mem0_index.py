"""Mem0（开源版）作为情节记忆的检索引擎。

适配 Mem0 2.x：
- 新增时使用 ``infer=False``：事实已由我们自己的提取器拆成原子句并标好重要度、私密度、日期，
  不再让 Mem0 二次调用 LLM。Mem0 2.x 的 add() 已改为“只追加”流水线，不再做 UPDATE/DELETE 判断，
  纠错与冲突处理统一由本项目的档案晋升（多次确认 + slot 替换）与事件版本机制负责。
- 检索使用 Mem0 的混合打分（向量 + BM25 + 实体），不调用 LLM。
- 关闭 Mem0 的遥测上报（MEM0_TELEMETRY=False），保证数据不出本机。
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any
from urllib.parse import urlparse

from ...config import Settings
from ..models import MemoryRecord

log = logging.getLogger(__name__)

# 尽量在 mem0 被导入之前关闭遥测；若 mem0 已被别处导入，_disable_mem0_telemetry 会再强制关闭一次
os.environ.setdefault("MEM0_TELEMETRY", "False")


def _disable_mem0_telemetry() -> None:
    """Mem0 默认会向 PostHog 上报匿名使用数据。这里无论导入顺序如何都强制关闭，保证数据不出本机。"""
    os.environ["MEM0_TELEMETRY"] = "False"
    try:
        import mem0.memory.main as mem0_main
        import mem0.memory.telemetry as mem0_telemetry

        mem0_telemetry.MEM0_TELEMETRY = False
        mem0_main.MEM0_TELEMETRY = False
    except Exception:  # noqa: BLE001 - 不同版本结构不同，尽力而为
        log.debug("未能定位 Mem0 遥测开关", exc_info=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_mem0_config(settings: Settings) -> dict[str, Any]:
    mem = settings.memory
    emb = settings.embedding
    worker = settings.worker_endpoint
    if mem.qdrant_path:
        vs_cfg: dict[str, Any] = {"collection_name": mem.collection, "path": mem.qdrant_path, "on_disk": True}
    else:
        parsed = urlparse(mem.qdrant_url)
        vs_cfg = {"collection_name": mem.collection, "host": parsed.hostname or "127.0.0.1", "port": parsed.port or 6333}
    vs_cfg["embedding_model_dims"] = emb.dims
    config = {
        "vector_store": {"provider": "qdrant", "config": vs_cfg},
        "embedder": {
            "provider": "openai",
            "config": {
                "model": emb.model,
                "openai_base_url": emb.base_url,
                "api_key": emb.api_key,
                "embedding_dims": emb.dims,
            },
        },
        # infer=False 时不会调用，但 Mem0 初始化时必须配置一个 LLM
        "llm": {
            "provider": "openai",
            "config": {
                "model": worker.model,
                "openai_base_url": worker.base_url,
                "api_key": worker.api_key,
                "temperature": 0.1,
            },
        },
        "history_db_path": str(settings.data_path / "mem0_history.db"),
    }
    return deep_merge(config, mem.mem0)


def _results(res: Any) -> list[dict[str, Any]]:
    if isinstance(res, dict):
        res = res.get("results", [])
    return [r for r in (res or []) if isinstance(r, dict)]


class Mem0Index:
    name = "mem0"
    default_min_score = 0.35  # 作用于余弦相似度
    default_dedupe_score = 0.92

    def __init__(self, config: dict[str, Any], max_workers: int = 4, memory: Any = None):
        if memory is None:
            try:
                from mem0 import Memory
            except ImportError as exc:  # pragma: no cover - 取决于安装
                raise RuntimeError("memory.engine=mem0 需要安装 mem0ai 与 qdrant-client：pip install -e .[mem0]") from exc
            _disable_mem0_telemetry()
            # spaCy（英文分词/实体）与 fastembed（BM25）缺失时 Mem0 会打警告；它们对中文检索没有帮助，
            # 向量检索不受影响，屏蔽掉以免在代理窗口里误导排查
            for name in ("mem0.utils.spacy_models", "mem0.vector_stores.qdrant"):
                logging.getLogger(name).setLevel(logging.ERROR)
            memory = Memory.from_config(config)
        self.memory = memory
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mem0")
        # Qdrant 本地嵌入模式不是线程安全的（后台写入与对话检索会同时发生），所有调用串行执行
        self._lock = threading.Lock()

    def _locked(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(*args, **kwargs)

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, partial(self._locked, fn, *args, **kwargs))

    async def add(self, rec: MemoryRecord) -> tuple[str, list[float] | None]:
        res = await self._run(
            self.memory.add,
            [{"role": "user", "content": rec.text}],
            user_id=rec.user_id,
            metadata=rec.index_metadata(),
            infer=False,
        )
        items = _results(res)
        if not items or not items[0].get("id"):
            raise RuntimeError(f"Mem0 未返回记忆 ID：{res!r}")
        return str(items[0]["id"]), None

    async def update(self, rec: MemoryRecord) -> list[float] | None:
        await self._run(self.memory.update, rec.memory_id, text=rec.text, metadata=rec.index_metadata())
        return None

    async def remove(self, memory_id: str) -> None:
        await self._run(self.memory.delete, memory_id)

    def _search_sync(self, user_ids: list[str], query: str, limit: int, min_score: float) -> list[tuple[str, float]]:
        """只算一次查询向量，直接按用户查询底层向量库。

        Mem0 自带的 search() 对每个用户各算一次向量，还会做 BM25 与实体加权（依赖英文 spaCy，
        对中文没有帮助），在读路径上白白多花几十毫秒。
        """
        store = getattr(self.memory, "vector_store", None)
        embedder = getattr(self.memory, "embedding_model", None)
        out: list[tuple[str, float]] = []
        if store is None or embedder is None or not hasattr(store, "search"):
            for uid in user_ids:
                res = self.memory.search(query, top_k=limit, filters={"user_id": uid}, threshold=min_score)
                out += [(str(i["id"]), float(i.get("score") or 0.0)) for i in _results(res) if i.get("id") is not None]
            return out
        vector = embedder.embed(query, "search")
        for uid in user_ids:
            for point in store.search(query=query, vectors=vector, top_k=limit, filters={"user_id": uid}):
                score = float(getattr(point, "score", 0.0) or 0.0)
                if score >= min_score:
                    out.append((str(point.id), score))
        return out

    async def search(self, user_ids: list[str], query: str, limit: int, min_score: float) -> list[tuple[str, float]]:
        out = await self._run(self._search_sync, user_ids, query, limit, min_score)
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]

    def invalidate(self, user_id: str) -> None:
        return None

    async def rebuild(self, records: list[MemoryRecord]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for uid in sorted({r.user_id for r in records}):
            await self._run(self.memory.delete_all, user_id=uid)
        for rec in records:
            new_id, _ = await self.add(rec)
            if new_id != rec.memory_id:
                mapping[rec.memory_id] = new_id
        log.info("Mem0 索引已重建：%d 条", len(records))
        return mapping

    async def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        # Mem0 的 close() 只关自己的 SQLite；Qdrant 客户端若留到解释器退出时由 __del__ 清理，
        # 会打印 “sys.meta_path is None” 之类的报错（本地嵌入模式下还会晚释放文件锁）
        for store in (getattr(self.memory, "vector_store", None), getattr(self.memory, "_entity_store", None)):
            client = getattr(store, "client", None)
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    log.debug("关闭 Qdrant 客户端失败", exc_info=True)
        if hasattr(self.memory, "close"):
            try:
                self.memory.close()
            except Exception:  # noqa: BLE001
                log.debug("关闭 Mem0 失败", exc_info=True)
