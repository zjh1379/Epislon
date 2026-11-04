"""轻量检索引擎：纯 SQLite + 字符二元组覆盖率，可选叠加向量相似度。

不依赖 Mem0/Qdrant，适合开发、测试，或在没有嵌入模型时作为退路。
单用户几千条记忆以内，一次检索在数毫秒内完成。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid

from ...llm.embedding import EmbeddingClient
from ...textutil import LexicalIndex, cosine
from ..db import Database
from ..models import MemoryRecord

log = logging.getLogger(__name__)


class _UserCache:
    def __init__(self, lexical: LexicalIndex, vectors: dict[str, list[float]]):
        self.lexical = lexical
        self.vectors = vectors


class LiteIndex:
    name = "lite"

    def __init__(self, db: Database, embedder: EmbeddingClient | None = None):
        self.db = db
        self.embedder = embedder
        self._cache: dict[str, _UserCache] = {}
        self._lock = threading.Lock()

    @property
    def default_min_score(self) -> float:
        return 0.45 if self.embedder else 0.3

    default_dedupe_score = None

    async def _embed_one(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            vecs = await self.embedder.embed([text])
            return vecs[0] if vecs else None
        except Exception as exc:  # noqa: BLE001 - 嵌入失败时退回纯文本检索
            log.warning("计算向量失败，本条记忆只走文本检索：%s", exc)
            return None

    async def add(self, rec: MemoryRecord) -> tuple[str, list[float] | None]:
        memory_id = rec.memory_id or "m_" + uuid.uuid4().hex[:16]
        vec = await self._embed_one(rec.text)
        self.invalidate(rec.user_id)
        return memory_id, vec

    async def update(self, rec: MemoryRecord) -> list[float] | None:
        vec = await self._embed_one(rec.text)
        self.invalidate(rec.user_id)
        return vec

    async def remove(self, memory_id: str) -> None:
        with self._lock:
            self._cache.clear()

    def invalidate(self, user_id: str) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def _load(self, user_id: str) -> _UserCache:
        with self._lock:
            cached = self._cache.get(user_id)
        if cached is not None:
            return cached
        rows = self.db.query(
            "SELECT memory_id, text, embedding FROM episodic_memory WHERE user_id = ? AND deleted = 0",
            (user_id,),
        )
        vectors: dict[str, list[float]] = {}
        for r in rows:
            if r["embedding"]:
                try:
                    vectors[r["memory_id"]] = json.loads(r["embedding"])
                except ValueError:
                    pass
        cache = _UserCache(LexicalIndex((r["memory_id"], r["text"]) for r in rows), vectors)
        with self._lock:
            self._cache[user_id] = cache
        return cache

    async def search(self, user_ids: list[str], query: str, limit: int, min_score: float) -> list[tuple[str, float]]:
        qvec = await self._embed_one(query) if self.embedder else None
        results: list[tuple[str, float]] = []
        for uid in user_ids:
            cache = self._load(uid)
            lexical = cache.lexical.score(query)
            if qvec is not None and cache.vectors:
                ids = set(lexical) | set(cache.vectors)
                for mid in ids:
                    sem = cosine(qvec, cache.vectors[mid]) if mid in cache.vectors else 0.0
                    score = 0.65 * sem + 0.35 * lexical.get(mid, 0.0)
                    if score >= min_score:
                        results.append((mid, score))
            else:
                results.extend((mid, s) for mid, s in lexical.items() if s >= min_score)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def rebuild(self, records: list[MemoryRecord]) -> dict[str, str]:
        if self.embedder is not None:
            for rec in records:
                vec = await self._embed_one(rec.text)
                if vec is not None:
                    await asyncio.to_thread(
                        self.db.execute,
                        "UPDATE episodic_memory SET embedding = ? WHERE memory_id = ?",
                        (json.dumps(vec), rec.memory_id),
                    )
        with self._lock:
            self._cache.clear()
        return {}

    async def close(self) -> None:
        return None
