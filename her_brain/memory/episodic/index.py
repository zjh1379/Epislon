"""情节记忆的检索索引接口。

正文与元数据的权威副本在 SQLite 的 episodic_memory 表里；索引只负责“给一句话，找相关记忆的 ID”。
因此索引坏了可以随时从 SQLite 重建，删除/恢复只改 SQLite 的标记，不必动索引。
"""

from __future__ import annotations

from typing import Protocol

from ..models import MemoryRecord


class VectorIndex(Protocol):
    name: str
    # 检索分数下限、判定为重复的分数（None 表示只按文本相似度去重）
    default_min_score: float
    default_dedupe_score: float | None

    async def add(self, rec: MemoryRecord) -> tuple[str, list[float] | None]:
        """写入索引，返回（memory_id, 需要存入 SQLite 的向量或 None）。"""

    async def update(self, rec: MemoryRecord) -> list[float] | None:
        """正文或元数据变化后更新索引。"""

    async def remove(self, memory_id: str) -> None:
        """从索引中彻底删除（仅用于 purge，普通删除是软删除）。"""

    async def search(self, user_ids: list[str], query: str, limit: int, min_score: float) -> list[tuple[str, float]]:
        """返回（memory_id, 分数），可能包含已软删除的记忆，由调用方过滤。"""

    def invalidate(self, user_id: str) -> None:
        """SQLite 中该用户的记忆有变化（轻量引擎需要刷新缓存）。"""

    async def rebuild(self, records: list[MemoryRecord]) -> dict[str, str]:
        """按 SQLite 内容重建索引，返回 旧 ID -> 新 ID 的映射（ID 不变则省略）。"""

    async def close(self) -> None: ...
