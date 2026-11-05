"""情节记忆：权威副本在 SQLite，检索索引可选 Mem0（推荐）或轻量引擎。"""

from __future__ import annotations

from ...config import Settings
from ...llm.embedding import EmbeddingClient
from ..db import Database
from .index import VectorIndex
from .lite import LiteIndex
from .store import EpisodicMemory


def create_index(settings: Settings, db: Database, embedder: EmbeddingClient | None) -> VectorIndex:
    if settings.memory.engine == "mem0":
        if not settings.embedding.enabled:
            raise RuntimeError("memory.engine=mem0 需要启用 embedding（配置 embedding.enabled: true 与嵌入服务地址）")
        from .mem0_index import Mem0Index, build_mem0_config

        try:
            return Mem0Index(build_mem0_config(settings))
        except RuntimeError as exc:
            if "already accessed" in str(exc):
                raise RuntimeError(
                    f"{settings.memory.qdrant_path} 正被另一个进程占用（Qdrant 本地嵌入模式同一时间只允许一个进程打开）。"
                    "通常是之前启动的大脑代理还在运行：先运行 python -m her_brain stop，或关闭那个窗口后再试。"
                    "process / reindex / restore 等命令也需要先停止代理"
                ) from exc
            raise
        except Exception as exc:  # noqa: BLE001 - 把常见的连接失败翻译成可操作的提示
            where = settings.memory.qdrant_path or settings.memory.qdrant_url
            raise RuntimeError(
                f"初始化 Mem0 失败（Qdrant：{where}）：{exc}。请确认 Qdrant 已启动"
                "（deploy/docker-compose.qdrant.yml），或改用 memory.qdrant_path 本地嵌入模式，"
                "或临时改用 memory.engine: lite。"
            ) from exc
    return LiteIndex(db, embedder)


__all__ = ["EpisodicMemory", "LiteIndex", "VectorIndex", "create_index"]
