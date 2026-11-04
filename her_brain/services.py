"""把所有模块装配在一起。测试时可以注入假的上游/嵌入传输层与索引。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from .backup import BackupManager
from .config import Settings
from .extensions import ExtensionManager
from .identity.resolver import IdentityResolver
from .llm.client import LLMClient
from .llm.embedding import EmbeddingClient
from .memory.candidates import ProfilePromoter
from .memory.db import Database
from .memory.episodic import EpisodicMemory, VectorIndex, create_index
from .memory.profile import ProfileStore
from .memory.rawlog import RawLogStore
from .memory.recent import RecentActivity
from .memory.retrieval import MemoryReader
from .metrics import Metrics
from .prompt.builder import DYNAMIC_HEADER, PromptBuilder
from .prompt.persona import PersonaHolder
from .proxy import ChatService
from .worker.activity import ActivityTracker
from .worker.extractor import MemoryExtractor
from .worker.runner import BackgroundWorker

log = logging.getLogger(__name__)


@dataclass
class Transports:
    upstream: httpx.AsyncBaseTransport | None = None
    worker: httpx.AsyncBaseTransport | None = None
    embedding: httpx.AsyncBaseTransport | None = None


class Services:
    def __init__(self, settings: Settings, *, transports: Transports | None = None, index: VectorIndex | None = None):
        t = transports or Transports()
        tz = settings.prompt.timezone
        self.settings = settings
        settings.data_path.mkdir(parents=True, exist_ok=True)

        self.db = Database(settings.db_path)
        self.persona = PersonaHolder(settings.persona_path)
        self.resolver = IdentityResolver(self.db, settings)
        self.profiles = ProfileStore(self.db, settings.memory.profile, tz)
        self.promoter = ProfilePromoter(self.db, self.profiles, settings.memory.profile)
        self.rawlog = RawLogStore(self.db)
        self.recent = RecentActivity(self.rawlog, settings.memory.recent, tz)

        self.embedder = EmbeddingClient(settings.embedding, t.embedding) if settings.embedding.enabled else None
        self.index = index if index is not None else create_index(settings, self.db, self.embedder)
        self.episodic = EpisodicMemory(
            self.db,
            self.index,
            min_score=settings.memory.min_score,
            dedupe_score=settings.memory.dedupe_threshold,
            timezone=tz,
        )
        self.reader = MemoryReader(settings, self.profiles, self.episodic, self.recent)
        self.builder = PromptBuilder(self.persona, settings.prompt)

        self.upstream = LLMClient(settings.upstream, t.upstream)
        self.worker_llm = LLMClient(settings.worker_endpoint, t.worker or t.upstream)
        self.extractor = MemoryExtractor(self.worker_llm, settings, self.persona)

        self.activity = ActivityTracker()
        self.metrics = Metrics()
        self.extensions = ExtensionManager.load(settings.extensions)
        self.backup = BackupManager(settings, self.db)
        self.worker = BackgroundWorker(self)
        self.chat = ChatService(self)

        self.resolver.seed()
        self._bg: set[asyncio.Task[object]] = set()

    def warmup_messages(self) -> list[dict[str, str]]:
        static = self.persona.static_text
        if self.settings.prompt.dynamic_position == "system":
            return [{"role": "system", "content": f"{static}\n\n{DYNAMIC_HEADER}"}]
        return [{"role": "system", "content": static}]

    async def warmup(self) -> bool:
        if self.activity.busy:
            return False
        ok = await self.upstream.warmup(self.warmup_messages())
        if ok:
            log.info("静态前缀已预热")
        return ok

    def spawn(self, coro: object) -> None:
        task = asyncio.ensure_future(coro)  # type: ignore[arg-type]
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def start(self, run_worker: bool = True) -> None:
        await self.extensions.startup(self)
        if run_worker and self.settings.worker.enabled:
            self.worker.start()
        if self.settings.upstream.warmup:
            self.spawn(self.warmup())

    async def close(self) -> None:
        await self.worker.stop()
        for task in list(self._bg):
            task.cancel()
        await self.extensions.shutdown()
        await self.upstream.aclose()
        await self.worker_llm.aclose()
        if self.embedder is not None:
            await self.embedder.aclose()
        await self.index.close()
        self.db.close()
