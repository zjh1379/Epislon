"""OpenAI 兼容的嵌入接口客户端（例如以 --embedding 启动的 llama-server）。"""

from __future__ import annotations

import httpx

from ..config import EmbeddingConfig


class EmbeddingClient:
    def __init__(self, cfg: EmbeddingConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=httpx.Timeout(cfg.timeout, connect=min(cfg.timeout, 2.0)),
            transport=transport,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.post("/embeddings", json={"model": self.cfg.model, "input": texts})
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data.sort(key=lambda d: d.get("index", 0))
        return [list(map(float, d["embedding"])) for d in data]

    async def aclose(self) -> None:
        await self._client.aclose()
