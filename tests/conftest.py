from __future__ import annotations

import copy
import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from her_brain.app import create_app
from her_brain.config import Settings
from her_brain.devtools.fake_llm import FakeLLM
from her_brain.services import Services, Transports

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    persona = tmp_path / "persona" / "her.yaml"
    persona.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "persona" / "her.yaml", persona)

    def _make(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "data_dir": str(tmp_path / "data"),
            "persona_path": str(persona),
            "users": [
                {"user_id": "owner", "display_name": "小明", "role": "owner", "bindings": ["qq:10001", "voice:owner"]}
            ],
            "upstream": {"base_url": "http://fake-llm/v1", "model": "fake-model", "warmup": False},
            "worker": {"enabled": False},
            "backup": {"dir": str(tmp_path / "backups")},
            "memory": {"engine": "lite"},
        }
        return Settings.model_validate(deep_merge(base, overrides))

    return _make


@pytest.fixture
def fake() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest_asyncio.fixture
async def services(settings: Settings, fake: FakeLLM) -> AsyncIterator[Services]:
    svc = Services(settings, transports=Transports(upstream=fake.transport(), embedding=fake.transport()))
    yield svc
    await svc.close()


@pytest_asyncio.fixture
async def client(services: Services) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(services.settings, services, run_worker=False)
    app.state.services = services  # ASGITransport 不触发 lifespan，直接挂上
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://brain") as c:
        yield c


def chat_body(text: str, *, metadata: dict[str, Any] | None = None, stream: bool = False,
              system: str | None = None, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages += history or []
    messages.append({"role": "user", "content": text})
    body: dict[str, Any] = {"model": "her", "messages": messages, "stream": stream}
    if metadata is not None:
        body["metadata"] = metadata
    return body


QQ_PRIVATE = {"channel": "qq", "scene": "private", "external_user_id": "10001", "user_name": "小明"}
QQ_GROUP = {"channel": "qq", "scene": "group", "external_user_id": "10001", "user_name": "小明", "group_id": "777",
            "group_name": "测试群"}
VOICE_OWNER = {"channel": "voice", "scene": "voice", "external_user_id": "owner", "speaker_verified": True}
