from __future__ import annotations

from pathlib import Path

from her_brain.config import load_settings
from her_brain.prompt.persona import PersonaHolder

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_is_valid(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("UNUSED", "x")
    s = load_settings(ROOT / "config/brain.example.yaml")
    assert s.server.model_name == "her"
    assert s.memory.engine == "mem0" and s.embedding.enabled
    assert s.upstream.slot_id == 0 and s.worker_endpoint.slot_id == 1
    assert s.worker_endpoint.base_url == s.upstream.base_url  # 未填写时沿用 upstream
    assert {k.channel for k in s.api_keys} == {"qq", "voice"}
    assert s.users[0].bindings == ["qq:123456", "voice:owner"]
    assert s.platform_aliases["aiocqhttp"] == "qq"


def test_requirements_match_pyproject() -> None:
    """安装脚本在可编辑安装失败时退回到 requirements.txt，两者必须一致。"""
    import re

    import pytest

    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = list(project["dependencies"])
    for extra in ("mem0", "dev"):
        wanted += project["optional-dependencies"][extra]

    def name(req: str) -> str:
        return re.split(r"[<>=;\s\[]", req.strip(), maxsplit=1)[0].lower()

    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    have = {name(l) for l in lines if l and not l.startswith("#")}
    assert {name(r) for r in wanted} == have


def test_env_expansion(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server:\n  admin_token: ${HER_TOKEN}\nupstream:\n  base_url: ${HER_LLM:-http://x/v1}\n", encoding="utf-8")
    monkeypatch.setenv("HER_TOKEN", "secret")
    s = load_settings(cfg)
    assert s.server.admin_token == "secret" and s.upstream.base_url == "http://x/v1"


def test_default_persona_renders_deterministically() -> None:
    a = PersonaHolder(ROOT / "persona/her.yaml")
    b = PersonaHolder(ROOT / "persona/her.yaml")
    assert a.static_text == b.static_text and a.digest == b.digest
    assert a.static_text.startswith("# 你是谁\n你是小E。")
    assert "# 行为规则" in a.static_text and "语音对话：" in a.static_text


def test_persona_reload_keeps_old_on_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "her.yaml"
    p.write_text("name: 甲\nidentity: 测试\n", encoding="utf-8")
    holder = PersonaHolder(p)
    p.write_text("name: [这不是合法的\n", encoding="utf-8")
    ok, err = holder.reload()
    assert not ok and err and holder.name == "甲"
