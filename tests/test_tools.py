"""运维工具的测试：一键启动、环境自检、配置助手、端到端自测。"""

from __future__ import annotations

import io
import os
import shutil
import socket
import stat
import sys
from pathlib import Path

import pytest

from her_brain.config import LauncherConfig, load_settings
from her_brain.devtools.fake_llm import FakeLLM
from her_brain.doctor import print_report, run_doctor
from her_brain.launcher import (
    discover_models,
    find_llama_server,
    is_healthy,
    plan_backends,
    start_backends,
    stop_backends,
)
from her_brain.selftest import run_selftest, summarize
from her_brain.services import Transports
from her_brain.setup_wizard import run_setup, set_engine, set_qq

ROOT = Path(__file__).resolve().parents[1]


def _gguf(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(size)
    return path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------- 模型与 llama-server 发现


def test_discover_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _gguf(models / "Qwen3-14B-Q4_K_M.gguf", 3000)
    _gguf(models / "Qwen3-8B-Q4_K_M.gguf", 2000)
    _gguf(models / "Qwen3-Embedding-0.6B-Q8_0.gguf", 1000)
    _gguf(models / "mmproj-F16.gguf", 5000)
    _gguf(models / "sub" / "big-00002-of-00002.gguf", 9000)
    found = discover_models(LauncherConfig(models_dir=str(models)))
    assert found.chat and found.chat.name == "Qwen3-14B-Q4_K_M.gguf"
    assert found.embedding and found.embedding.name == "Qwen3-Embedding-0.6B-Q8_0.gguf"
    pinned = discover_models(LauncherConfig(models_dir=str(models), chat_model="Qwen3-8B-Q4_K_M.gguf"))
    assert pinned.chat and pinned.chat.name == "Qwen3-8B-Q4_K_M.gguf"
    assert discover_models(LauncherConfig(models_dir=str(tmp_path / "none"))).chat is None


def test_find_llama_server(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    exe = tmp_path / "llama-b9999-bin-win-cuda-13.4-x64" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    exe.parent.mkdir()
    exe.write_text("")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_llama_server(LauncherConfig()) == exe  # 项目目录下以 llama 开头的文件夹
    monkeypatch.setenv("LLAMA_SERVER", str(exe.parent))
    assert find_llama_server(LauncherConfig()) == exe
    assert find_llama_server(LauncherConfig(llama_server=str(tmp_path / "missing.exe"))) is None


def test_plan_backends_commands(make_settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    models = tmp_path / "models"
    _gguf(models / "chat.gguf", 10)
    _gguf(models / "Qwen3-Embedding-0.6B-Q8_0.gguf", 5)
    settings = make_settings(
        upstream={"base_url": "http://127.0.0.1:18180/v1"},
        embedding={"enabled": True, "base_url": "http://127.0.0.1:18181/v1"},
        launcher={"models_dir": str(models), "llama_server": "/opt/llama-server"},
    )
    chat, emb = plan_backends(settings)
    assert chat.cmd[:3] == ["/opt/llama-server", "-m", str(models / "chat.gguf")]
    assert chat.cmd[3:7] == ["--host", "127.0.0.1", "--port", "18180"] and "--reasoning-budget" in chat.cmd
    assert "--embedding" in emb.cmd and emb.cmd[-2:] == ["--pooling", "last"]
    assert len(plan_backends(make_settings(launcher={"models_dir": str(models)}))) == 1  # 未启用嵌入


# ---------------------------------------------------------------------- 真实子进程启动


@pytest.mark.skipif(os.name == "nt", reason="用 shebang 脚本模拟 llama-server")
def test_start_and_stop_backends(make_settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    script = _fake_llama_script(tmp_path)
    models = tmp_path / "models"
    _gguf(models / "chat.gguf", 10)
    _gguf(models / "bge-m3.gguf", 5)
    p1, p2 = _free_port(), _free_port()
    settings = make_settings(
        upstream={"base_url": f"http://127.0.0.1:{p1}/v1"},
        embedding={"enabled": True, "base_url": f"http://127.0.0.1:{p2}/v1"},
        launcher={"models_dir": str(models), "llama_server": str(script), "log_dir": str(tmp_path / "logs"),
                  "start_timeout": 60},
    )
    lines: list[str] = []
    backends = start_backends(settings, printer=lines.append)
    try:
        assert [b.name for b in backends] == ["对话模型", "嵌入模型"]
        assert is_healthy(f"http://127.0.0.1:{p1}") and is_healthy(f"http://127.0.0.1:{p2}")
        assert any("[就绪] 对话模型" in l for l in lines)
        # 已在运行：不会重复启动
        assert start_backends(settings, printer=lines.append) == []
    finally:
        stop_backends(backends, printer=lines.append)
    assert all(b.proc is not None and b.proc.poll() is not None for b in backends)


@pytest.mark.skipif(os.name == "nt", reason="用 shebang 脚本模拟 llama-server、用 SIGTERM 结束进程")
def test_up_cleans_children_on_sigterm(tmp_path: Path) -> None:
    """回归：uvicorn 会重新触发 SIGTERM，曾导致 up 被结束时 llama-server 残留。"""
    import signal
    import subprocess
    import time

    import httpx

    script = _fake_llama_script(tmp_path)
    models = tmp_path / "models"
    _gguf(models / "chat.gguf", 10)
    shutil.copytree(ROOT / "persona", tmp_path / "persona")
    p_llm, p_brain = _free_port(), _free_port()
    cfg = tmp_path / "brain.yaml"
    cfg.write_text(
        f"data_dir: {tmp_path / 'data'}\npersona_path: {tmp_path / 'persona' / 'her.yaml'}\n"
        f"server: {{port: {p_brain}}}\nupstream: {{base_url: 'http://127.0.0.1:{p_llm}/v1', warmup: false}}\n"
        "memory: {engine: lite}\nworker: {enabled: false}\n"
        f"backup: {{enabled: false, dir: {tmp_path / 'backups'}}}\n"
        f"launcher: {{models_dir: {models}, llama_server: {script}, log_dir: {tmp_path / 'logs'}}}\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, "-m", "her_brain", "-c", str(cfg), "up"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(200):
            try:
                if httpx.get(f"http://127.0.0.1:{p_brain}/healthz", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            pytest.fail("代理没有启动：" + proc.stdout.read().decode() if proc.stdout else "")
        assert is_healthy(f"http://127.0.0.1:{p_llm}")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        for _ in range(50):
            if not is_healthy(f"http://127.0.0.1:{p_llm}", timeout=0.3):
                break
            time.sleep(0.1)
        assert not is_healthy(f"http://127.0.0.1:{p_llm}", timeout=0.3), "llama-server 子进程残留"
        assert "[停止] 对话模型" in (proc.stdout.read().decode() if proc.stdout else "")
    finally:
        if proc.poll() is None:
            proc.kill()


def _fake_llama_script(tmp_path: Path) -> Path:
    script = tmp_path / "llama.cpp" / "llama-server"
    script.parent.mkdir(exist_ok=True)
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import uvicorn\n"
        "from her_brain.devtools.fake_llm import create_app\n"
        "assert '-m' in sys.argv\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "uvicorn.run(create_app(), host='127.0.0.1', port=port, log_level='warning')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_start_backends_reports_missing_model(make_settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from her_brain.launcher import LauncherError

    settings = make_settings(upstream={"base_url": f"http://127.0.0.1:{_free_port()}/v1"},
                             launcher={"models_dir": str(tmp_path / "empty")})
    with pytest.raises(LauncherError, match="未找到对话模型"):
        start_backends(settings, printer=lambda s: None)


# ---------------------------------------------------------------------- 环境自检


def test_doctor_reports(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    shutil.copytree(ROOT / "persona", tmp_path / "persona")
    cfg = tmp_path / "config" / "brain.yaml"
    cfg.parent.mkdir()
    text = (ROOT / "config" / "brain.example.yaml").read_text(encoding="utf-8")
    # 换成空闲端口，避免机器上恰好运行着的服务影响结果
    text = (text.replace("127.0.0.1:8080", f"127.0.0.1:{_free_port()}")
                .replace("127.0.0.1:8081", f"127.0.0.1:{_free_port()}")
                .replace("port: 8000", f"port: {_free_port()}"))
    cfg.write_text(text, encoding="utf-8")
    _gguf(tmp_path / "models" / "Qwen3-14B-Q4_K_M.gguf", 100)
    checks = run_doctor(str(cfg))
    by_title = {c.title: c for c in checks}
    assert by_title["配置文件"].status == "ok"
    assert by_title["用户绑定"].status == "warn"  # 还是示例 QQ 号
    assert by_title["对话模型"].status == "ok"
    assert by_title["嵌入模型"].status == "fail"  # 示例启用了嵌入，但没有嵌入模型
    assert by_title["对话服务"].status == "warn"  # 未运行
    assert by_title["向量库"].status == "ok"  # 本地嵌入模式
    buf = io.StringIO()
    assert print_report(checks, buf) == 1
    assert "→" in buf.getvalue()
    missing = run_doctor(str(tmp_path / "nope.yaml"))
    assert any(c.title == "配置文件" and c.status == "fail" for c in missing)


def test_doctor_sees_astrbot_messages(services, make_settings, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from her_brain.doctor import check_astrbot

    services.rawlog.insert(services.rawlog.new(user_id="qq:openid-abc", channel="qq", scene="private",
                                               privacy="private", user_text="hi", reply_text="hello"))
    checks = check_astrbot(services.settings)
    assert checks[0].status == "ok" and "qq:openid-abc" in checks[0].detail
    assert any(c.title == "主人识别" and "openid-abc" in c.detail for c in checks)


def test_locate_project_from_other_directory(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """回归：在项目以外的目录运行 doctor 会报“找不到 config/brain.yaml”。"""
    import her_brain.__main__ as cli

    project = tmp_path / "Epislon"
    (project / "config").mkdir(parents=True)
    (project / "config" / "brain.example.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "PROJECT_ROOT", project)
    monkeypatch.delenv("HER_BRAIN_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # 上一级目录
    assert cli.locate_project(None) == project and Path.cwd() == project
    monkeypatch.chdir(tmp_path)
    assert cli.locate_project("custom.yaml") is None and Path.cwd() == tmp_path  # 显式指定配置时不切换
    monkeypatch.chdir(project)
    assert cli.locate_project(None) is None  # 已在项目目录


# ---------------------------------------------------------------------- 配置助手


def test_setup_text_edits() -> None:
    text = (ROOT / "config" / "brain.example.yaml").read_text(encoding="utf-8")
    assert "qq:987654" in set_qq(text, "987654") and "qq:123456" not in set_qq(text, "987654")
    lite = set_engine(text, "lite")
    assert "engine: lite" in lite and "enabled: false" in lite.split("embedding:", 1)[1][:80]
    assert set_engine(lite, "mem0").count("engine: mem0") == 1
    added = set_qq(set_qq(text, "1"), "2")
    assert "qq:1" in added and "qq:2" in added


def test_setup_wizard(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    shutil.copytree(ROOT / "persona", tmp_path / "persona")
    cfg = tmp_path / "config" / "brain.yaml"
    cfg.parent.mkdir()
    shutil.copy2(ROOT / "config" / "brain.example.yaml", cfg)
    astrbot = tmp_path / "AstrBot"
    (astrbot / "data").mkdir(parents=True)
    answers = iter(["987654", "", "y", str(astrbot)])  # QQ 号、改用 lite（默认是）、Docker、AstrBot 目录
    out: list[str] = []
    assert run_setup(str(cfg), input_fn=lambda prompt: next(answers), printer=out.append) == 0
    s = load_settings(cfg)
    assert s.users[0].bindings[0] == "qq:987654"
    assert s.memory.engine == "lite" and not s.embedding.enabled
    assert s.server.host == "0.0.0.0"
    assert (astrbot / "data" / "plugins" / "astrbot_plugin_her_bridge" / "main.py").exists()
    assert (cfg.parent / "brain.yaml.bak").exists()
    assert any("host.docker.internal" in line for line in out)
    # 非交互：再次运行不改动
    assert run_setup(str(cfg), assume_yes=True, printer=out.append) == 0


def test_install_plugin_desktop_client(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """AstrBot 桌面客户端版：安装目录（F:\\APP\\AstrBot）里没有 data，数据在 ~/.astrbot/data。"""
    from her_brain.doctor import _check_desktop_plugin
    from her_brain.setup_wizard import install_plugin

    home = tmp_path / "home"
    (home / ".astrbot" / "data").mkdir(parents=True)
    install_dir = tmp_path / "APP" / "AstrBot"
    install_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("ASTRBOT_ROOT", raising=False)

    assert _check_desktop_plugin()[0].status == "warn"  # 还没装
    dest = install_plugin(install_dir)
    assert dest == home / ".astrbot" / "data" / "plugins" / "astrbot_plugin_her_bridge"
    assert (dest / "main.py").exists() and (dest / "metadata.yaml").exists()
    assert _check_desktop_plugin()[0].status == "info"
    # ASTRBOT_ROOT 优先
    custom = tmp_path / "custom_root"
    (custom / "data").mkdir(parents=True)
    monkeypatch.setenv("ASTRBOT_ROOT", str(custom))
    assert install_plugin(install_dir).parent == custom / "data" / "plugins"


# ---------------------------------------------------------------------- 端到端自测


async def test_selftest_with_fake_llm(make_settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    fake = FakeLLM()
    settings = make_settings(memory={"engine": "lite"})
    lines: list[str] = []
    results = await run_selftest(settings, transports=Transports(upstream=fake.transport(), embedding=fake.transport()),
                                 printer=lines.append)
    statuses = {r.title: r.status for r in results}
    assert "fail" not in statuses.values(), "\n".join(lines)
    assert statuses["时效记忆（面试）"] == "ok"
    assert statuses["核心档案（明确要求记住）"] == "ok"
    assert statuses["场合过滤（群聊）"] == "ok" and statuses["语音回复"] == "ok"
    assert summarize(results, printer=lines.append) == 0
    # 正式数据目录没有被写入
    assert not Path(settings.data_dir).exists() or not (Path(settings.data_dir) / "brain.db").exists()


async def test_selftest_stops_early_on_embedding_dims_mismatch(make_settings) -> None:  # type: ignore[no-untyped-def]
    fake = FakeLLM(dims=256)
    settings = make_settings(embedding={"enabled": True, "base_url": "http://fake/v1", "dims": 1024})
    results = await run_selftest(settings, transports=Transports(upstream=fake.transport(), embedding=fake.transport()),
                                 printer=lambda s: None)
    assert results[-1].title == "嵌入服务" and results[-1].status == "fail" and "dims: 256" in results[-1].detail
    assert not any(r.title == "后台提取" for r in results)  # 不再连环失败


async def test_selftest_reports_unreachable_upstream(make_settings) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings(upstream={"base_url": f"http://127.0.0.1:{_free_port()}/v1", "connect_timeout": 1})
    results = await run_selftest(settings, printer=lambda s: None)
    assert results[-1].status == "fail" and "her_brain up" in results[-1].detail
