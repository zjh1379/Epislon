"""一键启动：python -m her_brain up

1. 在 models/ 里找模型：文件名含 embed/bge 的作为嵌入模型，其余最大的 .gguf 作为对话模型；
2. 找 llama-server：配置 → 环境变量 LLAMA_SERVER → PATH → 项目内/上级目录中名字以 llama 开头的文件夹；
3. 按 upstream.base_url / embedding.base_url 的端口启动服务（已在运行的跳过），等待模型加载完成；
4. 启动大脑代理；按 Ctrl+C 退出时一并关闭启动的服务。

日志写到 logs/llama-chat.log 与 logs/llama-embedding.log。
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import LauncherConfig, Settings

EMBED_HINTS = ("embed", "bge", "m3e", "gte-", "e5-")
SKIP_HINTS = ("mmproj",)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class LauncherError(RuntimeError):
    pass


@dataclass
class ModelFiles:
    chat: Path | None
    embedding: Path | None
    found: list[Path] = field(default_factory=list)


def _is_embedding(path: Path) -> bool:
    name = path.name.lower()
    return any(h in name for h in EMBED_HINTS)


def _is_extra_shard(path: Path) -> bool:
    """分片模型（xxx-00002-of-00003.gguf）只需要把第一片交给 llama-server。"""
    m = _SHARD_RE.search(path.name)
    return bool(m) and m.group(1) != "00001"


def discover_models(cfg: LauncherConfig) -> ModelFiles:
    root = Path(cfg.models_dir)
    files = sorted(root.rglob("*.gguf")) if root.exists() else []
    files = [f for f in files if not any(h in f.name.lower() for h in SKIP_HINTS) and not _is_extra_shard(f)]

    def resolve(name: str) -> Path | None:
        for cand in (Path(name), root / name):
            if cand.exists():
                return cand
        return None

    if cfg.embedding_model:
        embedding = resolve(cfg.embedding_model)
    else:
        embedding = next((f for f in files if _is_embedding(f)), None)
    if cfg.chat_model:
        chat = resolve(cfg.chat_model)
    else:
        chats = [f for f in files if not _is_embedding(f) and f != embedding]
        chat = max(chats, key=lambda f: f.stat().st_size, default=None)
    return ModelFiles(chat=chat, embedding=embedding, found=files)


def find_llama_server(cfg: LauncherConfig) -> Path | None:
    exe_names = ["llama-server.exe", "llama-server"] if os.name == "nt" else ["llama-server", "llama-server.exe"]
    explicit = cfg.llama_server or os.environ.get("LLAMA_SERVER")
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            for name in exe_names:
                hits = sorted(p.rglob(name))
                if hits:
                    return hits[0]
            return None
        return p if p.exists() else None
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    cwd = Path.cwd()
    search_dirs: list[Path] = []
    for base in (cwd, cwd.parent):
        if base.exists():
            search_dirs += [d for d in base.iterdir() if d.is_dir() and d.name.lower().startswith("llama")]
    search_dirs += [cwd / "bin", cwd / "tools"]
    for d in search_dirs:
        if not d.exists():
            continue
        for name in exe_names:
            hits = sorted(d.rglob(name))
            if hits:
                return hits[0]
    return None


def endpoint_root(base_url: str) -> tuple[str, str, int]:
    """http://127.0.0.1:8080/v1 → ('http://127.0.0.1:8080', '127.0.0.1', 8080)"""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{parsed.netloc}", host, port


def is_healthy(root: str, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"{root}/health", timeout=timeout)
        if r.status_code == 200:
            return True
        if r.status_code == 404:  # 不是 llama-server（例如其他 OpenAI 兼容服务）：退回检查 /v1/models
            return httpx.get(f"{root}/v1/models", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False
    return False


def pooling_args(model: Path, args: list[str]) -> list[str]:
    if "--pooling" in args:
        return []
    name = model.name.lower()
    if "qwen3-embedding" in name or "qwen3_embedding" in name:
        return ["--pooling", "last"]
    if "bge" in name:
        return ["--pooling", "cls"]
    return []


@dataclass
class Backend:
    name: str
    root: str
    cmd: list[str]
    log_path: Path
    proc: subprocess.Popen[bytes] | None = None


def plan_backends(settings: Settings, *, chat: bool = True, embedding: bool = True) -> list[Backend]:
    """根据配置与检测结果，列出需要启动的服务（不检查是否已在运行）。"""
    cfg = settings.launcher
    models = discover_models(cfg)
    exe = find_llama_server(cfg)
    log_dir = Path(cfg.log_dir)
    plans: list[Backend] = []
    wanted: list[tuple[str, str, Path | None, list[str], str]] = []
    if chat:
        wanted.append(("对话模型", settings.upstream.base_url, models.chat, list(cfg.chat_args), "llama-chat.log"))
    if embedding and settings.embedding.enabled:
        args = list(cfg.embedding_args)
        if models.embedding is not None:
            args += pooling_args(models.embedding, args)
        wanted.append(("嵌入模型", settings.embedding.base_url, models.embedding, args, "llama-embedding.log"))
    for name, base_url, model, args, log_name in wanted:
        root, host, port = endpoint_root(base_url)
        exe_text = str(exe) if exe else (cfg.llama_server or "llama-server")
        cmd = [exe_text, "-m", str(model) if model else "<未找到模型>",
               "--host", host, "--port", str(port), *args]
        plans.append(Backend(name=name, root=root, cmd=cmd, log_path=log_dir / log_name))
    return plans


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def start_backends(
    settings: Settings,
    *,
    chat: bool = True,
    embedding: bool = True,
    printer: Callable[[str], None] = print,
) -> list[Backend]:
    started: list[Backend] = []
    cfg = settings.launcher
    try:
        for backend in plan_backends(settings, chat=chat, embedding=embedding):
            host = urlparse(backend.root).hostname or ""
            if is_healthy(backend.root):
                printer(f"[跳过] {backend.name}已在运行：{backend.root}")
                continue
            if host not in LOCAL_HOSTS:
                raise LauncherError(f"{backend.name}地址 {backend.root} 不是本机，且无法连接；请先在那台机器上启动它")
            exe, model = backend.cmd[0], backend.cmd[2]
            if model == "<未找到模型>":
                hint = "嵌入模型文件名需包含 embed 或 bge" if backend.name == "嵌入模型" else "把 .gguf 放进 models 目录"
                raise LauncherError(f"未找到{backend.name}：{hint}，或在配置 launcher 中指定（当前目录 {cfg.models_dir}）")
            if not Path(exe).exists() and not shutil.which(exe):
                raise LauncherError(
                    "未找到 llama-server。请把 llama.cpp 解压到项目目录或上级目录中名字以 llama 开头的文件夹，"
                    "或设置环境变量 LLAMA_SERVER，或在配置 launcher.llama_server 中填写路径"
                )
            backend.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(backend.log_path, "wb")  # noqa: SIM115 - 子进程持有，退出时关闭
            printer(f"[启动] {backend.name}：{Path(model).name} → {backend.root}（日志 {backend.log_path}）")
            backend.proc = subprocess.Popen(backend.cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            started.append(backend)
            _wait_ready(backend, cfg.start_timeout, printer)
            log_fh.flush()
    except BaseException:
        stop_backends(started, printer)
        raise
    return started


def _wait_ready(backend: Backend, timeout: float, printer: Callable[[str], None]) -> None:
    assert backend.proc is not None
    t0 = time.monotonic()
    last_report = t0
    while True:
        code = backend.proc.poll()
        if code is not None:
            raise LauncherError(f"{backend.name}启动失败（退出码 {code}），日志最后几行：\n{_tail(backend.log_path)}")
        if is_healthy(backend.root, timeout=1.0):
            printer(f"[就绪] {backend.name}（{time.monotonic() - t0:.0f} 秒）")
            return
        now = time.monotonic()
        if now - t0 > timeout:
            raise LauncherError(f"{backend.name}在 {timeout:.0f} 秒内没有就绪，日志最后几行：\n{_tail(backend.log_path)}")
        if now - last_report >= 15:
            printer(f"  …正在加载{backend.name}（{now - t0:.0f} 秒）")
            last_report = now
        time.sleep(0.5)


def stop_backends(backends: list[Backend], printer: Callable[[str], None] = print) -> None:
    for backend in reversed(backends):
        proc = backend.proc
        if proc is None or proc.poll() is not None:
            continue
        printer(f"[停止] {backend.name}")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _exit_on_sigterm() -> None:
    """uvicorn 退出时会把捕获到的信号重新触发一次。SIGTERM 的默认处理是直接结束进程，
    下面的 finally 就来不及关掉 llama-server，留下占着显存的孤儿进程。
    这里改成抛出 SystemExit，保证清理代码一定执行。"""

    def _handler(signum: int, frame: object) -> None:
        raise SystemExit(0)

    for name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # 非主线程等情况
                pass


def proxy_root(settings: Settings, host: str | None = None, port: int | None = None) -> str:
    h = host or settings.server.host
    if h in ("0.0.0.0", "::", ""):
        h = "127.0.0.1"
    return f"http://{h}:{port or settings.server.port}"


def proxy_conflict(settings: Settings, host: str | None = None, port: int | None = None) -> str | None:
    """代理端口已被占用时返回说明。本地嵌入模式的 Qdrant 只允许一个进程打开，不能同时运行两个代理。"""
    import socket

    root = proxy_root(settings, host, port)
    try:
        data = httpx.get(f"{root}/healthz", timeout=2).json()
        if data.get("ok"):
            return (f"大脑代理已经在运行（{root}）。要重启（例如 git pull 之后），先运行 "
                    "python -m her_brain stop，或在原来的窗口按 Ctrl+C")
    except (httpx.HTTPError, ValueError):
        pass
    parsed = urlparse(root)
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex((parsed.hostname or "127.0.0.1", parsed.port or 80)) == 0:
            return f"端口 {parsed.port} 已被其他程序占用。关闭它，或在配置 server.port 中换一个端口"
    return None


def stop_running(settings: Settings, printer: Callable[[str], None] = print, timeout: float = 30) -> int:
    """python -m her_brain stop：请求正在运行的代理优雅退出，由 up 启动的 llama-server 随之关闭。"""
    root = proxy_root(settings)
    headers = {"X-Admin-Token": settings.server.admin_token} if settings.server.admin_token else {}
    try:
        r = httpx.post(f"{root}/admin/shutdown", headers=headers, timeout=5)
    except httpx.HTTPError:
        printer(f"大脑代理没有在运行（{root}）")
        return 0
    if r.status_code == 404:
        printer("正在运行的是旧版本，不支持 stop。请在它的窗口按 Ctrl+C，"
                "或在任务管理器里结束对应的 python.exe 与 llama-server.exe")
        return 1
    if r.status_code != 200:
        printer(f"停止失败：HTTP {r.status_code} {r.text[:200]}")
        return 1
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{root}/healthz", timeout=1)
        except httpx.HTTPError:
            printer("大脑代理已停止（由 up 启动的模型服务也已一并关闭）")
            return 0
        time.sleep(0.5)
    printer(f"已发出停止请求，但 {timeout:.0f} 秒后仍在运行")
    return 1


def run_up(settings: Settings, *, chat: bool = True, embedding: bool = True, host: str | None = None,
           port: int | None = None) -> None:
    import uvicorn

    from .app import create_app

    conflict = proxy_conflict(settings, host, port)
    if conflict:
        print(f"[跳过] {conflict}", file=sys.stderr)
        raise SystemExit(1)
    _exit_on_sigterm()
    try:
        backends = start_backends(settings, chat=chat, embedding=embedding)
    except LauncherError as exc:
        print(f"\n[失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if settings.embedding.enabled:
        from .doctor import check_embedding

        try:
            for c in check_embedding(settings):
                if c.status == "fail":
                    print(f"[注意] {c.detail}\n        → {c.hint}")
        except Exception:  # noqa: BLE001 - 只是提示，不影响启动
            pass
    try:
        print(f"[启动] 大脑代理 → http://{host or settings.server.host}:{port or settings.server.port}/v1"
              "（按 Ctrl+C 退出）")
        uvicorn.run(
            create_app(settings),
            host=host or settings.server.host,
            port=port or settings.server.port,
            log_level=settings.server.log_level.lower(),
            access_log=False,
        )
    except KeyboardInterrupt:  # Ctrl+C 或 stop：正常退出，不打印堆栈
        pass
    finally:
        stop_backends(backends)
