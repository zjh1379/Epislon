"""环境自检：python -m her_brain doctor

只读检查，逐项给出结论与修复提示：Python 与依赖、配置与人设、模型文件、llama-server、显卡、
对话/嵌入服务、向量库、大脑代理、AstrBot 是否已经接入（根据收到的 QQ 消息判断）。
"""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, load_settings
from .launcher import discover_models, endpoint_root, find_llama_server, is_healthy

LABELS = {"ok": "[通过]", "warn": "[注意]", "fail": "[失败]", "info": "[信息]"}
PLACEHOLDER_QQ = "qq:123456"


@dataclass
class Check:
    status: str  # ok / warn / fail / info
    title: str
    detail: str = ""
    hint: str = ""


def _safe(fn: Callable[[], list[Check]], title: str) -> list[Check]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - 自检本身不能崩
        return [Check("fail", title, f"检查时出错：{exc}")]


def check_python() -> list[Check]:
    v = sys.version_info
    ok = v >= (3, 10)
    return [Check("ok" if ok else "fail", "Python 版本", f"{v.major}.{v.minor}.{v.micro}（{sys.executable}）",
                  "" if ok else "需要 Python 3.10 或更新版本")]


def check_dependencies(settings: Settings | None) -> list[Check]:
    base = ["fastapi", "uvicorn", "httpx", "yaml", "pydantic"]
    missing = [m for m in base if importlib.util.find_spec(m) is None]
    out = [Check("fail" if missing else "ok", "基础依赖", "缺少：" + "、".join(missing) if missing else "齐全",
                 "在项目目录运行 pip install -e ." if missing else "")]
    if settings is not None and settings.memory.engine == "mem0":
        miss = [m for m in ("mem0", "qdrant_client") if importlib.util.find_spec(m) is None]
        out.append(Check("fail" if miss else "ok", "Mem0 依赖", "缺少：" + "、".join(miss) if miss else "齐全",
                         "运行 pip install -e .[mem0]，或改用 lite 引擎" if miss else ""))
    return out


def check_config(path: str | None) -> tuple[Settings | None, list[Check]]:
    try:
        settings = load_settings(path)
    except FileNotFoundError as exc:
        return None, [Check("fail", "配置文件", f"{exc}（当前目录 {Path.cwd()}）",
                            "先 cd 到项目目录；若项目目录里也没有，再运行 python -m her_brain init")]
    except Exception as exc:  # noqa: BLE001
        return None, [Check("fail", "配置文件", f"无法解析：{exc}", "检查 YAML 缩进与字段名")]
    out = [Check("ok", "配置文件", f"{settings.config_path}（记忆引擎 {settings.memory.engine}）")]
    bindings = [b for u in settings.users for b in u.bindings]
    if not settings.users:
        out.append(Check("warn", "用户绑定", "没有登记任何用户，所有人都会被当作访客", "运行 python -m her_brain setup"))
    elif PLACEHOLDER_QQ in bindings:
        out.append(Check("warn", "用户绑定", f"还在使用示例里的 {PLACEHOLDER_QQ}", "运行 python -m her_brain setup 填入你的 QQ 号"))
    else:
        out.append(Check("ok", "用户绑定", "、".join(bindings) or "（无平台绑定）"))
    return settings, out


def check_persona(settings: Settings) -> list[Check]:
    from .prompt.persona import PersonaHolder

    holder = PersonaHolder(settings.persona_path)
    out = [Check("ok", "人设", f"{holder.name}（静态前缀 {len(holder.static_text)} 字，{holder.digest}）")]
    if holder.name == "小E":
        out.append(Check("info", "人设内容", "仍是示例人设“小E”", f"编辑 {settings.persona_path} 换成她真正的设定"))
    return out


def check_models(settings: Settings) -> list[Check]:
    models = discover_models(settings.launcher)
    out: list[Check] = []
    root = Path(settings.launcher.models_dir).resolve()
    if models.found:
        listing = "；".join(f"{f.name}（{f.stat().st_size / 1024**3:.1f} GB）" for f in models.found)
        out.append(Check("info", "模型目录", f"{root}：{listing}"))
    else:
        out.append(Check("fail", "模型目录", f"{root} 里没有 .gguf 文件", "把模型放进项目的 models 目录"))
    _, host, _ = endpoint_root(settings.upstream.base_url)
    if models.chat:
        out.append(Check("ok", "对话模型", models.chat.name))
    elif host in ("127.0.0.1", "localhost"):
        out.append(Check("fail", "对话模型", "未找到", "例如 Qwen3-14B-Q4_K_M.gguf"))
    if settings.embedding.enabled:
        if models.embedding:
            out.append(Check("ok", "嵌入模型", models.embedding.name))
        else:
            out.append(Check("fail", "嵌入模型", "已启用 embedding 但没找到文件名含 embed/bge 的模型",
                             "下载 Qwen3-Embedding-0.6B-Q8_0.gguf 放进 models；或运行 python -m her_brain setup 改用 lite 引擎"))
    return out


def check_llama_server(settings: Settings) -> list[Check]:
    exe = find_llama_server(settings.launcher)
    if exe:
        return [Check("ok", "llama-server", str(exe))]
    return [Check("fail", "llama-server", "未找到",
                  "把 llama.cpp（win-cuda-13.4-x64 版）和 cudart 一起解压到项目目录或上级目录中以 llama 开头的文件夹，"
                  "或设置环境变量 LLAMA_SERVER")]


def check_gpu() -> list[Check]:
    if not shutil.which("nvidia-smi"):
        return [Check("warn", "显卡", "找不到 nvidia-smi", "确认已安装 NVIDIA 驱动")]
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    line = (res.stdout or "").strip().splitlines()
    if not line:
        return [Check("warn", "显卡", "nvidia-smi 没有输出")]
    name, driver, used, total = [x.strip() for x in line[0].split(",")]
    return [Check("info", "显卡", f"{name}，驱动 {driver}，显存已用 {used}/{total} MB")]


def check_upstream(settings: Settings) -> list[Check]:
    root, _, _ = endpoint_root(settings.upstream.base_url)
    if not is_healthy(root):
        return [Check("warn", "对话服务", f"{root} 未运行", "运行 python -m her_brain up 会自动启动")]
    detail = root
    try:
        data = httpx.get(f"{settings.upstream.base_url.rstrip('/')}/models", timeout=3).json().get("data", [])
        if data:
            detail += f"（{data[0].get('id')}）"
    except (httpx.HTTPError, ValueError):
        pass
    return [Check("ok", "对话服务", detail)]


def check_embedding(settings: Settings) -> list[Check]:
    if not settings.embedding.enabled:
        return [Check("info", "嵌入服务", "未启用（lite 引擎只用文本检索）")]
    root, _, _ = endpoint_root(settings.embedding.base_url)
    if not is_healthy(root):
        return [Check("warn", "嵌入服务", f"{root} 未运行", "运行 python -m her_brain up 会自动启动")]
    r = httpx.post(f"{settings.embedding.base_url.rstrip('/')}/embeddings",
                   json={"model": settings.embedding.model, "input": ["测试"]}, timeout=10)
    r.raise_for_status()
    dims = len(r.json()["data"][0]["embedding"])
    if dims != settings.embedding.dims:
        return [Check("fail", "嵌入服务", f"向量维度 {dims} 与配置 embedding.dims={settings.embedding.dims} 不一致",
                      f"把配置改为 dims: {dims}；已有记忆时再运行 python -m her_brain reindex")]
    return [Check("ok", "嵌入服务", f"{root}（{dims} 维）")]


def check_vector_store(settings: Settings) -> list[Check]:
    mem = settings.memory
    if mem.engine != "mem0":
        return [Check("info", "向量库", "lite 引擎，不需要 Qdrant")]
    if mem.qdrant_path:
        return [Check("ok", "向量库", f"Qdrant 本地嵌入模式：{mem.qdrant_path}（不需要单独运行服务）")]
    try:
        httpx.get(f"{mem.qdrant_url.rstrip('/')}/collections", timeout=3).raise_for_status()
        return [Check("ok", "向量库", f"Qdrant 服务 {mem.qdrant_url}")]
    except httpx.HTTPError:
        return [Check("fail", "向量库", f"连不上 Qdrant 服务 {mem.qdrant_url}",
                      "启动 Qdrant，或在配置 memory 下设置 qdrant_path: data/qdrant 使用本地嵌入模式")]


def check_proxy(settings: Settings) -> list[Check]:
    host = settings.server.host if settings.server.host not in ("0.0.0.0", "::") else "127.0.0.1"
    url = f"http://{host}:{settings.server.port}"
    try:
        data = httpx.get(f"{url}/healthz", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return [Check("info", "大脑代理", f"{url} 未运行", "python -m her_brain up 启动全部服务")]
    out = [Check("ok", "大脑代理", f"{url}（人设 {data.get('persona')}，引擎 {data.get('engine')}）")]
    try:
        headers = {"X-Admin-Token": settings.server.admin_token} if settings.server.admin_token else {}
        stats = httpx.get(f"{url}/admin/stats", headers=headers, timeout=3).json()["metrics"]
    except (httpx.HTTPError, ValueError, KeyError):
        return out
    counters = stats.get("counters", {})
    out.append(Check("info", "代理收到的对话", f"成功 {counters.get('turns', 0)} 轮（自本次启动）"))
    hints = {
        "auth_rejected": ("API Key 不匹配被拒绝", "入口（AstrBot 提供商）里的 API Key 要与 config/brain.yaml 的 api_keys 一致，例如 sk-her-astrbot"),
        "not_found": ("请求了不存在的地址", "入口里的 API 地址填 http://127.0.0.1:8000/v1"),
        "bad_requests": ("请求格式不对", "把代理窗口里对应的日志发给我"),
        "upstream_failures": ("模型服务出错", "查看 logs/llama-chat.log；显存不足时把 launcher.chat_args 里的 -c 调小"),
    }
    events = stats.get("recent_events", [])
    for kind, (title, hint) in hints.items():
        n = counters.get(kind, 0)
        if not n:
            continue
        last = next((e["detail"] for e in reversed(events) if e.get("kind") == kind), "")
        out.append(Check("warn", f"代理：{title}", f"{n} 次，最近一次：{last}", hint))
    return out


def check_astrbot(settings: Settings) -> list[Check]:
    db = settings.db_path
    if not db.exists():
        return [Check("info", "AstrBot 接入", "还没有数据库（代理尚未运行过）")]
    # as_uri 生成 file:///D:/... 形式并转义空格与中文，Windows 绝对路径也能正确打开
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            """SELECT user_id, scene, COUNT(*) AS n, MAX(created_at) AS last FROM raw_log
               WHERE channel = 'qq' GROUP BY user_id, scene ORDER BY last DESC"""
        )]
        owners = {r["user_id"] for r in conn.execute("SELECT user_id FROM users WHERE role = 'owner'")}
    finally:
        conn.close()
    if not rows:
        return [Check("info", "AstrBot 接入", "还没有收到来自 QQ 的消息",
                      "在 AstrBot 新增 OpenAI 兼容提供商指向本代理并设为默认，然后给机器人发条私聊")]
    out: list[Check] = []
    last = max(r["last"] for r in rows)
    summary = "；".join(f"{r['user_id']} {r['scene']} {r['n']} 条" for r in rows[:6])
    out.append(Check("ok", "AstrBot 接入", f"已收到 QQ 消息，最近一条在 {_ago(last)}：{summary}"))
    users = {r["user_id"] for r in rows}
    if any(u.startswith("anon:") for u in users):
        out.append(Check("warn", "AstrBot 插件", "有 QQ 消息拿不到发送者（anon:qq）",
                         "确认 astrbot_plugin_her_bridge 已安装并启用，且 enabled_provider_ids 包含本代理的提供商"))
    if not users & owners:
        guests = sorted(u for u in users if u.startswith("qq:"))
        if guests:
            out.append(Check("warn", "主人识别", f"QQ 消息被识别为访客：{'、'.join(guests[:5])}",
                             "如果其中有你：运行 python -m her_brain setup --qq <冒号后面的部分>，"
                             "或在 http://127.0.0.1:8000/docs 调用 POST /admin/identities 绑定（move_logs 设为 true）"))
    return out


def _ago(ts: float) -> str:
    diff = time.time() - float(ts)
    if diff < 120:
        return "刚刚"
    if diff < 3600:
        return f"{int(diff // 60)} 分钟前"
    if diff < 86400:
        return f"{int(diff // 3600)} 小时前"
    return f"{int(diff // 86400)} 天前"


def run_doctor(config_path: str | None) -> list[Check]:
    checks = check_python()
    settings, cfg_checks = check_config(config_path)
    checks += check_dependencies(settings)
    checks += cfg_checks
    if settings is None:
        return checks
    for fn, title in (
        (lambda: check_persona(settings), "人设"),
        (lambda: check_models(settings), "模型"),
        (lambda: check_llama_server(settings), "llama-server"),
        (check_gpu, "显卡"),
        (lambda: check_upstream(settings), "对话服务"),
        (lambda: check_embedding(settings), "嵌入服务"),
        (lambda: check_vector_store(settings), "向量库"),
        (lambda: check_proxy(settings), "大脑代理"),
        (lambda: check_astrbot(settings), "AstrBot 接入"),
    ):
        checks += _safe(fn, title)
    return checks


def print_report(checks: list[Check], out: Any = None) -> int:
    out = out or sys.stdout
    for c in checks:
        print(f"{LABELS.get(c.status, c.status)} {c.title}：{c.detail}", file=out)
        if c.hint:
            print(f"        → {c.hint}", file=out)
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    print(f"\n共 {len(checks)} 项：{fails} 项失败，{warns} 项需要注意。", file=out)
    return 1 if fails else 0
