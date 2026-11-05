"""配置助手：python -m her_brain setup

交互式完成第一次配置（也可全部用参数给出，配合 --yes 非交互运行）：
- 把示例里的 qq:123456 换成你的 QQ 号；
- 根据 models/ 里有没有嵌入模型，选择 mem0 或 lite 记忆引擎；
- AstrBot 用 Docker 运行时，让代理监听 0.0.0.0；
- 把 AstrBot 插件复制到 AstrBot 的插件目录。

只做有把握的文本替换，改完立即校验，校验失败会原样恢复；修改前的配置另存为 .bak。
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from .config import default_config_path, load_settings
from .launcher import discover_models, find_llama_server

PLACEHOLDER_QQ = "qq:123456"
PLUGIN_NAME = "astrbot_plugin_her_bridge"
PLUGIN_SRC = Path(__file__).resolve().parents[1] / "integrations" / PLUGIN_NAME

_ENGINE_RE = re.compile(r"(?m)^([ \t]+engine:[ \t]*)(mem0|lite)\b")
_EMBED_ENABLED_RE = re.compile(r"(?m)^(embedding:[^\n]*\n(?:[ \t]*#[^\n]*\n)*[ \t]+enabled:[ \t]*)(true|false)\b")
_SERVER_HOST_RE = re.compile(r"(?m)^(server:[^\n]*\n(?:[ \t]*#[^\n]*\n)*[ \t]+host:[ \t]*)(\S+)")
_BINDINGS_RE = re.compile(r"(?m)^([ \t]*)bindings:[ \t]*$")


def add_binding(text: str, binding: str) -> str:
    """在配置里第一个用户（示例中为主人）的 bindings 下加一条身份；示例占位的 qq:123456 会被替换。"""
    binding = binding.strip()
    if not binding or re.search(rf"{re.escape(binding)}(?![\w-])", text):
        return text
    if binding.startswith("qq:") and PLACEHOLDER_QQ in text:
        return text.replace(PLACEHOLDER_QQ, binding)
    m = _BINDINGS_RE.search(text)
    if m is None:
        raise ValueError("配置里没有找到 users[].bindings，无法自动添加，请手动编辑")
    indent = m.group(1)
    return text[: m.end()] + f"\n{indent}  - {binding}" + text[m.end():]


def set_qq(text: str, qq: str) -> str:
    qq = qq.strip()
    return add_binding(text, f"qq:{qq}") if qq else text


def set_engine(text: str, engine: str) -> str:
    text = _ENGINE_RE.sub(lambda m: m.group(1) + engine, text, count=1)
    return _EMBED_ENABLED_RE.sub(lambda m: m.group(1) + ("true" if engine == "mem0" else "false"), text, count=1)


def set_server_host(text: str, host: str) -> str:
    return _SERVER_HOST_RE.sub(lambda m: m.group(1) + host, text, count=1)


def find_plugin_dir(astrbot: Path) -> Path | None:
    """接受 AstrBot 根目录、data 目录或 plugins 目录。"""
    for cand in (astrbot / "data" / "plugins", astrbot / "plugins"):
        if cand.is_dir():
            return cand
    if astrbot.name == "plugins" and astrbot.is_dir():
        return astrbot
    if (astrbot / "data").is_dir():
        return astrbot / "data" / "plugins"
    if astrbot.name == "data" and astrbot.is_dir():
        return astrbot / "plugins"
    return None


def astrbot_data_roots() -> list[Path]:
    """AstrBot 可能的数据根目录：环境变量 ASTRBOT_ROOT；桌面客户端版固定使用 ~/.astrbot。"""
    roots = []
    if os.environ.get("ASTRBOT_ROOT"):
        roots.append(Path(os.environ["ASTRBOT_ROOT"]))
    roots.append(Path.home() / ".astrbot")
    return roots


def install_plugin(astrbot: Path) -> Path:
    plugins = find_plugin_dir(astrbot)
    if plugins is None:
        # 桌面客户端版的安装目录里没有 data：数据在用户目录的 .astrbot 下
        for root in astrbot_data_roots():
            plugins = find_plugin_dir(root)
            if plugins is not None:
                break
    if plugins is None:
        raise FileNotFoundError(
            f"{astrbot} 下没有找到 data 目录，{Path.home() / '.astrbot'} 也不存在。"
            "源码/安装包版填 AstrBot 的运行目录；客户端版先启动一次 AstrBot 再试"
        )
    dest = plugins / PLUGIN_NAME
    plugins.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PLUGIN_SRC, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def run_bind(
    config_path: str | None,
    external_id: str,
    *,
    platform: str = "qq",
    user_id: str | None = None,
    printer: Callable[[str], None] = print,
) -> int:
    """python -m her_brain bind <ID>：把一个平台身份绑定到主人（或 --user 指定的用户）。

    QQ 官方机器人给的是 openid（例如 53B97737D288FD889DAAF91A0C7A4EDA），不是 QQ 号；
    同一个人在私聊和群聊里的 openid 通常也不同，需要分别绑定。
    """
    import httpx

    from .identity.resolver import IdentityResolver
    from .launcher import proxy_root
    from .memory.db import Database

    cfg_path = Path(config_path) if config_path else default_config_path()
    settings = load_settings(cfg_path)
    target = user_id or next((u.user_id for u in settings.users if u.role == "owner"), None)
    if target is None:
        printer("配置里没有 role: owner 的用户，请用 --user 指定要绑定到谁")
        return 1
    external_id = external_id.strip()
    if ":" in external_id:  # 也接受 qq:xxx 这种带平台前缀的写法（与 doctor 显示的一致）
        platform, external_id = external_id.split(":", 1)

    root = proxy_root(settings)
    headers = {"X-Admin-Token": settings.server.admin_token} if settings.server.admin_token else {}
    body = {"platform": platform, "external_id": external_id, "user_id": target, "move_logs": True}
    try:
        r = httpx.post(f"{root}/admin/identities", json=body, headers=headers, timeout=10)
    except httpx.HTTPError:
        r = None
    if r is not None and r.status_code == 200:
        result = r.json()
    elif r is not None:
        printer(f"绑定失败：HTTP {r.status_code} {r.text[:200]}")
        return 1
    else:  # 代理没在运行：直接写数据库
        db = Database(settings.db_path)
        try:
            resolver = IdentityResolver(db, settings)
            resolver.seed()
            result = resolver.link(platform, external_id, target, move_logs=True)
        finally:
            db.close()
    moved = result.get("moved_logs", 0)
    printer(f"已绑定 {platform}:{external_id} → {target}"
            + (f"（之前以访客身份聊的 {moved} 条对话已迁移，后台会重新提取记忆）" if moved else ""))

    if settings.users and settings.users[0].user_id == target:
        original = cfg_path.read_text(encoding="utf-8")
        text = add_binding(original, f"{platform}:{external_id}")
        if text != original:
            cfg_path.with_name(cfg_path.name + ".bak").write_text(original, encoding="utf-8")
            cfg_path.write_text(text, encoding="utf-8")
            try:
                load_settings(cfg_path)
                printer(f"已写入 {cfg_path}（重建数据时也会保留）")
            except Exception as exc:  # noqa: BLE001
                cfg_path.write_text(original, encoding="utf-8")
                printer(f"写入配置后校验失败，已恢复原样（数据库中的绑定仍然有效）：{exc}")
    return 0


def _yes(answer: str, default: bool = True) -> bool:
    answer = answer.strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "是", "好", "1")


def run_setup(
    config_path: str | None = None,
    *,
    qq: str | None = None,
    astrbot: str | None = None,
    docker: bool | None = None,
    engine: str | None = None,
    assume_yes: bool = False,
    input_fn: Callable[[str], str] = input,
    printer: Callable[[str], None] = print,
) -> int:
    cfg_path = Path(config_path) if config_path else default_config_path()
    if not cfg_path.exists():
        printer(f"找不到 {cfg_path}，请先运行 python -m her_brain init")
        return 1
    original = cfg_path.read_text(encoding="utf-8")
    text = original
    settings = load_settings(cfg_path)

    def ask(prompt: str) -> str:
        return "" if assume_yes else input_fn(prompt)

    # 1. QQ 号
    if qq is None and PLACEHOLDER_QQ in text:
        qq = ask("你的 QQ 号（OneBot/NapCat 接入时填 QQ 号；直接回车跳过）：")
    if qq:
        text = set_qq(text, qq)
        printer(f"· 绑定 qq:{qq.strip()} 为主人")

    # 2. 记忆引擎
    models = discover_models(settings.launcher)
    printer(f"· 对话模型：{models.chat.name if models.chat else '未找到'}")
    printer(f"· 嵌入模型：{models.embedding.name if models.embedding else '未找到'}")
    exe = find_llama_server(settings.launcher)
    printer(f"· llama-server：{exe or '未找到（运行 up 前需要准备好）'}")
    target = engine
    if target is None:
        if models.embedding is None and settings.memory.engine == "mem0":
            target = "lite" if _yes(ask("没有嵌入模型，先改用 lite 引擎（不需要嵌入模型与 Qdrant）？[Y/n] ")) else None
        elif models.embedding is not None and settings.memory.engine == "lite":
            if importlib.util.find_spec("mem0") is None:
                printer("· 有嵌入模型，但未安装 Mem0（pip install -e .[mem0]），保持 lite 引擎")
            elif _yes(ask("检测到嵌入模型，改用 mem0 引擎（完整语义记忆）？[Y/n] ")):
                target = "mem0"
    if target in ("mem0", "lite") and target != settings.memory.engine:
        text = set_engine(text, target)
        printer(f"· 记忆引擎改为 {target}")

    # 3. AstrBot 是否在 Docker 中
    if docker is None and not assume_yes:
        docker = _yes(ask("AstrBot 是用 Docker 运行的吗？[y/N] "), default=False)
    if docker and settings.server.host != "0.0.0.0":
        text = set_server_host(text, "0.0.0.0")
        printer("· 代理改为监听 0.0.0.0，AstrBot 中的地址填 http://host.docker.internal:%d/v1" % settings.server.port)

    # 写回并校验
    if text != original:
        backup = cfg_path.with_name(cfg_path.name + ".bak")
        backup.write_text(original, encoding="utf-8")
        cfg_path.write_text(text, encoding="utf-8")
        try:
            settings = load_settings(cfg_path)
        except Exception as exc:  # noqa: BLE001
            cfg_path.write_text(original, encoding="utf-8")
            printer(f"修改后的配置无法通过校验，已恢复原样：{exc}")
            return 1
        printer(f"· 已更新 {cfg_path}（原配置保存在 {backup.name}）")

    # 4. AstrBot 插件
    if astrbot is None:
        astrbot = ask("AstrBot 目录（用于复制插件；客户端版填安装目录即可，会自动找到用户目录下的 .astrbot；"
                      "直接回车跳过）：").strip().strip('"')
    if astrbot:
        try:
            dest = install_plugin(Path(astrbot))
            printer(f"· 插件已复制到 {dest}，重启 AstrBot 后生效")
        except FileNotFoundError as exc:
            printer(f"· 插件未安装：{exc}")

    key = next((k.key for k in settings.api_keys if k.channel == "qq"), "（未配置，任意值）")
    host = "host.docker.internal" if docker else "127.0.0.1"
    printer("\nAstrBot 里新增“OpenAI 兼容”提供商：")
    printer(f"  API Base URL：http://{host}:{settings.server.port}/v1")
    printer(f"  API Key：{key}")
    printer(f"  模型：{settings.server.model_name}")
    printer("\n下一步：python -m her_brain doctor 检查环境，python -m her_brain up 启动全部服务。")
    return 0
