"""命令行入口：python -m her_brain <命令>

  up         一键启动：自动找到模型与 llama-server，启动对话/嵌入服务和大脑代理
  stop       停止正在运行的大脑代理（git pull 之后重启前先运行它）
  serve      只启动大脑代理（模型服务自行启动时使用）
  init       生成配置文件、数据目录与数据库
  setup      配置助手：填 QQ 号、选择记忆引擎、安装 AstrBot 插件
  doctor     环境自检
  selftest   端到端自测（临时数据，不影响正式记忆）
  process    立即处理所有待提取的原始日志（代理未运行时使用）
  backup     生成一份快照
  restore    从快照恢复（先停止代理）
  reindex    按 SQLite 中的权威副本重建向量索引
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from .config import Settings, default_config_path, load_settings

# 以可编辑方式安装（pip install -e .）时，这就是项目目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def locate_project(explicit_config: str | None) -> Path | None:
    """配置、数据、人设、模型等相对路径都以项目目录为准。

    在别的目录运行命令（例如激活虚拟环境后停在上一级目录）时，自动切换到项目目录；
    用 -c 或环境变量 HER_BRAIN_CONFIG 指定了配置时不切换。返回切换到的目录。
    """
    if explicit_config or os.environ.get("HER_BRAIN_CONFIG"):
        return None
    cwd = Path.cwd()
    if (cwd / "config" / "brain.yaml").exists() or (cwd / "config" / "brain.example.yaml").exists():
        return None
    if (PROJECT_ROOT / "config" / "brain.example.yaml").exists() and PROJECT_ROOT != cwd:
        os.chdir(PROJECT_ROOT)
        print(f"（当前目录不是项目目录，已切换到 {PROJECT_ROOT}）", file=sys.stderr)
        return PROJECT_ROOT
    return None


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _print(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_serve(settings: Settings, args: argparse.Namespace) -> None:
    import uvicorn

    from .app import create_app
    from .launcher import proxy_conflict

    conflict = proxy_conflict(settings, args.host, args.port)
    if conflict:
        sys.exit(f"[跳过] {conflict}")
    app = create_app(settings)
    uvicorn.run(
        app,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        log_level=settings.server.log_level.lower(),
        access_log=False,
    )


def cmd_init(args: argparse.Namespace) -> None:
    cfg = Path(args.config or default_config_path())
    example = Path("config/brain.example.yaml")
    if not cfg.exists():
        if not example.exists():
            sys.exit("找不到 config/brain.example.yaml，请在项目根目录运行")
        cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example, cfg)
        print(f"已生成配置文件 {cfg}，请按需修改")
    settings = load_settings(cfg)
    # 只初始化本地数据，不连接模型服务和 Qdrant（它们此时可能还没启动）
    from .identity.resolver import IdentityResolver
    from .memory.db import Database
    from .prompt.persona import PersonaHolder

    settings.data_path.mkdir(parents=True, exist_ok=True)
    persona = PersonaHolder(settings.persona_path)
    db = Database(settings.db_path)
    try:
        IdentityResolver(db, settings).seed()
        users = db.query("SELECT user_id, role FROM users")
    finally:
        db.close()
    print(f"数据库已就绪：{settings.db_path}（{len(users)} 个用户）")
    print(f"人设：{persona.name}（{persona.digest}，静态前缀 {len(persona.static_text)} 字）")
    print(f"记忆引擎：{settings.memory.engine}；对话模型：{settings.upstream.base_url}")


async def _with_services(settings: Settings, fn):  # type: ignore[no-untyped-def]
    from .services import Services

    svc = Services(settings)
    try:
        return await fn(svc)
    finally:
        await svc.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="her_brain", description="她的大脑代理")
    parser.add_argument("-c", "--config", help="配置文件路径（默认 config/brain.yaml 或环境变量 HER_BRAIN_CONFIG）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up", help="一键启动模型服务与大脑代理")
    p_up.add_argument("--host")
    p_up.add_argument("--port", type=int)
    p_up.add_argument("--no-llm", action="store_true", help="不启动对话模型（已在别处运行）")
    p_up.add_argument("--no-embed", action="store_true", help="不启动嵌入模型")
    sub.add_parser("stop", help="停止正在运行的大脑代理（以及由 up 启动的模型服务）")
    p_serve = sub.add_parser("serve", help="只启动大脑代理")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    sub.add_parser("init", help="生成配置、数据目录与数据库")
    p_setup = sub.add_parser("setup", help="配置助手")
    p_setup.add_argument("--qq", help="你的 QQ 号（或 QQ 官方机器人下的用户 ID）")
    p_setup.add_argument("--astrbot", help="AstrBot 安装目录，用于复制插件")
    p_setup.add_argument("--docker", action="store_true", default=None, help="AstrBot 运行在 Docker 中")
    p_setup.add_argument("--engine", choices=["mem0", "lite"], help="指定记忆引擎")
    p_setup.add_argument("--yes", action="store_true", help="不提问，未给出的项保持不变")
    sub.add_parser("doctor", help="环境自检")
    p_self = sub.add_parser("selftest", help="端到端自测（需要模型服务已启动）")
    p_self.add_argument("--keep", action="store_true", help="保留自测的临时数据目录")
    sub.add_parser("process", help="立即处理待提取的原始日志")
    sub.add_parser("backup", help="生成一份快照")
    p_restore = sub.add_parser("restore", help="从快照恢复（请先停止代理）")
    p_restore.add_argument("snapshot", nargs="?", help="快照目录或名称前缀；省略并加 --yesterday 时选前一天的快照")
    p_restore.add_argument("--yesterday", action="store_true", help="恢复到最近一份早于今天零点的快照")
    p_restore.add_argument("--keep-persona", action="store_true", help="不恢复人设文件")
    p_restore.add_argument("--no-reindex", action="store_true", help="恢复后不重建向量索引")
    sub.add_parser("reindex", help="按 SQLite 重建向量索引")
    args = parser.parse_args(argv)
    locate_project(args.config)

    if args.cmd == "init":
        _setup_logging("INFO")
        cmd_init(args)
        return
    if args.cmd == "setup":
        from .setup_wizard import run_setup

        sys.exit(run_setup(args.config, qq=args.qq, astrbot=args.astrbot, docker=args.docker,
                           engine=args.engine, assume_yes=args.yes))
    if args.cmd == "doctor":
        from .doctor import print_report, run_doctor

        sys.exit(print_report(run_doctor(args.config)))

    settings = load_settings(args.config)
    _setup_logging(settings.server.log_level)

    if args.cmd == "stop":
        from .launcher import stop_running

        sys.exit(stop_running(settings))
    if args.cmd == "up":
        from .launcher import run_up

        run_up(settings, chat=not args.no_llm, embedding=not args.no_embed, host=args.host, port=args.port)
    elif args.cmd == "selftest":
        from .selftest import run_selftest, summarize

        # 自测结论逐项打印，过程日志只会干扰阅读
        logging.disable(logging.ERROR)
        sys.exit(summarize(asyncio.run(run_selftest(settings, keep=args.keep))))
    elif args.cmd == "serve":
        cmd_serve(settings, args)
    elif args.cmd == "process":
        _print(asyncio.run(_with_services(settings, lambda s: s.worker.run_until_idle())))
    elif args.cmd == "backup":
        async def _backup(s):  # type: ignore[no-untyped-def]
            return s.backup.snapshot("manual")

        _print(asyncio.run(_with_services(settings, _backup)))
    elif args.cmd == "restore":
        from .backup import restore_snapshot, yesterday_snapshot

        snap = args.snapshot
        if snap is None and args.yesterday:
            found = yesterday_snapshot(settings)
            if found is None:
                sys.exit("没有早于今天的快照")
            snap = str(found)
        if snap is None:
            sys.exit("请指定快照，或使用 --yesterday")
        result = restore_snapshot(settings, snap, restore_persona=not args.keep_persona)
        if not args.no_reindex:
            result["reindex"] = asyncio.run(_with_services(settings, lambda s: s.episodic.rebuild_index()))
        _print(result)
    elif args.cmd == "reindex":
        _print(asyncio.run(_with_services(settings, lambda s: s.episodic.rebuild_index())))


if __name__ == "__main__":
    main()
