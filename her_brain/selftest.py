"""端到端自测：python -m her_brain selftest

在临时目录里建一套独立的数据（不碰你的正式记忆和日志），连接配置里真实的对话/嵌入服务，依次验证：

1. 上游连通：能正常回复，没有 <think> 泄漏；
2. 写入：QQ 私聊说“明天要去面试”“记住我对花生过敏”，再在群里说一句；
3. 后台提取：模拟过了一天，用真实模型提取记忆；面试应带上具体日期，“花生过敏”应直接进入核心档案；
4. 跨入口与场合：桌面语音说“我回来了”时提示词里有面试与过敏；群聊与未确认身份的语音里没有；
5. 真实回复：看她会不会主动问起面试；
6. 延迟：代理准备耗时、检索耗时、首字时间是否在预算内。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, UserConfig
from .timeutil import get_tz, local_date

LABELS = {"ok": "[通过]", "warn": "[注意]", "fail": "[失败]", "info": "[信息]"}
OWNER_QQ = "selftest-10001"
PRIVATE = {"channel": "qq", "scene": "private", "external_user_id": OWNER_QQ, "user_name": "小明"}
GROUP = {"channel": "qq", "scene": "group", "external_user_id": OWNER_QQ, "user_name": "小明",
         "group_id": "selftest-group", "group_name": "自测群"}
VOICE = {"channel": "voice", "scene": "voice", "external_user_id": "owner", "speaker_verified": True}
VOICE_UNVERIFIED = {"channel": "voice", "scene": "voice", "speaker_verified": False}


@dataclass
class Result:
    status: str
    title: str
    detail: str = ""


class Abort(Exception):
    pass


def isolated_settings(settings: Settings, root: Path) -> Settings:
    s = settings.model_copy(deep=True)
    s.data_dir = str(root / "data")
    s.config_path = None
    s.api_keys = []
    s.extensions = []
    s.backup.enabled = False
    s.backup.dir = str(root / "backups")
    s.worker.enabled = False
    s.upstream.warmup = False
    s.users = [UserConfig(user_id="tester", display_name="小明", role="owner",
                          bindings=[f"qq:{OWNER_QQ}", "voice:owner"])]
    if s.memory.engine == "mem0":
        s.memory.qdrant_path = str(root / "qdrant")
        s.memory.collection = "her_selftest"
    return s


def _body(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"model": "her", "messages": [{"role": "user", "content": text}], "metadata": metadata}


def _system_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content")) for m in messages if m.get("role") == "system")


async def run_selftest(settings: Settings, *, transports: Any = None, keep: bool = False,
                       printer: Any = print) -> list[Result]:
    from .app import create_app
    from .services import Services

    root = Path(tempfile.mkdtemp(prefix="her-selftest-"))
    results: list[Result] = []

    def report(status: str, title: str, detail: str = "") -> None:
        results.append(Result(status, title, detail))
        printer(f"{LABELS[status]} {title}" + (f"：{detail}" if detail else ""))

    test_settings = isolated_settings(settings, root)
    printer(f"自测数据目录：{root}（{'保留' if keep else '结束后删除'}），记忆引擎 {test_settings.memory.engine}\n")
    try:
        svc = Services(test_settings, transports=transports)
    except Exception as exc:  # noqa: BLE001
        report("fail", "初始化", str(exc))
        shutil.rmtree(root, ignore_errors=True)
        return results

    app = create_app(test_settings, svc, run_worker=False)
    app.state.services = svc
    tz = get_tz(test_settings.prompt.timezone)
    try:
        await svc.warmup_memory()  # 与正式启动一致：先预热检索，避免把冷启动算进延迟
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://selftest",
                                     timeout=600) as client:

            async def chat(text: str, meta: dict[str, Any]) -> str:
                r = await client.post("/v1/chat/completions", json=_body(text, meta))
                if r.status_code != 200:
                    raise Abort(f"HTTP {r.status_code}：{r.text[:300]}")
                return r.json()["choices"][0]["message"]["content"] or ""

            async def prompt_for(text: str, meta: dict[str, Any]) -> str:
                r = await client.post("/admin/debug/prompt", json=_body(text, meta))
                r.raise_for_status()
                return _system_text(r.json()["messages"])

            # 1. 上游连通
            try:
                t0 = time.perf_counter()
                reply = await chat("你好呀，今天过得怎么样？", PRIVATE)
                elapsed = (time.perf_counter() - t0) * 1000
            except (Abort, httpx.HTTPError) as exc:
                report("fail", "上游连通", f"{exc}（对话服务 {test_settings.upstream.base_url}；先运行 python -m her_brain up）")
                return results
            if not reply.strip():
                report("fail", "上游连通", "模型返回了空回复")
            elif "<think>" in reply or "</think>" in reply:
                report("fail", "上游连通", f"回复里有思考内容：{reply[:80]}")
            else:
                report("ok", "上游连通", f"{elapsed:.0f}ms 完成一轮回复：「{reply[:60]}」")

            # 嵌入服务：维度不对时后面的记忆步骤会全部连环失败，先单独报出来
            if svc.embedder is not None:
                try:
                    dims = len((await svc.embedder.embed(["测试"]))[0])
                except Exception as exc:  # noqa: BLE001
                    report("fail", "嵌入服务", f"{test_settings.embedding.base_url} 不可用：{exc}（先运行 python -m her_brain up）")
                    return results
                if dims != test_settings.embedding.dims:
                    report("fail", "嵌入服务", f"向量维度 {dims} 与配置 embedding.dims={test_settings.embedding.dims} 不一致；"
                                              f"把配置改为 dims: {dims} 后重试（已有正式记忆时再运行 python -m her_brain reindex）")
                    return results
                report("ok", "嵌入服务", f"{dims} 维")

            # 2. 写入
            for text, meta in (("我明天要去面试，有点紧张", PRIVATE), ("记住，我对花生过敏", PRIVATE),
                               ("大家晚上好呀", GROUP)):
                await chat(text, meta)
            logs = svc.rawlog.list(user_id="tester", limit=20)
            privacies = {log["user_text"]: log["privacy"] for log in logs}
            ok = privacies.get("我明天要去面试，有点紧张") == "private" and privacies.get("大家晚上好呀") == "public"
            report("ok" if ok else "fail", "原始日志与私密度", f"{len(logs)} 条（私聊=私密，群聊=公开）")

            # 3. 模拟过了一天，后台提取
            svc.db.execute("UPDATE raw_log SET created_at = created_at - 86400")
            svc.recent.forget()
            t0 = time.perf_counter()
            summary = await svc.worker.run_until_idle()
            elapsed = time.perf_counter() - t0
            counts = svc.rawlog.counts()
            mems = svc.episodic.list_for_user("tester")
            self_mems = svc.episodic.list_for_user(test_settings.self_user_id)
            listing = "；".join(
                f"{'（关于她）' if m.user_id != 'tester' else ''}{m.kind}{'@' + m.event_date if m.event_date else ''} {m.text}"
                for m in [*mems, *self_mems]
            ) or "（无）"
            if counts["failed"] or svc.worker.last_error:
                report("fail", "后台提取", f"{svc.worker.last_error or ''} 日志状态 {counts}")
            else:
                report("ok", "后台提取", f"{elapsed:.1f} 秒，{summary['delta'].get('facts', 0)} 条事实：{listing}")
            today = local_date(time.time(), tz).isoformat()
            interview = [m for m in mems if "面试" in m.text]
            if not interview:
                report("fail", "时效记忆（面试）", "没有提取出面试这件事")
            elif not any(m.event_date == today for m in interview):
                report("warn", "时效记忆（面试）", f"提取到了，但日期为 {interview[0].event_date}（应为 {today}）")
            else:
                report("ok", "时效记忆（面试）", f"日期换算正确：{today}")
            profile = [e.text for e in svc.profiles.get("tester").entries]
            if any("花生" in t for t in profile):
                report("ok", "核心档案（明确要求记住）", "；".join(profile))
            else:
                cands = [c["text"] for c in svc.promoter.list("tester")]
                report("fail", "核心档案（明确要求记住）", f"档案里没有花生过敏；候选：{cands or '无'}")
            unconfirmed = [t for t in profile if "面试" in t]
            report("fail" if unconfirmed else "ok", "核心档案（未确认的事实不进入）",
                   f"只提过一次、没要求记住的事进了档案：{unconfirmed}" if unconfirmed else "只提过一次的面试没有进入档案")

            # 4. 跨入口与场合过滤
            voice_ctx = await prompt_for("我回来了", VOICE)
            has = [w for w in ("面试", "花生") if w in voice_ctx]
            report("ok" if len(has) == 2 else "fail", "跨入口（QQ → 语音）",
                   "语音提示词里带上了面试与过敏" if len(has) == 2 else f"只找到：{has or '无'}")
            group_ctx = await prompt_for("最近怎么样", GROUP)
            leaked = [w for w in ("面试", "花生") if w in group_ctx]
            report("fail" if leaked else "ok", "场合过滤（群聊）", f"泄漏了：{leaked}" if leaked else "私密内容未出现")
            anon_ctx = await prompt_for("你知道你主人的事吗", VOICE_UNVERIFIED)
            leaked = [w for w in ("面试", "花生") if w in anon_ctx]
            report("fail" if leaked else "ok", "场合过滤（未确认身份的语音）",
                   f"泄漏了：{leaked}" if leaked else "私密内容未出现")

            # 5. 真实回复
            reply = await chat("我回来了", VOICE)
            if "面试" in reply:
                report("ok", "语音回复", f"「{reply[:80]}」")
            elif "面试" in voice_ctx:
                report("warn", "语音回复", f"「{reply[:80]}」（提示词里有面试，但这次她没提起；可在人设里强调主动关心）")
            else:
                report("warn", "语音回复", f"「{reply[:80]}」（提示词里没有面试，见上面的提取结果）")

        # 6. 延迟
        lat = svc.metrics.summary()["latency_ms"]
        prep = lat.get("proxy_prepare_ms", {})
        retr = lat.get("retrieval_ms", {})
        first = lat.get("first_token_ms", {})
        timeouts = svc.metrics.summary()["counters"].get("retrieval_timeouts", 0)
        if prep:
            report("ok" if prep["p50"] < 50 else "warn", "代理准备耗时", f"p50 {prep['p50']}ms，最大 {prep['max']}ms（预算 50ms）")
        if retr:
            budget = test_settings.memory.retrieval_timeout_ms
            report("ok" if not timeouts else "warn", "记忆检索耗时",
                   f"p50 {retr['p50']}ms，最大 {retr['max']}ms，超时 {timeouts} 次（上限 {budget}ms）")
        if first:
            report("ok" if first["p50"] < 1000 else "warn", "首字时间（经代理）",
                   f"p50 {first['p50']}ms，最大 {first['max']}ms（文字目标 1 秒内）")
    except Abort as exc:
        report("fail", "中止", str(exc))
    finally:
        await svc.close()
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    return results


def summarize(results: list[Result], printer: Any = print) -> int:
    fails = sum(r.status == "fail" for r in results)
    warns = sum(r.status == "warn" for r in results)
    printer(f"\n共 {len(results)} 项：{fails} 项失败，{warns} 项需要注意。")
    return 1 if fails else 0
