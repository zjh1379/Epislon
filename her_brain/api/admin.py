"""管理接口（/admin/*）：重载人设、查看与修改记忆、版本回滚、重新提取、备份等。

未设置 server.admin_token 时只允许本机访问。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from ..memory.rawlog import STATUS_NAMES
from ..timeutil import parse_time_arg
from .auth import require_admin

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

STATUS_CODES = {v: k for k, v in STATUS_NAMES.items()}


def _svc(request: Request) -> Any:
    return request.app.state.services


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"找不到{what}")


# ---------------------------------------------------------------------- 概览
@router.get("/stats")
async def stats(request: Request) -> dict[str, Any]:
    s = _svc(request)
    return {
        "persona": {"name": s.persona.name, "digest": s.persona.digest, "loaded_at": s.persona.loaded_at},
        "engine": s.index.name,
        "metrics": s.metrics.summary(),
        "worker": s.worker.status(),
        "memories": s.episodic.count(),
    }


# ---------------------------------------------------------------------- 人设
@router.post("/persona/reload")
async def reload_persona(request: Request) -> dict[str, Any]:
    s = _svc(request)
    ok, info = s.persona.reload()
    if not ok:
        raise HTTPException(status_code=400, detail=f"人设文件有误，仍在使用旧版本：{info}")
    if s.settings.upstream.warmup:
        s.spawn(s.warmup())
    return {"ok": True, "digest": info, "name": s.persona.name}


@router.get("/persona")
async def get_persona(request: Request) -> dict[str, Any]:
    s = _svc(request)
    return {"name": s.persona.name, "digest": s.persona.digest, "static_prefix": s.persona.static_text}


# ---------------------------------------------------------------------- 用户与身份
@router.get("/users")
async def list_users(request: Request) -> list[dict[str, Any]]:
    return _svc(request).resolver.list_users()


@router.post("/users")
async def create_user(request: Request, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not data.get("user_id"):
        raise HTTPException(400, "缺少 user_id")
    return _svc(request).resolver.create_user(data["user_id"], data.get("display_name", data["user_id"]),
                                              data.get("role", "friend"))


@router.get("/users/{user_id}")
async def get_user(request: Request, user_id: str) -> dict[str, Any]:
    user = _svc(request).resolver.get_user(user_id)
    if user is None:
        raise _not_found("用户")
    return user


@router.post("/identities")
async def link_identity(request: Request, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """把平台身份绑定到用户，例如把 qq:123456 绑到 owner。move_logs=true 会迁移旧访客的日志并重新提取。"""
    for key in ("platform", "external_id", "user_id"):
        if not data.get(key):
            raise HTTPException(400, f"缺少 {key}")
    s = _svc(request)
    if s.resolver.get_user(data["user_id"]) is None:
        raise _not_found("目标用户")
    return s.resolver.link(data["platform"], str(data["external_id"]), data["user_id"], bool(data.get("move_logs")))


@router.delete("/identities/{platform}/{external_id}")
async def unlink_identity(request: Request, platform: str, external_id: str) -> dict[str, Any]:
    return {"removed": _svc(request).resolver.unlink(platform, external_id)}


# ---------------------------------------------------------------------- 核心档案
@router.get("/users/{user_id}/profile")
async def get_profile(request: Request, user_id: str) -> dict[str, Any]:
    return _svc(request).profiles.get(user_id).to_dict()


@router.put("/users/{user_id}/profile")
async def replace_profile(request: Request, user_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return _svc(request).profiles.replace_all(user_id, data.get("entries", []), data.get("reason", "管理接口整体替换")).to_dict()


@router.post("/users/{user_id}/profile/entries")
async def add_profile_entry(request: Request, user_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not data.get("text"):
        raise HTTPException(400, "缺少 text")
    profile, entry = _svc(request).profiles.add_entry(user_id, data)
    return {"version": profile.version, "entry": asdict(entry)}


@router.patch("/users/{user_id}/profile/entries/{entry_id}")
async def update_profile_entry(request: Request, user_id: str, entry_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return _svc(request).profiles.update_entry(user_id, entry_id, data).to_dict()
    except KeyError:
        raise _not_found("档案条目") from None


@router.delete("/users/{user_id}/profile/entries/{entry_id}")
async def delete_profile_entry(request: Request, user_id: str, entry_id: str) -> dict[str, Any]:
    try:
        return _svc(request).profiles.delete_entry(user_id, entry_id).to_dict()
    except KeyError:
        raise _not_found("档案条目") from None


@router.get("/users/{user_id}/profile/versions")
async def profile_versions(request: Request, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    return _svc(request).profiles.versions(user_id, limit)


@router.get("/users/{user_id}/profile/versions/{version}")
async def profile_version(request: Request, user_id: str, version: int) -> dict[str, Any]:
    entries = _svc(request).profiles.get_version(user_id, version)
    if entries is None:
        raise _not_found("该版本")
    return {"user_id": user_id, "version": version, "entries": entries}


@router.post("/users/{user_id}/profile/rollback")
async def profile_rollback(request: Request, user_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return _svc(request).profiles.rollback(user_id, int(data["version"])).to_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


# ---------------------------------------------------------------------- 档案候选
@router.get("/users/{user_id}/candidates")
async def list_candidates(request: Request, user_id: str, status: str | None = None) -> list[dict[str, Any]]:
    return _svc(request).promoter.list(user_id, status)


@router.post("/candidates/{candidate_id}/promote")
async def promote_candidate(request: Request, candidate_id: str) -> dict[str, Any]:
    try:
        entry = _svc(request).promoter.promote(candidate_id)
    except KeyError:
        raise _not_found("候选") from None
    return {"promoted": True, "entry": asdict(entry) if entry else None}


@router.post("/candidates/{candidate_id}/reject")
async def reject_candidate(request: Request, candidate_id: str) -> dict[str, Any]:
    return {"rejected": _svc(request).promoter.reject(candidate_id)}


# ---------------------------------------------------------------------- 情节记忆
@router.get("/users/{user_id}/memories")
async def list_memories(
    request: Request,
    user_id: str,
    q: str | None = None,
    semantic: bool = False,
    include_deleted: bool = False,
    limit: int = Query(100, le=1000),
    offset: int = 0,
) -> list[dict[str, Any]]:
    s = _svc(request)
    if q and semantic:
        hits = await s.episodic.search([user_id], q, {"public", "private"}, limit)
        return [{**h.record.snapshot(), "score": round(h.score, 4)} for h in hits]
    return [r.snapshot() for r in s.episodic.list_for_user(user_id, include_deleted=include_deleted, q=q,
                                                          limit=limit, offset=offset)]


@router.post("/users/{user_id}/memories")
async def add_memory(request: Request, user_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not data.get("text"):
        raise HTTPException(400, "缺少 text")
    return (await _svc(request).episodic.add_manual(user_id, data)).snapshot()


@router.get("/memories/{memory_id}")
async def get_memory(request: Request, memory_id: str) -> dict[str, Any]:
    s = _svc(request)
    rec = s.episodic.get(memory_id)
    if rec is None:
        raise _not_found("记忆")
    return {**rec.snapshot(), "sources": [l.to_dict() for l in s.rawlog.get_many(rec.source_log_ids)]}


@router.patch("/memories/{memory_id}")
async def edit_memory(request: Request, memory_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return (await _svc(request).episodic.edit(memory_id, data)).snapshot()
    except KeyError:
        raise _not_found("记忆") from None


@router.delete("/memories/{memory_id}")
async def delete_memory(request: Request, memory_id: str, purge: bool = False) -> dict[str, Any]:
    s = _svc(request)
    try:
        if purge:
            await s.episodic.purge(memory_id)
            return {"purged": memory_id}
        return (await s.episodic.delete(memory_id)).snapshot()
    except KeyError:
        raise _not_found("记忆") from None


@router.post("/memories/{memory_id}/restore")
async def restore_memory(request: Request, memory_id: str) -> dict[str, Any]:
    try:
        return (await _svc(request).episodic.restore(memory_id)).snapshot()
    except KeyError:
        raise _not_found("记忆") from None


@router.get("/memory-events")
async def memory_events(
    request: Request, user_id: str | None = None, memory_id: str | None = None, limit: int = Query(100, le=1000)
) -> list[dict[str, Any]]:
    return _svc(request).episodic.events(user_id=user_id, memory_id=memory_id, limit=limit)


@router.post("/memory-events/{event_id}/revert")
async def revert_event(request: Request, event_id: int) -> dict[str, Any]:
    try:
        return await _svc(request).episodic.revert_event(event_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/users/{user_id}/memories/rollback")
async def rollback_memories(request: Request, user_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """把该用户的情节记忆恢复到 since 时刻（epoch 秒或 ISO 时间）。"""
    if "since" not in data:
        raise HTTPException(400, "缺少 since")
    return await _svc(request).episodic.rollback_user(user_id, parse_time_arg(data["since"]))


@router.post("/users/{user_id}/rebuild")
async def rebuild_user(request: Request, user_id: str, data: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """从原始日志重新生成该用户的情节记忆与档案候选（旧记忆软删除，可回滚）。"""
    s = _svc(request)
    deleted = await s.episodic.soft_delete_user(user_id, "admin", "从原始日志重建前清空")
    reset = s.promoter.reset_user(user_id)
    requeued = s.rawlog.reprocess(user_id=user_id)
    result = {"soft_deleted": deleted, "candidates_reset": reset, "requeued_logs": requeued}
    if data.get("run_now"):
        result["worker"] = await s.worker.run_until_idle()
    return result


@router.post("/reindex")
async def reindex(request: Request) -> dict[str, Any]:
    """按 SQLite 中的权威副本重建向量索引。"""
    return await _svc(request).episodic.rebuild_index()


# ---------------------------------------------------------------------- 原始日志
@router.get("/logs")
async def list_logs(
    request: Request, user_id: str | None = None, status: str | None = None,
    limit: int = Query(50, le=1000), offset: int = 0,
) -> list[dict[str, Any]]:
    code = STATUS_CODES.get(status) if status else None
    return _svc(request).rawlog.list(user_id, code, limit, offset)


@router.post("/logs/reprocess")
async def reprocess_logs(request: Request, data: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """把日志重新标记为待处理，对照原文重新提取。不传参数时只重试失败的日志。"""
    s = _svc(request)
    n = s.rawlog.reprocess(log_ids=data.get("log_ids"), user_id=data.get("user_id"))
    result: dict[str, Any] = {"requeued": n}
    if data.get("run_now"):
        result["worker"] = await s.worker.run_until_idle()
    return result


# ---------------------------------------------------------------------- 后台任务与备份
@router.get("/worker")
async def worker_status(request: Request) -> dict[str, Any]:
    return _svc(request).worker.status()


@router.post("/worker/run")
async def worker_run(request: Request) -> dict[str, Any]:
    return await _svc(request).worker.run_until_idle()


@router.post("/worker/tidy")
async def worker_tidy(request: Request) -> dict[str, Any]:
    return await _svc(request).worker.tidy()


@router.post("/backup")
async def make_backup(request: Request) -> dict[str, Any]:
    import asyncio

    return await asyncio.to_thread(_svc(request).backup.snapshot, "manual")


@router.get("/backups")
async def list_backups(request: Request) -> list[dict[str, Any]]:
    return _svc(request).backup.list()


# ---------------------------------------------------------------------- 调试
@router.post("/debug/prompt")
async def debug_prompt(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """按真实请求组装提示词但不调用模型，用于排查“她为什么会这么说”和记忆回归测试。"""
    s = _svc(request)
    turn = await s.chat.prepare(body, None)
    ctx = turn.ctx
    return {
        "user_id": turn.identity.user_id if turn.identity else None,
        "scene": asdict(turn.scene) if turn.scene else None,
        "meta_sources": turn.meta.sources,
        "prepare_ms": round((turn.t_prepared - turn.t_start) * 1000, 2),
        "retrieval_ms": round(ctx.retrieval_ms, 2) if ctx else None,
        "timed_out": ctx.timed_out if ctx else None,
        "memories": [
            {"memory_id": m.record.memory_id, "text": m.record.text, "score": round(m.score, 4), "reason": m.reason}
            for m in (ctx.memories if ctx else [])
        ],
        "sections": turn.debug.get("sections"),
        "messages": turn.upstream_body.get("messages"),
    }
