"""档案晋升：一次错误提取不会直接写进核心档案。

每条“描述这个人是什么样”的事实先成为候选，被不同对话独立确认足够次数
（或用户明确说“记住”）后才晋升进核心档案。同一 slot 出现新取值时，
新值同样需要确认，确认后替换旧值，旧值在档案历史中保留。
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import ProfileConfig
from ..identity.models import PRIVATE, PUBLIC
from ..textutil import dice, same_value
from ..timeutil import now
from .db import Database, dumps, loads
from .models import PROFILE_KINDS, Fact
from .profile import ProfileEntry, ProfileStore

log = logging.getLogger(__name__)

SLOT_SYNONYMS = {
    "姓名": "名字", "名称": "名字", "真名": "名字",
    "称呼": "昵称", "外号": "昵称", "小名": "昵称",
    "工作": "职业", "职位": "职业", "岗位": "职业", "身份": "职业",
    "公司": "工作单位", "单位": "工作单位",
    "城市": "所在城市", "居住地": "所在城市", "住址": "所在城市", "居住城市": "所在城市", "现居地": "所在城市",
    "老家": "家乡", "籍贯": "家乡",
    "学校": "学校", "就读学校": "学校",
    "恋爱状态": "感情状态", "婚姻状况": "感情状态",
    "出生日期": "生日",
}

_SLOT_CLEAN_RE = re.compile(r"[\s:：,，。.、\"'“”‘’()（）\[\]【】]")


def normalize_slot(slot: str) -> str:
    slot = _SLOT_CLEAN_RE.sub("", slot or "")
    return SLOT_SYNONYMS.get(slot, slot)


@dataclass
class ObserveResult:
    candidate_id: str
    confirmations: int
    status: str
    promoted: bool = False
    counted: bool = True  # False 表示这批日志已经计过数（重复处理）
    superseded_slot: str | None = None
    profile_entry: ProfileEntry | None = None


class ProfilePromoter:
    def __init__(self, db: Database, profiles: ProfileStore, cfg: ProfileConfig):
        self.db = db
        self.profiles = profiles
        self.cfg = cfg

    @staticmethod
    def is_profile_worthy(fact: Fact) -> bool:
        return fact.kind in PROFILE_KINDS or bool(fact.slot) or fact.explicit

    def observe(self, user_id: str, fact: Fact, privacy: str, source_log_ids: list[str]) -> ObserveResult | None:
        if not self.is_profile_worthy(fact):
            return None
        slot = normalize_slot(fact.slot)
        value = fact.value.strip()
        ts = now()
        rows = self.db.query(
            """SELECT * FROM profile_candidate WHERE user_id = ? AND slot = ?
               AND status IN ('pending', 'promoted') ORDER BY last_seen DESC""",
            (user_id, slot),
        )
        match: dict[str, Any] | None = None
        best = -1.0
        for row in rows:
            if slot:
                a, b = value or fact.text, row["value"] or row["text"]
            else:
                a, b = fact.text, row["text"]
            if not same_value(a, b, self.cfg.match_threshold):
                continue
            sim = dice(a, b)
            if sim > best:
                best, match = sim, row

        counted = True
        if match is not None:
            known = set(loads(match["source_log_ids"], []) or [])
            new_sources = set(source_log_ids) - known
            if not new_sources:
                counted = False
                cand = match
            else:
                confirmations = int(match["confirmations"]) + 1
                merged_privacy = PUBLIC if PUBLIC in (match["privacy"], privacy) else PRIVATE
                explicit = bool(match["explicit"]) or fact.explicit
                self.db.execute(
                    """UPDATE profile_candidate SET confirmations = ?, explicit = ?, importance = ?,
                       privacy = ?, source_log_ids = ?, last_seen = ?, event_date = COALESCE(?, event_date),
                       text = CASE WHEN ? THEN ? ELSE text END
                       WHERE candidate_id = ?""",
                    (
                        confirmations, int(explicit), max(float(match["importance"]), fact.effective_importance),
                        merged_privacy, dumps(sorted(known | new_sources)[-50:]), ts, fact.date,
                        int(fact.explicit), fact.text, match["candidate_id"],
                    ),
                )
                cand = self._get(match["candidate_id"])
        else:
            cid = "c_" + uuid.uuid4().hex[:12]
            self.db.execute(
                """INSERT INTO profile_candidate(candidate_id, user_id, slot, value, text, kind, privacy,
                       confirmations, explicit, importance, event_date, status, source_log_ids, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 'pending', ?, ?, ?)""",
                (
                    cid, user_id, slot, value, fact.text, fact.kind, privacy, int(fact.explicit),
                    fact.effective_importance, fact.date, dumps(sorted(set(source_log_ids))), ts, ts,
                ),
            )
            cand = self._get(cid)
        assert cand is not None

        result = ObserveResult(
            candidate_id=cand["candidate_id"],
            confirmations=int(cand["confirmations"]),
            status=cand["status"],
            counted=counted,
        )
        if not counted:
            return result

        need = max(1, self.cfg.confirmations_to_promote)
        if cand["status"] == "pending" and (int(cand["confirmations"]) >= need or bool(cand["explicit"])):
            reason = "明确要求记住" if cand["explicit"] else f"确认 {cand['confirmations']} 次"
            entry = self._promote(cand, f"晋升：{cand['text'][:24]}（{reason}）")
            result.promoted = True
            result.status = "promoted"
            result.profile_entry = entry
            result.superseded_slot = slot or None
        elif cand["status"] == "promoted" and cand["privacy"] == PUBLIC:
            # 已晋升的私密条目后来在公开场合被提到：同步把档案条目改为公开
            self._sync_privacy(cand)
        return result

    def _promote(self, cand: dict[str, Any], reason: str) -> ProfileEntry | None:
        _, entry = self.profiles.upsert_fact(
            cand["user_id"],
            text=cand["text"],
            slot=cand["slot"],
            value=cand["value"],
            kind=cand["kind"],
            privacy=cand["privacy"],
            confirmations=int(cand["confirmations"]),
            source_log_ids=loads(cand["source_log_ids"], []) or [],
            event_date=cand.get("event_date"),
            reason=reason,
        )
        with self.db.transaction():
            self.db.execute(
                "UPDATE profile_candidate SET status = 'promoted', profile_entry_id = ? WHERE candidate_id = ?",
                (entry.id if entry else None, cand["candidate_id"]),
            )
            if cand["slot"]:
                self.db.execute(
                    """UPDATE profile_candidate SET status = 'superseded'
                       WHERE user_id = ? AND slot = ? AND status = 'promoted' AND candidate_id != ?""",
                    (cand["user_id"], cand["slot"], cand["candidate_id"]),
                )
        log.info("档案晋升 %s：%s", cand["user_id"], cand["text"])
        return entry

    def _sync_privacy(self, cand: dict[str, Any]) -> None:
        entry_id = cand.get("profile_entry_id")
        if not entry_id:
            return
        profile = self.profiles.get(cand["user_id"])
        entry = next((e for e in profile.entries if e.id == entry_id), None)
        if entry is not None and entry.privacy != PUBLIC:
            self.profiles.update_entry(cand["user_id"], entry_id, {"privacy": PUBLIC}, reason="在公开场合被提及，改为公开")

    # ------------------------------------------------------------------ 管理
    def _get(self, candidate_id: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM profile_candidate WHERE candidate_id = ?", (candidate_id,))

    def list(self, user_id: str, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM profile_candidate WHERE user_id = ?"
        params: list[Any] = [user_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        for r in rows:
            r["source_log_ids"] = loads(r["source_log_ids"], [])
        return rows

    def promote(self, candidate_id: str) -> ProfileEntry | None:
        cand = self._get(candidate_id)
        if cand is None:
            raise KeyError(candidate_id)
        return self._promote(cand, f"手动晋升：{cand['text'][:24]}")

    def reject(self, candidate_id: str) -> bool:
        cur = self.db.execute(
            "UPDATE profile_candidate SET status = 'rejected' WHERE candidate_id = ?", (candidate_id,)
        )
        return cur.rowcount > 0

    def expire_stale(self, now_ts: float | None = None) -> int:
        cutoff = (now_ts or now()) - self.cfg.candidate_expire_days * 86400
        cur = self.db.execute(
            """UPDATE profile_candidate SET status = 'expired'
               WHERE status = 'pending' AND confirmations <= 1 AND explicit = 0 AND last_seen < ?""",
            (cutoff,),
        )
        return cur.rowcount

    def reset_user(self, user_id: str) -> int:
        cur = self.db.execute(
            "DELETE FROM profile_candidate WHERE user_id = ? AND status IN ('pending', 'expired')", (user_id,)
        )
        return cur.rowcount
