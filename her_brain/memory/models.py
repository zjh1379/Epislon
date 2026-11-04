from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

from .db import loads

KINDS = ("attribute", "preference", "relationship", "event", "plan", "other")
# 这些类型的事实描述的是“这个人是什么样的”，可以晋升进核心档案
PROFILE_KINDS = ("attribute", "preference", "relationship")
# 这些类型的事实带有日期，靠近日期时会被主动带上（时效记忆）
DATED_KINDS = ("event", "plan")


class Fact(BaseModel):
    """后台提取出的一条原子事实。"""

    text: str
    about: str = "user"  # user / self
    kind: str = "other"
    slot: str = ""  # 只有一个当前值的属性名，例如“职业”“所在城市”
    value: str = ""  # slot 对应的简短取值，例如“杭州”
    date: str | None = None  # 事件/计划日期 YYYY-MM-DD
    importance: int = 3
    explicit: bool = False  # 用户明确要求记住

    @field_validator("text", "slot", "value", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v or "").strip()

    @field_validator("about", mode="before")
    @classmethod
    def _about(cls, v: Any) -> str:
        return "self" if str(v or "").strip().lower() in ("self", "her", "assistant", "她", "自己") else "user"

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v: Any) -> str:
        v = str(v or "").strip().lower()
        return v if v in KINDS else "other"

    @field_validator("importance", mode="before")
    @classmethod
    def _importance(cls, v: Any) -> int:
        try:
            return max(1, min(5, int(round(float(v)))))
        except (TypeError, ValueError):
            return 3

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> str | None:
        v = str(v or "").strip()
        return v[:10] if v else None

    @field_validator("explicit", mode="before")
    @classmethod
    def _explicit(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "是")
        return bool(v)

    @property
    def effective_importance(self) -> int:
        return 5 if self.explicit else self.importance


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    text: str
    privacy: str
    importance: float = 3
    kind: str = "other"
    slot: str = ""
    event_date: str | None = None
    mentions: int = 1
    superseded: bool = False
    deleted: bool = False
    source_log_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> MemoryRecord:
        return cls(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            text=row["text"],
            privacy=row["privacy"],
            importance=float(row["importance"]),
            kind=row.get("kind") or "other",
            slot=row.get("slot") or "",
            event_date=row.get("event_date"),
            mentions=int(row.get("mentions") or 1),
            superseded=bool(row.get("superseded")),
            deleted=bool(row.get("deleted")),
            source_log_ids=loads(row.get("source_log_ids"), []) or [],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    def index_metadata(self) -> dict[str, Any]:
        """写入向量库的元数据（检索时按 privacy 过滤，按来源追溯）。"""
        return {
            "privacy": self.privacy,
            "importance": self.importance,
            "kind": self.kind,
            "slot": self.slot,
            "event_date": self.event_date or "",
            "source_log_ids": ",".join(self.source_log_ids),
        }


@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float
    reason: str = "semantic"  # semantic / agenda
