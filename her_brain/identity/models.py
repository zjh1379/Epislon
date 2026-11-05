from __future__ import annotations

from dataclasses import dataclass, field

PUBLIC = "public"
PRIVATE = "private"

ROLE_LABELS = {"owner": "你的主人", "friend": "朋友", "guest": "访客", "self": "你自己"}


@dataclass
class RequestMeta:
    """入口随请求附带的“谁在什么场合说话”。"""

    channel: str = "unknown"  # qq / voice / discord / duck …
    scene: str | None = None  # private / group / voice
    external_user_id: str | None = None
    user_name: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    speaker_verified: bool | None = None
    platform: str | None = None  # 入口原始平台名，例如 aiocqhttp
    mode: str = "chat"  # chat：注入人设与记忆；raw：原样透传
    no_memory: bool = False  # 记录原始日志，但不从中提取记忆
    no_log: bool = False  # 完全不落库
    sources: list[str] = field(default_factory=list)  # 元数据来自哪里，便于排查
    # 知道是谁，却没有任何来源说明是私聊还是群聊（例如只靠 AstrBot 的“用户识别”）。
    # 此时按保守模式处理：只读公开记忆，新写入一律标为私密
    scene_uncertain: bool = False


@dataclass
class Identity:
    user_id: str
    display_name: str
    role: str
    platform: str
    external_id: str
    anonymous: bool = False

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)


@dataclass
class Scene:
    channel: str
    scene: str
    group_id: str | None
    group_name: str | None
    speaker_verified: bool | None
    # 本轮可以读取的记忆级别：public 只读公开；private 公开与私密都可读
    access: str
    # 本轮对话写入日志时标注的私密度
    write_privacy: str
    # 无法确认是私聊还是群聊（见 RequestMeta.scene_uncertain）
    uncertain: bool = False

    @property
    def allowed_privacy(self) -> set[str]:
        return {PUBLIC, PRIVATE} if self.access == PRIVATE else {PUBLIC}

    @property
    def conversation_key(self) -> tuple[str, str]:
        """同一个对话窗口的标识，用于在“近期动态”里排除入口已经带上的上下文。"""
        return (self.channel, self.group_id or f"{self.scene}")
