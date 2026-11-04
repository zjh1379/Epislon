"""人设：唯一来源是 Git 管理的 persona/her.yaml。

渲染结果是提示词的 [1] 人设与说话风格、[2] 行为规则两段，内容只取决于 YAML 文件本身，
因此每次请求都完全相同，可以被 llama.cpp 的前缀缓存复用。
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

SCENE_LABELS = {"private": "私聊", "group": "群聊", "voice": "语音对话"}

BUILTIN_MEMORY_RULES = [
    "“动态信息”里的档案、记忆和近期动态是你自己记得的事，要像本来就知道一样自然地用，"
    "不要说“根据记录”“资料显示”“我的数据库里”。",
    "记忆可能过时或记错。和对方现在的说法冲突时以对方为准，可以顺口确认一句。",
    "只在相关时提起记忆，不要每轮都翻旧账，也不要一次把知道的全说出来。",
    "场合为公开（群聊，或无法确认对方身份的语音）时，不要提任何私密的事，"
    "也不要透露别人私下跟你说过的话。",
    "不确定的事不要编造；不记得就坦白说不记得。",
]


class DialogueExample(BaseModel):
    user: str
    her: str


class Persona(BaseModel):
    name: str = "她"
    identity: str = ""
    personality: list[str] = Field(default_factory=list)
    speaking_style: str = ""
    background: str = ""
    relationship: str = ""
    catchphrases: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    scene_styles: dict[str, str] = Field(default_factory=dict)
    examples: list[DialogueExample] = Field(default_factory=list)

    def render_static(self) -> str:
        """[1] 人设与说话风格 + [2] 行为规则。纯函数，输出只取决于人设内容。"""
        lines: list[str] = ["# 你是谁", f"你是{self.name}。{self.identity.strip()}".rstrip()]
        if self.personality:
            lines += ["", "## 性格"] + [f"- {p}" for p in self.personality]
        if self.speaking_style.strip():
            lines += ["", "## 说话方式", self.speaking_style.strip()]
        if self.catchphrases:
            lines += ["", "## 口头禅（偶尔用，别每句都用）", "、".join(self.catchphrases)]
        if self.background.strip():
            lines += ["", "## 背景", self.background.strip()]
        if self.relationship.strip():
            lines += ["", "## 你和主人的关系", self.relationship.strip()]
        if self.examples:
            lines += ["", "## 对话示例（只示范语气，不是真实发生过的事）"]
            for ex in self.examples:
                lines += [f"对方：{ex.user}", f"{self.name}：{ex.her}"]

        lines += ["", "# 行为规则"]
        lines += [f"- {r}" for r in self.rules]
        lines += [f"- {r}" for r in BUILTIN_MEMORY_RULES]
        if self.scene_styles:
            lines += ["", "## 不同场合的说话要求"]
            for scene in ("private", "group", "voice"):
                if scene in self.scene_styles:
                    lines.append(f"- {SCENE_LABELS[scene]}：{self.scene_styles[scene].strip()}")
            for scene, style in self.scene_styles.items():
                if scene not in SCENE_LABELS:
                    lines.append(f"- {scene}：{style.strip()}")
        return "\n".join(lines).strip()


class PersonaHolder:
    """持有当前人设，支持热重载；重载失败时保留旧版本。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.persona = Persona()
        self.static_text = ""
        self.digest = ""
        self.loaded_at: float | None = None
        self.reload(strict=True)

    def reload(self, strict: bool = False) -> tuple[bool, str]:
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            persona = Persona.model_validate(raw)
            static = persona.render_static()
        except Exception as exc:  # noqa: BLE001 - 需要把任何解析错误报告给管理接口
            if strict:
                raise
            log.error("重载人设失败，继续使用旧版本：%s", exc)
            return False, str(exc)
        import time

        with self._lock:
            self.persona = persona
            self.static_text = static
            self.digest = hashlib.sha256(static.encode("utf-8")).hexdigest()[:12]
            self.loaded_at = time.time()
        log.info("人设已加载：%s（%s，%d 字）", persona.name, self.digest, len(static))
        return True, self.digest

    @property
    def name(self) -> str:
        return self.persona.name
