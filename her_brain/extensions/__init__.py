"""扩展点：为小鸭机器人、直播、游戏等下一阶段入口预留。

本阶段只定义接口，在两处调用：
- 提示词组装：``prompt_sections`` 可以追加动态信息（例如小鸭的传感器状态、可执行的动作列表）；
- 回复完成后：``on_reply`` 拿到完整回复，可解析“动作意图”并发布出去（例如通过 MQTT 发给小鸭）。

在配置文件 ``extensions`` 中按 ``"模块路径:类名"`` 登记即可启用。
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..identity.models import Identity, RequestMeta, Scene

log = logging.getLogger(__name__)


class ActionIntent(BaseModel):
    """结构化的“动作意图”，下一阶段由模型输出、经 MQTT 发给小鸭。"""

    target: str = "duck"
    name: str  # 例如 nod / wave / dance / look_at
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TurnInfo:
    meta: RequestMeta
    identity: Identity
    scene: Scene
    user_text: str
    reply_text: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class Extension:
    name = "extension"

    async def startup(self, services: Any) -> None:  # noqa: B027 - 默认空实现
        """服务启动时调用，可在此建立 MQTT 连接等。"""

    def prompt_sections(self, turn: TurnInfo) -> list[tuple[str, list[str]]]:
        """返回要追加到动态信息中的（标题, 若干行）。必须很快，不要做 IO。"""
        return []

    async def on_reply(self, turn: TurnInfo) -> None:  # noqa: B027
        """回复结束后调用（不在回复路径上）。"""

    async def shutdown(self) -> None:  # noqa: B027
        """服务停止时调用。"""


class ExtensionManager:
    def __init__(self, extensions: list[Extension] | None = None):
        self.extensions = extensions or []

    @classmethod
    def load(cls, specs: list[str]) -> ExtensionManager:
        loaded: list[Extension] = []
        for spec in specs:
            module_name, _, attr = spec.partition(":")
            try:
                module = importlib.import_module(module_name)
                ext_cls = getattr(module, attr)
                loaded.append(ext_cls())
                log.info("已加载扩展 %s", spec)
            except Exception:  # noqa: BLE001
                log.exception("加载扩展 %s 失败", spec)
        return cls(loaded)

    async def startup(self, services: Any) -> None:
        for ext in self.extensions:
            try:
                await ext.startup(services)
            except Exception:  # noqa: BLE001
                log.exception("扩展 %s 启动失败", ext.name)

    def prompt_sections(self, turn: TurnInfo) -> list[tuple[str, list[str]]]:
        out: list[tuple[str, list[str]]] = []
        for ext in self.extensions:
            try:
                out.extend(ext.prompt_sections(turn))
            except Exception:  # noqa: BLE001
                log.exception("扩展 %s 生成提示词失败", ext.name)
        return out

    async def on_reply(self, turn: TurnInfo) -> None:
        for ext in self.extensions:
            try:
                await ext.on_reply(turn)
            except Exception:  # noqa: BLE001
                log.exception("扩展 %s 处理回复失败", ext.name)

    async def shutdown(self) -> None:
        for ext in self.extensions:
            try:
                await ext.shutdown()
            except Exception:  # noqa: BLE001
                log.exception("扩展 %s 停止失败", ext.name)
