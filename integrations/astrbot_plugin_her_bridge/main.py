"""AstrBot 插件：在请求发往大脑代理前，附上“谁在什么场合说话”。

AstrBot 的 ProviderRequest 没有可以按请求附加自定义字段的位置，因此把元数据写成
``<her-meta>{...}</her-meta>`` 标记追加到系统提示词末尾。大脑代理读取后会把标记从消息中删除，
不会传给模型；系统提示词也不会被写进 AstrBot 的对话历史。
"""

from __future__ import annotations

import json

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

DEFAULT_ALIASES = {
    "aiocqhttp": "qq",
    "qq_official": "qq",
    "qq_official_webhook": "qq",
    "qqofficial": "qq",
    "qqofficial_webhook": "qq",
}


@register("astrbot_plugin_her_bridge", "Epislon", "为大脑代理附加身份与场合元数据", "0.1.1")
class HerBridge(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._announced = False
        self.aliases = dict(DEFAULT_ALIASES)
        raw = self.config.get("platform_aliases") or ""
        if raw:
            try:
                self.aliases.update(json.loads(raw))
            except ValueError:
                logger.warning("[her_bridge] platform_aliases 不是合法 JSON，已忽略：%s", raw)

    def _provider_enabled(self, event: AstrMessageEvent) -> bool:
        allowed = [p for p in (self.config.get("enabled_provider_ids") or []) if p]
        if not allowed:
            return True
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            return provider is not None and provider.meta().id in allowed
        except Exception as exc:  # noqa: BLE001
            logger.debug("[her_bridge] 获取当前提供商失败：%s", exc)
            return False

    @filter.on_llm_request()
    async def attach_meta(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        try:
            if not self._provider_enabled(event):
                logger.debug("[her_bridge] 当前提供商不在 enabled_provider_ids 中，跳过")
                return
            meta = self._build_meta(event)
        except Exception as exc:  # noqa: BLE001 - 插件出错不能影响 AstrBot 正常回复
            logger.warning(f"[her_bridge] 生成身份元数据失败：{exc}")
            return
        marker = "<her-meta>" + json.dumps(meta, ensure_ascii=False) + "</her-meta>"
        req.system_prompt = f"{req.system_prompt}\n{marker}" if req.system_prompt else marker
        if not self._announced:
            self._announced = True
            logger.info(
                f"[her_bridge] 已为大脑代理附加身份：{meta['channel']}/{meta['scene']}/{meta['external_user_id']}"
                "（之后不再重复提示）"
            )

    def _build_meta(self, event: AstrMessageEvent) -> dict:
        platform = event.get_platform_name()
        group_id = event.get_group_id() or ""
        meta = {
            "platform": platform,
            "channel": self.aliases.get(platform, platform),
            "scene": "private" if event.is_private_chat() else "group",
            "external_user_id": str(event.get_sender_id()),
            "user_name": event.get_sender_name() or "",
        }
        if group_id:
            meta["group_id"] = str(group_id)
            if self.config.get("send_group_name", True):
                group = getattr(event.message_obj, "group", None)
                name = getattr(group, "group_name", None) if group else None
                if name:
                    meta["group_name"] = name
        return meta
