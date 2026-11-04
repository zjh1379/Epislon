"""配置加载。

配置来自一个 YAML 文件（默认 ``config/brain.yaml``），可以用 ``${ENV_NAME}`` 或
``${ENV_NAME:-默认值}`` 引用环境变量。所有相对路径都相对于启动进程时的工作目录（项目根目录）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

Scene = Literal["private", "group", "voice"]


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    # /v1/models 中暴露给入口的模型名
    model_name: str = "her"
    # 管理接口令牌；留空时只允许本机访问管理接口
    admin_token: str = ""
    log_level: str = "INFO"
    # 配置了 api_keys 时，是否仍允许未登记的 Key 访问
    allow_unknown_keys: bool = False


class ApiKeyConfig(BaseModel):
    """不同入口使用不同的 API Key，作为无法发送自定义字段时的退路。"""

    key: str
    channel: str
    scene: Scene | None = None
    name: str = ""
    # chat：注入人设与记忆；raw：原样透传，不注入、不记录
    mode: Literal["chat", "raw"] = "chat"


class LLMEndpoint(BaseModel):
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "sk-no-key"
    model: str = "local"
    timeout: float = 120.0
    connect_timeout: float = 3.0
    # llama-server 的 id_slot：把对话与后台任务固定在不同槽位，避免互相冲掉前缀缓存
    slot_id: int | None = None
    # 通过 chat_template_kwargs 关闭 Qwen3 等模型的思考模式
    disable_thinking: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)


class UpstreamConfig(LLMEndpoint):
    # 入口未指定时使用的生成参数
    default_params: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.7})
    # 按场景覆盖的生成参数，例如 {"voice": {"max_tokens": 200}}
    scene_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # 过滤 <think>…</think> 与 reasoning_content，避免被 TTS 念出来
    strip_think: bool = True
    # 启动、重载人设、后台任务结束后预热静态前缀，保证前缀缓存命中
    warmup: bool = True


class WorkerLLMConfig(BaseModel):
    """后台提取用的模型。未填写的字段沿用 upstream。"""

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float = 180.0
    slot_id: int | None = None
    temperature: float = 0.1
    max_tokens: int = 1024
    # json_schema：llama-server 用语法约束输出；json_object：只要求 JSON；none：只靠提示词
    json_mode: Literal["json_schema", "json_object", "none"] = "json_schema"
    extra_body: dict[str, Any] = Field(default_factory=dict)

    def resolve(self, upstream: UpstreamConfig) -> LLMEndpoint:
        return LLMEndpoint(
            base_url=self.base_url or upstream.base_url,
            api_key=self.api_key or upstream.api_key,
            model=self.model or upstream.model,
            timeout=self.timeout,
            connect_timeout=upstream.connect_timeout,
            slot_id=self.slot_id,
            disable_thinking=upstream.disable_thinking,
            extra_body=self.extra_body,
        )


class EmbeddingConfig(BaseModel):
    """OpenAI 兼容的嵌入接口（例如以 --embedding 启动的 llama-server）。"""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8081/v1"
    api_key: str = "sk-no-key"
    model: str = "embedding"
    dims: int = 1024
    timeout: float = 2.0


class ProfileConfig(BaseModel):
    # 同一事实被独立确认几次后进入核心档案（明确说“记住”的立即进入）
    confirmations_to_promote: int = 2
    # 判断两条事实/取值是否相同的相似度阈值（字符二元组 Dice 系数）
    match_threshold: float = 0.55
    max_render_entries: int = 40
    # 计划类条目在日期过后多少天不再注入
    plan_expire_days: int = 3
    # 只出现过一次、长期未再提起的候选在多少天后过期
    candidate_expire_days: int = 90


class RecentConfig(BaseModel):
    window_hours: float = 12
    max_items: int = 4
    max_chars: int = 60


class MemoryConfig(BaseModel):
    # mem0：Mem0 + Qdrant（推荐）；lite：纯 SQLite 的轻量检索（开发、测试或无 GPU 嵌入时使用）
    engine: Literal["lite", "mem0"] = "lite"
    retrieval_timeout_ms: int = 150
    top_k: int = 5
    # 检索分数下限；留空时按引擎取默认值
    min_score: float | None = None
    # 少于这么多字的消息（如“嗯”“好”）不做语义检索
    min_query_chars: int = 2
    # 新事实与已有记忆相似度超过该值时视为同一条，只累计提及次数
    dedupe_threshold: float | None = None
    # 时效记忆：日期落在 [今天-back, 今天+ahead] 内的计划/事件，无论相似度都会带上
    agenda_days_back: int = 2
    agenda_days_ahead: int = 2
    agenda_max: int = 3
    remember_guests: bool = True
    # 声纹未确认的语音是否仍按主人处理（默认否：按匿名访客处理，不读私密记忆）
    trust_unverified_voice: bool = False
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    recent: RecentConfig = Field(default_factory=RecentConfig)
    # 深度合并进自动生成的 Mem0 配置，可覆盖 vector_store / embedder / llm 等
    mem0: dict[str, Any] = Field(default_factory=dict)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_path: str | None = None
    collection: str = "her_memory"


class PromptConfig(BaseModel):
    # system：动态信息拼在系统消息里（文档默认顺序）
    # before_last_user：系统消息只放静态人设，动态信息放在最后一条用户消息之前，
    #                   这样历史对话也能命中前缀缓存（需模板支持多条 system 消息，Qwen 系列可以）
    dynamic_position: Literal["system", "before_last_user"] = "system"
    # 入口自带的系统提示词：drop 丢弃；append 放进动态部分的“入口附加说明”
    entry_system_prompt: Literal["drop", "append"] = "append"
    max_profile_chars: int = 800
    max_memory_chars: int = 900
    timezone: str = "Asia/Shanghai"
    # 只保留入口传来的最近 N 条消息；0 表示全部保留
    keep_last_messages: int = 0


class WorkerConfig(BaseModel):
    enabled: bool = True
    # 距离最后一次对话多少秒后才开始后台任务
    idle_seconds: float = 90
    poll_seconds: float = 5
    batch_size: int = 8
    max_attempts: int = 3
    tidy_interval_hours: float = 6


class BackupConfig(BaseModel):
    enabled: bool = True
    dir: str = "backups"
    daily_at: str = "04:30"
    keep_days: int = 14
    qdrant_snapshot: bool = True


class UserConfig(BaseModel):
    user_id: str
    display_name: str = ""
    role: Literal["owner", "friend", "guest"] = "friend"
    # 形如 "qq:123456"、"voice:owner"
    bindings: list[str] = Field(default_factory=list)


DEFAULT_PLATFORM_ALIASES = {
    "aiocqhttp": "qq",
    "qq_official": "qq",
    "qqofficial": "qq",
    "qq_official_webhook": "qq",
    "qqofficial_webhook": "qq",
    "fengyun": "voice",
}


class Settings(BaseModel):
    data_dir: str = "data"
    persona_path: str = "persona/her.yaml"
    self_user_id: str = "_self"
    server: ServerConfig = Field(default_factory=ServerConfig)
    api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    platform_aliases: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_PLATFORM_ALIASES))
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    worker_llm: WorkerLLMConfig = Field(default_factory=WorkerLLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    users: list[UserConfig] = Field(default_factory=list)
    # 扩展，形如 "her_brain.extensions.duck:DuckActionStub"
    extensions: list[str] = Field(default_factory=list)
    # 加载时记录配置文件路径，备份时一并保存
    config_path: str | None = None

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_path / "brain.db"

    @property
    def worker_endpoint(self) -> LLMEndpoint:
        return self.worker_llm.resolve(self.upstream)


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def default_config_path() -> Path:
    return Path(os.environ.get("HER_BRAIN_CONFIG", "config/brain.yaml"))


def load_settings(path: str | Path | None = None) -> Settings:
    cfg_path = Path(path) if path else default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {cfg_path}。可以先复制 config/brain.example.yaml 为 config/brain.yaml，"
            "或运行 `python -m her_brain init`。"
        )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = _expand_env(raw)
    raw["config_path"] = str(cfg_path)
    aliases = dict(DEFAULT_PLATFORM_ALIASES)
    aliases.update(raw.get("platform_aliases") or {})
    raw["platform_aliases"] = aliases
    return Settings.model_validate(raw)
