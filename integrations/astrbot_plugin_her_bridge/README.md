# AstrBot 接入大脑代理

让 AstrBot（QQ 等文字聊天）使用“她”的统一人设与长期记忆。

## 1. 安装插件

把本目录整个复制到 AstrBot 的插件目录，并重命名为 `astrbot_plugin_her_bridge`：

```
AstrBot/data/plugins/astrbot_plugin_her_bridge/
├── main.py
├── metadata.yaml
└── _conf_schema.json
```

重启 AstrBot，在管理面板「插件」里能看到“为大脑代理附加身份与场合元数据”。

插件在每次请求模型前，把下面这段标记追加到系统提示词末尾：

```
<her-meta>{"platform": "aiocqhttp", "channel": "qq", "scene": "group", "external_user_id": "123456", "user_name": "小明", "group_id": "7890"}</her-meta>
```

大脑代理读取后会把它从消息里删除，模型看不到。系统提示词不写进 AstrBot 的对话历史，标记也就不会在历史里越积越多。

## 2. 新建提供商

管理面板 → 服务提供商 → 新增 → **OpenAI 兼容**：

| 字段 | 值 |
| --- | --- |
| API Base URL | `http://127.0.0.1:8000/v1`（大脑代理地址） |
| API Key | `sk-her-astrbot`（与 `config/brain.yaml` 中 `api_keys` 一致） |
| 模型 | `her` |
| 流式输出 | 开启（文字首字更快） |

记下这个提供商的 ID，填到插件配置 `enabled_provider_ids` 中。这样即使你同时还接了云端模型，QQ 号等信息也只会发给本地的大脑代理。

## 3. 其他设置

| 设置 | 建议 | 原因 |
| --- | --- | --- |
| 人格（Persona） | 留空或使用默认空人格 | 人设由大脑代理统一注入 |
| 长期记忆插件（LivingMemory 等） | 不启用 | 记忆由大脑代理统一管理，跨入口共享 |
| 对话上下文条数 | 保留少量（例如最近 10 轮） | 近期对话由入口提供，长期信息靠记忆 |
| 群聊上下文感知 | 可开启 | 群里其他人的发言会作为上下文传给模型 |

## 4. 确认生效

在 QQ 私聊里说一句话，然后查看大脑代理的原始日志：

```bash
curl http://127.0.0.1:8000/admin/logs?limit=1
```

`channel` 应为 `qq`，`scene` 为 `private`，`user_id` 为你在配置文件里绑定的 `owner`。如果 `user_id` 是 `qq:123456` 这样的访客 ID，说明 QQ 号还没绑定，用管理接口绑定即可（`move_logs` 会把之前的日志一并迁移并重新提取）：

```bash
curl -X POST http://127.0.0.1:8000/admin/identities -H 'content-type: application/json' -d '{"platform": "qq", "external_id": "123456", "user_id": "owner", "move_logs": true}'
```

## 无法安装插件时

不装插件也能用：提供商的 API Key 填 `sk-her-astrbot`，代理会按 Key 认出入口是 QQ。但拿不到发送者，所有人都会被当作匿名访客：只读公开信息，也不写入记忆。所以建议安装插件。
