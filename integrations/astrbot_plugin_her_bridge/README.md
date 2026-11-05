# AstrBot 接入大脑代理

让 AstrBot（QQ 等文字聊天）使用“她”的统一人设与长期记忆。

## 1. 安装插件

最简单的方式是让配置助手自动复制（填 AstrBot 的安装目录）：

```bash
python -m her_brain setup --astrbot D:\her\AstrBot
```

也可以手动把本目录整个复制到 AstrBot 的插件目录，并重命名为 `astrbot_plugin_her_bridge`：

```
AstrBot/data/plugins/astrbot_plugin_her_bridge/
├── main.py
├── metadata.yaml
└── _conf_schema.json
```

**复制后必须重启 AstrBot**。在管理面板「插件」里应能看到 `astrbot_plugin_her_bridge`（为大脑代理附加身份与场合元数据），并且处于启用状态。收到第一条消息时，AstrBot 控制台会打印一行：

```
[her_bridge] 已为大脑代理附加身份：qq/private/你的QQ号（之后不再重复提示）
```

没看到这一行时，按顺序检查：

- AstrBot 启动日志里有没有这个插件的加载报错；
- 插件是否被禁用；
- 会话管理里是否限制了插件的启用范围；
- 插件配置 `enabled_provider_ids` 是否填了别的提供商。

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

## 插件未生效时

| 情况 | 效果 |
| --- | --- |
| 插件正常 | 认出是谁、私聊还是群聊；私聊可用私密记忆，群聊只用公开记忆 |
| 没有插件，但打开了 AstrBot 配置里的「用户识别」 | 能从 AstrBot 附加的 `User ID` 认出是谁，但分不清私聊和群聊。为了不在群里说漏私事，按保守模式处理：只读公开记忆，新写入一律标为私密 |
| 两者都没有 | 只能按 API Key 认出入口是 QQ，所有人都当作匿名访客：只读公开信息，不写入记忆 |

`python -m her_brain doctor` 会显示最近一条 QQ 消息的身份来自哪里。

AstrBot 默认开启的“附加当前时间”等提醒内容（`<system_reminder>`）不会被当作用户说的话写入日志或用于检索。
