# 枫云 AI 助手接入大脑代理

让枫云（桌面语音、声纹识别、Live2D/VRM）使用“她”的统一人设与长期记忆。

## 1. 软件设置

| 设置项 | 值 |
| --- | --- |
| 对话语言模型 | 自定义API-LLM |
| 自定义API-LLM base_url | `http://127.0.0.1:8000/v1` |
| 自定义API-LLM api_key | `sk-her-fengyun`（与 `config/brain.yaml` 中 `api_keys` 一致） |
| 自定义API-LLM model | `her` |
| AI助手提示词 | 留空（人设由大脑代理注入；枫云附加的 `/no_think` 会被代理自动去掉） |
| 记忆模式 | 短期记忆（枫云自己只保留近期对话，长期记忆交给代理） |
| 声纹识别 | 开启，并在「声纹管理」里录制你的声纹 |
| 语音识别 | SenseVoice（内置） |
| 流式语音合成 | 开启 |

只改设置、不打补丁也能用：代理按 API Key 认出这是语音入口。但拿不到声纹结果，会把说话人当作未确认身份的访客：不读私密记忆，也不写入记忆。所以建议打下面的补丁。

## 2. 打补丁（推荐）

`fengyun_her_brain.patch` 只改两处（基于社区版提交 `3fc3108`）：

1. `llm.py` 的 `chat_llm`（自定义 API 分支）：请求里附带身份元数据。
   ```python
   "metadata": {"channel": "voice", "scene": "voice", "external_user_id": "owner",
                "user_name": username, "speaker_verified": voiceprint_switch == "开启"}
   ```
   枫云开启声纹识别后，不是你的声音在识别阶段就被丢弃了（`asr.py` 的 `verify_speakers`），
   所以到达模型的语音都可以视为已确认。
2. `function.py` 的 `function_llm`（自定义 API 分支）：标记 `{"mode": "raw"}`。
   意图识别、绘画提示词生成等工具类调用原样透传，不注入人设、不记录、不产生记忆。

在枫云目录下执行：

```bash
git apply fengyun_her_brain.patch
```

如果不是 Git 仓库，可以用 `patch -p1 < fengyun_her_brain.patch`，或按上面两段手动修改。
枫云源码使用 CRLF 换行，补丁也按 CRLF 保存，不要用会改动换行的编辑器打开后再保存。

## 3. 关于语音延迟

社区版的 `chat_llm` 是**非流式**调用：要等整段回复生成完，才会切句、合成、播放。
所以从说完到开口 ≈ ASR + 整段回复的生成时间 + 第一句 TTS。在这个结构下，代理侧已经做了这些：

- 语音场景下 `max_tokens` 默认 200，人设要求语音回复只说一到三句短句；
- 动态信息放在静态人设之后，人设部分始终命中 llama.cpp 的前缀缓存；
- 记忆检索最多等 150ms，超时就只带核心档案。

想进一步逼近 1.5 秒（实施步骤第 6 步），需要把枫云改成流式：一边生成一边按句送 TTS。
`her_bridge.py` 提供了现成的流式切句工具：

```python
from her_bridge import her_metadata, stream_sentences

client = OpenAI(base_url=custom_llm_url, api_key=custom_llm_key)
for sentence in stream_sentences(client, custom_llm_model, messages, her_metadata(username, voiceprint_switch)):
    tts_queue.put(sentence)   # 由一个播放线程按顺序合成、播放
```

`get_tts_play` 每次调用都会新开一个播放线程，直接逐句调用会导致声音重叠，
所以需要改成“单个播放线程 + 队列”。这部分涉及枫云的界面与音频代码，需要在你的 Windows 机器上实测，本阶段没有直接改动。

## 4. 确认生效

对着麦克风说一句话，然后查看：

```bash
curl http://127.0.0.1:8000/admin/logs?limit=1
```

`channel` 应为 `voice`，`user_id` 为 `owner`。如果是 `anon:voice`，说明声纹识别没开，或者补丁没打上。
