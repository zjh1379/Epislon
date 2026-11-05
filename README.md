# 她的大脑代理（第一阶段）

让 AstrBot（QQ 等文字聊天）和枫云 AI 助手（桌面语音）共用同一个“她”：同一份人设、同一套长期记忆，并且开口足够快。全部在本机运行。

```
 QQ ── AstrBot ──(插件附带身份)──┐
                                ├─► 大脑代理 :8000 ──► llama-server :8080（对话）
 麦克风 ── 枫云 AI ──(补丁附带声纹)┘   │  OpenAI 兼容      llama-server :8081（嵌入）
                                      │                  Qdrant :6333（经 Mem0）
                                      ├─ 读路径：人设 + 核心档案 + 时效/相关记忆 + 跨入口近期动态（≤150ms）
                                      └─ 写路径：回复后只落库原文，空闲时后台提取记忆
```

- 架构文档：[docs/architecture.md](docs/architecture.md)
- 详细设计（含与架构文档的差异说明）：[docs/design.md](docs/design.md)
- 入口接入：[AstrBot](integrations/astrbot_plugin_her_bridge/README.md) · [枫云 AI](integrations/fengyun/README.md)

## 记忆怎样保持稳定

| 层 | 内容 | 读 | 写 |
| --- | --- | --- | --- |
| 核心档案 | 最重要、最稳定的事实（名字、过敏、宠物……） | 每次注入（内存缓存） | 同一事实被独立提到 2 次，或明确说“记住”才进入；每次改写都留版本 |
| 情节记忆 | 零散经历、计划、喜好 | 语义检索（150ms 超时）+ 临近日期的计划/事件 | 空闲时由后台提取；每次变更都有前后快照，可撤销、可回滚到任意时间点 |
| 原始日志 | 每轮对话原文 | 不读 | 回复结束后落库；任何一层出错都能从这里重建 |

私聊与语音标为私密，群聊标为公开；在群里、或声纹未确认时，私密内容不会进入提示词。

## 快速体验（不需要显卡）

```bash
pip install -e ".[dev]"
cp config/brain.example.yaml config/brain.yaml
```

把 `config/brain.yaml` 里的 `memory.engine` 改为 `lite`、`embedding.enabled` 改为 `false`，然后分别在两个终端运行：

```bash
python -m her_brain.devtools.fake_llm --port 8080
```

```bash
python -m her_brain serve
```

发一句话试试（QQ 号 123456 在示例配置里绑定为主人）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer sk-her-astrbot" -H "content-type: application/json" -d '{"model": "her", "messages": [{"role": "user", "content": "记住，我对花生过敏"}], "metadata": {"channel": "qq", "scene": "private", "external_user_id": "123456"}}'
```

90 秒不说话后，后台会提取记忆。也可以立即触发：

```bash
curl -X POST http://127.0.0.1:8000/admin/worker/run
```

```bash
curl http://127.0.0.1:8000/admin/users/owner/profile
```

## 在你的电脑上部署（Windows，RTX 5070 Ti）

### 需要下载的东西

| 内容 | 来源 | 放到 |
| --- | --- | --- |
| llama.cpp | `ggml-org/llama.cpp` Releases：`llama-b?????-bin-win-cuda-13.4-x64.zip` 和 `cudart-llama-bin-win-cuda-13.4-x64.zip`（50 系显卡要选 CUDA 13 的版本） | 两个 zip 解压到**同一个**文件夹，文件夹名以 `llama` 开头，放在项目目录或它的上级目录 |
| 对话模型 | Hugging Face `Qwen/Qwen3-14B-GGUF` → `Qwen3-14B-Q4_K_M.gguf`（约 9 GB） | 项目的 `models\` |
| 嵌入模型 | Hugging Face `Qwen/Qwen3-Embedding-0.6B-GGUF` → `Qwen3-Embedding-0.6B-Q8_0.gguf`（约 0.6 GB） | 项目的 `models\` |

```
D:\her\
├─ llama.cpp\                     llama-server.exe 和 cudart 的 dll
└─ Epislon\
   ├─ models\                     两个 .gguf
   └─ deploy\setup_windows.bat    ……
```

不需要 Docker 和 Qdrant 服务：默认使用 Qdrant 本地嵌入模式（`data\qdrant`）。

### 三步完成

1. **安装**：双击 `deploy\setup_windows.bat`。它会创建虚拟环境、安装依赖、生成配置，然后问你 QQ 号、AstrBot 安装目录（会自动复制插件）和 AstrBot 是否在 Docker 里，最后做一次环境自检。
2. **启动**：双击 `deploy\up.bat`。它会自动找到 `models\` 里的模型和 llama-server，依次启动对话模型、嵌入模型和大脑代理；按 Ctrl+C 一并关闭。已有代理在运行时会先把它停掉，所以 `git pull` 之后直接再双击一次即可。同一时间只能运行一个代理（Qdrant 本地嵌入模式是独占的）。
3. **自测**：另开一个窗口双击 `deploy\selftest.bat`。它在临时目录里用你的真实模型把整个流程跑一遍，不会写入正式记忆：

   ```
   [通过] 上游连通 / 嵌入服务 / 原始日志与私密度
   [通过] 后台提取：……小明在 2026-11-04 要去面试……；小明对花生过敏
   [通过] 时效记忆（面试） / 核心档案（明确要求记住） / 核心档案（未确认的事实不进入）
   [通过] 跨入口（QQ → 语音） / 场合过滤（群聊） / 场合过滤（未确认身份的语音）
   [通过] 语音回复：「欢迎回来！面试怎么样？」
   [通过] 代理准备耗时 / 记忆检索耗时 / 首字时间
   ```

然后在 AstrBot 里新增“OpenAI 兼容”提供商（`setup` 最后会打印要填的地址、Key 和模型名），给机器人发条私聊，再运行一次自检。看到“AstrBot 接入：已收到 QQ 消息……owner”就说明接通了：

```bash
python -m her_brain doctor
```

| 命令 | 作用 |
| --- | --- |
| `python -m her_brain setup` | 配置助手（可重复运行；`--qq`、`--astrbot`、`--docker`、`--engine`、`--yes`） |
| `python -m her_brain bind <ID>` | 把 QQ 号或 QQ 官方机器人的 openid 绑定到主人（之前以访客身份的对话会迁移并重新提取） |
| `python -m her_brain doctor` | 环境自检：依赖、配置、模型、llama-server、显卡、各服务、嵌入维度、AstrBot 是否接入 |
| `python -m her_brain up` | 一键启动（`--no-llm` / `--no-embed` 跳过某个模型服务） |
| `python -m her_brain stop` | 停止正在运行的代理及由 `up` 启动的模型服务（`git pull` 后重启前先运行；`up.bat` 会自动执行） |
| `python -m her_brain selftest` | 端到端自测（`--keep` 保留临时数据以便排查） |

想手动控制 llama-server 参数时，仍可用 `deploy\start_llama_server.bat` 和 `deploy\start_embedding_server.bat`，或在配置的 `launcher` 段里修改。对话服务开两个槽位（`-np 2`），0 号给对话、1 号给后台提取，互不冲掉前缀缓存。

### 其他测量

1. **测速基线**（实施步骤 1）：

   ```bash
   python scripts/bench_baseline.py --llm http://127.0.0.1:8080/v1 --embed http://127.0.0.1:8081/v1 --tts "http://127.0.0.1:9880/tts?text={text}&text_lang=zh&ref_audio_path=ref.wav&prompt_lang=zh&prompt_text=参考文本&streaming_mode=true"
   ```

2. **接入入口**：按 [AstrBot](integrations/astrbot_plugin_her_bridge/README.md) 和 [枫云](integrations/fengyun/README.md) 的说明操作。
3. **代理开销**（实施步骤 3/4）：

   ```bash
   python scripts/bench_proxy.py --key sk-her-astrbot --user 你的QQ号
   ```

## 日常维护

改了人设：

```bash
curl -X POST http://127.0.0.1:8000/admin/persona/reload
```

她记错了一件事，先找到那条记忆和它的变更事件：

```bash
curl "http://127.0.0.1:8000/admin/users/owner/memories?q=过敏"
```

```bash
curl "http://127.0.0.1:8000/admin/memory-events?user_id=owner&limit=20"
```

撤销某次变更，或把她的记忆整体恢复到某个时间点：

```bash
curl -X POST http://127.0.0.1:8000/admin/memory-events/42/revert
```

```bash
curl -X POST http://127.0.0.1:8000/admin/users/owner/memories/rollback -H "content-type: application/json" -d '{"since": "2026-11-03T00:00:00+08:00"}'
```

核心档案回滚到某个版本：

```bash
curl -X POST http://127.0.0.1:8000/admin/users/owner/profile/rollback -H "content-type: application/json" -d '{"version": 3}'
```

整个数据回滚到前一天的快照（先停止代理）：

```bash
python -m her_brain restore --yesterday
```

排查“她为什么这么说”——按真实请求组装提示词但不调用模型：

```bash
curl http://127.0.0.1:8000/admin/debug/prompt -H "content-type: application/json" -d '{"messages": [{"role": "user", "content": "我回来了"}], "metadata": {"channel": "voice", "speaker_verified": true}}'
```

完整接口列表见 [docs/design.md 第 12 节](docs/design.md#12-管理接口)。未设置 `server.admin_token` 时，管理接口只允许本机访问。

## 回归测试

复制 `regression/cases.example.yaml` 为 `regression/cases.yaml`，按你们的真实对话写 30~50 个问题，每次升级模型或改提示词后运行：

```bash
python scripts/memory_regression.py regression/cases.yaml
```

默认的 `context` 模式只检查组装出的提示词，不调用模型，结果稳定。`--mode reply` 会真正发起对话并检查回复，会产生对话日志，建议在一份恢复出来的副本数据上运行：复制配置，把 `data_dir` 指向副本目录，换个端口启动。

## 开发

```bash
pip install -e ".[dev,mem0]"
```

```bash
python -m pytest
```

测试使用假的 LLM 服务（`her_brain/devtools/fake_llm.py`），覆盖元数据解析、身份与场合、提示词顺序、流式转发、思考过滤、档案晋升与回滚、情节记忆撤销与回滚、后台任务打断、备份恢复、图 6 跨入口场景，以及真实 Mem0 2.x 的适配（未安装 mem0ai 时跳过）。

## 许可提醒

枫云 AI 助手社区版为 GPL-3.0 且禁止商用，`integrations/fengyun/fengyun_her_brain.patch` 是针对它的补丁；Mem0、Qdrant 为宽松许可。大脑代理作为独立服务运行，便于以后替换入口。
