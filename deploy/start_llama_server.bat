@echo off
chcp 65001 >nul
REM 对话模型：llama-server（RTX 5070 Ti 16GB）
REM -np 2 -c 16384：两个槽位各 8K 上下文。0 号给对话，1 号给后台提取（见 brain.yaml 的 slot_id），
REM                 后台任务不会冲掉对话的前缀缓存。
REM -fa on + q8_0 KV 缓存：省显存，给 TTS 与嵌入模型留空间。
REM --reasoning-budget 0：关闭思考模式（陪伴聊天不需要，否则首字会被拖慢好几秒）。
REM --cache-reuse 256：复用前缀缓存。
REM 显存不够时：换 8B 级模型，或把 -c 降到 12288。

set MODEL=models\Qwen3-14B-Q4_K_M.gguf
set LLAMA=llama-server.exe

%LLAMA% -m %MODEL% -ngl 99 -c 16384 -np 2 -fa on -ctk q8_0 -ctv q8_0 ^
  --jinja --reasoning-budget 0 --cache-reuse 256 ^
  --host 127.0.0.1 --port 8080 --metrics
