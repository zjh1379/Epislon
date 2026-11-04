#!/usr/bin/env bash
# Linux/WSL 下一键启动：对话模型、嵌入模型、Qdrant（可选）、大脑代理。
# 各进程的日志写到 logs/ 目录。Windows 请分别运行 deploy/*.bat。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CHAT_MODEL=${CHAT_MODEL:-models/Qwen3-14B-Q4_K_M.gguf}
EMBED_MODEL=${EMBED_MODEL:-models/bge-m3-Q8_0.gguf}
LLAMA=${LLAMA:-llama-server}

$LLAMA -m "$CHAT_MODEL" -ngl 99 -c 16384 -np 2 -fa on -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning-budget 0 --cache-reuse 256 --host 127.0.0.1 --port 8080 --metrics \
  > logs/llama-chat.log 2>&1 &

$LLAMA -m "$EMBED_MODEL" --embedding --pooling cls -ngl 99 -c 2048 -ub 2048 \
  --host 127.0.0.1 --port 8081 > logs/llama-embed.log 2>&1 &

if command -v docker >/dev/null 2>&1; then
  docker compose -f deploy/docker-compose.qdrant.yml up -d
fi

# 等待模型加载
for port in 8080 8081; do
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    sleep 1
  done
done

exec python -m her_brain serve
