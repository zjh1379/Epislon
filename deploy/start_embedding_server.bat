@echo off
chcp 65001 >nul
REM 嵌入模型：llama-server --embedding（Mem0 情节记忆检索用）
REM bge-m3：中文效果好，1024 维，--pooling cls
REM 换成 Qwen3-Embedding-0.6B 时：--pooling last，维度同样是 1024
REM 修改模型后记得同步 brain.yaml 中 embedding.model / embedding.dims，并执行 python -m her_brain reindex

set MODEL=models\bge-m3-Q8_0.gguf
set LLAMA=llama-server.exe

%LLAMA% -m %MODEL% --embedding --pooling cls -ngl 99 -c 2048 -ub 2048 ^
  --host 127.0.0.1 --port 8081
