"""实施步骤 1：测速基线。

分别测量：llama-server 首 token 时间与解码速度、嵌入接口延迟、TTS 首包时间、显存占用。
验收：三者同时常驻不爆显存；首 token 低于 500ms。

用法示例：
    python scripts/bench_baseline.py --llm http://127.0.0.1:8080/v1 \
        --embed http://127.0.0.1:8081/v1 \
        --tts "http://127.0.0.1:9880/tts?text={text}&text_lang=zh&ref_audio_path=ref.wav&prompt_lang=zh&prompt_text=参考文本&streaming_mode=true"

--tts 是一个 URL 模板，{text} 会被替换为测试句子（GPT-SoVITS api_v2 的写法如上）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SYSTEM = "你是一个温柔的虚拟伙伴，说话口语化，回复简短。" * 20  # 约 600 字，模拟人设前缀
PROMPTS = ["今天好累啊", "你觉得我明天穿什么好", "给我讲个笑话吧", "晚饭吃什么呢", "我有点想家了"]


def bench_llm(base: str, model: str, rounds: int, max_tokens: int) -> dict:
    ttfts, speeds = [], []
    with httpx.Client(base_url=base.rstrip("/"), timeout=120) as client:
        for i in range(rounds + 1):  # 第一轮用于预热前缀缓存，不计入
            body = {
                "model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            }
            t0 = time.perf_counter()
            first = None
            n_chunks = 0
            with client.stream("POST", "/chat/completions", json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        n_chunks += 1
                        if first is None:
                            first = time.perf_counter()
            t1 = time.perf_counter()
            if i == 0 or first is None:
                continue
            ttfts.append((first - t0) * 1000)
            if n_chunks > 1 and t1 > first:
                speeds.append((n_chunks - 1) / (t1 - first))
    return {
        "first_token_ms_p50": round(statistics.median(ttfts), 1) if ttfts else None,
        "first_token_ms_max": round(max(ttfts), 1) if ttfts else None,
        "decode_tokens_per_s": round(statistics.median(speeds), 1) if speeds else None,
    }


def bench_embed(base: str, model: str, rounds: int) -> dict:
    times = []
    with httpx.Client(base_url=base.rstrip("/"), timeout=30) as client:
        for i in range(rounds + 1):
            t0 = time.perf_counter()
            r = client.post("/embeddings", json={"model": model, "input": [PROMPTS[i % len(PROMPTS)]]})
            r.raise_for_status()
            if i:
                times.append((time.perf_counter() - t0) * 1000)
            dims = len(r.json()["data"][0]["embedding"])
    return {"embed_ms_p50": round(statistics.median(times), 1), "dims": dims}


def bench_tts(template: str, rounds: int) -> dict:
    times = []
    text = "你好呀，今天过得怎么样？"
    with httpx.Client(timeout=60) as client:
        for i in range(rounds + 1):
            url = template.replace("{text}", urllib.parse.quote(text))
            t0 = time.perf_counter()
            with client.stream("GET", url) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes():
                    if chunk:
                        break
            if i:
                times.append((time.perf_counter() - t0) * 1000)
    return {"tts_first_packet_ms_p50": round(statistics.median(times), 1)}


def gpu_memory() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"gpu": "未找到 nvidia-smi"}
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    if not out:
        return {"gpu": "nvidia-smi 无输出"}
    name, used, total = [x.strip() for x in out[0].split(",")]
    return {"gpu": name, "vram_used_mb": int(used), "vram_total_mb": int(total)}


def main() -> None:
    ap = argparse.ArgumentParser(description="测速基线（实施步骤 1）")
    ap.add_argument("--llm", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--embed", help="嵌入接口地址，例如 http://127.0.0.1:8081/v1")
    ap.add_argument("--embed-model", default="embedding")
    ap.add_argument("--tts", help="TTS URL 模板，{text} 为测试句子")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    result: dict = {}
    result.update(bench_llm(args.llm, args.model, args.rounds, args.max_tokens))
    if args.embed:
        result.update(bench_embed(args.embed, args.embed_model, args.rounds))
    if args.tts:
        result.update(bench_tts(args.tts, args.rounds))
    result.update(gpu_memory())
    print(json.dumps(result, ensure_ascii=False, indent=2))

    checks = []
    ttft = result.get("first_token_ms_p50")
    checks.append(("首 token < 500ms", ttft is not None and ttft < 500, f"{ttft} ms"))
    if "vram_used_mb" in result:
        headroom = result["vram_total_mb"] - result["vram_used_mb"]
        checks.append(("显存余量 ≥ 500MB（三者同时常驻）", headroom >= 500, f"剩余 {headroom} MB"))
    print("\n验收：")
    for name, ok, detail in checks:
        print(f"  [{'通过' if ok else '未通过'}] {name}：{detail}")


if __name__ == "__main__":
    main()
