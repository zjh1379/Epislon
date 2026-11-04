"""实施步骤 3/4：测量大脑代理增加的延迟。

同样的请求分别直接发给 llama-server 和发给大脑代理，比较首 token 时间；
同时读取代理返回的 X-Her-Prepare-Ms（身份 + 记忆 + 组装提示词耗时）与 X-Her-Retrieval-Ms。
验收：代理增加的延迟低于 50ms；检索耗时低于 150ms。

    python scripts/bench_proxy.py --proxy http://127.0.0.1:8000/v1 --llm http://127.0.0.1:8080/v1 --key sk-her-astrbot
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx

PROMPTS = ["今天好累啊", "我明天要去面试", "你还记得我喜欢吃什么吗", "晚安", "周末去哪玩好呢"]


def ttft(client: httpx.Client, body: dict) -> tuple[float, dict[str, str]]:
    t0 = time.perf_counter()
    with client.stream("POST", "/chat/completions", json=body) as r:
        r.raise_for_status()
        headers = dict(r.headers)
        for line in r.iter_lines():
            if line.startswith("data:") and line.strip() != "data: [DONE]":
                delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    return (time.perf_counter() - t0) * 1000, headers
    return (time.perf_counter() - t0) * 1000, headers


def main() -> None:
    ap = argparse.ArgumentParser(description="测量代理增加的延迟")
    ap.add_argument("--proxy", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--key", default="sk-no-key", help="代理的 API Key")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--channel", default="qq")
    ap.add_argument("--user", default="123456", help="外部用户 ID（用于检索该用户的记忆）")
    args = ap.parse_args()

    meta = {"channel": args.channel, "scene": "private", "external_user_id": args.user}
    proxy = httpx.Client(base_url=args.proxy.rstrip("/"), timeout=120, headers={"Authorization": f"Bearer {args.key}"})
    llm = httpx.Client(base_url=args.llm.rstrip("/"), timeout=120)
    direct, via, prepare, retrieval = [], [], [], []
    try:
        for i in range(args.rounds + 1):
            text = PROMPTS[i % len(PROMPTS)]
            body = {"model": "her", "stream": True, "max_tokens": 32,
                    "messages": [{"role": "user", "content": text}], "metadata": meta}
            t_via, headers = ttft(proxy, body)
            # 直连上游时使用与代理相同长度的系统提示，只比较“代理本身”的开销
            debug = proxy.post(args.proxy.rsplit("/v1", 1)[0] + "/admin/debug/prompt", json=body).json()
            t_direct, _ = ttft(llm, {"model": "local", "stream": True, "max_tokens": 32,
                                     "chat_template_kwargs": {"enable_thinking": False},
                                     "messages": debug["messages"]})
            if i == 0:
                continue
            via.append(t_via)
            direct.append(t_direct)
            prepare.append(float(headers.get("x-her-prepare-ms", "nan")))
            retrieval.append(float(headers.get("x-her-retrieval-ms", "nan")))
    finally:
        proxy.close()
        llm.close()

    overhead = statistics.median(v - d for v, d in zip(via, direct))
    result = {
        "direct_first_token_ms_p50": round(statistics.median(direct), 1),
        "proxy_first_token_ms_p50": round(statistics.median(via), 1),
        "proxy_overhead_ms_p50": round(overhead, 1),
        "prepare_ms_p50": round(statistics.median(prepare), 1),
        "retrieval_ms_p50": round(statistics.median(retrieval), 1),
        "retrieval_ms_max": round(max(retrieval), 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n验收：")
    print(f"  [{'通过' if result['prepare_ms_p50'] < 50 else '未通过'}] 代理准备耗时 < 50ms：{result['prepare_ms_p50']} ms")
    print(f"  [{'通过' if result['retrieval_ms_max'] < 150 else '未通过'}] 检索耗时 < 150ms：最大 {result['retrieval_ms_max']} ms")


if __name__ == "__main__":
    main()
