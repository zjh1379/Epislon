"""实施步骤 5：记忆回归测试。

准备 30~50 个“她应该记得什么”的问题（见 regression/cases.example.yaml），
每次升级模型、改提示词或调整记忆参数后跑一遍，看通过率是否稳定。

两种模式：
- context（默认，快且确定）：调用 /admin/debug/prompt，只检查组装出的提示词里有没有该出现/不该出现的内容。
  不调用模型，适合检验“记得对、不串场合”。
- reply：真正发对话请求，检查她的回复。受模型随机性影响，适合定期抽查。

    python scripts/memory_regression.py regression/cases.yaml --base http://127.0.0.1:8000 --mode context

注意：reply 模式会产生真实对话日志。建议在一份恢复出来的副本数据上运行（见 README“回归测试”一节）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml


def check(text: str, case: dict[str, Any], prefix: str) -> list[str]:
    problems = []
    for word in case.get(f"{prefix}_all", []) or []:
        if word not in text:
            problems.append(f"缺少「{word}」")
    any_words = case.get(f"{prefix}_any", []) or []
    if any_words and not any(w in text for w in any_words):
        problems.append(f"应至少出现其一：{any_words}")
    for word in case.get(f"forbid_{prefix}", []) or []:
        if word in text:
            problems.append(f"不应出现「{word}」")
    return problems


def run_case(client: httpx.Client, case: dict[str, Any], defaults: dict[str, Any], mode: str, key: str | None) -> list[str]:
    metadata = {**defaults.get("metadata", {}), **case.get("metadata", {})}
    body = {"model": "her", "messages": [{"role": "user", "content": case["ask"]}], "metadata": metadata}
    if mode == "context":
        r = client.post("/admin/debug/prompt", json=body)
        r.raise_for_status()
        messages = r.json()["messages"]
        text = "\n".join(m["content"] for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str))
        return check(text, case, "context")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = client.post("/v1/chat/completions", json=body, headers=headers)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"] or ""
    return check(reply, case, "reply")


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆回归测试")
    ap.add_argument("cases", type=Path)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=["context", "reply"], default="context")
    ap.add_argument("--key", help="reply 模式使用的 API Key")
    ap.add_argument("--admin-token", default="")
    args = ap.parse_args()

    spec = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    defaults = spec.get("defaults", {})
    cases = spec.get("cases", [])
    headers = {"X-Admin-Token": args.admin_token} if args.admin_token else {}
    passed = 0
    report = []
    with httpx.Client(base_url=args.base.rstrip("/"), timeout=120, headers=headers) as client:
        for case in cases:
            key = "context" if args.mode == "context" else "reply"
            relevant = any(case.get(k) for k in (f"{key}_all", f"{key}_any", f"forbid_{key}"))
            if not relevant:
                continue
            try:
                problems = run_case(client, case, defaults, args.mode, args.key)
            except httpx.HTTPError as exc:
                problems = [f"请求失败：{exc}"]
            ok = not problems
            passed += ok
            report.append({"id": case.get("id"), "ok": ok, "problems": problems})
            print(f"[{'通过' if ok else '失败'}] {case.get('id')}: {case['ask']}" + ("" if ok else f"  → {'；'.join(problems)}"))
    total = len(report)
    rate = passed / total if total else 0
    print(f"\n通过 {passed}/{total}（{rate:.0%}）")
    out = args.cases.with_suffix(f".{args.mode}.result.json")
    out.write_text(json.dumps({"passed": passed, "total": total, "cases": report}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
