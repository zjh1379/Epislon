"""流式过滤 <think>…</think>。标签可能被拆在多个 token 里，这里用状态机处理。"""

from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"


def _partial_suffix(text: str, tag: str) -> int:
    """text 末尾与 tag 前缀重合的最长长度（可能是被截断的标签）。"""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class ThinkStripper:
    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._emitted_any = False
        self._strip_leading = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_think:
                idx = self._buf.find(CLOSE)
                if idx < 0:
                    keep = _partial_suffix(self._buf, CLOSE)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[idx + len(CLOSE) :]
                self._in_think = False
                self._strip_leading = True
                continue
            idx = self._buf.find(OPEN)
            if idx >= 0:
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(OPEN) :]
                self._in_think = True
                continue
            keep = _partial_suffix(self._buf, OPEN)
            if keep:
                out.append(self._buf[:-keep])
                self._buf = self._buf[-keep:]
            else:
                out.append(self._buf)
                self._buf = ""
            break
        return self._post("".join(out))

    def flush(self) -> str:
        if self._in_think:
            self._buf = ""
            return ""
        rest, self._buf = self._buf, ""
        return self._post(rest)

    def _post(self, text: str) -> str:
        if self._strip_leading or not self._emitted_any:
            stripped = text.lstrip()
            if stripped:
                self._strip_leading = False
            text = stripped
        if text:
            self._emitted_any = True
        return text


def strip_think(text: str) -> str:
    s = ThinkStripper()
    return s.feed(text) + s.flush()
