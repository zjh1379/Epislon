"""进程内的简单延迟统计，用于对照文档中的延迟预算（/admin/stats）。"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any


class Metrics:
    def __init__(self, window: int = 500):
        self._series: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._series[name].append(float(value))

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def summary(self) -> dict[str, Any]:
        with self._lock:
            series = {k: sorted(v) for k, v in self._series.items()}
            counters = dict(self._counters)
        out: dict[str, Any] = {"counters": counters, "latency_ms": {}}
        for name, values in series.items():
            if not values:
                continue
            n = len(values)
            out["latency_ms"][name] = {
                "count": n,
                "p50": round(values[n // 2], 1),
                "p95": round(values[min(n - 1, int(n * 0.95))], 1),
                "max": round(values[-1], 1),
            }
        return out
