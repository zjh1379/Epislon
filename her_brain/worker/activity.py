"""对话活动追踪：后台任务只在空闲时运行，对话一来就让路。

- 每个对话请求开始时调用 begin()，结束时调用 end()。
- 后台任务里需要调用 LLM 的部分用 run_preemptible() 包起来：对话开始时它会被立即取消，
  上游生成随之中止，显卡马上空出来；被打断的日志保持“待处理”，下次空闲再做。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")


class Preempted(Exception):
    """后台任务因为有新对话而被打断。"""


class ActivityTracker:
    def __init__(self) -> None:
        self.active = 0
        self.last_activity = time.monotonic()
        self.preemptions = 0
        self.total_turns = 0
        self._job: asyncio.Future[Any] | None = None

    def begin(self) -> None:
        self.active += 1
        self.total_turns += 1
        self.last_activity = time.monotonic()
        job = self._job
        if job is not None and not job.done():
            job.cancel()
            self.preemptions += 1

    def end(self) -> None:
        self.active = max(0, self.active - 1)
        self.last_activity = time.monotonic()

    def idle_for(self) -> float:
        if self.active:
            return 0.0
        return time.monotonic() - self.last_activity

    @property
    def busy(self) -> bool:
        return self.active > 0

    async def run_preemptible(self, aw: Awaitable[T]) -> T:
        if self.active:
            if asyncio.iscoroutine(aw):
                aw.close()
            raise Preempted()
        task = asyncio.ensure_future(aw)
        self._job = task
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            self._job = None
        if task.cancelled():
            raise Preempted()
        return task.result()
