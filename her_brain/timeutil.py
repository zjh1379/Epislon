"""时间工具：统一用 epoch 秒存储，渲染时转换成本地时区的中文表述。"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone, tzinfo
from functools import lru_cache

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def now() -> float:
    return time.time()


@lru_cache(maxsize=8)
def get_tz(name: str) -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # Windows 未安装 tzdata 时退回东八区
        return timezone(timedelta(hours=8), name="UTC+8")


def to_local(ts: float, tz: tzinfo) -> datetime:
    return datetime.fromtimestamp(ts, tz)


def local_date(ts: float, tz: tzinfo) -> date:
    return to_local(ts, tz).date()


def period_of_day(hour: int) -> str:
    if hour < 5:
        return "凌晨"
    if hour < 8:
        return "早上"
    if hour < 11:
        return "上午"
    if hour < 13:
        return "中午"
    if hour < 17:
        return "下午"
    if hour < 19:
        return "傍晚"
    if hour < 23:
        return "晚上"
    return "深夜"


def format_now(ts: float, tz: tzinfo) -> str:
    """例如：2026-11-04 周三 晚上 21:30"""
    dt = to_local(ts, tz)
    return f"{dt:%Y-%m-%d} {WEEKDAYS[dt.weekday()]} {period_of_day(dt.hour)} {dt:%H:%M}"


def format_stamp(ts: float, tz: tzinfo) -> str:
    """例如：2026-11-04 21:30 周三"""
    dt = to_local(ts, tz)
    return f"{dt:%Y-%m-%d %H:%M} {WEEKDAYS[dt.weekday()]}"


def relative_day(target: date, today: date) -> str:
    delta = (target - today).days
    named = {0: "今天", -1: "昨天", -2: "前天", 1: "明天", 2: "后天"}
    if delta in named:
        return named[delta]
    if -30 <= delta < 0:
        return f"{-delta}天前"
    if 0 < delta <= 30:
        return f"{delta}天后"
    return target.isoformat()


def human_ago(ts: float, now_ts: float, tz: tzinfo) -> str:
    diff = now_ts - ts
    if diff < 90:
        return "刚刚"
    if diff < 3600:
        return f"{int(diff // 60)}分钟前"
    if diff < 6 * 3600:
        return f"{int(diff // 3600)}小时前"
    return relative_day(local_date(ts, tz), local_date(now_ts, tz))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(value[:11], fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_time_arg(value: str | float | int) -> float:
    """接受 epoch 秒或 ISO 时间字符串（无时区时按 UTC 处理）。"""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
