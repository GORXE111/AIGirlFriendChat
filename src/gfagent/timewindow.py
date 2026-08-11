"""DeepSeek 峰谷时段。

高峰（北京时间 09:00-12:00、14:00-18:00）所有计费项 ×2。

对本项目是白捡的便宜：恋爱陪伴的活跃高峰在晚上和深夜，全在低谷区。
慢回路（记忆归档、日记、主动消息预生成）应当主动避开高峰 —— 见 is_peak / seconds_until_offpeak。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")

PEAK_WINDOWS: tuple[tuple[time, time], ...] = (
    (time(9, 0), time(12, 0)),
    (time(14, 0), time(18, 0)),
)

PEAK_MULTIPLIER = 2.0


def now_beijing() -> datetime:
    return datetime.now(BEIJING)


def is_peak(at: datetime | None = None) -> bool:
    at = (at or now_beijing()).astimezone(BEIJING)
    t = at.time()
    return any(start <= t < end for start, end in PEAK_WINDOWS)


def price_multiplier(at: datetime | None = None) -> float:
    return PEAK_MULTIPLIER if is_peak(at) else 1.0


def seconds_until_offpeak(at: datetime | None = None) -> float:
    """距离进入低谷还有多少秒。已在低谷返回 0。

    慢回路调度用：宁可等半小时也别在高峰烧双倍价。
    """
    at = (at or now_beijing()).astimezone(BEIJING)
    t = at.time()
    for start, end in PEAK_WINDOWS:
        if start <= t < end:
            end_dt = datetime.combine(at.date(), end, tzinfo=BEIJING)
            return (end_dt - at).total_seconds()
    return 0.0


def next_offpeak_window(at: datetime | None = None) -> tuple[datetime, datetime]:
    """返回下一个（或当前的）低谷区间 [start, end)。用于批量任务排程。"""
    at = (at or now_beijing()).astimezone(BEIJING)
    if not is_peak(at):
        start = at
    else:
        start = at + timedelta(seconds=seconds_until_offpeak(at))

    t = start.time()
    for w_start, _ in PEAK_WINDOWS:
        if t < w_start:
            return start, datetime.combine(start.date(), w_start, tzinfo=BEIJING)
    # 当天已过全部高峰，低谷延续到次日首个高峰
    tomorrow = start.date() + timedelta(days=1)
    return start, datetime.combine(tomorrow, PEAK_WINDOWS[0][0], tzinfo=BEIJING)
