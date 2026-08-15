"""日程引擎。

**它只做两件事：给回复延迟一个合理的区间，给 prompt 一句情境描述。**

它不限制她说多少字、不禁止她回消息。日程是氛围，不是枷锁 ——
「她在上课，所以慢一点」是活人感；「她在上课，所以只能说 8 个字」是折磨玩家。

延迟按聊天体验校准（秒到几分钟），不是按现实校准（几十分钟）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo("Asia/Shanghai")
"""**她的**时区，不是玩家的。

她是中国高中生，晚自习、月考、周日下午的家庭时间都按她的当地时间发生。
UTC+8 区内（新加坡／马来西亚／菲律宾）读数一致，无需改动。

要改的话用 `Settings.story_timezone`，见 `ScheduleEngine(tz=...)`。

⚠️ 与 `timewindow.py` 的 DeepSeek 峰谷计费时区无关 —— 那个恒为北京时间。
"""


class Pace(str, Enum):
    """她多快能看到消息。只影响延迟，不影响能不能回、能说多少。"""

    QUICK = "quick"    # 手机在手边
    SLOW = "slow"      # 在忙，隔一会儿才看
    AWAY = "away"      # 上课／睡觉，看到得晚


@dataclass(frozen=True, slots=True)
class Window:
    name: str
    start: time
    end: time
    pace: Pace
    delay_min_s: int
    delay_max_s: int
    proactive: bool = False


WEEKDAY_WINDOWS: tuple[Window, ...] = (
    Window("刚起床", time(6, 30), time(7, 20), Pace.SLOW, 40, 180),
    Window("在路上", time(7, 20), time(7, 50), Pace.QUICK, 5, 40),
    Window("上午课", time(7, 50), time(11, 50), Pace.AWAY, 120, 600),
    Window("午休", time(11, 50), time(13, 50), Pace.QUICK, 5, 60),
    Window("下午课", time(13, 50), time(17, 20), Pace.AWAY, 120, 600),
    Window("放学", time(17, 20), time(18, 30), Pace.QUICK, 5, 45, proactive=True),
    Window("晚自习", time(18, 30), time(21, 30), Pace.SLOW, 90, 420),
    Window("回家路上", time(21, 30), time(22, 0), Pace.QUICK, 5, 40, proactive=True),
    Window("在房间", time(22, 0), time(23, 30), Pace.QUICK, 3, 30, proactive=True),
    Window("躺下了", time(23, 30), time(23, 59, 59), Pace.QUICK, 10, 90, proactive=True),
    Window("深夜", time(0, 0), time(1, 0), Pace.QUICK, 10, 120, proactive=True),
    Window("睡着了", time(1, 0), time(6, 30), Pace.AWAY, 300, 900),
)

WEEKEND_WINDOWS: tuple[Window, ...] = (
    Window("睡着了", time(1, 0), time(8, 30), Pace.AWAY, 300, 900),
    Window("早上", time(8, 30), time(9, 0), Pace.SLOW, 40, 180),
    Window("钢琴课", time(9, 0), time(10, 30), Pace.AWAY, 120, 540),
    Window("上午", time(10, 30), time(12, 0), Pace.QUICK, 5, 60, proactive=True),
    Window("下午", time(12, 0), time(18, 0), Pace.QUICK, 10, 120, proactive=True),
    Window("晚上", time(18, 0), time(23, 30), Pace.QUICK, 3, 40, proactive=True),
    Window("躺下了", time(23, 30), time(23, 59, 59), Pace.QUICK, 10, 90, proactive=True),
    Window("深夜", time(0, 0), time(1, 0), Pace.QUICK, 10, 120),
)

# 周日下午是家庭时间
SUNDAY_FAMILY_TIME = (time(13, 0), time(18, 0))


@dataclass(frozen=True, slots=True)
class ScheduleState:
    window: Window
    pace: Pace
    delay_seconds: int
    can_initiate: bool
    local_time: datetime
    note: str = ""

    def describe(self) -> str:
        t = self.local_time
        weekday = "一二三四五六日"[t.weekday()]
        base = f"现在是周{weekday} {t:%H:%M}，她在{self.window.name}。"
        return f"{base}{self.note}" if self.note else base


class ScheduleEngine:
    def __init__(
        self,
        rng: random.Random | None = None,
        tz: ZoneInfo | str | None = None,
    ) -> None:
        self._rng = rng or random.Random()
        self.tz = ZoneInfo(tz) if isinstance(tz, str) else (tz or LOCAL)

    def now_local(self) -> datetime:
        return datetime.now(self.tz)

    def _windows_for(self, when: datetime) -> tuple[Window, ...]:
        return WEEKEND_WINDOWS if when.weekday() >= 5 else WEEKDAY_WINDOWS

    def _find_window(self, when: datetime) -> Window:
        t = when.time()
        for w in self._windows_for(when):
            if w.start <= t < w.end:
                return w
        return Window("空隙", t, t, Pace.QUICK, 5, 60)

    def state(self, when: datetime | None = None) -> ScheduleState:
        when = (when or self.now_local()).astimezone(self.tz)
        window = self._find_window(when)
        note = ""

        if when.weekday() == 6 and SUNDAY_FAMILY_TIME[0] <= when.time() < SUNDAY_FAMILY_TIME[1]:
            window = Window("家里有事", *SUNDAY_FAMILY_TIME, Pace.SLOW, 120, 420)
            note = "周日下午，家里规定不出门。"

        return ScheduleState(
            window=window,
            pace=window.pace,
            delay_seconds=self._rng.randint(window.delay_min_s, window.delay_max_s),
            can_initiate=window.proactive,
            local_time=when,
            note=note,
        )

    def is_mother_night_shift(self, when: datetime | None = None) -> bool:
        """母亲值夜班 —— 一个月 3–4 次。用日期哈希保证当天结果稳定。"""
        when = (when or self.now_local()).astimezone(self.tz)
        if when.hour < 6:
            when = when - timedelta(days=1)
        return (when.toordinal() * 2654435761) % 30 < 4

    def context_note(self, when: datetime | None = None) -> str:
        when = when or self.now_local()
        st = self.state(when)
        parts = [st.describe()]
        if self.is_mother_night_shift(when) and when.hour >= 18:
            parts.append("今晚她妈值夜班，家里只有她一个人。")
        return " ".join(parts)
