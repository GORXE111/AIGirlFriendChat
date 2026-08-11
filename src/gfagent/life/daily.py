"""她的今天。

**「具体性」的来源。**

在 prompt 里写「要说具体的事」是没用的 —— 她得真的**有**事可说。
不给素材，模型只能现编，而模型现编的默认结果就是抽象（「今天挺累的」）。

实现上刻意**不用 LLM**：

  - 零成本、零延迟
  - **确定性** —— 同一天同一存档抽到的是同一批事。
    她的今天是确定的，不是每次对话现编的，这本身就是活人感
  - 编剧改 `content/characters/h01/daily-events.md` 就能扩池，不用动代码
  - 不会幻觉出不符合设定的事
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from ..persona.loader import content_root

log = logging.getLogger(__name__)

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

SCHOOL = "上课日 · 学校"
HOME = "家里"
OUTSIDE = "路上与外面"
BODY = "身体与状态"
WEEKEND = "周末"
WEATHER_KEY = "天气（按季节抽，与事件独立）"


@dataclass(frozen=True, slots=True)
class EventPools:
    school: tuple[str, ...] = ()
    home: tuple[str, ...] = ()
    outside: tuple[str, ...] = ()
    body: tuple[str, ...] = ()
    weekend: tuple[str, ...] = ()
    weather: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return sum(len(p) for p in
                   (self.school, self.home, self.outside,
                    self.body, self.weekend, self.weather))


@lru_cache(maxsize=8)
def load_pools(character_id: str = "h01") -> EventPools:
    path = content_root() / "characters" / character_id / "daily-events.md"
    if not path.exists():
        log.warning("没有每日事件池：%s", path)
        return EventPools()

    text = path.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    matches = list(_H2.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        lines = [
            ln.strip() for ln in body.splitlines()
            if ln.strip() and not ln.startswith(("#", ">", "-", "*", "|"))
        ]
        sections[m.group(1)] = lines

    pools = EventPools(
        school=tuple(sections.get(SCHOOL, ())),
        home=tuple(sections.get(HOME, ())),
        outside=tuple(sections.get(OUTSIDE, ())),
        body=tuple(sections.get(BODY, ())),
        weekend=tuple(sections.get(WEEKEND, ())),
        weather=tuple(sections.get(WEATHER_KEY, ())),
    )
    log.info("加载了 %d 条每日事件", pools.total)
    return pools


def _pick(pool: tuple[str, ...], seed: int, n: int = 1) -> list[str]:
    """按种子确定性地抽 n 条，不重复。"""
    if not pool:
        return []
    n = min(n, len(pool))
    out: list[str] = []
    idx = seed % len(pool)
    step = 1 + (seed // len(pool)) % max(1, len(pool) - 1)
    for _ in range(n):
        while pool[idx] in out:
            idx = (idx + 1) % len(pool)
        out.append(pool[idx])
        idx = (idx + step) % len(pool)
    return out


# 抽关键词时忽略的**单字**虚词。
#
# ⚠️ 这里必须逐字列，不能把多字词拼进来 —— 踩过：
# 写「一直」进去，`直` 就成了停用字，「直闪」这个关键词被误删，
# 于是「教室的灯一直闪」被判成没说过，她又说了一遍。
_STOP = frozenset(
    "的了着过和跟在是有会要就都很太也还又没不别更最个些次"
    "一二两点面前后左右上下里"
    "我你他她们自己什么怎这那可大概"
    "今明昨早中晚时候然刚才已经"
)


def _keywords(event: str) -> set[str]:
    """从事件里抽出可辨识的词，用来判断"这件事她说过没有"。

    她会改述（「教室的灯坏了一盏」→「教室有盏灯一直闪」），
    所以按子串比对，不能整句匹配。
    """
    chars = re.sub(r"[^一-鿿]", " ", event)
    out: set[str] = set()
    for run in chars.split():
        for n in (3, 2):
            for i in range(len(run) - n + 1):
                w = run[i:i + n]
                if not any(c in _STOP for c in w):
                    out.add(w)
    return out


def mentioned(event: str, texts: list[str]) -> bool:
    """这件事在这些消息里被提过了吗。"""
    if not texts:
        return False
    blob = "".join(texts)
    keys = _keywords(event)
    if not keys:
        return False
    hits = sum(1 for k in keys if k in blob)
    # 一个三字词命中就够；两字词要两个以上，避免误判
    return any(len(k) >= 3 and k in blob for k in keys) or hits >= 2


@dataclass(slots=True)
class Today:
    day: date
    weather: str = ""
    events: list[str] = field(default_factory=list)

    def render(self, said: list[str] | None = None) -> str:
        """said：她这次对话里已经说过的消息，用来标出哪些事别再说了。"""
        if not self.events and not self.weather:
            return ""

        said = said or []
        fresh = [e for e in self.events if not mentioned(e, said)]
        used = [e for e in self.events if mentioned(e, said)]

        lines = []
        if self.weather:
            lines.append(f"天气：{self.weather}")
        if fresh:
            lines.append("她今天遇到的事（还没跟他说过）：")
            lines += [f"- {e}" for e in fresh]
        if used:
            lines.append("**这些已经说过了，不要再提**：" + "、".join(used))

        return "\n".join(lines) + (
            "\n\n（这些是**她自己的**今天。"
            "**一次最多带出一件**，而且要有由头 —— 顺着话头提，"
            "不是逐条汇报。没有合适的由头就一件都不提。"
            "把清单念一遍是最假的写法。）"
        )


def today_for(
    save_id: int,
    when: datetime,
    *,
    character_id: str = "h01",
) -> Today:
    """今天她遇到了什么。

    同一天 + 同一存档 → 同一批事。不同存档的同一天各不相同。
    """
    pools = load_pools(character_id)
    day = when.date()
    seed = (day.toordinal() * 7919 + save_id * 104729) & 0x7FFFFFFF
    weekend = when.weekday() >= 5

    events: list[str] = []
    if weekend:
        events += _pick(pools.weekend, seed, 1)
        events += _pick(pools.home, seed >> 3, 1)
    else:
        events += _pick(pools.school, seed, 2)
        events += _pick(pools.home, seed >> 3, 1)

    # 三分之一的日子加一件路上的事，一小半加一条身体状态
    if seed % 3 == 0:
        events += _pick(pools.outside, seed >> 7, 1)
    if seed % 5 < 2:
        events += _pick(pools.body, seed >> 11, 1)

    weather = (_pick(pools.weather, seed >> 5, 1) or [""])[0]
    return Today(day=day, weather=weather, events=events)
