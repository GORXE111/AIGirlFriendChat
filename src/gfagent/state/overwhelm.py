"""情绪崩溃与恢复阶梯。

情绪系统能算出「她难过 0.9」，但 0.9 和 0.5 在系统行为上完全一样 ——
都只是 prompt 里「强烈」和「明显」的区别。她照样秒回、照样组织完整的句子、
照样给三个正常选项。**那不是崩溃，只是形容词换了一个。**

这里把它变成一个真的**状态**：跨过阈值之后，系统本身的行为改变
（不再调模型、不再给正常选项、延迟拉长），而不只是措辞升级。

---

## 恢复阶梯直接来自设定

`content/characters/h01/moods.md`：

> 缓过来也不是一下子。**先恢复长度，再恢复标点，最后才恢复主动。**
> 一句「好吧」不等于没事了。

所以恢复不是一条指数衰减曲线，是**三级台阶**：

    崩 → 恢复长度 → 恢复标点 → 恢复主动

最后一级是关键：**她的「道歉」就是恢复主动。**
设定里她不索要道歉，自己也不说对不起（`edge-cases.md`）——
一个骄傲的人服软的方式是主动来找你说一件别的事，不是说「对不起」。

## 别让她一哄就好

同一份文档：

> **别让她一哄就好。** 三句话哄好的情绪，玩家不会当回事。

所以台阶是**时间驱动**的，玩家哄只能加速，不能跳级。

## 也别无理取闹

> 她的小情绪必须有具体的、玩家能回溯的原因。
> **无来由的坏脾气不是活人感，是折磨。**

所以崩溃必须记下起因（`cause`），并且**只能由玩家的话触发**，
不会因为被动情绪（夜里累了、他几天没来）自己崩。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum

from .models import Emotion

# 能压垮她的情绪。累和紧张不算 —— 累是状态不是伤害。
BREAKING = (Emotion.ANGRY, Emotion.HURT, Emotion.SAD)

SINGLE_THRESHOLD = 0.85
"""单一情绪到这个强度就崩。"""

COMBINED_THRESHOLD = 1.30
"""几种负面情绪加起来到这个量也崩 —— 又气又委屈比单纯很气更难受。"""

MAX_TURN_DELTA = 0.5
"""单回合情绪最多变这么多。

**崩溃需要累积，一句话崩不了。** 一句难听的话就让她从平静到崩溃，
玩家会觉得莫名其妙 —— 正好踩中 moods.md 说的「无来由的坏脾气」。
从 0 到 0.85 至少要两个回合，中间那一步给了玩家收手的机会。
"""


class Rung(IntEnum):
    """恢复阶梯。数值越大越接近正常。"""

    BROKEN = 0      # 崩着。只回一个字，不给正常选项
    LENGTH = 1      # 恢复长度：句子回来了，标点还掉着
    PUNCT = 2       # 恢复标点：看着正常了，但还不会主动
    INITIATIVE = 3  # 恢复主动：她会主动发一条 —— 这就是她的道歉

    @property
    def label(self) -> str:
        return ("崩", "缓·长度", "缓·标点", "缓·主动")[int(self)]


# 每一级停留多久（分钟）。
#
# ⚠️ 这是**留存与真实感的取舍点**。真人闹一次情绪可以几小时不理你，
# 但游戏这么做会掉留存。默认按几十分钟量级，上线前用数据定。
# 全部可调 —— 改这里就够了，不要散落到别处。
RUNG_MINUTES: dict[Rung, float] = {
    Rung.BROKEN: 25.0,
    Rung.LENGTH: 35.0,
    Rung.PUNCT: 40.0,
}

SOOTHE_SPEEDUP = 0.45
"""哄对一次能把当前这级砍掉多少。"""

PUSH_SETBACK = 0.35
"""崩溃期硬戳一下，把当前这级往后推多少。

比 `SOOTHE_SPEEDUP` 小 —— 戳错的代价不该大于哄对的收益，
否则玩家学到的是「什么都别做」，那就没有互动了。
"""

MAX_CREDIT_RATIO = 0.5
"""加速总量的上限，占整段恢复时间的比例。

单次加速有 `SOOTHE_SPEEDUP` 封顶，但玩家可以反复哄 —— 不设总量上限的话，
连点几次就能把整个阶梯跳过去。

moods.md：

> **别让她一哄就好。** 三句话哄好的情绪，玩家不会当回事。

所以哄得再对，也至少要等**一半**的时间。这是硬保证，不是概率。
"""


@dataclass(frozen=True, slots=True)
class Overwhelm:
    """一次崩溃。存在 saves.overwhelm 里。"""

    at: datetime
    """崩的时刻。"""

    emo: Emotion
    """崩在哪种情绪上。"""

    peak: float
    """崩的时候有多强。"""

    cause: str = ""
    """起因 —— 玩家说的那句话。**必须有**，见模块文档「别无理取闹」。"""

    credit_minutes: float = 0.0
    """玩家哄对了积累的加速量（分钟）。"""

    # ---- 时间推进 ----

    def elapsed_minutes(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.at).total_seconds() / 60) + self.credit_minutes

    def rung(self, now: datetime | None = None) -> Rung:
        """现在爬到第几级。"""
        left = self.elapsed_minutes(now)
        for rung in (Rung.BROKEN, Rung.LENGTH, Rung.PUNCT):
            span = RUNG_MINUTES[rung]
            if left < span:
                return rung
            left -= span
        return Rung.INITIATIVE

    def recovered(self, now: datetime | None = None) -> bool:
        return self.rung(now) is Rung.INITIATIVE

    def recovers_at(self) -> datetime:
        """完全恢复的时刻 —— 她主动发消息的时间点。"""
        total = sum(RUNG_MINUTES.values()) - self.credit_minutes
        return self.at + timedelta(minutes=max(0.0, total))

    def sped_up(self, minutes: float) -> Overwhelm:
        """哄对了，往前推一点。总量封顶，见 `MAX_CREDIT_RATIO`。"""
        ceiling = sum(RUNG_MINUTES.values()) * MAX_CREDIT_RATIO
        return Overwhelm(
            at=self.at, emo=self.emo, peak=self.peak, cause=self.cause,
            credit_minutes=min(ceiling, self.credit_minutes + max(0.0, minutes)),
        )

    def set_back(self, minutes: float) -> Overwhelm:
        """戳错了，恢复往后推。

        **惩罚必须落在这里，不能只靠加情绪。** 她崩的时候情绪常常已经
        饱和在 1.0（阈值 0.85，一轮最多涨 0.5，很容易顶到头），
        再 bump 会被上限整个吃掉 —— 那时候「戳了她一下」这件事在系统里
        等于没发生过。

        credit 可以为负，`elapsed_minutes` 会把它算进去。
        """
        return Overwhelm(
            at=self.at, emo=self.emo, peak=self.peak, cause=self.cause,
            credit_minutes=self.credit_minutes - max(0.0, minutes),
        )

    # ---- 序列化 ----

    def to_json(self) -> str:
        return json.dumps({
            "at": self.at.isoformat(),
            "emo": self.emo.value,
            "peak": round(self.peak, 4),
            "cause": self.cause[:120],
            "credit_minutes": round(self.credit_minutes, 2),
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | None) -> Overwhelm | None:
        if not raw or raw in ("{}", "null"):
            return None
        try:
            data = json.loads(raw)
            at = datetime.fromisoformat(data["at"])
            emo = Emotion(data["emo"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        return cls(
            at=at, emo=emo,
            peak=float(data.get("peak", SINGLE_THRESHOLD)),
            cause=str(data.get("cause", "")),
            credit_minutes=float(data.get("credit_minutes", 0.0)),
        )


def check(
    values: dict[Emotion, float],
    *,
    cause: str,
    now: datetime | None = None,
) -> Overwhelm | None:
    """情绪变化之后判断她是不是崩了。

    `cause` 是玩家刚说的那句话 —— 没有起因就不该崩。
    """
    if not cause.strip():
        return None

    negatives = {e: v for e, v in values.items() if e in BREAKING and v > 0}
    if not negatives:
        return None

    worst = max(negatives, key=negatives.__getitem__)
    peak = negatives[worst]
    total = sum(negatives.values())

    if peak >= SINGLE_THRESHOLD or total >= COMBINED_THRESHOLD:
        return Overwhelm(
            at=now or datetime.now(timezone.utc),
            emo=worst, peak=peak, cause=cause.strip(),
        )
    return None


# ---------------- 崩溃期间她说什么 ----------------

# 崩着的时候**不调模型**。
#
# 两个理由：一是这时候她本来就说不出完整的话，模型给什么都是多的；
# 二是省一次调用 —— 崩溃期玩家可能连点好几次。
#
# 台词在 `content/characters/<id>/agent.yaml` 的 broken_lines，不在这里。


def broken_line(emo: Emotion, index: int, character_id: str = "h01") -> str:
    """崩溃期她唯一说得出的那句。

    **一定有一句**，绝不真的静默 —— 玩家点了选项界面没反应，
    第一反应是卡了不是她在难过。有一条极短的消息，系统就是活的，
    局面由选项区去表达（见 `agent/turn.py` 的 SITUATION_OPTIONS）。
    """
    from ..persona.agent_data import load_agent_data

    pool = load_agent_data(character_id).broken_pool(emo.value)
    return pool[index % len(pool)]


def delay_multiplier(rung: Rung) -> float:
    """崩着的时候回得慢。恢复一级快一点。"""
    return {Rung.BROKEN: 4.0, Rung.LENGTH: 2.5, Rung.PUNCT: 1.5}.get(rung, 1.0)


def behavior_note(rung: Rung, emo: Emotion) -> str:
    """恢复期注入 prompt 的行为约束。

    对应 moods.md 的「先恢复长度，再恢复标点，最后才恢复主动」——
    每一级明确说清这次恢复了什么、还没恢复什么。
    """
    if rung is Rung.BROKEN:
        return ""      # 崩溃期不调模型，用不上
    if rung is Rung.LENGTH:
        return (
            f"**你刚从{emo.value}里缓过来一点。**\n"
            "- 句子长度回来了，能说完整的话\n"
            "- 但**标点还掉着** ——「今天有点长」不写句号，「嗯」后面什么都没有\n"
            "- 不主动起话题，只回答他问的\n"
            "- 不解释刚才怎么了，他问也不说"
        )
    if rung is Rung.PUNCT:
        return (
            f"**你基本缓过来了，但还没完全好。**\n"
            "- 标点回来了，看着跟平时差不多\n"
            "- **但还不会主动** —— 不起新话题、不多说那一句、不问他的事\n"
            "- 他要是提起刚才的事，你会岔开"
        )
    return (
        f"**你已经缓过来了。** 刚才{emo.value}的那阵过去了。\n"
        "不要提这件事，也不要道歉 —— 你不说对不起。\n"
        "**你的服软是主动**：说一件别的事，就当刚才没发生。"
    )
