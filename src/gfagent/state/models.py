"""情绪与关系阶段。

情绪是**持久状态**，跨会话保留并按各自速率衰减 —— 见 content 的 emotions.md。
两条特殊规则：

  - **生气与委屈不会自动衰减**，必须由互动消解
  - **委屈状态下她不会完全沉默**，会发微弱信号等对方发现
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class Stage(str, Enum):
    S0 = "S0"  # 陌生：好友列表里的一个名字
    S1 = "S1"  # 试探期：越界后必撤回
    S2 = "S2"  # 确认期：撤回变少
    S3 = "S3"  # 热恋期：不撤回了

    @property
    def label(self) -> str:
        return {"S0": "陌生", "S1": "试探期", "S2": "确认期", "S3": "热恋期"}[self.value]

    @property
    def rank(self) -> int:
        return int(self.value[1])


AFFINITY_THRESHOLDS: dict[Stage, float] = {
    Stage.S0: 0,
    Stage.S1: 20,
    Stage.S2: 50,
    Stage.S3: 80,
}


def stage_for_affinity(affinity: float) -> Stage:
    stage = Stage.S0
    for s, threshold in AFFINITY_THRESHOLDS.items():
        if affinity >= threshold:
            stage = s
    return stage


class Emotion(str, Enum):
    TIRED = "累"
    HAPPY = "开心"
    ANGRY = "生气"
    HURT = "委屈"
    FLUSTERED = "慌"
    SAD = "难过"
    NERVOUS = "紧张"
    RELAXED = "松弛"


# 半衰期（小时）。None ＝ 不自动衰减，必须由互动消解。
HALF_LIFE_HOURS: dict[Emotion, float | None] = {
    Emotion.TIRED: 10.0,      # 睡一觉恢复一部分
    Emotion.HAPPY: 5.0,
    Emotion.ANGRY: None,      # 不会自己好
    Emotion.HURT: None,       # 不会自己消，需要被发现
    Emotion.FLUSTERED: 1.5,
    Emotion.SAD: 24.0,
    Emotion.NERVOUS: 8.0,
    Emotion.RELAXED: 48.0,
}

STAGE_GATED: dict[Emotion, Stage] = {
    Emotion.RELAXED: Stage.S3,  # 松弛是 S3 限定的最终奖励
}


@dataclass(slots=True)
class EmotionState:
    """情绪强度 0..1。"""

    values: dict[Emotion, float] = field(default_factory=dict)
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ---- 衰减 ----

    def decayed(self, now: datetime | None = None) -> dict[Emotion, float]:
        now = now or datetime.now(timezone.utc)
        elapsed_h = max(0.0, (now - self.updated_at).total_seconds() / 3600)
        out: dict[Emotion, float] = {}
        for emo, v in self.values.items():
            half = HALF_LIFE_HOURS.get(emo)
            if half is None:
                out[emo] = v            # 生气／委屈：原样保留
            else:
                decayed = v * (0.5 ** (elapsed_h / half))
                if decayed >= 0.05:     # 低于阈值视为消退
                    out[emo] = decayed
        return out

    def apply_decay(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self.values = self.decayed(now)
        self.updated_at = now

    # ---- 变更 ----

    def bump(self, emo: Emotion, amount: float, stage: Stage = Stage.S0) -> None:
        gate = STAGE_GATED.get(emo)
        if gate is not None and stage.rank < gate.rank:
            return
        self.apply_decay()
        self.values[emo] = min(1.0, self.values.get(emo, 0.0) + amount)

    def soothe(self, emo: Emotion, amount: float) -> None:
        """互动消解。生气与委屈只能靠这个降下来。"""
        self.apply_decay()
        if emo in self.values:
            self.values[emo] = max(0.0, self.values[emo] - amount)
            if self.values[emo] < 0.05:
                del self.values[emo]

    # ---- 读取 ----

    def dominant(self, now: datetime | None = None) -> tuple[Emotion, float] | None:
        cur = self.decayed(now)
        if not cur:
            return None
        emo = max(cur, key=cur.__getitem__)
        return emo, cur[emo]

    def active(self, now: datetime | None = None, threshold: float = 0.15) -> dict[Emotion, float]:
        return {e: v for e, v in self.decayed(now).items() if v >= threshold}

    def describe(self, now: datetime | None = None) -> str:
        """给 prompt 的易变层用。不描述行为，只报状态 —— 行为规则在 lexicon 里。"""
        active = self.active(now)
        if not active:
            return "情绪：平。"
        parts = []
        for emo, v in sorted(active.items(), key=lambda kv: -kv[1]):
            level = "强烈" if v >= 0.7 else ("明显" if v >= 0.4 else "轻微")
            parts.append(f"{emo.value}（{level}）")
        return "情绪：" + "、".join(parts) + "。"

    # ---- 序列化 ----

    def to_json(self) -> str:
        return json.dumps(
            {
                "values": {e.value: round(v, 4) for e, v in self.values.items()},
                "updated_at": self.updated_at.isoformat(),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> EmotionState:
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        values = {}
        for k, v in (data.get("values") or {}).items():
            try:
                values[Emotion(k)] = float(v)
            except ValueError:
                continue
        ts = data.get("updated_at")
        updated = (
            datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
        )
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return cls(values=values, updated_at=updated)


# ---------------- 阶段行为参数 ----------------


@dataclass(frozen=True, slots=True)
class StageBehavior:
    """每个阶段的行为参数。对应 dialogue-rules.md。"""

    max_chars: int
    """单条消息字数的**软上限**。

    主要靠 prompt 执行，后处理只在明显超标时才动手（见 postprocess 的容差）。
    她说话短是性格，不是字数表 —— 卡太死会切出半截话。
    """

    proactive_weight: float
    """主动概率权重 0..1。S0 接近 0。"""

    retract_rate: float
    """越界后撤回的概率。S3 基本不撤 —— 这就是「甜」。"""

    max_messages: int
    """一次最多发几条。

    真人在 IM 里想到什么发什么 —— 「嗯。」「想起来了。」是两条，不是一句。
    她话少，但话少 ≠ 永远只发一条。由模型决定实际发几条，这里只是上限。
    """

    farewell: str
    """收场语。「睡了」是通知，「晚安」是给予。"""

    allow_we: bool
    """能否使用「我们」。第一次说出「我们」是巨大的时刻。"""

    address_player: str
    """怎么称呼玩家。"""


STAGE_BEHAVIOR: dict[Stage, StageBehavior] = {
    Stage.S0: StageBehavior(
        max_chars=20, proactive_weight=0.02, retract_rate=0.0,
        max_messages=2, farewell="", allow_we=False,
        address_player="不称呼",
    ),
    Stage.S1: StageBehavior(
        max_chars=25, proactive_weight=0.15, retract_rate=0.75,
        max_messages=2, farewell="先去写作业了。", allow_we=False,
        address_player="不称呼",
    ),
    Stage.S2: StageBehavior(
        max_chars=30, proactive_weight=0.35, retract_rate=0.35,
        max_messages=3, farewell="睡了。", allow_we=False,
        address_player="偶尔全名",
    ),
    Stage.S3: StageBehavior(
        max_chars=35, proactive_weight=0.55, retract_rate=0.08,
        max_messages=4, farewell="晚安。", allow_we=True,
        address_player="偶尔去姓，多数仍不称呼",
    ),
}
