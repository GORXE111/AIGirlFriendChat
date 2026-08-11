"""桥段（Beat）—— 一场小戏的骨架。

编剧写「这场戏要干什么」，AI 在骨架内生成台词和选项。

最重要的字段是 `hidden`（她不会说的）—— 它定义了戏的张力：玩家在猜，她在藏。
没有它，AI 会把所有事直白说出来，戏就没了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BeatKind(str, Enum):
    HER = "her"        # 她主动发起
    PLAYER = "player"  # 出现在玩家的开场选项里
    BOTH = "both"


class TimeOfDay(str, Enum):
    MORNING = "morning"      # 06:00–11:00
    NOON = "noon"            # 11:00–14:00
    AFTERNOON = "afternoon"  # 14:00–18:00
    EVENING = "evening"      # 18:00–22:00
    NIGHT = "night"          # 22:00–24:00
    LATE = "late"            # 00:00–06:00

    @classmethod
    def of(cls, when: datetime) -> TimeOfDay:
        h = when.hour
        if h < 6:
            return cls.LATE
        if h < 11:
            return cls.MORNING
        if h < 14:
            return cls.NOON
        if h < 18:
            return cls.AFTERNOON
        if h < 22:
            return cls.EVENING
        return cls.NIGHT


@dataclass(frozen=True, slots=True)
class Entry:
    stage_min: str | None = None
    stage_max: str | None = None
    affinity_min: float | None = None
    affinity_max: float | None = None
    time_of_day: tuple[TimeOfDay, ...] = ()
    weekday: tuple[int, ...] = ()
    flags_all: tuple[str, ...] = ()
    flags_any: tuple[str, ...] = ()
    flags_none: tuple[str, ...] = ()
    cooldown_days: int = 0
    mother_night_shift: bool | None = None


@dataclass(frozen=True, slots=True)
class Outcome:
    id: str
    label: str
    affinity: float = 0.0
    flags_add: tuple[str, ...] = ()
    flags_remove: tuple[str, ...] = ()
    emotion_bump: dict[str, float] = field(default_factory=dict)
    emotion_soothe: str | None = None


@dataclass(frozen=True, slots=True)
class Beat:
    id: str
    title: str
    kind: BeatKind
    priority: int
    once: bool
    entry: Entry
    min_turns: int
    max_turns: int
    outcomes: tuple[Outcome, ...]

    # 正文（给 AI 的骨架）
    scene: str = ""
    her_state: str = ""
    hidden: str = ""
    """**她不会说的。** 这场戏的张力全在这里。"""
    stakes: str = ""
    """这场戏在赌什么 —— 玩家的选择空间。"""
    ending: str = ""

    source: str = ""

    def outcome(self, oid: str) -> Outcome | None:
        return next((o for o in self.outcomes if o.id == oid), None)

    def brief(self) -> str:
        """喂给 AI 的骨架。刻意不含台词 —— 台词由 AI 生成。"""
        parts = [f"# 当前这场戏：{self.title}"]
        if self.scene:
            parts.append(f"## 场景\n{self.scene}")
        if self.her_state:
            parts.append(f"## 她现在的状态\n{self.her_state}")
        if self.hidden:
            parts.append(
                f"## 她不会说的（**重要**）\n{self.hidden}\n\n"
                "这些是她藏着的东西。**不要让她直白说出来。**"
                "玩家要靠察觉，不是靠被告知。"
            )
        if self.stakes:
            parts.append(f"## 这场戏在赌什么\n{self.stakes}")
        if self.ending:
            parts.append(f"## 怎么收尾\n{self.ending}")
        if self.outcomes:
            lines = "\n".join(f"- `{o.id}`：{o.label}" for o in self.outcomes)
            parts.append(
                f"## 可能的结局\n{lines}\n\n"
                "戏演完了才给结局；没演完就返回 null，不要急着收。"
            )
        return "\n\n".join(parts)


@dataclass(slots=True)
class BeatProgress:
    """存档里的桥段进度。"""

    beat_id: str | None = None
    turn: int = 0
    history: dict[str, str] = field(default_factory=dict)
    """{beat_id: 上次演完的 ISO 时间}，用于 once 与 cooldown。"""

    def to_dict(self) -> dict[str, Any]:
        return {"beat_id": self.beat_id, "turn": self.turn, "history": self.history}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> BeatProgress:
        if not raw:
            return cls()
        return cls(
            beat_id=raw.get("beat_id"),
            turn=int(raw.get("turn", 0)),
            history=dict(raw.get("history") or {}),
        )
