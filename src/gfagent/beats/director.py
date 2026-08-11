"""导演：决定现在该演哪场戏。

两个入口：

- `pick_her_beat`  —— 她主动发起的（玩家一打开就看到她发来消息）
- `player_options` —— 玩家能主动开启的桥段
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .loader import load_beats
from .models import Beat, BeatKind, BeatProgress, TimeOfDay

log = logging.getLogger(__name__)


def _stage_rank(s: str) -> int:
    return int(s[1]) if len(s) == 2 and s[0] == "S" else 0


def eligible(
    beat: Beat,
    *,
    stage: str,
    affinity: float,
    flags: set[str],
    now_local: datetime,
    progress: BeatProgress,
    mother_night_shift: bool,
) -> bool:
    e = beat.entry

    last = progress.history.get(beat.id)
    if last:
        if beat.once:
            return False
        if e.cooldown_days > 0:
            try:
                when = datetime.fromisoformat(last)
            except ValueError:
                when = None
            if when is not None:
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                if now_local - when < timedelta(days=e.cooldown_days):
                    return False

    if e.stage_min and _stage_rank(stage) < _stage_rank(e.stage_min):
        return False
    if e.stage_max and _stage_rank(stage) > _stage_rank(e.stage_max):
        return False
    if e.affinity_min is not None and affinity < e.affinity_min:
        return False
    if e.affinity_max is not None and affinity > e.affinity_max:
        return False
    if e.time_of_day and TimeOfDay.of(now_local) not in e.time_of_day:
        return False
    if e.weekday and now_local.weekday() not in e.weekday:
        return False
    if e.flags_all and not set(e.flags_all) <= flags:
        return False
    if e.flags_any and not set(e.flags_any) & flags:
        return False
    if e.flags_none and set(e.flags_none) & flags:
        return False
    if e.mother_night_shift is not None and e.mother_night_shift != mother_night_shift:
        return False
    return True


def _candidates(
    kinds: tuple[BeatKind, ...],
    *,
    character_id: str,
    stage: str,
    affinity: float,
    flags: set[str],
    now_local: datetime,
    progress: BeatProgress,
    mother_night_shift: bool,
) -> list[Beat]:
    out = [
        b for b in load_beats(character_id)
        if b.kind in kinds
        and eligible(b, stage=stage, affinity=affinity, flags=flags,
                     now_local=now_local, progress=progress,
                     mother_night_shift=mother_night_shift)
    ]
    out.sort(key=lambda b: (-b.priority, b.id))
    return out


def pick_her_beat(**kw) -> Beat | None:
    """她主动发起的那一场。优先级最高的一个。"""
    c = _candidates((BeatKind.HER, BeatKind.BOTH), **kw)
    return c[0] if c else None


def player_beats(limit: int = 3, **kw) -> list[Beat]:
    """玩家能主动开启的几场。"""
    return _candidates((BeatKind.PLAYER, BeatKind.BOTH), **kw)[:limit]
