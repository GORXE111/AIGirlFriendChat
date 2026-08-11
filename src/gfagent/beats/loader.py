"""从 content/beats/ 读桥段。

格式：YAML 头 + Markdown 正文。编剧改 md，不用碰代码。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..persona.loader import content_root
from .models import Beat, BeatKind, Entry, Outcome, TimeOfDay

log = logging.getLogger(__name__)

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# 正文小节 → Beat 字段
_SECTIONS = {
    "场景": "scene",
    "她的状态": "her_state",
    "她不会说的": "hidden",
    "这场戏在赌什么": "stakes",
    "收尾": "ending",
}


def _sections(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    matches = list(_H2.finditer(body))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out[m.group(1)] = body[m.end():end].strip()
    return out


def _entry(raw: dict[str, Any] | None) -> Entry:
    raw = raw or {}
    tod = raw.get("time_of_day") or []
    if isinstance(tod, str):
        tod = [tod]
    weekday = raw.get("weekday") or []
    if isinstance(weekday, int):
        weekday = [weekday]

    def tup(key: str) -> tuple[str, ...]:
        v = raw.get(key) or []
        return tuple([v] if isinstance(v, str) else v)

    return Entry(
        stage_min=raw.get("stage_min"),
        stage_max=raw.get("stage_max"),
        affinity_min=raw.get("affinity_min"),
        affinity_max=raw.get("affinity_max"),
        time_of_day=tuple(TimeOfDay(t) for t in tod),
        weekday=tuple(int(d) for d in weekday),
        flags_all=tup("flags_all"),
        flags_any=tup("flags_any"),
        flags_none=tup("flags_none"),
        cooldown_days=int(raw.get("cooldown_days", 0)),
        mother_night_shift=raw.get("mother_night_shift"),
    )


def _outcomes(raw: list[dict[str, Any]] | None) -> tuple[Outcome, ...]:
    out = []
    for o in raw or []:
        def tup(key: str) -> tuple[str, ...]:
            v = o.get(key) or []
            return tuple([v] if isinstance(v, str) else v)

        out.append(Outcome(
            id=str(o["id"]),
            label=str(o.get("label", o["id"])),
            affinity=float(o.get("affinity", 0)),
            flags_add=tup("flags_add"),
            flags_remove=tup("flags_remove"),
            emotion_bump={str(k): float(v) for k, v in (o.get("emotion_bump") or {}).items()},
            emotion_soothe=o.get("emotion_soothe"),
        ))
    return tuple(out)


def parse_beat(path: Path) -> Beat | None:
    text = path.read_text(encoding="utf-8")
    m = _FRONT.match(text)
    if not m:
        log.error("%s 缺少 YAML 头，跳过", path.name)
        return None

    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        log.error("%s 的 YAML 头解析失败：%s", path.name, exc)
        return None

    if "id" not in meta:
        log.error("%s 缺少 id，跳过", path.name)
        return None

    sec = _sections(m.group(2))
    turns = meta.get("turns") or [3, 6]
    if isinstance(turns, int):
        turns = [turns, turns]

    fields: dict[str, Any] = {v: sec.get(k, "") for k, v in _SECTIONS.items()}

    return Beat(
        id=str(meta["id"]),
        title=str(meta.get("title", meta["id"])),
        kind=BeatKind(meta.get("kind", "her")),
        priority=int(meta.get("priority", 50)),
        once=bool(meta.get("once", False)),
        entry=_entry(meta.get("entry")),
        min_turns=int(turns[0]),
        max_turns=int(turns[1]),
        outcomes=_outcomes(meta.get("outcomes")),
        source=path.name,
        **fields,
    )


@lru_cache(maxsize=8)
def load_beats(character_id: str = "h01") -> tuple[Beat, ...]:
    d = content_root() / "beats" / character_id
    if not d.is_dir():
        log.warning("没有桥段目录：%s", d)
        return ()

    beats: list[Beat] = []
    seen: set[str] = set()
    for path in sorted(d.glob("*.md")):
        beat = parse_beat(path)
        if beat is None:
            continue
        if beat.id in seen:
            log.error("桥段 id 重复：%s（%s）", beat.id, path.name)
            continue
        seen.add(beat.id)
        beats.append(beat)

    log.info("加载了 %d 个桥段", len(beats))
    return tuple(beats)


def get_beat(beat_id: str, character_id: str = "h01") -> Beat | None:
    return next((b for b in load_beats(character_id) if b.id == beat_id), None)
