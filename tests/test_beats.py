from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gfagent.agent.turn import MAX_OPTIONS, parse
from gfagent.beats import (
    BeatKind,
    BeatProgress,
    eligible,
    get_beat,
    load_beats,
    pick_her_beat,
    player_beats,
)
from gfagent.beats.models import TimeOfDay
from gfagent.schedule.engine import LOCAL


def at(h: int, day: int = 4) -> datetime:
    return datetime(2026, 8, day, h, 0, tzinfo=LOCAL)


# eligible() 只判条件；pick_her_beat()/player_beats() 还需要 character_id
COND = dict(
    stage="S1", affinity=30.0, flags=set(),
    progress=BeatProgress(), mother_night_shift=False,
)
BASE = {**COND, "character_id": "h01"}


# ---------------- 加载 ----------------


def test_beats_load():
    beats = load_beats("h01")
    assert beats
    assert all(b.id and b.title for b in beats)


def test_every_beat_has_hidden_and_outcomes():
    """「她不会说的」是戏的张力所在；没有结局的戏收不了尾。"""
    for b in load_beats("h01"):
        assert b.hidden, f"{b.id} 缺少「她不会说的」"
        assert b.outcomes, f"{b.id} 没有结局"
        assert b.min_turns <= b.max_turns


def test_brief_carries_the_skeleton_not_the_lines():
    b = get_beat("post_exam_night")
    assert b is not None
    brief = b.brief()
    assert "她不会说的" in brief
    assert "名次掉了" in brief
    assert "opened" in brief and "closed" in brief


def test_unknown_beat():
    assert get_beat("nope") is None


# ---------------- 触发条件 ----------------


def test_stage_gate():
    b = get_beat("post_exam_night")
    assert not eligible(b, now_local=at(20), **{**COND, "stage": "S0"})
    assert eligible(b, now_local=at(20), **{**COND, "stage": "S2"})


def test_affinity_gate():
    b = get_beat("post_exam_night")   # affinity_min: 20
    assert not eligible(b, now_local=at(20), **{**COND, "affinity": 5})
    assert eligible(b, now_local=at(20), **{**COND, "affinity": 25})


def test_time_of_day_gate():
    b = get_beat("late_night_awake")  # time_of_day: [late]
    assert eligible(b, now_local=at(2), **COND)
    assert not eligible(b, now_local=at(15), **COND)


def test_mother_night_shift_gate():
    b = get_beat("mother_night_shift")
    kw = {**COND, "affinity": 40.0}
    assert eligible(b, now_local=at(21), **{**kw, "mother_night_shift": True})
    assert not eligible(b, now_local=at(21), **{**kw, "mother_night_shift": False})


def test_once_beats_do_not_repeat():
    b = get_beat("first_words")
    assert b.once
    played = BeatProgress(history={"first_words": at(4).isoformat()})
    assert not eligible(b, now_local=at(20), **{**COND, "stage": "S0",
                                                "progress": played})


def test_cooldown():
    b = get_beat("late_night_awake")   # cooldown_days: 5
    recent = BeatProgress(history={"late_night_awake": (at(2) - timedelta(days=2)).isoformat()})
    old = BeatProgress(history={"late_night_awake": (at(2) - timedelta(days=30)).isoformat()})
    assert not eligible(b, now_local=at(2), **{**COND, "progress": recent})
    assert eligible(b, now_local=at(2), **{**COND, "progress": old})


def test_time_of_day_boundaries():
    assert TimeOfDay.of(at(3)) is TimeOfDay.LATE
    assert TimeOfDay.of(at(9)) is TimeOfDay.MORNING
    assert TimeOfDay.of(at(15)) is TimeOfDay.AFTERNOON
    assert TimeOfDay.of(at(20)) is TimeOfDay.EVENING
    assert TimeOfDay.of(at(23)) is TimeOfDay.NIGHT


# ---------------- 导演 ----------------


def test_first_words_wins_at_s0():
    """开局第一场必须是「第一次说话」—— 它的 priority 最高。"""
    beat = pick_her_beat(now_local=at(20), **{**BASE, "stage": "S0", "affinity": 0})
    assert beat is not None and beat.id == "first_words"


def test_player_beats_exclude_her_only():
    beats = player_beats(now_local=at(20), **BASE)
    assert beats
    assert all(b.kind in (BeatKind.PLAYER, BeatKind.BOTH) for b in beats)


def test_nothing_eligible_returns_none():
    """条件全不满足时不能硬塞一场戏。"""
    played = BeatProgress(history={b.id: at(4).isoformat() for b in load_beats("h01")})
    beat = pick_her_beat(
        now_local=at(15),
        **{**BASE, "stage": "S0", "affinity": 0, "progress": played},
    )
    assert beat is None or beat.kind is not BeatKind.HER


# ---------------- 回合解析 ----------------


GOOD = """{"messages": ["嗯。", "想起来了。"],
 "options": [{"text": "你还记得我", "tone": "直接"},
             {"text": "打扰了", "tone": "回避"},
             {"text": "在忙吗", "tone": "关心"}],
 "outcome": null}"""


def test_parse_good():
    p = parse(GOOD, ("connected", "awkward"))
    assert p.messages == ["嗯。", "想起来了。"]
    assert len(p.options) == 3
    assert p.options[0].tone == "直接"
    assert p.outcome is None


def test_parse_with_code_fence():
    p = parse(f"```json\n{GOOD}\n```", ())
    assert p.messages == ["嗯。", "想起来了。"]


def test_parse_with_preamble():
    p = parse(f"好的，这是这一回合：\n{GOOD}", ())
    assert len(p.options) == 3


def test_parse_outcome_must_be_known():
    """模型编一个不存在的结局时忽略，不能让它乱收尾。"""
    raw = GOOD.replace('"outcome": null', '"outcome": "编的"')
    assert parse(raw, ("connected", "awkward")).outcome is None
    raw2 = GOOD.replace('"outcome": null', '"outcome": "connected"')
    assert parse(raw2, ("connected", "awkward")).outcome == "connected"


def test_parse_caps_options():
    raw = """{"messages":["嗯。"],"options":[
      {"text":"a"},{"text":"b"},{"text":"c"},{"text":"d"},{"text":"e"}]}"""
    assert len(parse(raw, ()).options) == MAX_OPTIONS


def test_parse_garbage_is_empty_not_crash():
    p = parse("模型今天不想说话", ())
    assert p.messages == [] and p.options == []


def test_parse_plain_string_options():
    raw = '{"messages":["嗯。"],"options":["好","不好","算了"]}'
    p = parse(raw, ())
    assert [o.text for o in p.options] == ["好", "不好", "算了"]


# ---------------- 阶段分寸 ----------------


def test_stage_guides_are_distinct_and_escalate():
    """S0 不许约见面，S2 才可以 —— 越级选项会直接毁掉关系的可信度。"""
    from gfagent.agent.turn import STAGE_OPTION_GUIDE

    assert set(STAGE_OPTION_GUIDE) == {"S0", "S1", "S2", "S3"}
    assert "约见面" in STAGE_OPTION_GUIDE["S0"] or "约吃饭" in STAGE_OPTION_GUIDE["S0"]
    assert "绝不能出现" in STAGE_OPTION_GUIDE["S0"]
    assert "可以约见面" in STAGE_OPTION_GUIDE["S2"]


def test_instructions_carry_stage_constraint():
    from gfagent.agent.turn import instructions

    s0 = instructions(her_max_chars=20, her_max_messages=2, in_beat=False,
                      can_finish=False, outcome_ids=(), stage="S0")
    s3 = instructions(her_max_chars=35, her_max_messages=4, in_beat=False,
                      can_finish=False, outcome_ids=(), stage="S3")
    assert "绝不能出现" in s0
    assert s0 != s3


def test_topic_instructions_carry_stage():
    from gfagent.agent.turn import topic_instructions

    assert "绝不能出现" in topic_instructions("S0")
    assert "JSON" in topic_instructions("S0")


def test_first_outcome_is_the_default_close():
    """轮数用尽取第一个结局，所以第一个必须是「正常演完」那个。"""
    for b in load_beats("h01"):
        first = b.outcomes[0].label
        assert not any(bad in first for bad in ("关上", "尬住", "没聊起来")), (
            f"{b.id} 把负面结局放在了第一位 —— 好好聊完的玩家会莫名吃个坏结局"
        )


# ---------------- 话题 ----------------


def test_parse_topics():
    from gfagent.agent.turn import parse_topics

    raw = """{"topics":[
      {"title":"问他胃疼","opener":"你胃还疼吗"},
      {"title":"说说今天","opener":"今天下雨了"}]}"""
    ts = parse_topics(raw, 3)
    assert [t.title for t in ts] == ["问他胃疼", "说说今天"]
    assert ts[0].opener == "你胃还疼吗"


def test_parse_topics_caps_and_skips_empty():
    from gfagent.agent.turn import parse_topics

    raw = """{"topics":[{"title":"a","opener":"x"},{"title":"b"},
                        {"title":"c","opener":"z"},{"title":"d","opener":"w"}]}"""
    assert len(parse_topics(raw, 2)) <= 2


def test_parse_topics_garbage():
    from gfagent.agent.turn import parse_topics

    assert parse_topics("不是 JSON", 3) == []
