from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from gfagent.life import load_pools, today_for

SH = ZoneInfo("Asia/Shanghai")


def at(month: int, day: int, hour: int = 20) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=SH)


def test_pools_load():
    p = load_pools("h01")
    assert p.total > 50
    assert p.school and p.home and p.weather


def test_events_are_short_and_concrete():
    """「食堂的番茄炒蛋今天没有蛋」比「食堂菜不好吃」强。太长就不像随口一提。"""
    p = load_pools("h01")
    for pool in (p.school, p.home, p.outside, p.body, p.weekend):
        for e in pool:
            assert len(e) <= 25, f"太长：{e}"
            assert e.endswith(("。", "？")), f"缺句号：{e}"


def test_events_never_mention_the_player():
    """这些是她自己的生活。写进男主会让所有存档共享同一段虚构往事。"""
    p = load_pools("h01")
    everything = p.school + p.home + p.outside + p.body + p.weekend
    for e in everything:
        assert "你" not in e, f"提到了男主：{e}"


def test_same_day_same_save_is_stable():
    """她的今天是确定的，不是每次对话现编的 —— 这本身就是活人感。"""
    a = today_for(1, at(8, 6, 9))
    b = today_for(1, at(8, 6, 23))
    assert a.events == b.events
    assert a.weather == b.weather


def test_different_days_differ():
    days = [tuple(today_for(1, at(8, d)).events) for d in range(1, 15)]
    assert len(set(days)) > 8, "十四天里应该有明显不同的日子"


def test_different_saves_differ_on_same_day():
    """两个玩家在同一天不该听到一模一样的事。"""
    a = today_for(1, at(8, 6))
    b = today_for(2, at(8, 6))
    assert a.events != b.events


def test_weekday_gets_school_events():
    t = today_for(1, at(8, 6))          # 周四
    p = load_pools("h01")
    assert any(e in p.school for e in t.events)


def test_weekend_has_no_school_events():
    """周六她不上课，说「周老师拖堂」就穿帮了。"""
    p = load_pools("h01")
    for day in (8, 9):                   # 周六、周日
        t = today_for(1, at(8, day))
        assert not any(e in p.school for e in t.events), t.events


def test_no_duplicate_events_in_one_day():
    for d in range(1, 30):
        t = today_for(1, at(8, d))
        assert len(t.events) == len(set(t.events))


def test_render_includes_events_and_guidance():
    t = today_for(1, at(8, 6))
    out = t.render()
    assert t.events[0] in out
    assert "一次最多带出一件" in out


def test_render_empty_when_nothing():
    from gfagent.life.daily import Today

    assert Today(day=at(8, 6).date()).render() == ""


# ---------------- 指标必须跟着事件池走 ----------------


def test_concrete_vocabulary_covers_event_pool():
    """具体性指标的词表必须覆盖每一条事件。

    真机踩过：她说「阳台的花被风吹倒了」，指标却判成"不具体" ——
    因为词表里没有「阳台」。指标静默漏报比没有指标更糟，
    它会让人以为改动无效而去改本来正确的东西。
    """
    from gfagent.evals.critic import _CONCRETE

    p = load_pools("h01")
    missed = [
        e for pool in (p.school, p.home, p.outside, p.body, p.weather)
        for e in pool if not _CONCRETE.search(e)
    ]
    assert not missed, (
        "这些事件不会被计入具体性，请补进 critic._CONCRETE：\n  "
        + "\n  ".join(missed)
    )


def test_relational_and_concrete_are_distinguishable():
    """两个指标不该互相打架 —— 一句纯关系话不该被算成具体。"""
    from gfagent.evals.critic import _CONCRETE, _RELATIONAL

    assert _RELATIONAL.search("早点睡，别硬撑")
    assert not _CONCRETE.search("我陪着你")
    assert _CONCRETE.search("教室风扇又坏了")


# ---------------- 不要把同一件事说两遍 ----------------


def test_mentioned_detects_paraphrase():
    """她会改述，所以不能整句匹配。"""
    from gfagent.life.daily import mentioned

    e = "教室的灯坏了一盏，一直闪。"
    assert mentioned(e, ["教室有盏灯一直闪。"])
    assert mentioned(e, ["今天教室的灯坏了。"])
    assert not mentioned(e, ["今天挺累的。", "你吃饭了吗。"])


def test_mentioned_ignores_common_words():
    """「今天」「有点」这种词到处都是，不能算命中。"""
    from gfagent.life.daily import mentioned

    assert not mentioned("今天有点热。", ["今天有点累。"])


def test_render_marks_things_already_said():
    t = today_for(1, at(8, 6))
    first = t.events[0]
    out = t.render([first])
    assert "已经说过了，不要再提" in out
    assert first in out.split("已经说过了")[1]


def test_render_tells_her_not_to_list_everything():
    """把清单念一遍是最假的写法 —— 真机上她真的这么干了。"""
    out = today_for(1, at(8, 6)).render()
    assert "一次最多带出一件" in out
    assert "不是逐条汇报" in out


def test_render_survives_everything_said():
    t = today_for(1, at(8, 6))
    out = t.render(t.events)
    assert "已经说过了" in out


# ---------------- 语义级反复读 ----------------


def test_care_kinds_catch_paraphrased_repetition():
    """「带伞」「擦干」「别淋着」是三句不同的话，同一种关心。

    评审 agent 反复抓到这个，字面查重完全查不出来。
    """
    blob = "记得带伞。你湿透了擦一下。外面还在下雨。"
    assert "提醒他别淋雨" in _care_hits(blob)


def test_care_kinds_do_not_fire_on_single_mention():
    assert not _care_hits("记得带伞。今天风扇又坏了。")


def _care_hits(blob: str, character_id: str = "h01") -> list[str]:
    """关心的类型现在在 content/characters/<id>/agent.yaml 里，不在代码里。"""
    import re as _re

    from gfagent.persona.agent_data import load_agent_data

    return [
        kind.label for kind in load_agent_data(character_id).care_kinds
        if len(_re.findall(kind.pattern, blob)) >= 2
    ]
