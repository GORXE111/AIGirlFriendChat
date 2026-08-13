"""情绪崩溃与恢复阶梯。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gfagent.state.models import Emotion, Stage
from gfagent.state.overwhelm import (
    COMBINED_THRESHOLD,
    MAX_CREDIT_RATIO,
    MAX_TURN_DELTA,
    RUNG_MINUTES,
    SINGLE_THRESHOLD,
    SOOTHE_SPEEDUP,
    Overwhelm,
    Rung,
    behavior_note,
    broken_line,
    check,
    delay_multiplier,
)

T0 = datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc)


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _broken(**kw) -> Overwhelm:
    return Overwhelm(at=T0, emo=Emotion.SAD, peak=0.9, cause="他说了很重的话", **kw)


# ---------------- 触发 ----------------

def test_single_strong_emotion_breaks_her():
    assert check({Emotion.SAD: SINGLE_THRESHOLD}, cause="他说了很重的话") is not None


def test_combined_load_breaks_her():
    """又气又委屈比单纯很气更难受 —— 两个都没到单项阈值也该崩。"""
    values = {Emotion.ANGRY: 0.7, Emotion.HURT: 0.65}
    assert max(values.values()) < SINGLE_THRESHOLD
    assert sum(values.values()) >= COMBINED_THRESHOLD
    assert check(values, cause="他又那样说") is not None


def test_moderate_emotion_does_not_break_her():
    assert check({Emotion.SAD: 0.6}, cause="他说了句难听的") is None


def test_tired_never_breaks_her():
    """累是状态不是伤害。累到 1.0 也不该崩。"""
    assert check({Emotion.TIRED: 1.0, Emotion.NERVOUS: 1.0}, cause="x") is None


def test_no_cause_no_breakdown():
    """moods.md：无来由的坏脾气不是活人感，是折磨。

    被动情绪（夜里累了、他几天没来）不该让她自己崩 —— 那些没有玩家能
    回溯的起因。
    """
    assert check({Emotion.SAD: 1.0}, cause="") is None
    assert check({Emotion.SAD: 1.0}, cause="   ") is None


def test_breakdown_records_which_emotion_and_why():
    b = check({Emotion.ANGRY: 0.5, Emotion.HURT: 0.95}, cause="他说算了")
    assert b is not None
    assert b.emo is Emotion.HURT        # 最强的那个
    assert b.peak == pytest.approx(0.95)
    assert b.cause == "他说算了"


def test_one_turn_cannot_reach_the_threshold():
    """一句话崩不了 —— 从 0 到阈值至少两轮，中间给玩家收手的机会。"""
    assert MAX_TURN_DELTA < SINGLE_THRESHOLD


# ---------------- 恢复阶梯 ----------------

def test_ladder_follows_the_written_order():
    """moods.md：先恢复长度，再恢复标点，最后才恢复主动。"""
    b = _broken()
    assert b.rung(_at(0)) is Rung.BROKEN
    assert b.rung(_at(RUNG_MINUTES[Rung.BROKEN] + 1)) is Rung.LENGTH
    mid = RUNG_MINUTES[Rung.BROKEN] + RUNG_MINUTES[Rung.LENGTH] + 1
    assert b.rung(_at(mid)) is Rung.PUNCT
    assert b.rung(_at(10_000)) is Rung.INITIATIVE


def test_recovery_is_monotonic():
    b = _broken()
    seen = [int(b.rung(_at(m))) for m in range(0, 130, 5)]
    assert seen == sorted(seen)


def test_recovered_only_at_the_top():
    b = _broken()
    assert not b.recovered(_at(0))
    assert not b.recovered(_at(RUNG_MINUTES[Rung.BROKEN] + 1))
    assert b.recovered(b.recovers_at() + timedelta(minutes=1))


def test_soothing_helps():
    b = _broken()
    modest = b.sped_up(RUNG_MINUTES[Rung.BROKEN] * SOOTHE_SPEEDUP)
    assert modest.recovers_at() < b.recovers_at()


def test_coaxing_can_never_skip_the_whole_ladder():
    """moods.md：别让她一哄就好。三句话哄好的情绪，玩家不会当回事。

    单次加速有上限，但玩家能反复哄 —— 所以总量也必须封顶，
    否则连点几次就跳过整个阶梯。哄得再对也要等一半时间。
    """
    b = _broken()
    total = sum(RUNG_MINUTES.values())

    spammed = b
    for _ in range(50):                    # 疯狂点
        spammed = spammed.sped_up(total)

    assert spammed.credit_minutes <= total * MAX_CREDIT_RATIO + 1e-9
    assert not spammed.recovered(_at(0)), "刚崩就缓透了"

    # 核心保证：哄得再对，真实时间也得走过一半才可能缓透
    half = total * MAX_CREDIT_RATIO
    assert not spammed.recovered(_at(half - 1))
    assert spammed.recovered(_at(half + 1))


def test_speedup_shortens_total_recovery():
    b = _broken()
    assert _broken(credit_minutes=20).recovers_at() == b.recovers_at() - timedelta(minutes=20)


def test_delay_grows_the_more_broken_she_is():
    ms = [delay_multiplier(r) for r in (Rung.BROKEN, Rung.LENGTH,
                                        Rung.PUNCT, Rung.INITIATIVE)]
    assert ms == sorted(ms, reverse=True)
    assert ms[-1] == 1.0        # 缓透了就是正常速度


# ---------------- 行为注入 ----------------

def test_broken_rung_injects_nothing():
    """崩溃期不调模型，注入给谁看。"""
    assert behavior_note(Rung.BROKEN, Emotion.SAD) == ""


def test_each_recovery_rung_says_what_came_back():
    length = behavior_note(Rung.LENGTH, Emotion.SAD)
    punct = behavior_note(Rung.PUNCT, Emotion.SAD)
    init = behavior_note(Rung.INITIATIVE, Emotion.SAD)

    assert "标点还掉着" in length and "长度回来了" in length
    assert "标点回来了" in punct and "还不会主动" in punct
    # 她不说对不起 —— 见 edge-cases.md
    assert "不要道歉" in init and "主动" in init


def test_she_never_apologises_in_words():
    for rung in Rung:
        note = behavior_note(rung, Emotion.HURT)
        assert "对不起" not in note.replace("你不说对不起", "")


# ---------------- 崩溃期的那一句 ----------------

def test_there_is_always_a_line():
    """绝不真静默 —— 点了没反应，玩家第一反应是卡了不是她在难过。"""
    for emo in Emotion:
        for i in range(5):
            line = broken_line(emo, i)
            assert line and len(line) <= 8


def test_broken_lines_fit_her_voice():
    """短、句号、无波浪号无感叹号。"""
    for emo in (Emotion.ANGRY, Emotion.HURT, Emotion.SAD):
        for i in range(3):
            line = broken_line(emo, i)
            assert "~" not in line and "！" not in line and "!" not in line


# ---------------- 序列化 ----------------

def test_roundtrip():
    b = _broken(credit_minutes=7.5)
    again = Overwhelm.from_json(b.to_json())
    assert again is not None
    assert (again.at, again.emo, again.cause) == (b.at, b.emo, b.cause)
    assert again.credit_minutes == pytest.approx(7.5)


@pytest.mark.parametrize("raw", ["", None, "{}", "null", "不是JSON", '{"at":"坏"}'])
def test_bad_json_is_not_a_breakdown(raw):
    assert Overwhelm.from_json(raw) is None
