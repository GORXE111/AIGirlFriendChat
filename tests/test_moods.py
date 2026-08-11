from __future__ import annotations

import pytest

from gfagent.state import Emotion, behavior_note
from gfagent.state.moods import MILD, STRONG, _BEHAVIOR


def test_calm_gives_nothing():
    """平静的时候不该注入任何行为指令 —— 那只会稀释真正的规则。"""
    assert behavior_note({}) == ""
    assert behavior_note({Emotion.TIRED: 0.1}) == ""


def test_mild_and_strong_differ():
    mild = behavior_note({Emotion.ANGRY: MILD})
    strong = behavior_note({Emotion.ANGRY: STRONG})
    assert mild and strong and mild != strong
    assert len(strong) > len(mild)


def test_angry_says_how_not_just_that():
    """情绪系统只报「生气（明显）」，模型不知道该怎么演，
    结果生气和不生气看起来一样。行为指令必须给出具体动作。"""
    out = behavior_note({Emotion.ANGRY: 0.8})
    assert "说反话" in out
    assert "突然客气" in out
    assert "绝不" in out          # 也要说不做什么


def test_hurt_encodes_the_denial_ladder():
    """「没事」→「我说了没事」→ 第三次才松口。前两次否认要当真地演。"""
    out = behavior_note({Emotion.HURT: 0.8})
    assert "第三次" in out
    assert "否认" in out


def test_soothing_only_when_strongly_upset():
    """哄的方法只在她真的闹情绪时给，平时读它是浪费。"""
    assert "你要的不是道歉" in behavior_note({Emotion.ANGRY: 0.8})
    assert "你要的不是道歉" in behavior_note({Emotion.HURT: 0.9})
    assert "你要的不是道歉" not in behavior_note({Emotion.ANGRY: 0.3})
    assert "你要的不是道歉" not in behavior_note({Emotion.HAPPY: 0.9})


def test_at_most_two_emotions():
    """同时演五种情绪等于什么都没演。"""
    out = behavior_note({e: 0.9 for e in _BEHAVIOR})
    assert out.count("**你现在的状态**") == 1
    # 只有最强的两种会出现；这里全部同权，至少不能全塞进去
    assert len(out) < 1200


def test_strongest_emotion_wins():
    out = behavior_note({Emotion.TIRED: 0.3, Emotion.ANGRY: 0.9})
    assert "说反话" in out


def test_happy_is_understated():
    """她的开心不是变热情，只是多说了一句。"""
    out = behavior_note({Emotion.HAPPY: 0.8})
    assert "多说了一句" in out
    assert "感叹号" in out


def test_tired_drops_punctuation():
    out = behavior_note({Emotion.TIRED: 0.8})
    assert "标点会掉" in out


def test_every_behavior_has_both_levels():
    for emo, (mild, strong) in _BEHAVIOR.items():
        assert mild and strong, emo
        assert mild != strong, emo


def test_moods_not_in_static_card():
    """情境性规则该情境性注入。放静态卡里，她平静时也在读「生气怎么演」。"""
    from gfagent.persona import load_card
    from gfagent.persona.manifest import LEXICON_SECTIONS

    assert not any(s.file == "moods.md" for s in LEXICON_SECTIONS)
    card = load_card("h01")
    assert "默不作声型" not in card.lexicon
