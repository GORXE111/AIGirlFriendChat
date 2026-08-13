"""手滑与撤回。"""

from __future__ import annotations

import random

import pytest

from gfagent.output.slips import (
    BASE_TYPO_RATE,
    MIN_TYPO_LEN,
    _HOMOPHONE_MAP,
    _NEVER_TOUCH,
    apply_regret,
    apply_typo,
    make_typo,
    strip_regret_mark,
    typo_rate,
)
from gfagent.state.models import Emotion


# ---------------- 错字生成 ----------------

def test_typo_changes_exactly_one_char():
    rng = random.Random(0)
    src = "我妈今天值夜班。"
    for _ in range(50):
        out, right = make_typo(src, rng)
        if not right:
            continue
        assert len(out) == len(src)
        diff = [i for i, (a, b) in enumerate(zip(src, out)) if a != b]
        assert len(diff) == 1, f"改了 {len(diff)} 个字：{out}"
        assert src[diff[0]] == right


def test_typo_never_touches_negation():
    """「不」「没」被错掉会让句子意思反过来 —— 那是 bug 不是手滑。"""
    rng = random.Random(1)
    for src in ("我不想去。", "今天没带伞。", "我没这么说过。"):
        for _ in range(60):
            out, _ = make_typo(src, rng)
            assert out.count("不") == src.count("不")
            assert out.count("没") == src.count("没")


def test_never_touch_chars_are_not_substitutable():
    assert not (_NEVER_TOUCH & set(_HOMOPHONE_MAP)), (
        "禁改表和同音表有交集，禁令会失效"
    )


def test_substitutes_are_single_common_chars():
    """错成的字必须是单个常用汉字 —— 输入法不会蹦出词组或生僻字。"""
    for right, wrongs in _HOMOPHONE_MAP.items():
        assert len(right) == 1
        for w in wrongs:
            assert len(w) == 1, f"{right}→{w} 不是单字"
            assert "一" <= w <= "鿿", f"{right}→{w} 不是汉字"
            assert w != right


def test_short_messages_never_get_typos():
    """「嗯。」错一个字就整条不可读。"""
    rng = random.Random(2)
    for src in ("嗯。", "还行。", "在。"):
        assert len(src) < MIN_TYPO_LEN
        for _ in range(40):
            assert apply_typo(src, rng, rate=1.0).sent == src


def test_no_candidates_returns_original():
    out, right = make_typo("嗯嗯嗯嗯嗯", random.Random(3))
    assert out == "嗯嗯嗯嗯嗯" and right == ""


# ---------------- 错字概率 ----------------

def test_flustered_typos_far_more_than_calm():
    """错字是 tell 不是装饰：她慌了才手抖。"""
    calm = typo_rate({})
    flustered = typo_rate({Emotion.FLUSTERED: 1.0})
    assert calm == pytest.approx(BASE_TYPO_RATE)
    assert flustered > calm * 5


def test_typo_rate_scales_with_strength():
    weak = typo_rate({Emotion.FLUSTERED: 0.2})
    strong = typo_rate({Emotion.FLUSTERED: 1.0})
    assert BASE_TYPO_RATE < weak < strong


def test_emotions_do_not_stack():
    """又累又慌不该让她打错两倍的字。"""
    both = typo_rate({Emotion.FLUSTERED: 1.0, Emotion.TIRED: 1.0})
    only = typo_rate({Emotion.FLUSTERED: 1.0})
    assert both == pytest.approx(only)


def test_base_rate_is_rare():
    """她精确是人设。错字太密就不再意味着任何事。"""
    assert BASE_TYPO_RATE <= 0.03


# ---------------- 手滑的三种收场 ----------------

def test_typo_outcomes_are_all_reachable():
    rng = random.Random(7)
    kinds = {"留着": 0, "星号": 0, "撤回": 0}
    for _ in range(400):
        s = apply_typo("我妈今天值夜班。", rng, rate=1.0)
        if not s.kind:
            continue
        if s.retract:
            kinds["撤回"] += 1
        elif s.followups:
            kinds["星号"] += 1
        else:
            kinds["留着"] += 1
    assert all(v > 0 for v in kinds.values()), kinds


def test_star_correction_carries_the_right_char():
    rng = random.Random(11)
    for _ in range(300):
        s = apply_typo("我妈今天值夜班。", rng, rate=1.0)
        if s.followups and s.followups[0].startswith("*") and not s.retract:
            right = s.followups[0][1:]
            assert len(right) == 1
            # 更正的字必须是原文里被改掉的那个
            assert right in "我妈今天值夜班。"
            return
    pytest.fail("没抽到星号更正")


def test_hand_slip_retract_resends_the_correct_text():
    """手滑撤回之后要把对的重发出来 —— 不然玩家什么都没看到。"""
    rng = random.Random(5)
    src = "我妈今天值夜班。"
    for _ in range(400):
        s = apply_typo(src, rng, rate=1.0)
        if s.retract:
            assert src in s.followups, s.followups
            assert s.sent != src
            return
    pytest.fail("没抽到撤回")


# ---------------- 说多了 ----------------

def test_regret_does_not_resend():
    """说多了跟手滑的关键区别：收回去就是收回去了。"""
    rng = random.Random(13)
    src = "我妈今天值夜班。"
    s = apply_regret(src, rng, retract_rate=1.0)
    assert s.retract and s.kind == "说多了"
    assert src not in s.followups
    assert s.followups and s.followups[0] in (
        "……没什么。", "当我没说。", "算了。", "没事，你忙。"
    )


def test_regret_respects_stage_rate():
    """S3 不撤回 —— 这就是甜。"""
    rng = random.Random(17)
    assert not apply_regret("想见你。", rng, retract_rate=0.0).retract

    hits = sum(
        apply_regret("我今天没带伞。", rng, retract_rate=0.75).retract
        for _ in range(400)
    )
    assert 250 < hits < 350, hits


def test_stage_behavior_retract_rates_are_wired():
    """这些值定义了很久但一直没人消费，回归时容易又断掉。"""
    from gfagent.state.models import STAGE_BEHAVIOR, Stage

    s1 = STAGE_BEHAVIOR[Stage.S1].retract_rate
    s3 = STAGE_BEHAVIOR[Stage.S3].retract_rate
    rng = random.Random(19)
    assert sum(apply_regret("x" * 8, rng, retract_rate=s1).retract
               for _ in range(200)) > 100
    assert sum(apply_regret("x" * 8, rng, retract_rate=s3).retract
               for _ in range(200)) < 40


# ---------------- 收回标记 ----------------

@pytest.mark.parametrize("raw,clean,marked", [
    ("我妈今天值夜班。[收回]", "我妈今天值夜班。", True),
    ("我妈今天值夜班。 [收回]", "我妈今天值夜班。", True),
    ("我妈今天值夜班。收回", "我妈今天值夜班。", True),
    ("我妈今天值夜班。", "我妈今天值夜班。", False),
    ("你把话收回去。", "你把话收回去。", False),
])
def test_strip_regret_mark(raw, clean, marked):
    assert strip_regret_mark(raw) == (clean, marked)


def test_clean_message_passes_through():
    s = apply_typo("记得吃饭。", random.Random(23), rate=0.0)
    assert s.clean and s.sent == "记得吃饭。" and not s.followups
