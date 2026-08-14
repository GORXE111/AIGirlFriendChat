"""扩充的机械指标 + 成对判优。

这两样是 A/B 从「跑得动」变成「能做决策」的关键 —— 实测绝对打分在 n=6
两个变体 p=0.77 / 0.36，完全是噪声。
"""

from __future__ import annotations

import pytest

from gfagent.agent.turn import Option
from gfagent.evals.autoplay import Session
from gfagent.evals.critic import Mechanical, _text_variety, mechanical, sign_test


def _session(msgs: list[str], option_sets: list[list[Option]] | None = None) -> Session:
    s = Session(preset="s3", style="galgamer")
    s.her_messages = msgs
    s.option_sets = option_sets or []
    return s


# ---------------- 语言指纹 ----------------

def test_subjectless_ratio_distinguishes_her_from_generic():
    """「我今天很累」和「今天有点长」是两个人。"""
    hers = mechanical(_session(["今天有点长。", "刚弹完琴。", "还行。", "知道了。"]))
    generic = mechanical(_session(["我今天很累。", "我刚弹完琴。",
                                   "我觉得还行。", "我知道了。"]))
    assert hers.subjectless_ratio == 1.0
    assert generic.subjectless_ratio == 0.0
    assert "省主语" in " ".join(generic.problems())
    assert "省主语" not in " ".join(hers.problems())


def test_subjectless_regex_is_alternation_not_char_class():
    """`[我你|我们]` 里的多字项无效，只会匹配单字符 —— 踩过。"""
    m = mechanical(_session(["咱们走吧。", "谁说的。", "她来了。"]))
    assert m.subjectless_ratio == 0.0


def test_comma_ratio_is_bounded():
    """不用「句号/逗号」原始比值 —— 逗号少的时候分母一小比值就炸。"""
    m = mechanical(_session(["今天有点长。", "刚弹完琴，手指疼。", "嗯。"]))
    assert 0.0 <= m.comma_ratio <= 1.0
    assert m.comma_ratio == pytest.approx(1 / 3)


def test_comma_heavy_speech_is_flagged():
    wordy = ["今天有点长，作业也多，还得练琴。"] * 4
    assert "带逗号" in " ".join(mechanical(_session(wordy)).problems())


def test_no_commas_at_all_does_not_crash():
    m = mechanical(_session(["嗯。", "还行。"]))
    assert m.comma_ratio == 0.0
    assert "带逗号" not in " ".join(m.problems())


# ---------------- 句长方差 ----------------

def test_flat_length_is_flagged():
    """均值正常但方差塌了，是很典型的机器味。"""
    flat = mechanical(_session(["今天有点累了啊", "刚刚弹完琴了", "作业还没写完"]))
    varied = mechanical(_session(["嗯。", "今天有点长。",
                                  "刚弹完琴，手指有点疼，明天还有考试。"]))
    assert flat.len_stdev < varied.len_stdev
    assert "句长方差" in " ".join(flat.problems())
    assert "句长方差" not in " ".join(varied.problems())


# ---------------- 话题跨度 ----------------

def test_topic_spread_counts_distinct_things():
    """具体性只看「这条有没有」，一整局说同一把伞也能拿满分。"""
    narrow = mechanical(_session(["伞带了。", "那把伞呢。", "伞在我这。", "伞。"]))
    wide = mechanical(_session(["风扇坏了。", "周老师叫我。", "月考出来了。",
                                "练琴弹错了。", "猫在车顶上。", "下雨了。"]))
    assert narrow.concrete_ratio == 1.0, "具体性看不出问题"
    assert narrow.topic_spread == 1 and wide.topic_spread >= 6
    assert "原地打转" in " ".join(narrow.problems())


# ---------------- 选项字面差异 ----------------

def test_option_text_variety_catches_relabelled_duplicates():
    """三个不同 tone 配三句几乎一样的话 —— 标签合格，玩家读到的还是一个选项。"""
    same = [[Option("你早点睡吧", "往前"), Option("你早点睡呀", "守住"),
             Option("你早点睡啊", "越界")]]
    diff = [[Option("那我过去接你", "往前"), Option("行，你忙", "守住"),
             Option("刚才谁给你发消息", "越界")]]

    m_same, m_diff = mechanical(_session(["x"], same)), mechanical(_session(["x"], diff))
    assert m_same.option_tone_variety == 3.0, "tone 标签是齐的，这就是问题"
    assert m_same.option_text_variety < m_diff.option_text_variety
    assert "字面重合度高" in " ".join(m_same.problems())
    assert "字面重合度高" not in " ".join(m_diff.problems())


@pytest.mark.parametrize("texts,expect", [
    (["abc", "abc"], 0.0),
    (["abc", "xyz"], 1.0),
    (["只有一条"], 1.0),
    ([], 1.0),
])
def test_text_variety_edges(texts, expect):
    assert _text_variety(texts) == pytest.approx(expect)


# ---------------- 空会话 ----------------

def test_empty_session_has_no_problems():
    m = mechanical(_session([]))
    assert m.problems() == []


def test_single_message_does_not_divide_by_zero():
    m = mechanical(_session(["嗯。"]))
    assert m.len_stdev == 0.0
    assert "句长方差" not in " ".join(m.problems())


# ---------------- 符号检验 ----------------

@pytest.mark.parametrize("w,l,sig", [
    (6, 0, True),    # 全胜才显著
    (5, 1, False),
    (3, 3, False),
    (9, 1, True),
    (10, 2, True),
])
def test_sign_test_significance(w, l, sig):
    assert (sign_test(w, l) < 0.05) is sig


def test_sign_test_is_symmetric():
    assert sign_test(6, 1) == sign_test(1, 6)


def test_sign_test_no_data():
    assert sign_test(0, 0) == 1.0


def test_six_pairs_need_a_sweep():
    """n=6 只有 6-0 才显著 —— 成对判优要发挥作用得跑 n≥10。

    这条钉住的是**方法论前提**，不是代码行为：跑 6 对然后得出 5-1
    就宣布有效，是拿 p=0.22 当结论。
    """
    assert sign_test(6, 0) < 0.05
    assert sign_test(5, 1) > 0.2
