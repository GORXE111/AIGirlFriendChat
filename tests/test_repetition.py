"""重复与角色混淆。

基准复盘时读对局发现的两类问题，检测层原来一个都抓不住：

- 她换着说法重复同一件事 —— `duplicate_lines` 只抓逐字相同
- **她把自己说过的事当成他的** —— 完全没有检测

以及一个从上线就没起过作用的检查：复读检测方向写反了。
"""

from __future__ import annotations

import pytest

from gfagent.evals.autoplay import Line, Session
from gfagent.evals.critic import mechanical
from gfagent.output.postprocess import process


def _s(msgs: list[str]) -> Session:
    s = Session(preset="s3", style="x")
    s.her_messages = msgs
    s.lines = [Line("她", m) for m in msgs]
    return s


def _dialog(pairs: list[tuple[str, str]]) -> Session:
    s = Session(preset="s1", style="x")
    s.lines = [Line(who, text) for who, text in pairs]
    s.her_messages = [t for w, t in pairs if w == "她"]
    return s


# ---------------- 复读检测：方向原来是反的 ----------------

HE = "我今天看到学校门口修路了，绕了一圈"


def test_shorter_paraphrase_is_caught():
    """**这条是这个检查存在的理由。**

    原来写的是 `if 他的整句 in 她的这一行` —— 要求他说的话完整出现在
    她的话里。她的话更短就永远抓不到，而「用更少的字重复同一件事」
    正是真实的失败模式。这个检查从上线到基准复盘为止没起过作用。
    """
    assert process("校门口修路，绕了一大圈。", echo_of=HE).violations == ["复读玩家"]


def test_verbatim_echo_still_caught():
    assert process(HE, echo_of=HE).violations == ["复读玩家"]


@pytest.mark.parametrize("she", [
    "嗯。",                    # 短应答不算复读
    "嗯。修路了。",            # 接一个词是正常接话
    "那你绕远了吧。",          # 顺着说但是她自己的话
    "今天风扇又坏了。",        # 完全无关
    "绕了多久。",              # 追问
])
def test_normal_replies_are_not_echoes(she):
    """接话本来就会重复对方的词。要抓的是**整句实词几乎全来自他**。"""
    assert not process(she, echo_of=HE).violations


ANSWERS = [
    ("你作业写完了吗", "作业写完了。"),
    ("你吃饭了没", "吃过了。"),
    ("你在练琴吗", "在练琴。"),
    ("教室的风扇修好了吗", "风扇还没修。"),
    ("今天降温了", "降温了。多穿点。"),
    ("周六去面馆吧", "周六。面馆。"),
    ("明天考试吧", "明天考试。"),
]


@pytest.mark.parametrize("he,she", ANSWERS)
def test_answering_a_question_is_not_echoing(he, she):
    """**这一组是回归测试，第二版检测器全判错了。**

    他提问时她复用他的词是**回答**，不是复读。少了这个判断，
    12 局基准实测人设违规从 2 涨到 11、兜底 0→4 ——
    她被逼着说「……」，比复读糟糕得多。
    """
    assert not process(she, echo_of=he).violations, f"{he} → {she}"


def test_long_statement_restated_is_caught():
    """他讲了一件事、她原样倒回去 —— 这才是要抓的。"""
    he = "我今天在便利店买了关东煮和一瓶冰水"
    assert process("便利店买了关东煮和冰水。", echo_of=he).violations == ["复读玩家"]


def test_short_player_line_never_triggers():
    """他说「嗯」，她说什么都不该算复读。"""
    for she in ["嗯。", "嗯，知道了。", "在。"]:
        assert not process(she, echo_of="嗯").violations


# ---------------- 近义重复 ----------------

def test_reworded_repeat_is_caught():
    m = mechanical(_s(["说好了。", "那说好了。", "今天风扇坏了。"]))
    assert m.near_duplicates
    assert "换着说法重复" in " ".join(m.problems())


def test_verbatim_repeat_goes_to_the_other_bucket():
    """逐字相同归 duplicate_lines，不要两边都报。"""
    m = mechanical(_s(["我多要一碗汤。", "嗯。", "我多要一碗汤。"]))
    assert "我多要一碗汤。" in m.duplicate_lines
    assert not any(a == b for a, b in m.near_duplicates)


def test_short_replies_repeat_freely():
    """「嗯。」「哦。」重复是她的常态，不是缺陷。"""
    m = mechanical(_s(["嗯。", "哦。", "嗯。", "还行。", "哦。"]))
    assert not m.near_duplicates


def test_distant_callback_is_not_repetition():
    """隔了很远再提一次是**回指**，那是内梗不是重复。

    「第二次提起同一件事比说十件新事更亲密」—— 这条不能被检测误伤。
    """
    msgs = [
        "今天风扇又坏了。",
        "刚吃完。", "周老师拖堂了。", "我妈今天值夜班。",
        "外面下雨了。", "刚弹完琴。", "月考成绩出来了。", "有点困。",
        "风扇还是没修。",       # 隔了 8 条才回指 —— 这是内梗
    ]
    assert not mechanical(_s(msgs)).near_duplicates


# ---------------- 角色混淆 ----------------

def test_she_asks_him_about_her_own_thing():
    """真实对局抓到的：她说「教室的灯坏了一盏」，几轮后问「灯修好了吗？」

    **比重复更伤** —— 重复只是无聊，这个直接出戏。
    """
    s = _dialog([
        ("她", "教室的灯坏了一盏，一直闪。"),
        ("他", "那挺烦的"),
        ("她", "嗯。"),
        ("她", "灯修好了吗？"),
    ])
    m = mechanical(s)
    assert m.attribution_flips
    assert "灯" in m.attribution_flips[0]
    assert "把自己说过的事当成他的" in " ".join(m.problems())


def test_asking_about_his_things_is_fine():
    """她问他的事是正常的 —— 只有**她自己先提过**的才算混淆。"""
    s = _dialog([
        ("他", "我作业还没写"),
        ("她", "你作业写完了吗。"),
        ("她", "你那边下雨了吗。"),
    ])
    assert not mechanical(s).attribution_flips


def test_mentioning_her_own_thing_again_is_fine():
    """她再提一次自己的事不是混淆 —— 只有**拿它去问他**才是。"""
    s = _dialog([
        ("她", "教室的灯坏了一盏。"),
        ("她", "灯还在闪。"),
        ("她", "那盏灯真烦。"),
    ])
    assert not mechanical(s).attribution_flips


def test_needs_timeline_not_just_messages():
    """判定要按时序 —— 只有 her_messages 时不该误报。"""
    m = mechanical(_s(["灯修好了吗？", "教室的灯坏了一盏。"]))
    assert not m.attribution_flips, "问在前、提在后，不算混淆"


# ---------------- 不误伤 ----------------

def test_clean_conversation_reports_nothing_new():
    s = _dialog([
        ("她", "刚到家。"),
        ("他", "吃了吗"),
        ("她", "刚吃完。"),
        ("她", "今天风扇又坏了。"),
        ("他", "热吗"),
        ("她", "还行。晚自习有空调。"),
    ])
    m = mechanical(s)
    assert not m.near_duplicates
    assert not m.attribution_flips
