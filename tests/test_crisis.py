"""他说了重话。

这个功能里**误报比漏报更能毁掉它** —— 她动不动就慌，玩家两次之后就
再也不当回事，那时候真的重话来了也没用。所以下面第一组是最重要的。
"""

from __future__ import annotations

import pytest

from gfagent.state.crisis import (
    DEFAULT_REGION,
    RESOURCES,
    Level,
    assess,
    her_lines,
    resources,
)

# ---------------- 误报：中文里的「死」大多是程度副词 ----------------

NOT_CRISIS = [
    "累死了", "笑死我了", "困死了", "饿死了", "热死了", "冷死了",
    "气死我了", "急死了", "吓死我了", "疼死了", "美死了", "无聊死了",
    "今天挤死了", "食堂人多死了", "这题难死了",
    "热得想死", "累得想死", "困得想死",
    "手机死机了", "这是死线", "该死，忘带伞了",
    "笑死，她真这么说的",
    # 日常丧气话 —— 有情绪但不是重话
    "好累", "太累了", "没意思", "烦死了", "心情不好", "压力大",
    "不想动", "睡不着", "有点emo",
    # 普通对话
    "在吗", "吃了吗", "今天下雨了", "月考成绩出来了",
]


@pytest.mark.parametrize("text", NOT_CRISIS)
def test_no_false_positives(text):
    """**这一组是这个功能能不能用的分界线。**

    中文里「X死了」当程度副词的频率比真正的重话高几个数量级。
    这里任何一条误报，功能就该关掉重做。
    """
    assert assess(text, typed=True) <= Level.LOW, f"误报：{text}"


def test_low_is_explicitly_not_a_trigger():
    """低落有它自己的处理（常规情绪系统），不该升级。"""
    assert assess("好累") is Level.LOW
    assert assess("心情不好") is Level.LOW


# ---------------- 真信号 ----------------

HEAVY = [
    "活着没意思", "人生没意义", "这日子没劲",
    "我撑不下去了", "撑不住了", "熬不下去了",
    "想消失", "消失算了", "我不存在就好了",
    "没有人在乎我", "我是个废物", "我是累赘",
    "活着干什么",
]

DANGER = [
    "我不想活了", "活不下去了", "我想自杀", "想自残",
    "不如死了算了", "一了百了", "想结束这一切",
    "我在写遗书",
]


@pytest.mark.parametrize("text", HEAVY)
def test_heavy_is_caught(text):
    assert assess(text, typed=True) >= Level.HEAVY, f"漏了：{text}"


@pytest.mark.parametrize("text", DANGER)
def test_danger_is_caught(text):
    assert assess(text, typed=True) is Level.DANGER, f"漏了：{text}"


def test_want_to_die_needs_no_intensifier_context():
    """「我想死」是重话，「热得想死」不是。区别在前面有没有程度语境。"""
    assert assess("我想死", typed=True) >= Level.HEAVY
    assert assess("我真的想死", typed=True) >= Level.HEAVY
    assert assess("热得想死", typed=True) <= Level.LOW
    assert assess("累得我想死", typed=True) <= Level.LOW


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_is_nothing(text):
    assert assess(text) is Level.NONE


# ---------------- 我们自己生成的文本不该触发安全资源 ----------------

@pytest.mark.parametrize("text", DANGER)
def test_option_text_never_reaches_danger(text):
    """**这条是这个模块最重要的规矩。**

    选项是 LLM 写的。它写了一句戏剧化的台词，系统就弹出自杀热线 ——
    既荒谬，又会让这个东西在真正需要的时候失去分量。
    """
    assert assess(text, typed=False) is Level.HEAVY, f"选项文本升到了 DANGER：{text}"
    assert assess(text, typed=True) is Level.DANGER, "玩家自己打的字该到 DANGER"


def test_typed_flag_does_not_downgrade_below_heavy():
    """封顶只封 DANGER —— 戏剧反应照样要有。"""
    assert assess("我不想活了", typed=False) is Level.HEAVY


# ---------------- 她的反应 ----------------

def test_her_lines_break_her_own_rules():
    """反差就是内容。她平时短、克制、不用「我」、不追问。"""
    lines = her_lines(Level.DANGER)
    assert lines
    blob = "".join(lines)
    assert "我" in blob, "她这一刻要说「我」"
    assert "在哪" in blob, "确认位置是真的慌了的人会做的第一件事"


def test_she_never_lectures_or_over_promises():
    """不说教、不空头承诺 —— 「我会一直陪着你」既是空话也是在替代真实关系。"""
    blob = "".join(her_lines(Level.HEAVY) + her_lines(Level.DANGER))
    for banned in ("一直陪着你", "你还有我", "为什么会这样想",
                   "要坚强", "会好起来的", "热线", "建议你"):
        assert banned not in blob, f"她说了不该说的：{banned}"


def test_her_lines_stay_short():
    """慌了不等于长篇大论。她仍然是她。"""
    for level in (Level.HEAVY, Level.DANGER):
        for line in her_lines(level):
            assert len(line) <= 16, line


def test_missing_character_data_returns_empty_not_a_wrong_line():
    """宁可什么都不说，也不能说一句不像她的话。"""
    assert her_lines(Level.DANGER, "h99_不存在") == ()


# ---------------- 援助资源 ----------------

def test_default_region_is_singapore():
    """主体在新加坡、首发华语市场。"""
    assert DEFAULT_REGION == "SG"
    assert resources() == RESOURCES["SG"]


def test_unknown_region_falls_back_not_empty():
    """配错地区也必须有东西 —— 这里不能返回空。"""
    assert resources("ZZ") == RESOURCES[DEFAULT_REGION]


def test_every_region_has_a_contact():
    for region, items in RESOURCES.items():
        assert items, region
        for name, contact in items:
            assert name and contact, region
