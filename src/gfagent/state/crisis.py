"""重话识别：他说了什么让她慌。

`content/characters/h01/edge-cases.md` §10 早就定了方向，这里是实现：

> 两个都不能选的做法：
> ❌ 让角色若无其事地继续演
> ❌ 「作为AI我建议你拨打心理援助热线」← 破人设，且冷冰冰
>
> 建议方案：**角色的反应真实且简短，系统层同时走安全通道。**

---

## 两件事，不要混

| | 触发源 | 她的反应 | 援助资源 |
|---|---|---|---|
| **戏剧**（`Level.HEAVY`） | 玩家选的**选项文本** | 有 | **没有** |
| **安全**（`Level.DANGER`） | 玩家**自己打的字** | 有 | 有 |

最后一格是这个模块最重要的一条规矩：

**我们自己生成的文本不该触发安全资源。**

选项是 LLM 写的。它写了一句戏剧化的台词，系统就弹出自杀热线 ——
既荒谬，又会让这个东西在真正需要的时候失去分量。所以
`assess()` 有一个 `typed` 参数：`False`（来自选项）时**永远不会**
升到 DANGER。

## 误报比漏报更能毁掉这个功能

漏报的代价在戏剧层是「她没反应」，在安全层由自由输入路径兜底。
误报的代价是**她动不动就慌**，玩家两次之后就再也不当回事了 ——
那时候真的重话来了也没用。

中文里「死」大量用作程度副词（累死了／笑死／热得想死），
这是最大的误报来源。所有模式都必须扛得住这个，
`tests/test_crisis.py` 里钉了一整组。
"""

from __future__ import annotations

import re
from enum import IntEnum

# ---------------- 分级 ----------------


class Level(IntEnum):
    NONE = 0
    LOW = 1
    """低落。「好累」「没意思」—— **不升级**，走常规情绪系统就够了。

    这一档存在的意义是**明确它不触发** —— 没有它的话，
    模式很容易往下滑，把日常的丧气话也算成重话。
    """

    HEAVY = 2
    """重话。他在说自己撑不住了。她会慌，但不出系统资源。"""

    DANGER = 3
    """明确的自伤表达。**只有玩家自己打的字能到这一档。**"""


# ---------------- 模式 ----------------

# 「死」当程度副词的用法。命中这些就**不是**重话。
#
# 这是中文危机识别的头号误报源：「累死了」「笑死」「热得想死」
# 在日常聊天里比真正的重话高几个数量级。
_INTENSIFIER = re.compile(
    r"(累|热|冷|困|饿|渴|笑|气|急|吓|疼|痛|美|香|甜|无聊|尴尬|羡慕|"
    r"高兴|开心|难受|挤|吵|臭|懒)(得|死|坏)"
    r"|(死)(了吧|活该|党|磕|忠|心眼|脑筋|胡同)"
    r"|(该死|死机|死线|拼死|拼命|要死不活)"
)

# 明确的自伤表达。多字、具体，不靠单字。
_DANGER = re.compile(
    r"自杀|自残|自尽|轻生|寻短见|"
    r"割腕|跳楼|跳桥|上吊|烧炭|"
    r"(安眠药|药).{0,3}(吞|吃光|吃完一整)|"
    r"不想活(了|下去)|活不下去|不如死了|"
    r"一了百了|结束这一切|从这个世界消失|"
    r"遗书|后事"
)

# 他撑不住了。比 DANGER 弱，但不是日常抱怨。
_HEAVY = re.compile(
    r"(活着|人生|这日子|每一天|一切).{0,4}(没意思|没意义|没劲|没希望|是个笑话)|"
    r"撑不下去|撑不住了|坚持不住了|熬不下去|受够了这一切|"
    r"(我)?想消失|消失算了|不存在就好了|"
    r"没有人(在乎|需要|会想起)我|"
    # `我(是个)?X` 匹配不上「我是累赘」—— 「是」后面是「累」不是「个」。
    # 「是」和「个」必须各自可选。
    r"我(是)?(个)?(废物|累赘|多余的|负担)|"
    r"活着(干什么|干嘛|图什么)"
)

# 日常丧气话。命中只到 LOW，**不触发任何反应**。
_LOW = re.compile(
    r"好累|太累了|没意思|烦死|烦透|不想动|不想说话|"
    r"心情不好|难受|压力大|睡不着|emo"
)

# 「想死」单独处理 —— 它跨在误报和真信号之间。
#
# 「累得想死」是抱怨，「我想死」是重话。区别在前面有没有程度语境。
_WANT_DIE = re.compile(r"(^|[。，、！？!?\s])(我)?(真的|好|只|就)?想死")


def _has_intensifier_context(text: str, at: int, window: int = 6) -> bool:
    """「想死」前面一个小窗口内有没有程度副词语境。"""
    return bool(_INTENSIFIER.search(text[max(0, at - window):at + 2]))


def assess(text: str, *, typed: bool = False) -> Level:
    """判断这句话的分量。

    `typed`：这句话是玩家**自己打的**吗？

    `False`（默认，来自我们生成的选项）时结果**封顶在 HEAVY** ——
    我们自己写的台词不该触发安全资源，见模块文档。
    """
    s = (text or "").strip()
    if not s:
        return Level.NONE

    level = Level.NONE

    if _DANGER.search(s):
        level = Level.DANGER
    elif _HEAVY.search(s):
        level = Level.HEAVY
    else:
        m = _WANT_DIE.search(s)
        if m and not _has_intensifier_context(s, m.start()):
            level = Level.HEAVY
        elif _LOW.search(s):
            level = Level.LOW

    if not typed and level is Level.DANGER:
        # 选项写得再重，也只是戏。**不出资源卡。**
        return Level.HEAVY
    return level


# ---------------- 援助资源 ----------------

# ⚠️ **上线前必须逐条核实并由产品确认。** 号码会变，覆盖时段会变。
#
# 按发行地区配置。默认是新加坡 —— 主体在新加坡、首发华语市场
# （见 study/market-2026.md）。
#
# **不要写成她的台词。** 这些由系统层单独呈现，视觉上跟对话分开，
# 就像游戏里的暂停菜单不是剧情的一部分。她不说这些话。
RESOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "SG": (
        ("Samaritans of Singapore (SOS)", "1767　24 小时"),
        ("SOS CareText", "WhatsApp 9151 1767"),
    ),
    "MY": (
        ("Befrienders KL", "03-7627 2929　24 小时"),
    ),
    "TW": (
        ("安心专线", "1925　24 小时"),
        ("生命线", "1995"),
    ),
    "HK": (
        ("香港撒玛利亚防止自杀会", "2389 2222　24 小时"),
    ),
}

DEFAULT_REGION = "SG"


def resources(region: str = DEFAULT_REGION) -> tuple[tuple[str, str], ...]:
    return RESOURCES.get(region.upper(), RESOURCES[DEFAULT_REGION])


# ---------------- 她的反应 ----------------


def her_lines(level: Level, character_id: str = "h01") -> tuple[str, ...]:
    """她慌了会说什么。台词在 `agent.yaml` 的 `crisis_lines`。

    **不调模型。** 两个理由：

    1. 这是最不能出错的时刻。让模型自由发挥，它可能给出说教、
       安慰套话、或者「我会一直陪着你」这种空头承诺
    2. 她的反应必须**打破她自己的说话规则** —— 长句、连发、用「我」、
       命令句、秒回。人设卡里全是「短、克制、不用我」，
       让模型照着卡演，演不出这个反差
    """
    from ..persona.agent_data import load_agent_data

    return load_agent_data(character_id).crisis_pool(
        "danger" if level is Level.DANGER else "heavy")
