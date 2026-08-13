"""手滑与撤回 —— 让她不完美。

两种「收回去」的动作，机制共用，含义相反：

  **手滑**   打错一个字，发现了，撤回重发。机械失误，无关痛痒。
  **说多了** 话说出口才觉得越界，撤回。情绪失误，**这才是戏**。

第二种在 `content/characters/h01/voice-samples.md` 的「六、撤回」里早就写好了
（「……没什么。」「当我没说。」「算了。」），`STAGE_BEHAVIOR.retract_rate` 也
定义了各阶段概率 —— 但一直没有任何代码消费它。这里补上。

---

## 为什么错别字要跟状态挂钩

麦麦的 `typo_generator.py` 用固定 `error_rate=0.3` 给所有输出撒错字。
那是群聊 bot 的做法：错字是**质感**。

对我们不成立。林静姝的人设是精确 —— 她会揪着对方说「你刚才说『一直』」。
这种人平静时打错字的概率很低。所以固定错误率对她是**破人设**。

改成：**错字率是她状态的函数**。这样一个错字不再是装饰，而是一个 tell ——

    她慌了，所以打错了。

玩家读到的不是「这个 AI 会打错字」，是「她刚才手抖了」。

## 为什么不用拼音库

麦麦用 jieba + pypinyin 在同音字里挑，靠 `max_freq_diff=200` 保证错成常见字
（真人的错别字来自输入法选词，不会蹦出生僻同音字 —— 这个洞察是对的）。

我们把同一个约束推到极限：**手工表**。只收「她真的会打的字 × 输入法真的会
选错的词」，代价是覆盖面小，换来的是输出空间完全可审。
她的消息平均十几个字且每条都被玩家逐字读，这里赌不起意外。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..state.models import Emotion

# ---------------- 同音误选表 ----------------

# 每项：(正确, 错成什么)。**只收 IME 真的会选错的高频对**。
#
# 三条收录标准：
#   1. 同音或近音 —— 拼音输入法能把两个字排进同一个候选列表
#   2. 两个字都是常用字 —— 生僻字不会被选中，写出来只会像乱码
#   3. 错了还读得懂 —— 玩家要能一眼看出是手滑，不是我们的 bug
_HOMOPHONES: tuple[tuple[str, str], ...] = (
    ("在", "再"), ("再", "在"),
    ("的", "得"), ("得", "的"), ("地", "的"),
    ("做", "作"), ("作", "做"),
    ("那", "哪"), ("哪", "那"),
    ("是", "事"), ("事", "是"),
    ("到", "道"), ("道", "到"),
    ("以", "已"), ("已", "以"),
    ("他", "她"), ("她", "他"),
    ("有", "又"), ("又", "有"),
    ("象", "像"), ("像", "象"),
    ("跟", "根"), ("根", "跟"),
    ("因", "音"), ("带", "戴"), ("戴", "带"),
    ("完", "玩"), ("玩", "完"),
    ("直", "值"), ("值", "直"),
    ("睡", "税"), ("疼", "腾"),
)

_HOMOPHONE_MAP: dict[str, tuple[str, ...]] = {}
for _right, _wrong in _HOMOPHONES:
    _HOMOPHONE_MAP.setdefault(_right, ())
    _HOMOPHONE_MAP[_right] += (_wrong,)

# 绝不碰的字：碰了会改变意思或踩人设。
#
# 「不」「没」是否定词，错了句子意思反过来 —— 那不是手滑，是我们制造的 bug。
# 「妈」「爸」涉及家人，错字读起来像不敬。
_NEVER_TOUCH = frozenset("不没别妈爸嗯啊哦")

# 她打字时更容易在这些位置手滑：句子后半段（前面还在斟酌，后面手快了）。
_LATE_BIAS = 0.6

# ---------------- 错字概率 ----------------

BASE_TYPO_RATE = 0.02
"""平静状态下每条消息带一个错字的概率。

**故意压得很低。** 她精确，这是人设的一部分。错字要稀有才有信息量 ——
每三条错一个字的人，错字就不再意味着任何事了。
"""

_EMOTION_TYPO: dict[Emotion, float] = {
    Emotion.FLUSTERED: 0.13,   # 慌 → 手抖。**这是这个机制存在的主要理由**
    Emotion.ANGRY: 0.07,       # 生气 → 打得快
    Emotion.TIRED: 0.06,       # 累 → 手不听使唤
    Emotion.SAD: 0.05,
    Emotion.NERVOUS: 0.06,
    Emotion.HAPPY: 0.03,
    Emotion.HURT: 0.03,
    Emotion.RELAXED: 0.02,
}

MIN_TYPO_LEN = 5
"""短于这个长度不打错字。

「嗯。」「还行。」错一个字就整条不可读了。真人在很短的消息里也很少打错 ——
字少，眼睛扫得过来。
"""


def typo_rate(emotions: dict[Emotion, float] | None = None) -> float:
    """按当前情绪算这条消息带错字的概率。

    取**最强那一种**情绪的贡献，按强度插值到基础率。多种情绪不叠加 ——
    又累又慌不会让她打错两倍的字。
    """
    if not emotions:
        return BASE_TYPO_RATE
    best = BASE_TYPO_RATE
    for emo, strength in emotions.items():
        peak = _EMOTION_TYPO.get(emo)
        if peak is None:
            continue
        s = max(0.0, min(1.0, float(strength)))
        best = max(best, BASE_TYPO_RATE + (peak - BASE_TYPO_RATE) * s)
    return best


def _typo_candidates(text: str) -> list[int]:
    """可以下手的位置。"""
    return [
        i for i, ch in enumerate(text)
        if ch in _HOMOPHONE_MAP and ch not in _NEVER_TOUCH
    ]


def make_typo(text: str, rng: random.Random) -> tuple[str, str]:
    """把一个字换成同音字。

    返回 `(打错的文本, 正确的那个字)`。没得换就返回 `(原文, "")`。
    **只换一个字** —— 一条消息里两个错字，玩家读到的是乱码不是手滑。
    """
    positions = _typo_candidates(text)
    if not positions:
        return text, ""

    # 偏向后半段
    if len(positions) > 1 and rng.random() < _LATE_BIAS:
        positions = positions[len(positions) // 2:] or positions

    at = rng.choice(positions)
    right = text[at]
    wrong = rng.choice(_HOMOPHONE_MAP[right])
    return text[:at] + wrong + text[at + 1:], right


# ---------------- 撤回 ----------------

NOTICE_RATE = 0.85
"""打错之后她发现的概率。

高，因为她是那种会揪别人用词的人 —— 自己打错了不可能装看不见。
剩下 15% 就那么挂着，也真实：有时候确实没注意。
"""

CORRECTION_RATE = 0.55
"""发现之后用「*正确字」更正、而不是撤回重发的概率。

微信/QQ 里 `*字` 是更常见的做法 —— 撤回要长按两步，打个星号快得多。
撤回留给「说多了」，那才值得多花两步。
"""

_HAND_SLIP_LINES: tuple[str, ...] = ("手滑。", "打错了。", "*")
"""撤回重发时偶尔跟一句。「手滑。」是 voice-samples「六、撤回」的第 54 条。"""


@dataclass(slots=True)
class Slip:
    """一条消息经过手滑/撤回处理之后要发的东西。"""

    sent: str
    """实际发出去的第一条（可能带错字）。"""

    retract: bool = False
    """这条发出去之后撤回。UI 显示「她撤回了一条消息」。"""

    followups: list[str] = field(default_factory=list)
    """撤回或更正之后补发的。撤回时是重发的正确版本，更正时是「*正确字」。"""

    kind: str = ""
    """"" | "手滑" | "说多了"。用于观测和存档 meta。"""

    @property
    def clean(self) -> bool:
        return not self.kind


def apply_typo(
    text: str,
    rng: random.Random,
    *,
    emotions: dict[Emotion, float] | None = None,
    rate: float | None = None,
) -> Slip:
    """按概率给一条消息加手滑。没触发就原样返回。"""
    if len(text) < MIN_TYPO_LEN:
        return Slip(sent=text)

    if rng.random() >= (typo_rate(emotions) if rate is None else rate):
        return Slip(sent=text)

    wrong_text, right_char = make_typo(text, rng)
    if not right_char:
        return Slip(sent=text)

    # 没发现 —— 错字就那么留着
    if rng.random() >= NOTICE_RATE:
        return Slip(sent=wrong_text, kind="手滑")

    # 发现了，用星号更正（更常见、更轻）
    if rng.random() < CORRECTION_RATE:
        return Slip(sent=wrong_text, followups=[f"*{right_char}"], kind="手滑")

    # 撤回重发
    followups = [text]
    if rng.random() < 0.3:
        followups.insert(0, rng.choice(_HAND_SLIP_LINES[:2]))
    return Slip(sent=wrong_text, retract=True, followups=followups, kind="手滑")


def apply_regret(text: str, rng: random.Random, *, retract_rate: float) -> Slip:
    """说多了 —— 发出去之后撤回，**不重发**。

    跟手滑的关键区别：话收回去了就是收回去了，她不会再说一遍。
    留下的只有「她撤回了一条消息」和后面那句找补。

    **玩家可能看到了，也可能没看到。** 这个不确定性就是这个机制的全部价值 ——
    他会盯着那行灰字想「她刚才说了什么」。
    """
    if retract_rate <= 0 or rng.random() >= retract_rate:
        return Slip(sent=text)

    return Slip(
        sent=text,
        retract=True,
        followups=[rng.choice(("……没什么。", "当我没说。", "算了。", "没事，你忙。"))],
        kind="说多了",
    )


# 模型用它标记「这条是我一说出口就后悔的」。
# 放在文本里而不是 JSON 字段：JSON 加字段会让本来就偶尔崩的解析更脆，
# 而这个标记漏掉的代价只是这次不撤回。
REGRET_MARK = re.compile(r"\s*\[?收回\]?\s*$")


def strip_regret_mark(text: str) -> tuple[str, bool]:
    """剥掉句尾的「收回」标记，返回 `(净文本, 是否标记了)`。"""
    stripped = REGRET_MARK.sub("", text)
    return stripped.strip(), stripped != text
