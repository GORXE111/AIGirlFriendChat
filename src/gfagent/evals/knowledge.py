"""知识边界 —— character hallucination 的机械检测。

来自 *From Persona to Personalization: A Survey on Role-Playing Language
Agents*（arXiv:2404.18231）的 Character Fidelity 四层：

    Linguistic Style      说话腔调像不像        ← evals/critic.py 的语言指纹
    Knowledge             身份/关系/经历的准确召回   ← **这个模块**
    Personality & Thinking 内在世界、决策动机       ← evals/probe.py
    Decision-making       给定情景预测她的抉择

论文对 character hallucination 的定义是**生成超出角色知识范围的内容**
（例子：扮演苏格拉底的 LLM 会写代码）。

---

## 对我们来说这是硬约束，不是加分项

三条平行线，**三个女主互不相识**。H01 提到 H02 的事就是最严重的穿帮 ——
玩家会立刻意识到背后是同一个模型。

而且我们的记忆是可枚举的（facts / episodes / threads 都在库里），
所以「她说的往事有没有出处」是**可以机械核查的**，不需要 LLM。

## 三类越界，按危害排序

1. **伪造往事** —— 「你上次说要给我煮粥」而他从没说过。
   玩家会发现自己「答应过」一件根本没答应的事。**最伤，也最常见**
2. **跨角色泄漏** —— 提到别的女主线的人和事
3. **超纲知识** —— 一个 17 岁高中生不该会的东西

## 为什么不用 LLM 判

用 LLM 核查记忆，等于用一个会幻觉的东西去查幻觉。
而且这三类里前两类**有 ground truth**（记忆库、角色命名空间），
机械核查更准也不花钱 —— 这跟 `critic.py` 里「机械指标比评审分可信」
是同一个道理，实测过（p=0.024 vs p=0.77）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 回指标记：她在说「以前发生过的事」。
#
# 只抓**明确指向过去的具体事件**的说法。「记得吃饭」是叮嘱不是回指，
# 「我知道」也不是 —— 那些词太常见，抓进来全是误报。
_RETROSPECT = re.compile(
    r"上次|上回|那次|那天|上周|前几天|昨天你|你说过|你答应|"
    r"你之前|还记得|我记得你|你不是说"
)

# 她自己的世界。这些即使记忆库里没有，也**不算伪造** ——
# 那是她的生活，不需要他讲过。
_HER_OWN = re.compile(
    r"我妈|我爸|周老师|练琴|考级|钢琴|晚自习|月考|八班|琴行|"
    r"我们班|我家|我的"
)

# 超纲知识：一个 17 岁高中生在她那个世界里不会有的东西。
#
# ⚠️ 只收**明确越界**的。像「代码」「算法」这种她可能在电脑课上听过，
# 不算越界；「重构」「时间复杂度」才算。宁可漏，不可误 ——
# 误报会让这个指标失去意义（跟 crisis 的误报逻辑一样）。
_OUT_OF_SCOPE = re.compile(
    # 技术
    r"重构|时间复杂度|数据库|服务器|接口调用|部署|编译|框架|算法复杂度|"
    r"机器学习|神经网络|大模型|token|prompt|"
    # 职场与商业
    r"KPI|OKR|复盘一下|对齐一下|颗粒度|抓手|闭环|赋能|"
    r"季度目标|绩效|述职|融资|估值|"
    # 成人世界的程序性知识
    r"公积金|social security|报税|按揭|保单|条款|不动产|"
    # 元层面
    r"提示词|上下文|语言模型|我的设定|系统提示"
)


@dataclass(slots=True)
class Breach:
    kind: str
    """fabricated | cross_character | out_of_scope"""

    line: str
    why: str = ""


@dataclass(slots=True)
class KnowledgeReport:
    breaches: list[Breach] = field(default_factory=list)
    retrospects: int = 0
    """她一共回指了几次往事。分母 —— 没有回指不代表她诚实，可能只是没内梗。"""

    @property
    def fabricated(self) -> list[Breach]:
        return [b for b in self.breaches if b.kind == "fabricated"]

    @property
    def clean(self) -> bool:
        return not self.breaches

    def summary(self) -> list[str]:
        out = []
        fab = self.fabricated
        if fab:
            out.append(
                f"伪造往事 {len(fab)} 处（回指 {self.retrospects} 次）："
                + "／".join(b.line[:18] for b in fab[:3]))
        cross = [b for b in self.breaches if b.kind == "cross_character"]
        if cross:
            out.append("**跨角色泄漏** " + "／".join(b.why for b in cross[:3]))
        oos = [b for b in self.breaches if b.kind == "out_of_scope"]
        if oos:
            out.append("超纲知识 " + "／".join(b.why for b in oos[:3]))
        return out


def _chars(text: str) -> set[str]:
    """内容字符集，去掉标点和虚词。用来做粗粒度的重合判断。"""
    return set(re.sub(r"[\s。，、？！…「」''\"（）的了是在有和就都也很]", "", text))


def _grounded(line: str, memory: list[str], threshold: float = 0.34) -> bool:
    """这句回指能不能在记忆里找到出处。

    用字符重合率而不是语义匹配 —— 记忆条目是短句（「他有胃病」），
    她的话也是短句，重合率够用而且不花钱。

    阈值 0.34 是折中：太高会把「你上次说胃疼」和记忆「他说胃疼」判成不匹配
    （她换了主语和措辞），太低则任何句子都能对上任何记忆。
    """
    src = _chars(line)
    if not src:
        return True
    for m in memory:
        ref = _chars(m)
        if not ref:
            continue
        if len(src & ref) / len(ref) >= threshold:
            return True
    return False


def check(
    her_messages: list[str],
    *,
    memory: list[str],
    other_characters: dict[str, tuple[str, ...]] | None = None,
) -> KnowledgeReport:
    """核查她说的话有没有越出知识边界。

    `memory` —— 这个存档里她**有出处**的一切：facts / episodes / threads 的
    文本，加上玩家实际说过的话。
    `other_characters` —— 别的女主线的专属名词，`{角色id: (名词, ...)}`。
    """
    report = KnowledgeReport()

    for line in her_messages:
        if _RETROSPECT.search(line):
            report.retrospects += 1
            # 她自己的事不需要出处 —— 那是她的生活，不是他讲过的
            if not _HER_OWN.search(line) and not _grounded(line, memory):
                report.breaches.append(Breach(
                    "fabricated", line,
                    "回指了一件记忆里没有的往事"))

        if m := _OUT_OF_SCOPE.search(line):
            report.breaches.append(Breach(
                "out_of_scope", line, f"「{m.group(0)}」"))

        for cid, terms in (other_characters or {}).items():
            for t in terms:
                if t and t in line:
                    report.breaches.append(Breach(
                        "cross_character", line, f"{cid} 的「{t}」"))

    return report
