"""复盘：对照设计文档挑毛病。

两层：

- **机械检查** —— 不用 LLM 就能算的（重复率、选项是否雷同、句长、禁用符号）。
  这些是硬指标，不该有争议。
- **评审 agent** —— 判断"像不像活人"这种机械算不出来的东西。

评审标准全部来自我们自己的设定文档，不是泛泛的「好不好」。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from ..llm import DeepSeekProvider, LLMError, LLMRequest, Message, Task
from ..metrics import UsageRecorder
from .autoplay import Session

log = logging.getLogger(__name__)

_BANNED = re.compile(r"[!！~～]|233|xswl|绝绝子|哈哈哈")
_ASSISTANT = re.compile(
    r"作为(一个)?(AI|人工智能)|我不是医生|建议你(及时|尽快)|我理解你的感受|辛苦了")
# 「关系话」：在谈两个人本身，而不是在谈一件事
_RELATIONAL = re.compile(
    r"想我|喜欢你|我们的关系|你对我|我对你|在一起|陪着你|别硬撑|早点睡|注意身体")

# 「具体物」：她世界里的名词。
#
# ⚠️ 这份词表**必须覆盖 daily-events.md 里的每一条事件**，
# 否则指标会静默漏报 —— 真机踩过：她明明在说「阳台的花被风吹倒了」，
# 指标却判成"不具体"，因为词表里没有「阳台」。
# `test_concrete_vocabulary_covers_event_pool` 守着这条。
_CONCRETE = re.compile(
    # 学校
    r"风扇|周老师|老师|食堂|月考|排名|成绩|晚自习|早读|课间|上课|调课|"
    r"作业|卷子|物理|数学|古文|体育|大扫除|黑板|窗帘|讲台|同桌|抽屉|"
    r"校服|扣子|笔袋|绿萝|身高|纸条|打饭|走廊|自习|巡视|灯|打瞌睡|全班|"
    # 家里与钢琴
    r"练琴|钢琴|考级|报名表|我妈|我爸|值夜班|出差|点心|排骨|晚饭|"
    r"洗衣机|作息表|阳台|花|亲戚|台灯|曲目|养生|文章|开门|半夜|心不在焉|"
    # 外面
    r"猫|车顶|校门|修路|公交|便利店|关东煮|琴行|橱窗|超市|伞|自行车|"
    r"链子|同学|奶茶|巷子|"
    # 身体
    r"睡着|哈欠|手指|嗓子|咖啡|眼睛|冷|"
    # 天气
    r"下雨|雨|降温|太阳|晒|闷|雾|风|天气|"
    # 内梗
    r"面馆|辣酱|铅笔|吉他|座位|停电|体检"
)


@dataclass(slots=True)
class Mechanical:
    her_count: int = 0
    avg_len: float = 0.0
    over_len: int = 0
    duplicate_lines: list[str] = field(default_factory=list)
    same_opener_runs: int = 0
    banned_hits: list[str] = field(default_factory=list)
    assistant_hits: list[str] = field(default_factory=list)
    concrete_ratio: float = 0.0
    relational_ratio: float = 0.0
    identical_option_sets: int = 0
    option_tone_variety: float = 0.0

    def problems(self) -> list[str]:
        out = []
        if self.duplicate_lines:
            out.append(f"她重复说了同样的话：{self.duplicate_lines}")
        if self.same_opener_runs:
            out.append(f"连续 {self.same_opener_runs} 次用同一个字开头")
        if self.banned_hits:
            out.append(f"出现禁用符号／网络梗：{self.banned_hits}")
        if self.assistant_hits:
            out.append(f"助理腔泄漏：{self.assistant_hits}")
        if self.concrete_ratio < 0.3:
            out.append(
                f"具体性不足：只有 {self.concrete_ratio:.0%} 的消息提到了具体的东西"
                "（风扇／周老师／月考…）。她在说感受，不是在说事情")
        if self.relational_ratio > 0.4:
            out.append(
                f"过度谈关系：{self.relational_ratio:.0%} 的消息在谈两个人本身。"
                "真实情侣大部分时间在聊第三件事")
        if self.identical_option_sets:
            out.append(f"{self.identical_option_sets} 组选项里出现了重复文本")
        if self.option_tone_variety < 0.8:
            out.append(
                f"选项方向雷同：平均每组只有 {self.option_tone_variety:.1f} 种不同语气")
        if self.over_len:
            out.append(f"{self.over_len} 条消息明显超长")
        return out


def mechanical(session: Session, max_chars: int = 40) -> Mechanical:
    m = Mechanical()
    msgs = session.her_messages
    m.her_count = len(msgs)
    if not msgs:
        return m

    m.avg_len = sum(len(s) for s in msgs) / len(msgs)
    m.over_len = sum(1 for s in msgs if len(s) > max_chars)

    counts = Counter(msgs)
    m.duplicate_lines = [s for s, n in counts.items() if n > 1 and len(s) > 2]

    run = best = 1
    for a, b in zip(msgs, msgs[1:]):
        run = run + 1 if a[:1] == b[:1] else 1
        best = max(best, run)
    m.same_opener_runs = best if best >= 3 else 0

    m.banned_hits = [s for s in msgs if _BANNED.search(s)]
    m.assistant_hits = [s for s in msgs if _ASSISTANT.search(s)]
    m.concrete_ratio = sum(bool(_CONCRETE.search(s)) for s in msgs) / len(msgs)
    m.relational_ratio = sum(bool(_RELATIONAL.search(s)) for s in msgs) / len(msgs)

    if session.option_sets:
        dupes = sum(
            1 for opts in session.option_sets
            if len({o.text for o in opts}) < len(opts)
        )
        m.identical_option_sets = dupes
        m.option_tone_variety = sum(
            len({o.tone for o in opts}) for opts in session.option_sets
        ) / len(session.option_sets)
    return m


CRITIC_PROMPT = """你是一款恋爱游戏的对话质量评审。下面是一局自动对局的完整记录。

女主设定：高二女生林静姝。表面对所有人保持距离（高嶺の花），只对男主不是。
说话短、标点完整、**绝不用感叹号／波浪号／emoji／网络流行语**。
她的关心是短的、不解释的。她不擅长表达感情，所以一旦表达就极其直白。

关系阶段：S0 陌生 → S1 试探期 → S2 确认期 → S3 热恋期。
本局是 **{stage}**。

按下面七条逐项打分（1–5，5 最好），并**引用具体台词**说明问题。

1. **活人感** —— 她像不像一个有自己生活的人？还是像一个只会回话的机器？
2. **说事情 vs 说关系** —— 真实情侣大部分时间在聊第三件事（今天发生了什么），
   而不是一直在谈"我们俩"。她是不是一直在关心他／谈感情？
3. **阶段感** —— 这段对话读起来像 {stage} 吗？
   S0 该疏离克制，S3 该直接、会撒娇（但她的撒娇是命令句，不是"嘛~"）、会说废话。
4. **选项质量** —— 三个选项是不是走了**不同的关系动作**？
   还是三条都在关心她（那等于只有一个选项）？选项像不像真人打的字？
5. **内梗与回指** —— 她有没有提起之前发生过的事？
   「第二次提起同一件事」比「说十件新事」更亲密。
6. **重复与套路** —— 有没有反复用同一个句式／同一个开头／同一种回应？
7. **出戏** —— 有没有任何一句让你意识到这是 AI？（助理腔、说教、过度周到、
   编造事实、答非所问、突然很懂事）

严格输出 JSON：

{{
  "scores": {{"活人感":3, "说事情":2, "阶段感":4, "选项质量":3,
             "内梗":2, "重复":4, "出戏":4}},
  "worst": "最严重的一个问题，一句话",
  "problems": [
    {{"quote":"具体台词", "issue":"什么问题", "fix":"该怎么改"}}
  ],
  "good": ["做得对的地方，引用台词"],
  "verdict": "整体两三句话"
}}

problems 最多 6 条，挑最重要的。**必须引用原文**，不要泛泛而谈。"""


@dataclass(slots=True)
class Review:
    scores: dict[str, int] = field(default_factory=dict)
    worst: str = ""
    problems: list[dict[str, str]] = field(default_factory=list)
    good: list[str] = field(default_factory=list)
    verdict: str = ""
    raw: str = ""

    @property
    def average(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


async def review(
    session: Session,
    provider: DeepSeekProvider,
    recorder: UsageRecorder | None = None,
) -> Review:
    stage = f"{session.stage_end}（{session.preset}）"
    try:
        completion = await provider.complete(LLMRequest(
            messages=[
                Message("system", CRITIC_PROMPT.format(stage=stage)),
                Message("user", session.transcript()),
            ],
            task=Task.REFLECT, json_mode=True, max_tokens=2000,
        ))
    except LLMError as exc:
        log.error("评审失败：%s", exc)
        return Review(verdict=f"评审调用失败：{exc}")

    if recorder:
        recorder.record(completion)

    r = Review(raw=completion.text)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError:
        r.verdict = "评审输出不是 JSON"
        return r

    r.scores = {str(k): int(v) for k, v in (data.get("scores") or {}).items()
                if isinstance(v, (int, float))}
    r.worst = str(data.get("worst", ""))
    r.problems = [p for p in (data.get("problems") or []) if isinstance(p, dict)][:6]
    r.good = [str(g) for g in (data.get("good") or [])][:4]
    r.verdict = str(data.get("verdict", ""))
    return r
