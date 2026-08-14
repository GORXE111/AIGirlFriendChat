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
import statistics
from collections import Counter
from dataclasses import dataclass, field

from ..llm import DeepSeekProvider, LLMError, LLMRequest, Message, Task
from ..metrics import UsageRecorder
from .autoplay import Session

log = logging.getLogger(__name__)

# 省主语：句子直接以谓语／状语开头，没有「我／你／他」这类主语。
#
# 「今天有点长。」省了主语，「我今天很累。」没省 —— 这是她最硬的两个指纹之一
# （72%，见 content/craft/chinese-character-voice.md）。
#
# 用排除法：开头不是人称代词就算省了。粗，但这个指纹的差别足够大，
# 跑偏的时候会从 70% 掉到 30%，不需要精确到个位数。
#
# ⚠️ 必须写成交替而不是字符类 —— `[我你|我们]` 里的多字项是无效的，
# 它只会匹配单个字符。「我们」要靠「我」挡掉，但「咱们」得显式写。
_SUBJECTLESS = re.compile(r"^(?!我|你|他|她|咱|谁)")

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

    # ---- 语言指纹。她的坐标见 content/craft/chinese-character-voice.md ----
    #
    # 「句号／逗号 11:1」和「省主语 72%」是她最硬的两个指纹。
    # 这两个数偏了，说的就不是她了 —— 而且这是评审打分**测不出**的东西：
    # 评审看的是「像不像真人」，不是「像不像**这个**真人」。
    comma_ratio: float = 0.0
    """带逗号的消息占比。

    ⚠️ 不用「句号／逗号 11:1」那个原始比值 —— 她的逗号本来就少，分母一小
    比值就炸：实测同一个变体的两边能差出 5.6:1 和 15.5:1，纯噪声。
    换成有界的占比之后，24 局实测均 17%、σ 11%，能读了。
    """

    subjectless_ratio: float = 0.0

    len_stdev: float = 0.0
    """句长的标准差。

    真人的消息长短不一 —— 「嗯。」和一句三十字的抱怨会挨着出现。
    模型的通病是所有消息趋近同一个长度，读起来像在填表。
    均值正常但方差塌了，是很典型的机器味。
    """

    topic_spread: int = 0
    """整局提到过多少个不同的具体物。

    具体性只看「这条有没有具体物」，一整局翻来覆去说同一把伞也能拿高分。
    跨度看的是她的世界有多大。"""

    option_sets_seen: int = 0
    """看到过几组选项。0 的时候选项类指标全部无意义，不该报。"""

    option_text_variety: float = 0.0
    """三个选项之间的**字面**差异度（0–1）。

    `option_tone_variety` 只数 tone 标签，模型可以给三个不同标签配三句
    几乎一样的话 —— 标签合格，玩家读到的还是一个选项。这个直接比文本。
    """

    def problems(self) -> list[str]:
        # 没有消息就没有问题。
        #
        # 不加这层的话，空会话会报「具体性 0%」和「选项方向雷同」——
        # 那不是质量问题，是没数据。A/B 里会把「这一局崩了」算成两个质量缺陷。
        if not self.her_count:
            return []

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
        if self.option_sets_seen and self.option_tone_variety < 0.8:
            out.append(
                f"选项方向雷同：平均每组只有 {self.option_tone_variety:.1f} 种不同语气")
        if self.over_len:
            out.append(f"{self.over_len} 条消息明显超长")
        # 语言指纹。区间放得比目标宽 —— 这是查「跑偏」不是查「不精确」。
        # 阈值 35% 是照 24 局实测定的（均 17%、σ 11%）—— 抓离群，不抓噪声。
        if self.comma_ratio > 0.35:
            out.append(
                f"{self.comma_ratio:.0%} 的消息带逗号，她的基线是两成左右。"
                "逗号一多就变成了绵长的解释句，那不是她")
        if self.subjectless_ratio < 0.45:
            out.append(
                f"省主语只有 {self.subjectless_ratio:.0%}，她的指纹是 72%。"
                "「我今天很累」和「今天有点长」是两个人")
        if self.her_count > 1 and self.len_stdev < 3.0:
            out.append(
                f"句长方差只有 {self.len_stdev:.1f}，所有消息一样长 —— "
                "真人是「嗯。」和一长串挨着出现")
        if self.topic_spread < 5:
            out.append(
                f"话题跨度只有 {self.topic_spread} 个具体物，整局在原地打转")
        if self.option_sets_seen and self.option_text_variety < 0.55:
            out.append(
                f"选项字面重合度高（差异度 {self.option_text_variety:.0%}）："
                "tone 标签不同但话几乎一样，玩家读到的还是一个选项")
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

    # ---- 语言指纹 ----
    blob = "".join(msgs)
    m.comma_ratio = sum(1 for s in msgs if "，" in s or "、" in s) / len(msgs)
    m.subjectless_ratio = sum(bool(_SUBJECTLESS.match(s)) for s in msgs) / len(msgs)

    if len(msgs) > 1:
        m.len_stdev = statistics.stdev([len(s) for s in msgs])
    m.topic_spread = len({mo.group(0) for mo in _CONCRETE.finditer(blob)})

    if session.option_sets:
        dupes = sum(
            1 for opts in session.option_sets
            if len({o.text for o in opts}) < len(opts)
        )
        m.option_sets_seen = len(session.option_sets)
        m.identical_option_sets = dupes
        m.option_tone_variety = sum(
            len({o.tone for o in opts}) for opts in session.option_sets
        ) / len(session.option_sets)
        m.option_text_variety = statistics.fmean(
            [_text_variety([o.text for o in opts]) for opts in session.option_sets]
        )
    return m


def _text_variety(texts: list[str]) -> float:
    """一组选项之间的字面差异度，0（完全一样）到 1（毫无重合）。

    取两两之间字符集合的 Jaccard 距离的均值。够粗但够用 ——
    要抓的是「三条话几乎一样」这种明显情况，不是细微的语义差别。
    """
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return 1.0
    dists = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = set(texts[i]), set(texts[j])
            union = a | b
            dists.append(1 - len(a & b) / len(union) if union else 0.0)
    return statistics.fmean(dists)


def _critic_prompt(stage: str, character_id: str = "h01") -> str:
    """女主设定从 agent.yaml 读 —— 评审 prompt 不该硬编码某一个角色。"""
    from ..persona.agent_data import load_agent_data

    brief = load_agent_data(character_id).critic_brief or "见对局记录自行判断。"
    return CRITIC_PROMPT.replace("{brief}", brief).replace("{stage}", stage)


CRITIC_PROMPT = """你是一款恋爱游戏的对话质量评审。下面是一局自动对局的完整记录。

女主设定：{brief}

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
                Message("system", _critic_prompt(stage, session.character_id)),
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


# ---------------- 成对判优 ----------------
#
# 绝对打分（1–5）在 A/B 里测不出东西。实测 n=6 两个变体 p=0.77 / p=0.36 ——
# 同一份对局给 3 还是 4 全看评审当时怎么想，噪声盖过了真实差异。
#
# 成对比较把「这份有多好」换成「这两份哪个更好」。评审不需要维持一把稳定的
# 尺子，只需要分辨方向，对噪声的抵抗力高一个量级。
#
# ⚠️ LLM 评审有**位置偏好** —— 倾向于选先出现的那个。所以每一对都要
# 正反各判一次，两次结论一致才算数（见 `compare_pair`）。

COMPARE_PROMPT = """你在为一款恋爱游戏比较两份自动对局记录。

女主设定：{brief}

关系阶段：{stage}

下面是同一个场景下的两份对局。**判断哪一份更接近「一个有自己生活的真人」**，
而不是哪一份更礼貌、更周到、更会照顾人。

判断依据，按重要性排序：

1. **她像不像一个有自己生活的人** —— 会说自己的事，不是只会接话
2. **在说事情还是在说关系** —— 真实情侣大部分时间在聊第三件事
3. **有没有出戏** —— 助理腔、说教、过度周到、编造事实、答非所问
4. **选项是不是三个不同的关系动作** —— 三条都在关心她等于只有一个选项
5. **有没有内梗与回指** —— 第二次提起同一件事比说十件新事更亲密

⚠️ 不要因为一份更长、更热情、更体贴就选它。**过度周到本身就是扣分项。**
⚠️ 如果两份确实分不出高下，就选 "tie"，不要硬凑。

════════ A ════════
{a}

════════ B ════════
{b}

严格输出 JSON：

{{"winner": "A" 或 "B" 或 "tie",
  "why": "一句话，必须引用两边的具体台词说明差别",
  "confidence": 0.0 到 1.0}}"""


@dataclass(slots=True)
class Verdict:
    """一次成对比较。"""

    winner: str = "tie"
    """"A" | "B" | "tie"。"""

    why: str = ""
    confidence: float = 0.0
    error: str = ""


async def compare(
    a: str,
    b: str,
    provider: DeepSeekProvider,
    *,
    stage: str = "S3",
    character_id: str = "h01",
    recorder: UsageRecorder | None = None,
) -> Verdict:
    """判两份对局哪个更好。`a`/`b` 是 `Session.transcript()`。"""
    from ..persona.agent_data import load_agent_data

    brief = load_agent_data(character_id).critic_brief or "见对局记录自行判断。"
    prompt = (COMPARE_PROMPT
              .replace("{brief}", brief).replace("{stage}", stage)
              .replace("{a}", a).replace("{b}", b))
    try:
        completion = await provider.complete(LLMRequest(
            messages=[Message("system", prompt)],
            task=Task.REFLECT, json_mode=True, max_tokens=600,
        ))
    except LLMError as exc:
        return Verdict(error=str(exc))

    if recorder:
        recorder.record(completion)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError:
        return Verdict(error="判优输出不是 JSON")

    winner = str(data.get("winner", "tie")).strip().upper()
    return Verdict(
        winner=winner if winner in ("A", "B") else "tie",
        why=str(data.get("why", ""))[:300],
        confidence=max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
    )


async def compare_pair(
    on: str,
    off: str,
    provider: DeepSeekProvider,
    **kw,
) -> tuple[str, str]:
    """正反各判一次，消除位置偏好。

    返回 `(结论, 说明)`，结论是 `"on"` / `"off"` / `"tie"`。

    **两次判断不一致就算平局。** 一个位置偏好严重的评审会两次都选同一个
    位置，那两次的结论必然矛盾，正好被这条规则挡掉 —— 比事后统计偏好再
    校正简单，也不会把噪声当成信号。
    """
    first = await compare(on, off, provider, **kw)      # on 在 A 位
    second = await compare(off, on, provider, **kw)     # on 在 B 位

    if first.error or second.error:
        return "tie", f"判优失败：{first.error or second.error}"

    a_says = {"A": "on", "B": "off", "tie": "tie"}[first.winner]
    b_says = {"A": "off", "B": "on", "tie": "tie"}[second.winner]

    if a_says == b_says:
        return a_says, first.why
    return "tie", f"两次判断不一致（{a_says} vs {b_says}），按平局算"


def sign_test(wins: int, losses: int) -> float:
    """符号检验的双侧 p 值。平局不计入 —— 它们不提供方向信息。

    n 很小的时候直接算二项分布，不用近似。
    """
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb

    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)
