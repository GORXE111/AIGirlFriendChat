"""一回合的生成：她说什么 ＋ 玩家能选什么。

**同一次调用一起生成**，选项才能贴住当下语境（她刚说了什么、她今天不对劲、
你上周三说过胃疼）。分两次调用会让选项和台词脱节。

关键约束：**选项是玩家的声音，不是她的。** 不能套她的语言指纹 ——
她说 8 字冷淡句，玩家是个普通高二男生，说话方式完全不同。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MAX_OPTIONS = 3

# ⚠️ tone 是**关系动作**，不是说话方式。
#
# 原来用「关心／直接／调侃」这类语气标签，结果三个选项在**关系上完全一样** ——
# 都是「体贴地回应她」，只是措辞不同。玩家选哪个都没差别，于是没有决策感、
# 没有恋爱感。
#
# galgame 的选项本质是：**这一步，关系要不要往前。**
TONES = ("往前", "守住", "后退", "越界")

# 崩溃期的选项。**这不是她说话的选项，是玩家对局面的处置。**
#
# 她崩着的时候只回得出一个字（见 overwhelm.broken_line）。如果这时候还给
# 三个正常选项，玩家会继续正常聊，而她继续只回「……」—— 读起来就是坏了。
#
# 换成局面选项，「她不理你」从一个疑似故障的空白，变成一个玩家能操作的处境。
SITUATION_TONE = "局面"

SITUATION_OPTIONS: tuple[tuple[str, str], ...] = (
    ("等一会。", "wait"),
    ("再说一句。", "push"),
    ("明天再找她。", "leave"),
)
"""(显示文本, 动作)。动作语义见 `Agent.choose` 的崩溃分支：

- `wait`  什么都不做，让时间走。**最安全，也最有效** —— 对应 moods.md
          「装作没发现，正常聊 → 她会自己缓，但更慢」的极简版
- `push`  再说一句。可能哄好，也可能更糟 —— 这是玩家唯一的主动权
- `leave` 今天到此为止。不惩罚，但也不加速
"""

TONE_MEANING = {
    "往前": "主动推进一点 —— 提议、答应、给承诺、说想见她",
    "守住": "停在原地 —— 接住她的话但不加码，安全但不推进",
    "后退": "给她空间 —— 岔开、不追问、把选择权还给她",
    "越界": "冒一点险 —— 说一句超出当前关系的话，可能被推开",
}

def tail_rules(stage: str = "S0") -> str:
    """历史后指令 —— 排在对话记录之后，紧贴输出位置。

    **只放最常被违反的几条，且只放祈使句。** 理由和反例留在 `instructions()`
    里说一次就够；这里是动手前的最后一眼。

    挑选依据是真机上反复出问题的那几条（自动对局的评审 agent 抓到过），
    不是「看起来最重要」的那几条：

    - 选项编造没发生过的事 —— 玩家会发现自己「答应过」根本没答应的事
    - 三条选项走同一个关系动作 —— 等于只有一个选项
    - 旁白／舞台指示混进台词
    - 选项套用她的说话方式 —— 玩家不是她

    ⚠️ 这段会随规则增加而膨胀。膨胀了就失去意义 —— 尾部的注意力优势是
    稀缺资源。**加新条目之前先删一条。**

    ## feeling 为什么在这

    A/B 实测（n=6，s3/galgamer）：feeling 放在 `instructions()` 中段时，
    七个评审维度里六个下降、具体性 −6 个百分点、机械问题 3→5。
    没有一项显著，但**没有一项对它有利**。

    怀疑是位置问题不是存在问题 —— 它在中段是模型要同时兼顾的第四件事。
    而 tail 的位置优势已经被同一轮 A/B 证实（具体性 +11pt，p=0.024）。
    所以整段挪过来**顶替**原来那句残缺的第 5 条，条数不变。
    """
    moves = "／".join(tone for tone, _ in STAGE_MOVES.get(stage, STAGE_MOVES["S0"]))
    return (
        "---\n\n"
        "**动手前最后确认（这几条最常出错）：**\n\n"
        "1. 选项里出现的往事，必须在上面的「发生过的事」或「悬着的事」里"
        "**逐字对得上**。对不上就不要提。\n"
        f"2. 三条选项是三个不同的关系动作（{moves}），不是三种措辞。"
        "把三条都发出去，关系要走向三个不同的地方。\n"
        "3. 她的消息里不要有括号、旁白、动作、心理活动。只有她打出来的字。\n"
        "4. 选项是**玩家**打的字 —— 自然、随意，不要写成她那种短冷句。\n"
        "5. feeling ＝ 他这句话让她的情绪变了多少。"
        "情绪名 → 增量，正数涨、负数消解；"
        "可用：累/开心/生气/委屈/慌/难过/紧张/松弛。\n"
        "   **大多数回合是 `{}`** —— 普通一来一回不改变任何人的情绪状态。"
        "单项不超过 0.5，最多两项。按他实际说的判，不要按剧情走向判。\n\n"
        "现在输出 JSON。"
    )


SCHEMA_HINT = """严格输出 JSON，不要任何额外文字：

{
  "messages": ["她发的第一条", "第二条（可选）"],
  "options": [
    {"text": "玩家可以说的话", "tone": "往前"},
    {"text": "另一种", "tone": "守住"},
    {"text": "第三种", "tone": "越界"}
  ],
  "outcome": null,
  "feeling": {}
}"""


@dataclass(slots=True)
class Option:
    text: str
    tone: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "tone": self.tone}


@dataclass(slots=True)
class TurnPlan:
    messages: list[str] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    outcome: str | None = None
    """桥段结局 id。非 None 表示这场戏演完了。"""

    feeling: dict[str, float] = field(default_factory=dict)
    """这一轮他的话让她的情绪怎么变。情绪名 → 增量（负数＝消解）。

    **不加一次调用** —— 跟台词同一次出。缺字段就是没变化。
    """

    raw: str = ""


@dataclass(slots=True)
class Topic:
    """开场话题。玩家打开聊天时先选今天聊什么。"""

    title: str
    opener: str
    """选中后作为玩家的第一句话发出去。"""

    beat_id: str | None = None
    """非空则同时开启这场戏。"""

    def as_dict(self) -> dict[str, str | None]:
        return {"title": self.title, "opener": self.opener, "beat_id": self.beat_id}


def topic_instructions(stage: str = "S0", name: str = "她") -> str:
    # 用 replace 不用 format —— 模板末尾有 JSON 示例，花括号会被 format 吃掉。
    return TOPIC_INSTRUCTIONS.replace("{name}", name) + (
        "\n\n### 关系阶段的分寸（**硬约束**）\n"
        f"- {STAGE_OPTION_GUIDE.get(stage, STAGE_OPTION_GUIDE['S0'])}"
    )


TOPIC_INSTRUCTIONS = """玩家正要找{name}说话。给他几个**今天可以聊什么**的话题。

每个话题两部分：
- `title`：4–8 字的短标签，概括这个话题（给玩家看的按钮文字）
- `opener`：他实际发出去的第一句话，8–20 字，口语，像真人在手机上打的

要求：

- **优先用你已经知道的事** —— 他上次说过的、你们之间发生过的、今天的日子和天气。
  「问问他胃还疼不疼」比「随便聊聊」好得多
- **至少两个话题要是「一件具体的事」，不是「一种感受」。**
  ✅「食堂今天那个菜」「昨天那只猫」「月考排名出来了」
  ❌「聊聊心情」「关心一下她」「随便说说」
- 几个话题要**方向不同**：一件外部的事 / 他自己的事 / 上次没聊完的
- opener 是**他**说的话，不是她说的。不要套用{name}的说话方式
- 不要出现旁白、括号、引号

严格输出 JSON：

{"topics": [
  {"title": "问他胃疼", "opener": "你胃还疼吗"},
  {"title": "说说今天", "opener": "今天下雨了，你带伞了没"},
  {"title": "随便找话", "opener": "在干嘛"}
]}"""


def parse_topics(text: str, want: int = 3) -> list[Topic]:
    import json as _json

    blob = (text or "").strip()
    if "```" in blob:
        start = blob.find("```")
        end = blob.find("```", start + 3)
        inner = blob[start + 3:end if end > 0 else len(blob)]
        blob = inner.split("\n", 1)[-1] if inner.startswith("json") else inner
    lo, hi = blob.find("{"), blob.rfind("}")
    if lo >= 0 and hi > lo:
        blob = blob[lo:hi + 1]

    try:
        data = _json.loads(blob)
    except _json.JSONDecodeError:
        log.error("话题输出不是 JSON：%s", text[:250])
        return []

    out: list[Topic] = []
    for t in (data.get("topics") or [])[:want]:
        if not isinstance(t, dict):
            continue
        opener = str(t.get("opener", "")).strip()
        if not opener:
            continue
        out.append(Topic(
            title=str(t.get("title") or opener[:8]).strip(),
            opener=opener,
        ))
    return out


STAGE_OPTION_GUIDE: dict[str, str] = {
    "S0": (
        "**他们几乎还是陌生人**（考场借过一支笔，好友躺了两周）。\n"
        "  可选：客气的搭话、问一件具体的小事、找个由头、准备撤退。\n"
        "  **绝不能出现**：约见面、约吃饭、要联系方式、调情、"
        "「你真好看」、任何暗示已经很熟的话。"
    ),
    "S1": (
        "**刚开始说得上话。** 可以问她的日常、可以关心、可以开个小玩笑。\n"
        "  **不能**约见面、不能表白、不能开亲密玩笑。"
    ),
    "S2": (
        "**已经算朋友了，而且是特别的那种。** 可以约见面、可以互相调侃、"
        "可以稍微越界然后自己找补。\n"
        "  **不要**说「我喜欢你」这种定性的话 —— 还没到。"
    ),
    "S3": (
        "**在一起了，正热。** 这里最容易写砸 —— 不要写成"
        "「体贴的成年人在照顾她」。\n"
        "  热恋的男朋友会**打趣、耍赖、黏人、吃醋、说废话、直球**，"
        "不是三条都在关心她。\n"
        "  可选：调侃她 / 撒娇耍赖 / 直接说想见她 / 问刚才谁找她 / "
        "没话找话就想多聊两句 / 答应她任何事。\n"
        "  ⚠️ **不要三条都是「你早点睡」「别硬撑」「我陪着你」这种照顾型。**"
        "那是护工，不是男朋友。"
    ),
}

# 三条选项必须是**三个不同的关系动作**，不是三种措辞。
#
# 判据：把三条选项都发出去，关系会走向三个不同的地方吗？
# 如果答案是「差不多」，那就等于只有一个选项。
STAGE_MOVES: dict[str, tuple[tuple[str, str], ...]] = {
    "S0": (
        ("往前", "多问一句她的事，让对话继续"),
        ("守住", "客气地接住，不深入"),
        ("后退", "找个由头收尾，不打扰"),
    ),
    "S1": (
        ("往前", "关心她一句，或者说一件自己的事换她的"),
        ("守住", "接话但不深入"),
        ("越界", "问一件她还没主动说过的事 —— 可能被挡回来"),
    ),
    "S2": (
        ("往前", "约她、答应她、往前推一步"),
        ("守住", "陪着但不追问"),
        ("越界", "调侃她，或者戳破一件她以为你没看出来的事"),
    ),
    "S3": (
        ("往前", "直球 —— 答应她、说想见她、给承诺"),
        ("守住", "耍赖／黏着不放，但不推进"),
        ("越界", "戳她一下 —— 吃醋、翻旧账、说一句让她慌的话"),
    ),
}


def instructions(
    *,
    her_max_chars: int,
    her_max_messages: int,
    in_beat: bool,
    can_finish: bool,
    outcome_ids: tuple[str, ...],
    stage: str = "S0",
    name: str = "她",
) -> str:
    """给模型的输出规格。"""
    moves = STAGE_MOVES.get(stage, STAGE_MOVES["S0"])
    parts = [
        f"你在为一部恋爱游戏生成**一个回合**的内容：{name}发出的消息，"
        "以及玩家接下来可以选的几句话。",
        "",
        "## 她的消息（messages）",
        f"- 每条不超过 {her_max_chars} 字，最多 {her_max_messages} 条",
        "- 严格遵守上面定义的她的说话方式",
        "- 想到什么发什么：一条说得完就一条，还有下一句就分成两条",
        "- 只写她说的话。不写旁白、动作、心理活动、括号内容",
        f"- **按 {stage} 阶段来写。** 见人设里「关系越近，规则越松」那一节 ——"
        "越到后面她越直接、越会主动、越会说没用的话。"
        "S3 还写得跟 S1 一样冷，等于这段关系白推了。",
        "- **说事情，不要只说关系。** 大部分时候她该在转述一件具体的事"
        "（风扇坏了 / 周老师说了什么 / 她妈又转文章），而不是在谈你们俩"
        "（你累不累 / 我陪着你）。见「日常聊天的质感」那一节。",
        "- **优先回指已经发生过的事**，不要每次都造新的。"
        "第二次提起同一件事，比说十件新事更亲密 —— 那就是内梗。",
        *(
            [
                "",
                "### 说出口才后悔的那句",
                "  她有时候会说一句**越界的话，发出去才觉得不该说**"
                "（试探他、暴露自己想他、显得太在意）。",
                "  这种句子在**最后一条**的末尾加上 `[收回]`，"
                "系统会让她真的撤回它。",
                "  例：`我妈今天值夜班。[收回]`",
                "  ⚠️ 不常用。她大部分话是想清楚了才说的 —— "
                "每回合都收回，那不是害羞，是神经质。"
                "标记只加在真的越了界的那句上，普通的日常话不要加。",
            ]
            if stage in ("S1", "S2") else []
        ),
        "",
        "## 玩家的选项（options）",
        f"- 正好 {MAX_OPTIONS} 条",
        "- **这是玩家说的话，不是她说的。**"
        "玩家是个普通的高二男生 —— 说话自然、随意，"
        f"**不要套用{name}的说话方式**（不要都是短冷句、不要都不带标点）",
        "- 每条 4–20 字，口语，像真人在手机上打的字",
        "- 不要出现「（沉默）」这类舞台指示，选项必须是能直接发出去的话",
        "- 不要三条都在问问题",
        "- ⚠️ **选项不许编造没发生过的事。** 「你上次说要给我煮粥」这种伪回忆，"
        "只要记忆里没有，就是凭空捏造 —— 玩家会立刻发现自己"
        "「答应过」一件根本没答应的事。要回指往事，只能用上面列出的"
        "「你们之间发生过的事」和「还悬着的事」，一个字都不能加。",
        "",
        "### 三条＝三个**关系动作**，不是三种措辞（最重要的一条）",
        "  这是恋爱游戏，玩家选的是「**这一步，关系要不要往前**」，"
        "不是「我用什么语气说」。",
        "  三条都是「体贴地回应她」，只是措辞不同 ＝ 实际上只有一个选项，"
        "玩家选哪个都没差别，就没有恋爱感了。",
        "",
        "  这个阶段的三条：",
        *[f"    · **{tone}** —— {desc}" for tone, desc in moves],
        "",
        "  `tone` 字段就填上面的动作名。",
        "  **判据：把三条都发出去，关系会走向三个不同的地方吗？**"
        "如果答案是「差不多」，重写。",
        "",
        "### 情绪价值从哪来",
        "  日常琐事是**载体**，不是内容。「今天风扇坏了」本身没有价值；"
        "「今天风扇坏了，热得我一直在想你那瓶冰水」才有。",
        "  每条选项都该指向下面至少一样：",
        "    · **被需要** —— 她需要他，不是他在照顾她",
        "    · **独占** —— 只有他能这么跟她说话",
        "    · **被记住** —— 回指一件具体的旧事",
        "    · **心慌** —— 打破她的规律，让她一时接不上",
        "  三条都只是「关心她的身体」＝ 零情绪价值。那是护工，不是恋人。",
        "",
        "### 关系阶段的分寸（**硬约束**）",
        f"- {STAGE_OPTION_GUIDE.get(stage, STAGE_OPTION_GUIDE['S0'])}",
        "  超出当前阶段会毁掉可信度；**低于当前阶段一样糟** ——"
        "热恋期还在说「别硬撑」，玩家会觉得这两个人根本没在一起。",
    ]

    # feeling 的完整规格在 `tail_rules()` 里 —— 放在对话记录之后而不是这里。
    # A/B 显示它在中段会拖累其他输出，理由见 `tail_rules` 的 docstring。

    if in_beat:
        parts += [
            "",
            "## 这场戏",
            "- 按上面的骨架推进，但**不要把「她不会说的」直接说出来**",
            "- 一回合只推进一点点。不要一次演完",
        ]
        if can_finish and outcome_ids:
            parts += [
                f"- 如果这场戏已经演到了收尾，把 outcome 设成 "
                f"{ '、'.join(repr(o) for o in outcome_ids) } 之一；"
                "否则 outcome 必须是 null",
            ]
        else:
            parts.append("- outcome 这一轮必须是 null（戏还没演够）")
    else:
        parts += ["", "## 这是日常闲聊", "- 没有剧情任务，自然聊", "- outcome 固定 null"]

    parts += ["", SCHEMA_HINT]
    return "\n".join(parts)


def parse(text: str, outcome_ids: tuple[str, ...]) -> TurnPlan:
    """解析模型输出。容错：模型偶尔会包 ```json 或加前言。"""
    plan = TurnPlan(raw=text)
    blob = (text or "").strip()

    if "```" in blob:
        start = blob.find("```")
        end = blob.find("```", start + 3)
        inner = blob[start + 3:end if end > 0 else len(blob)]
        blob = inner.split("\n", 1)[-1] if inner.startswith("json") else inner

    lo, hi = blob.find("{"), blob.rfind("}")
    if lo >= 0 and hi > lo:
        blob = blob[lo:hi + 1]

    try:
        data: dict[str, Any] = json.loads(blob)
    except json.JSONDecodeError:
        log.error("回合输出不是 JSON：%s", text[:300])
        return plan

    msgs = data.get("messages")
    if isinstance(msgs, str):
        msgs = [msgs]
    plan.messages = [str(m).strip() for m in (msgs or []) if str(m).strip()]

    for o in (data.get("options") or [])[:MAX_OPTIONS]:
        if isinstance(o, str):
            plan.options.append(Option(text=o.strip()))
        elif isinstance(o, dict) and o.get("text"):
            plan.options.append(
                Option(text=str(o["text"]).strip(), tone=str(o.get("tone", "")))
            )

    outcome = data.get("outcome")
    if isinstance(outcome, str) and outcome in outcome_ids:
        plan.outcome = outcome
    elif outcome not in (None, "", "null"):
        log.warning("模型给了未知结局 %r，忽略", outcome)

    feeling = data.get("feeling")
    if isinstance(feeling, dict):
        # 最多取两种。同时动五种情绪等于没动 —— 而且模型很容易在每轮
        # 都把所有情绪列一遍，那样情绪会漂。
        for name, delta in list(feeling.items())[:2]:
            try:
                value = float(delta)
            except (TypeError, ValueError):
                continue
            if abs(value) >= 0.01:
                plan.feeling[str(name)] = value

    return plan
