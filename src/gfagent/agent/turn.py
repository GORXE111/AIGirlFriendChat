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

TONES = ("关心", "直接", "玩笑", "试探", "回避", "陪着")

SCHEMA_HINT = """严格输出 JSON，不要任何额外文字：

{
  "messages": ["她发的第一条", "第二条（可选）"],
  "options": [
    {"text": "玩家可以说的话", "tone": "关心"},
    {"text": "另一种说法", "tone": "直接"},
    {"text": "第三种", "tone": "回避"}
  ],
  "outcome": null
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


def topic_instructions(stage: str = "S0") -> str:
    return TOPIC_INSTRUCTIONS + (
        "\n\n### 关系阶段的分寸（**硬约束**）\n"
        f"- {STAGE_OPTION_GUIDE.get(stage, STAGE_OPTION_GUIDE['S0'])}"
    )


TOPIC_INSTRUCTIONS = """玩家正要找林静姝说话。给他几个**今天可以聊什么**的话题。

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
- opener 是**他**说的话，不是她说的。不要套用林静姝的说话方式
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

# 每条选项要走**不同的关系动作**。三条都在关心她，等于只有一个选项。
STAGE_MOVES: dict[str, tuple[str, ...]] = {
    "S0": ("接住她的话", "问一件具体的事", "礼貌地撤"),
    "S1": ("关心她", "说自己的事", "开个小玩笑或岔开"),
    "S2": ("往前一步", "调侃她", "不追问，陪着"),
    "S3": ("调侃／打趣", "撒娇／耍赖／黏人", "直球／答应她／说想见她"),
}


def instructions(
    *,
    her_max_chars: int,
    her_max_messages: int,
    in_beat: bool,
    can_finish: bool,
    outcome_ids: tuple[str, ...],
    stage: str = "S0",
) -> str:
    """给模型的输出规格。"""
    moves = STAGE_MOVES.get(stage, STAGE_MOVES["S0"])
    parts = [
        "你在为一部恋爱游戏生成**一个回合**的内容：林静姝发出的消息，"
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
        "",
        "## 玩家的选项（options）",
        f"- 正好 {MAX_OPTIONS} 条",
        "- **这是玩家说的话，不是她说的。**"
        "玩家是个普通的高二男生 —— 说话自然、随意，"
        "**不要套用林静姝的说话方式**（不要都是短冷句、不要都不带标点）",
        "- 每条 4–20 字，口语，像真人在手机上打的字",
        f"- 三条要**明显不同**：语气不同、方向不同。可用的 tone：{'、'.join(TONES)}",
        "- 不要出现「（沉默）」这类舞台指示，选项必须是能直接发出去的话",
        "- 不要三条都在问问题",
        "- ⚠️ **选项不许编造没发生过的事。** 「你上次说要给我煮粥」这种伪回忆，"
        "只要记忆里没有，就是凭空捏造 —— 玩家会立刻发现自己"
        "「答应过」一件根本没答应的事。要回指往事，只能用上面列出的"
        "「你们之间发生过的事」，一个字都不能加。",
        "",
        "### 三条必须走**不同的关系动作**",
        "  三条都在关心她 ＝ 实际上只有一个选项。玩家选哪个都一样，就没得玩了。",
        f"  这个阶段的三个方向：{ ' ／ '.join(moves) }",
        "  一条对应一个方向，不要混。",
        "",
        "### 关系阶段的分寸（**硬约束**）",
        f"- {STAGE_OPTION_GUIDE.get(stage, STAGE_OPTION_GUIDE['S0'])}",
        "  超出当前阶段会毁掉可信度；**低于当前阶段一样糟** ——"
        "热恋期还在说「别硬撑」，玩家会觉得这两个人根本没在一起。",
    ]

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

    return plan
