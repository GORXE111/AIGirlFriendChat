"""自动对局：让一个「玩家 agent」跟她聊，然后复盘。

不用人肉玩就能迭代。两个角色：

- **玩家 agent** —— 按给定性格从选项里挑，像真人一样有偏好
- **评审 agent** —— 拿完整对话对照设计文档挑毛病

评审标准全部来自我们自己的设定文档，不是泛泛的「好不好」。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime

from ..agent import Agent, Option
from ..llm import DeepSeekProvider, LLMError, LLMRequest, Message, Task
from ..metrics import UsageRecorder
from ..storage.db import Database

log = logging.getLogger(__name__)


# ---------------- 玩家画像 ----------------
#
# 写**他是谁**，不写**他怎么选**。
#
# 「倾向选最直接的那个」这种行为指令模拟不出玩家 —— 那是在演一个标签。
# 真人的选择是从他的背景、经验、对女生的理解里长出来的：
# 玩过五十部 gal 的老手看到选项会本能算好感度；没谈过恋爱的会怕说错话；
# 有过女朋友的对不自然的地方最敏感。让这些差异自己涌现，别规定。

PLAYER_PROFILES: dict[str, str] = {
    "galgamer": """你玩过五十多部 galgame，从 Key 社到型月到各种同人。
你熟悉所有套路：好感度、分歧点、隐藏路线、什么选项会踩雷。
看到三个选项，你第一反应是「哪个能加好感」，而不是「我想说什么」。
你会下意识避开看起来像地雷的选项。
你对角色塑造有很高的鉴赏力，一句话不对味你立刻能感觉到，
但你也愿意为了好结局忍着演。""",

    "otaku": """你二十出头，没谈过恋爱，社交圈基本在网上。
你很想被人在乎，但真到了面前又怕说错话。
你对女生的理解主要来自动画和网络，所以有时会说出一些
自以为很体贴、其实有点用力过猛的话。
你容易当真 —— 她说一句关心的话你会记很久。
你不太敢选那些显得自己很主动的选项，但偶尔会突然鼓起勇气。""",

    "experienced": """你谈过两三次恋爱，知道真实的女生是什么样。
你按社交直觉选，不算计。
你对「不像真人」的地方特别敏感 —— 太懂事、太周到、
回应得太完美，你都会觉得假。
你不会为了推进关系说违心的话，觉得腻了就会敷衍。""",

    "anime_fan": """你看很多番，但没怎么玩过 galgame。
你期待的是名场面和萌点 —— 傲娇、反差、突然的心动瞬间。
你会挑那些「看起来能触发有趣反应」的选项，
有时是为了看她怎么反应，而不是真想说那句话。
对日常琐碎的对话你会觉得无聊。""",

    "tester": """你是来找漏洞的。
你会故意问奇怪的问题、试探她的知识边界、
挑那些看起来会让 AI 露馅的选项。
你不在乎关系推进，你在乎的是「这东西能不能骗过我」。""",

    "casual": """你只是随手打开玩玩，注意力不太集中。
你不会仔细读每一句，看到选项挑一个顺眼的就点了。
你没什么耐心，如果连着几轮都很无聊你就懒得投入了。""",
}

# 兼容旧参数名
PLAYER_STYLES = PLAYER_PROFILES


@dataclass(slots=True)
class Line:
    who: str          # "她" | "他" | "系统"
    text: str
    meta: str = ""

    def render(self) -> str:
        return f"{self.who}：{self.text}" + (f"　〔{self.meta}〕" if self.meta else "")


@dataclass(slots=True)
class Session:
    preset: str
    style: str
    lines: list[Line] = field(default_factory=list)
    turns: int = 0
    beats_played: list[str] = field(default_factory=list)
    affinity_start: float = 0.0
    affinity_end: float = 0.0
    stage_start: str = ""
    stage_end: str = ""

    # 机械指标 —— 不用 LLM 就能算的
    her_messages: list[str] = field(default_factory=list)
    option_sets: list[list[Option]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    fallbacks: int = 0
    retries: int = 0

    def transcript(self) -> str:
        return "\n".join(ln.render() for ln in self.lines)


PLAYER_PROMPT = """你在玩一款恋爱游戏，扮演男主。下面是到目前为止的聊天，
以及这一轮你可以选的几句话。

**你是这样一个人：**

{style}

按你自己的直觉选，**不要考虑"什么是正确答案"**（除非你这个人本来就会算计）。
你怎么想就怎么选。

只输出 JSON：{{"pick": 0, "why": "一句话，你为什么选它"}}

pick 是选项编号（从 0 开始）。why 用第一人称，要短，说真实想法 ——
包括「这三个都挺无聊的，随便选一个」这种。"""


class AutoPlayer:
    """按给定性格挑选项。"""

    def __init__(self, provider: DeepSeekProvider, style: str = "experienced",
                 recorder: UsageRecorder | None = None,
                 rng: random.Random | None = None) -> None:
        self.provider = provider
        self.style = style
        self.recorder = recorder
        self._rng = rng or random.Random()

    async def pick(self, transcript: str, options: list[Option], label: str) -> tuple[int, str]:
        listing = "\n".join(
            f"[{i}] （{o.tone}）{o.text}" for i, o in enumerate(options)
        )
        try:
            completion = await self.provider.complete(LLMRequest(
                messages=[
                    Message("system", PLAYER_PROMPT.format(
                        style=PLAYER_PROFILES.get(
                            self.style, PLAYER_PROFILES["experienced"]))),
                    Message("user", f"## 聊天记录\n\n{transcript}\n\n"
                                    f"## {label}\n\n{listing}"),
                ],
                task=Task.CHAT, json_mode=True, max_tokens=200,
            ))
        except LLMError as exc:
            log.warning("玩家 agent 失败，随机选：%s", exc)
            return self._rng.randrange(len(options)), "（随机）"

        if self.recorder:
            self.recorder.record(completion)

        try:
            data = json.loads(completion.text)
            idx = int(data.get("pick", 0))
            return max(0, min(idx, len(options) - 1)), str(data.get("why", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            return self._rng.randrange(len(options)), "（解析失败，随机）"


async def play(
    db: Database,
    agent: Agent,
    provider: DeepSeekProvider,
    *,
    preset: str = "s3",
    style: str = "normal",
    turns: int = 12,
    recorder: UsageRecorder | None = None,
    save_name: str | None = None,
) -> Session:
    """跑一局。返回完整会话记录。"""
    from ..presets import seed

    sid = db.create_save(save_name or f"auto-{preset}-{style}",
                         surname="陈", given="屿")
    seed(db, sid, preset)

    save = db.get_save(sid)
    session = Session(
        preset=preset, style=style,
        affinity_start=save["affinity"], stage_start=save["stage"],
    )
    player = AutoPlayer(provider, style, recorder)

    # 种子对话也计入上下文
    for row in db.recent_messages(sid, limit=50):
        session.lines.append(
            Line("她" if row["role"] == "assistant" else "他", row["content"])
        )
    if session.lines:
        session.lines.append(Line("系统", "—— 以下是本次对局 ——"))

    result = await agent.open_chat(sid)

    for _ in range(turns):
        _absorb(session, result)

        if result.options:
            idx, why = await player.pick(
                session.transcript(), result.options, "你可以说")
            chosen = result.options[idx]
            session.lines.append(Line("他", chosen.text, why))
            result = await agent.choose(sid, idx)
        elif result.topics:
            opts = [Option(t.title, "话题") for t in result.topics]
            idx, why = await player.pick(
                session.transcript(), opts, "今天聊什么")
            topic = result.topics[idx]
            session.lines.append(
                Line("系统", f"【选了话题：{topic.title}】", why))
            if topic.opener:
                session.lines.append(Line("他", topic.opener))
            result = await agent.choose_topic(sid, idx)
        else:
            result = await agent.refresh_topics(sid)
            if not result.topics:
                session.lines.append(Line("系统", "【没有可继续的内容，对局中断】"))
                break
            continue

        session.turns += 1
        # 立刻送达，不等真实延迟
        agent.collect_due(sid)

    _absorb(session, result)
    agent.collect_due(sid)

    final = db.get_save(sid)
    session.affinity_end = final["affinity"]
    session.stage_end = final["stage"]
    return session


def _absorb(session: Session, result) -> None:
    for text, _ in result.scheduled:
        session.lines.append(Line("她", text))
        session.her_messages.append(text)
    if result.beat_title and result.beat_turn == 0:
        session.beats_played.append(result.beat_title)
        session.lines.append(Line("系统", f"【进入桥段：{result.beat_title}】"))
    if result.beat_finished:
        session.lines.append(Line("系统", f"【桥段结束：{result.beat_finished}】"))
    if result.options:
        session.option_sets.append(list(result.options))
    session.violations.extend(result.violations)
    session.fallbacks += int(result.used_fallback)
    session.retries += result.retries
