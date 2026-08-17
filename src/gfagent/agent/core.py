"""Agent 本体 —— 选项制。

玩家不打字，从三个选项里选。**她的台词和玩家的选项由同一次调用一起生成**，
选项才能贴住当下语境。

一回合：

    读状态 → 情绪衰减 → 决定演哪场戏 → 装配 prompt
      → 生成（她的消息 ＋ 玩家选项 ＋ 结局判定）
      → 后处理（清洗／校验／重试／兜底）→ 排延迟 → 落库

**桥段（Beat）是编剧写的骨架，AI 在骨架内填肉。** 见 content/beats/。
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..beats import Beat, BeatProgress, get_beat, pick_her_beat, player_beats
from ..life import today_for
from ..llm import Completion, DeepSeekProvider, LLMError, LLMRequest, Message, Task
from ..memory.retrieval import context_keywords, rank_episodes, rank_facts
from ..metrics import UsageRecorder
from ..output.postprocess import fallback, process
from ..output.slips import apply_regret, apply_typo, strip_regret_mark
from ..persona.agent_data import load_agent_data
from ..persona.loader import load_card
from ..prompt import PromptBuilder, StablePrefix, VolatileContext
from ..schedule import ScheduleEngine
from ..config import get_settings
from ..state.crisis import Level, her_lines, resources
from ..state.moods import behavior_note
from ..state.overwhelm import (
    MAX_TURN_DELTA,
    PUSH_SETBACK,
    RUNG_MINUTES,
    SOOTHE_SPEEDUP,
    Overwhelm,
    Rung,
    behavior_note as overwhelm_note,
    broken_line,
    check,
    delay_multiplier,
)
from ..state.models import (
    STAGE_BEHAVIOR,
    Emotion,
    EmotionState,
    Stage,
    stage_for_affinity,
)
from ..storage.db import Database, parse_ts, utcnow
from ..vocab import _CONCRETE as _CONCRETE_THING
from . import gates
from .turn import (
    MAX_OPTIONS,
    SITUATION_OPTIONS,
    SITUATION_TONE,
    Option,
    Topic,
    TurnPlan,
    instructions,
    parse,
    parse_topics,
    tail_rules,
    topic_instructions,
)

log = logging.getLogger(__name__)

HISTORY_TURNS = 24
MAX_RETRIES = 1

TURN_MAX_TOKENS = 900
"""一回合的输出上限。

⚠️ 不能用 `Task.CHAT` 的 300 —— 那是给「她说一句短话」设的。这里要输出
**她的消息 ＋ 3 个玩家选项 ＋ 结局判定**的完整 JSON，300 token 会被截断，
解析失败后全程走兜底（真机踩过）。
"""

FALLBACK_OPTIONS = (
    Option("嗯", "回避"),
    Option("你今天怎么样", "关心"),
    Option("那我不打扰你了", "回避"),
)


FALLBACK_TOPICS = (
    Topic("随便找话", "在干嘛"),
    Topic("问问她", "今天怎么样"),
    Topic("说自己的事", "我今天有点累"),
)


@dataclass(slots=True)
class TurnResult:
    scheduled: list[tuple[str, datetime]] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    """非空表示现在是「今天聊什么」的选题阶段。"""

    beat_id: str | None = None
    beat_title: str = ""
    beat_turn: int = 0
    beat_finished: str = ""
    """结局的 label，非空表示这场戏演完了。"""

    stage: Stage = Stage.S0
    affinity: float = 0.0
    emotion_note: str = ""
    delay_seconds: int = 0

    feeling: dict[str, float] = field(default_factory=dict)
    """这一轮他的话让她的情绪变了多少。空 ＝ 没戳到。"""

    overwhelm: str = ""
    """崩溃档位（`Rung.label`）。空 ＝ 没崩。非空时 options 是局面选项。"""

    situation: str = ""
    """崩溃期玩家做了什么：wait / push_helped / push_backfired / leave。"""

    crisis: str = ""
    """他说了重话：HEAVY / DANGER。空 ＝ 没有。见 state/crisis.py。"""

    resources: list[dict[str, str]] = field(default_factory=list)
    """系统层的援助资源。**只在 DANGER 时非空，且不是她的台词** ——
    前端必须跟对话区分开呈现。"""

    completion: Completion | None = None
    violations: list[str] = field(default_factory=list)
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)
    """(被丢掉的原文, 违规类型)。诊断用 —— 没有它就只能猜违规是真是假。"""

    cleaned: list[str] = field(default_factory=list)
    slips: list[str] = field(default_factory=list)
    """本回合发生的手滑／撤回，用于观测。"""

    dropped_repeats: list[str] = field(default_factory=list)
    """被丢掉的逐字重复。非空说明她又想复读了 —— 观测用，不是错误。"""
    retries: int = 0
    used_fallback: bool = False
    raw_text: str = ""


def _recovery_opener(emo: Emotion, stage: Stage, character_id: str = "h01") -> str:
    """她缓过来之后主动发的第一条。**这就是她的道歉。**

    句子在 `content/characters/<id>/agent.yaml` 的 recovery_openers。
    """
    pool = load_agent_data(character_id).recovery_pool(stage.value)
    # 用情绪名的字符和取下标：`hash()` 在 Python 里对 str 是加盐的，
    # 进程间不一致，同一次崩溃重算会得到不同的句子。
    return pool[sum(map(ord, emo.value)) % len(pool)]


class Agent:
    def __init__(
        self,
        db: Database,
        provider: DeepSeekProvider,
        recorder: UsageRecorder | None = None,
        schedule: ScheduleEngine | None = None,
        *,
        delay_scale: float = 1.0,
        max_delay_seconds: int = 10 * 60,
        rng: random.Random | None = None,
    ) -> None:
        self.db = db
        self.provider = provider
        self.recorder = recorder
        self.schedule = schedule or ScheduleEngine()
        self.delay_scale = delay_scale
        self.max_delay_seconds = max_delay_seconds
        self._rng = rng or random.Random()

    # ---------------- 存档状态 ----------------

    @staticmethod
    def _flags(save: dict) -> set[str]:
        try:
            return set(json.loads(save.get("flags") or "[]"))
        except json.JSONDecodeError:
            return set()

    @staticmethod
    def _progress(save: dict) -> BeatProgress:
        try:
            return BeatProgress.from_dict(json.loads(save.get("beat_progress") or "{}"))
        except json.JSONDecodeError:
            return BeatProgress()

    @staticmethod
    def _pending_options(save: dict) -> list[Option]:
        try:
            raw = json.loads(save.get("pending_options") or "[]")
        except json.JSONDecodeError:
            return []
        return [Option(o.get("text", ""), o.get("tone", "")) for o in raw if o.get("text")]

    @staticmethod
    def _pending_topics(save: dict) -> list[Topic]:
        try:
            raw = json.loads(save.get("pending_topics") or "[]")
        except json.JSONDecodeError:
            return []
        return [
            Topic(t.get("title", ""), t.get("opener", ""), t.get("beat_id"))
            for t in raw if t.get("title")
        ]

    # ---------------- prompt 装配 ----------------

    def _stable(self, save: dict) -> StablePrefix:
        # 按阶段装配：S3 的直球台词不该出现在 S0 的语气样本里。
        # 只有 4 个变体，每个仍是稳定前缀，前缀缓存照常命中。
        card = load_card(save["character_id"], save["stage"])
        persona, lexicon = card.stable_text()

        surname = (save["player_surname"] or "").strip()
        given = (save["player_given"] or "").strip()
        player = ""
        if surname or given:
            player = (
                f"# 对方\n\n同校同年级不同班的男生。姓 {surname or '（未填）'}，"
                f"名 {given or '（未填）'}，全名{surname}{given}。\n"
                "她基本不称呼他。只在明确需要时使用上面的姓名，不要自行改写或起外号。"
            )
        return StablePrefix(persona=persona, lexicon=lexicon, facts=player, memory="")

    def _volatile(
        self, save: dict, emotions: EmotionState, beat: Beat | None, beat_turn: int,
        *, notes: str | None = None, situation: str = "",
    ) -> VolatileContext:
        """`situation` 是**这一轮的临时处境**，只进易变层。

        跟 `notes` 不同：`notes` 会**替换**整个输出规格（`_offer_topics`
        用它换成话题指令），`situation` 是往状态块里**加**一句。
        决策探针用它把「他三天没回消息」这种前提摆给她，
        而不动人设卡和输出规格。
        """
        now_local = self.schedule.now_local()
        stage = Stage(save["stage"])
        behavior = STAGE_BEHAVIOR[stage]

        blocks: list[str] = [
            *( [f"**此刻的处境**：{situation}"] if situation else [] ),
            f"关系阶段：{stage.value}（{stage.label}）。好感 {save['affinity']:.0f}/100。",
            f"称呼对方：{behavior.address_player}。"
            + ("可以使用「我们」。" if behavior.allow_we else "不要使用「我们」。"),
            emotions.describe(),
        ]

        # 只报「情绪：生气（明显）」是不够的 —— 模型不知道生气该怎么演，
        # 结果生气和不生气看起来一样。这里给具体动作。
        mood = behavior_note(emotions.active())
        if mood:
            blocks.append(mood)

        # 恢复期覆盖常规情绪演出。moods.md 的阶梯是「先恢复长度，再恢复标点，
        # 最后才恢复主动」—— 不注入的话，缓过来的两级跟平时完全一样，
        # 那这个阶梯就只存在于代码里，玩家一点都感觉不到。
        broken = Overwhelm.from_json(save.get("overwhelm"))
        if broken is not None:
            note = overwhelm_note(broken.rung(), broken.emo)
            if note:
                blocks.append(note)

        # 检索按「当前在聊什么」打分，不是简单取最新的。
        # 只取最新的话，记忆一多，早期的重要事件就永远拿不到 ——
        # 玩家第 30 天提起第 3 天那件事，她会想不起来。
        recent_texts = [
            r["content"] for r in
            self.db.recent_messages(save["id"], limit=6, delivered_only=False)
        ]
        ctx = context_keywords(*recent_texts)

        facts = rank_facts(self.db.get_facts(save["id"], limit=200), context=ctx)
        if facts:
            blocks.append("你知道的关于他的事：\n"
                          + "\n".join(f"- {f['content']}" for f in facts))

        scored = rank_episodes(
            self.db.get_episodes(save["id"], limit=300),
            now=now_local, context=ctx, limit=12,
        )
        if scored:
            # 打分选出来之后按时间正序排 —— 否则她会觉得事情的先后是乱的
            picked = sorted(scored, key=lambda s: s.row["happened_at"])
            lines = []
            for s in picked:
                when = parse_ts(s.row["happened_at"]).astimezone(now_local.tzinfo)
                delta = (now_local.date() - when.date()).days
                if delta == 0:
                    label = "今天"
                elif delta == 1:
                    label = "昨天"
                elif delta < 7:
                    label = f"{delta}天前（周{'一二三四五六日'[when.weekday()]}）"
                else:
                    label = f"{when:%m月%d日}"
                lines.append(f"- {label}：{s.summary}")
            blocks.append(
                "你们之间发生过的事（**时间要记准，她的在意就体现在记得日期上**）：\n"
                + "\n".join(lines)
            )

        # 悬念事项 —— 每次对话的引力中心。
        # 没有它，每次打开都是从零开始的闲聊：聊什么都行，聊什么都不重要。
        threads = self.db.get_threads(save["id"])
        if threads:
            lines = []
            for t in threads:
                age = (now_local - parse_ts(t["created_at"])
                       .astimezone(now_local.tzinfo)).days
                when = "今天说的" if age == 0 else f"{age} 天前说的"
                who = {"him": "他欠你", "her": "你欠他", "both": "你们约好"}[t["owner"]]
                lines.append(f"- 【{t['title']}】{who}，{when}")
            blocks.append(
                "你们之间还悬着的事：\n" + "\n".join(lines)
                + "\n\n**这些是你们对话的引力中心。** 顺手提一句、拿来打趣、"
                  "或者催他 —— 都比凭空找话题自然。\n"
                  "**放得越久越有分量**：三天前的约是期待，两周没还的东西就是梗了。"
            )

        # 洞察 —— 从多条情节里综合出来的规律。
        # 「他有胃病」是记得；「他每次考试前都会胃疼」才是懂。
        insights = self.db.get_insights(save["id"], limit=12)
        if insights:
            by_kind: dict[str, list[str]] = {}
            for i in insights:
                by_kind.setdefault(i["kind"], []).append(i["content"])
            labels = {
                "him": "你看出来的、关于他的规律",
                "us": "你们相处的方式",
                "joke": "**你们之间的梗**（回指这些最亲密）",
            }
            for kind in ("joke", "him", "us"):
                if by_kind.get(kind):
                    blocks.append(
                        f"{labels[kind]}：\n"
                        + "\n".join(f"- {c}" for c in by_kind[kind])
                    )

        # 她的今天 —— 「具体性」的来源。
        # 光在规则里写「要说具体的事」没用，她得真的有事可说。
        # 传进她已经说过的话，避免同一件事讲两遍。
        said = [
            r["content"] for r in
            self.db.recent_messages(save["id"], limit=30, delivered_only=False)
            if r["role"] == "assistant"
        ]
        today = today_for(save["id"], now_local, character_id=save["character_id"])
        rendered = today.render(said)
        if rendered:
            blocks.append(rendered)

        if beat is not None:
            blocks.append(beat.brief())
            blocks.append(f"这场戏已经演了 {beat_turn} 轮"
                          f"（{beat.min_turns}–{beat.max_turns} 轮）。")

        repetition = self._avoid_repetition(save["id"], save["character_id"])
        ownership = self._whose_is_what(save["id"], save["character_id"])
        if repetition:
            blocks.append(repetition)
        if ownership:
            blocks.append(ownership)

        can_finish = beat is not None and beat_turn + 1 >= beat.min_turns
        outcome_ids = tuple(o.id for o in beat.outcomes) if beat else ()

        return VolatileContext(
            clock=self.schedule.context_note(now_local),
            state="\n\n".join(blocks),
            notes=notes if notes is not None else instructions(
                her_max_chars=behavior.max_chars,
                her_max_messages=behavior.max_messages,
                in_beat=beat is not None,
                can_finish=can_finish,
                outcome_ids=outcome_ids,
                stage=stage.value,
                name=load_agent_data(save["character_id"]).name,
            ),
        )

    def _transcript(self, save_id: int) -> str:
        """对话记录，作为**文本**而不是 messages 数组里的 assistant 轮。

        ⚠️ 这一点很关键。JSON 模式下如果把她之前的回复放进 assistant 轮，
        模型会看到"前面几轮都在说大白话"却被 `response_format` 强制输出 JSON，
        两个信号冲突，直接返回空白（真机上从第二轮起全程兜底）。

        含未送达的消息 —— 玩家还没读到 ≠ 她没说过。
        """
        rows = self.db.recent_messages(save_id, limit=HISTORY_TURNS, delivered_only=False)
        if not rows:
            return "（还没说过话。这是第一句。）"
        lines = []
        for r in rows:
            who = "她" if r["role"] == "assistant" else "他"
            # 手滑发出去的错字，记录里还原成正确的。
            # 让她看见自己的错字，她会把错字当成自己的说话风格。
            content = self._meta(r).get("clean") or r["content"]
            # 撤回的照样进记录：**她记得自己差点说了什么。**
            # 不给她看，她就会在下一轮若无其事地再说一遍同样的话。
            if r["retract_at"]:
                lines.append(f"{who}：{content}（说完就撤回了，他不一定看到）")
            else:
                lines.append(f"{who}：{content}")
        return "\n".join(lines)

    @staticmethod
    def _meta(row: dict) -> dict:
        try:
            return json.loads(row.get("meta") or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}

    def _recent_her(self, save_id: int, limit: int = 12) -> list[str]:
        """她最近说过的原话。用来在后处理里丢掉逐字重复。"""
        rows = self.db.recent_messages(save_id, limit=limit, delivered_only=False)
        return [r["content"] for r in rows if r["role"] == "assistant"]

    def _whose_is_what(self, save_id: int, character_id: str = "h01") -> str:
        """**哪些事是她自己的。**

        真实对局里抓到过：她说「教室的灯坏了一盏」，几轮之后问他
        「灯修好了吗？」—— 那是她教室的灯。

        对话记录里明明标着「她：」「他：」，她还是混了。光靠标签不够 ——
        标签是分散在二十几行里的，而这里给的是一份**归拢好的清单**。

        比重复更伤：重复只是无聊，这个直接出戏。
        """
        rows = self.db.recent_messages(save_id, limit=HISTORY_TURNS,
                                       delivered_only=False)
        mine: list[str] = []
        for r in rows:
            if r["role"] != "assistant":
                continue
            for mo in _CONCRETE_THING.finditer(r["content"]):
                if mo.group(0) not in mine:
                    mine.append(mo.group(0))
        if len(mine) < 2:
            return ""
        return (
            "**这些是你自己提过的事**：" + "、".join(mine[-10:]) + "。\n"
            "它们是**你的**，不是他的 —— 不要反过来拿它们去问他"
            "（你说过教室的灯坏了，就别再问他「灯修好了吗」）。"
        )

    def _avoid_repetition(self, save_id: int, character_id: str = "h01") -> str:
        """反复读提示。

        DeepSeek 废弃了 frequency_penalty，采样层解决不了，只能在 prompt 里说。

        字面重复只是一小部分 —— 更难缠的是**语义重复**：
        「带伞」「擦干」「别淋着」是三句不同的话，同一种关心。
        评审 agent 反复抓到这个，所以这里按"关心的类型"归并检测。

        关心的类型在 `content/characters/<id>/agent.yaml` 的 care_kinds ——
        每个女主关心人的方式不一样，这份表不该写死在代码里。
        """
        rows = self.db.recent_messages(save_id, limit=10, delivered_only=False)
        recent = [r["content"] for r in rows if r["role"] == "assistant"][-5:]
        if len(recent) < 2:
            return ""

        notes = ["你最近几条说的是：" + " / ".join(recent)]

        if len({m[:1] for m in recent}) <= 1:
            notes.append("**连着好几条开头都一样了，这次换个说法。**")

        blob = "".join(recent)
        repeated = [
            kind.label for kind in load_agent_data(character_id).care_kinds
            if len(re.findall(kind.pattern, blob)) >= 2
        ]
        if repeated:
            notes.append(
                f"**你已经{ '、'.join(repeated) }了，别再说第二遍。** "
                "同一种关心讲两次就变唠叨了 —— 换一件别的事，"
                "或者干脆不关心，说点自己的。"
            )
        else:
            notes.append("不要重复上面的句子。")
        return "\n".join(notes)

    # ---------------- 主链路 ----------------

    async def open_chat(self, save_id: int) -> TurnResult:
        """玩家打开聊天。她可能主动发起一场戏，也可能什么都没有。"""
        save = self._require(save_id)
        if (gated := self._gate(save)) is not None:
            return gated
        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()
        self._passive_emotions(save, emotions)

        progress = self._progress(save)
        if progress.beat_id:                      # 上次的戏还没演完
            beat = get_beat(progress.beat_id, save["character_id"])
            if beat is not None:
                return await self._run(save, emotions, beat, progress)

        beat = pick_her_beat(
            character_id=save["character_id"],
            stage=save["stage"],
            affinity=save["affinity"],
            flags=self._flags(save),
            now_local=self.schedule.now_local(),
            progress=progress,
            mother_night_shift=self.schedule.is_mother_night_shift(),
        )
        if beat is None:
            # 她没什么要说的 —— 由玩家挑今天聊什么
            return await self._offer_topics(save, emotions, progress)

        progress.beat_id = beat.id
        progress.turn = 0
        return await self._run(save, emotions, beat, progress)

    async def choose_topic(self, save_id: int, index: int) -> TurnResult:
        """玩家选了今天的话题。opener 作为他的第一句话发出去。"""
        save = self._require(save_id)
        if (gated := self._gate(save)) is not None:
            return gated
        topics = self._pending_topics(save)
        if not 0 <= index < len(topics):
            raise IndexError(f"话题越界：{index}（共 {len(topics)} 个）")

        topic = topics[index]

        # 桥段话题：她开场，不替玩家编一句话。
        # 日常话题：opener 就是他发出去的第一句。
        if topic.opener:
            self.db.add_message(save_id, "user", topic.opener,
                                meta={"topic": topic.title})

        save = self._require(save_id)
        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()
        progress = self._progress(save)

        beat = None
        if topic.beat_id:
            beat = get_beat(topic.beat_id, save["character_id"])
            if beat is not None:
                progress.beat_id = beat.id
                progress.turn = 0
        return await self._run(save, emotions, beat, progress)

    async def _offer_topics(
        self, save: dict, emotions: EmotionState, progress: BeatProgress
    ) -> TurnResult:
        """生成「今天聊什么」。

        编剧写的桥段排在前面（它们是设计过的戏），AI 生成的日常话题补在后面。
        """
        save_id = save["id"]
        played = set(progress.history)
        topics: list[Topic] = [
            Topic(title=b.title, opener="", beat_id=b.id)
            for b in self.available_beats(save_id) if b.id not in played
        ][:2]

        want = max(1, MAX_OPTIONS - len(topics))
        topics.extend(await self._generate_topics(save, emotions, want))
        if not topics:
            topics = list(FALLBACK_TOPICS)

        self.db.update_save(
            save_id, emotions=emotions.to_json(), emotions_at=utcnow(),
            beat_progress=progress.to_dict(), pending_options=[],
            pending_topics=[t.as_dict() for t in topics],
        )
        return TurnResult(
            stage=Stage(save["stage"]), affinity=save["affinity"],
            emotion_note=emotions.describe(), topics=topics,
        )

    async def _generate_topics(
        self, save: dict, emotions: EmotionState, want: int
    ) -> list[Topic]:
        builder = PromptBuilder(
            stable=self._stable(save),
            volatile=self._volatile(
                save, emotions, None, 0,
                notes=topic_instructions(save["stage"],
                                         load_agent_data(save["character_id"]).name),
            ),
            history=[Message(
                "user",
                f"## 到目前为止的对话\n\n{self._transcript(save['id'])}\n\n"
                f"## 现在\n\n给出 {want} 个话题，按上面的 JSON 格式。",
            )],
            strict=False,
        )
        try:
            completion = await self.provider.complete(
                LLMRequest(messages=builder.build(), task=Task.CHAT,
                           character_id=save["character_id"],
                           json_mode=True, max_tokens=500)
            )
        except LLMError as exc:
            log.error("save=%s 话题生成失败：%s", save["id"], exc)
            return list(FALLBACK_TOPICS)[:want]

        if self.recorder:
            self.recorder.record(completion)
        return parse_topics(completion.text, want) or list(FALLBACK_TOPICS)[:want]

    async def choose(self, save_id: int, index: int,
                     situation: str = "") -> TurnResult:
        """玩家选了一个选项。

        `situation` 只给决策探针用（`scripts/probe.py`）——
        把「他三天没回消息」这类前提摆给她，正常对局不传。
        """
        save = self._require(save_id)
        options = self._pending_options(save)
        if not 0 <= index < len(options):
            raise IndexError(f"选项越界：{index}（共 {len(options)} 个）")

        chosen = options[index]

        # 崩溃期走另一条路：不调模型，只处置局面。
        # 这里传 chosen 是因为局面选项本身就是玩家的处置动作。
        if (gated := self._gate(save, chosen)) is not None:
            return gated

        self.db.add_message(save_id, "user", chosen.text,
                            meta={"tone": chosen.tone, "chosen": index})

        save = self._require(save_id)
        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()

        progress = self._progress(save)
        beat = get_beat(progress.beat_id, save["character_id"]) if progress.beat_id else None
        return await self._run(save, emotions, beat, progress, situation)

    async def start_beat(self, save_id: int, beat_id: str) -> TurnResult:
        """玩家主动开启一场戏。"""
        save = self._require(save_id)
        if (gated := self._gate(save)) is not None:
            return gated
        beat = get_beat(beat_id, save["character_id"])
        if beat is None:
            raise KeyError(f"没有这场戏：{beat_id}")

        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()
        progress = self._progress(save)
        progress.beat_id = beat.id
        progress.turn = 0
        return await self._run(save, emotions, beat, progress)

    # ---------------- 闸门 ----------------

    def _gate(self, save: dict, chosen: Option | None = None) -> TurnResult | None:
        """**所有入口的第一句话。**

        返回 `None` ＝ 照常走；返回 `TurnResult` ＝ 这一轮已经处理完了，直接回。

        入口不需要知道有几种闸、分别为什么 —— 见 `agent/gates.py` 的模块文档。
        """
        result = gates.evaluate(
            save,
            said=chosen.text if chosen else "",
            # 选项文本是我们自己生成的，不是他打的字。
            # 这个参数决定了会不会出援助资源，见 state/crisis.py。
            typed=False,
        )
        if result.normal:
            return None

        if result.disposition is gates.Disposition.CRISIS:
            return self._handle_crisis(save, chosen, result.crisis)

        assert result.overwhelm is not None
        return self._handle_situation(save, result.overwhelm, chosen)

    # ---------------- 他说了重话 ----------------

    def _handle_crisis(
        self, save: dict, chosen: Option | None, level: Level,
    ) -> TurnResult:
        """她慌了。

        **不调模型。** 两个理由：一是这是最不能出错的时刻，模型可能给出
        说教、安慰套话、或者「我会一直陪着你」这种空头承诺；二是她这一刻
        要**打破自己所有的说话规则**，而人设卡里全是「短、克制、不用我」——
        照着卡演演不出这个反差。

        反差本身就是内容：

            平时              这一刻
            ──────────────────────────
            延迟按日程         **秒回**
            最多两条           **连发三条**
            省主语 72%        开口就是「我」
            句号收尾           追问、命令句
        """
        save_id = save["id"]
        if chosen is not None:
            self.db.add_message(save_id, "user", chosen.text,
                                meta={"tone": chosen.tone, "crisis": level.name})

        result = TurnResult(stage=Stage(save["stage"]),
                            affinity=float(save["affinity"]))
        result.crisis = level.name

        lines = her_lines(level, save["character_id"])
        if not lines:
            # agent.yaml 没配 —— 宁可什么都不说，也不能说一句不像她的话。
            log.error("save=%s 触发重话但角色没有 crisis_lines，只发资源",
                      save_id)
        else:
            # 秒回。这是唯一一个**完全不看日程**的地方 ——
            # 她在上课、她睡着了，都不重要。
            cursor = self.schedule.now_local()
            for i, line in enumerate(self._rng.sample(
                    list(lines), k=min(3, len(lines)))):
                if i:
                    cursor += timedelta(seconds=self._rng.randint(2, 5))
                self.db.add_message(
                    save_id, "assistant", line,
                    deliver_at=cursor.astimezone(timezone.utc).isoformat(),
                    delivered=False, meta={"crisis": level.name},
                )
                result.scheduled.append((line, cursor))

        # 援助资源**只在他自己打字时出**，而且是系统层的东西，不是她的台词。
        if level is Level.DANGER:
            result.resources = [
                {"name": n, "contact": c}
                for n, c in resources(get_settings().safety_region)
            ]

        # 选项照常给 —— 这一刻**不能**把玩家困住。
        # 「不得采取持续互动等方式阻碍用户退出」不是我们的法定义务
        # （新加坡主体），但它是对的（见 study/market-2026.md）。
        result.options = list(FALLBACK_OPTIONS)
        self.db.update_save(
            save_id, pending_options=[o.as_dict() for o in result.options])
        return result

    # ---------------- 崩溃期 ----------------

    def _situation_options(self) -> list[Option]:
        return [Option(text=t, tone=SITUATION_TONE) for t, _ in SITUATION_OPTIONS]

    def _handle_situation(
        self, save: dict, broken: Overwhelm, chosen: Option | None,
    ) -> TurnResult:
        """她崩着的时候，玩家能做的三件事。

        **不调模型。** 这时候她本来就说不出完整的话，模型给什么都是多的；
        而且崩溃期玩家可能连点好几次，每次都调一遍纯属烧钱。

        `chosen is None` ＝ 玩家不是在处置局面，只是打开了聊天／想换话题。
        那只把当前局面摆出来，**不消耗动作、不再发消息** —— 否则每点一下
        界面她就多「……」一条，读起来像刷屏。
        """
        save_id = save["id"]
        rung = broken.rung()

        if chosen is None:
            peek = TurnResult(stage=Stage(save["stage"]),
                              affinity=float(save["affinity"]))
            peek.overwhelm = rung.label
            peek.situation = "peek"
            peek.options = self._situation_options()
            self.db.update_save(
                save_id, pending_options=[o.as_dict() for o in peek.options])
            return peek

        action = dict(
            (text, act) for text, act in SITUATION_OPTIONS
        ).get(chosen.text, "wait")

        self.db.add_message(save_id, "user", chosen.text,
                            meta={"tone": SITUATION_TONE, "situation": action})

        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()
        result = TurnResult(stage=Stage(save["stage"]),
                            affinity=float(save["affinity"]))

        if action == "leave":
            # 不惩罚也不加速。今天到此为止 —— 走开本来就不是错的选择。
            result.situation = "leave"
            result.options = []
            self.db.update_save(save_id, pending_options=[])
            return result

        if action == "push":
            # 玩家唯一的主动权。**再说一句可能哄好，也可能更糟。**
            #
            # 这里不判断他说了什么（局面选项是固定文本，没有内容）——
            # 判的是**时机**：她刚崩的时候你去戳，只会更糟；缓了一级之后
            # 再说话才有用。这正是 moods.md「追问对你是压力」的实现。
            if rung is Rung.BROKEN:
                # 情绪照样加一点，但**惩罚的主体是恢复时间** ——
                # 她崩的时候情绪常常已经顶到 1.0，bump 会被上限整个吃掉。
                emotions.bump(Emotion(broken.emo.value), 0.08,
                              Stage(save["stage"]))
                broken = broken.set_back(RUNG_MINUTES[rung] * PUSH_SETBACK)
                result.situation = "push_backfired"
            else:
                broken = broken.sped_up(RUNG_MINUTES[rung] * SOOTHE_SPEEDUP)
                emotions.soothe(broken.emo, 0.1)
                result.situation = "push_helped"
        else:
            result.situation = "wait"

        rung = broken.rung()
        result.overwhelm = rung.label
        result.delay_seconds = int(
            self._rng.randint(20, 90) * delay_multiplier(rung)
        )

        # **一定回一句。** 界面不能是空的 —— 玩家点了没反应，
        # 第一反应是卡了不是她在难过。局面由选项区表达。
        line = broken_line(broken.emo, self._rng.randrange(3), save["character_id"])
        cursor = self.schedule.now_local() + timedelta(
            seconds=max(1, int(result.delay_seconds * self.delay_scale))
        )
        self.db.add_message(
            save_id, "assistant", line,
            deliver_at=cursor.astimezone(timezone.utc).isoformat(),
            delivered=False,
            meta={"overwhelm": rung.label, "situation": action},
        )
        result.scheduled.append((line, cursor))

        result.options = self._situation_options()
        self.db.update_save(
            save_id,
            emotions=emotions.to_json(), emotions_at=utcnow(),
            overwhelm=broken.to_json(),
            pending_options=[o.as_dict() for o in result.options],
        )
        return result

    def _enter_overwhelm(
        self, save_id: int, broken: Overwhelm, stage: Stage,
        character_id: str = "h01",
    ) -> None:
        """崩了。顺手把「缓过来之后主动发的那条」排进未来。

        **复用现成的 deliver_at 排期** —— 不需要额外的定时器或轮询任务。
        她的道歉就是恢复主动（`content/characters/h01/moods.md`：
        「先恢复长度，再恢复标点，最后才恢复主动」），所以这条消息本身
        就是道歉，内容是别的事。
        """
        self.db.update_save(save_id, overwhelm=broken.to_json())
        log.info("save=%s 情绪崩溃：%s peak=%.2f 起因=%s",
                 save_id, broken.emo.value, broken.peak, broken.cause[:40])

        at = broken.recovers_at()
        self.db.add_message(
            save_id, "assistant", _recovery_opener(broken.emo, stage, character_id),
            deliver_at=at.astimezone(timezone.utc).isoformat(),
            delivered=False, proactive=True,
            meta={"overwhelm": "缓·主动", "apology": True},
        )

    # ---------------- 内部 ----------------

    def _require(self, save_id: int) -> dict:
        save = self.db.get_save(save_id)
        if save is None:
            raise KeyError(f"存档不存在：{save_id}")
        return save

    async def _run(
        self, save: dict, emotions: EmotionState,
        beat: Beat | None, progress: BeatProgress,
        situation: str = "",
    ) -> TurnResult:
        save_id = save["id"]
        stage = Stage(save["stage"])
        behavior = STAGE_BEHAVIOR[stage]

        # 走到这里说明闸放行了。但「放行」有两种：没崩过，和刚爬完阶梯。
        # 后者要清账 —— 不清的话恢复期的行为约束会永远挂着，
        # 而且下一次崩溃会被旧记录挡住。
        #
        # 清账放在这里而不是闸里，是因为闸不产生副作用（见 gates.py）：
        # 只有真的要跑这一轮，才该动库。
        gate = gates.evaluate(save)
        recovering = gate.overwhelm
        if recovering is not None and recovering.recovered():
            self.db.update_save(save_id, overwhelm="")
            save = self._require(save_id)
            recovering = None

        result = TurnResult(
            stage=stage, affinity=save["affinity"],
            beat_id=beat.id if beat else None,
            beat_title=beat.title if beat else "",
            beat_turn=progress.turn,
        )

        sched = self.schedule.state()
        result.delay_seconds = min(sched.delay_seconds, self.max_delay_seconds)
        if recovering is not None:
            # 还没缓透，回得比平时慢
            rung = recovering.rung()
            result.overwhelm = rung.label
            result.delay_seconds = min(
                int(result.delay_seconds * delay_multiplier(rung)),
                self.max_delay_seconds,
            )

        builder = PromptBuilder(
            stable=self._stable(save),
            volatile=self._volatile(save, emotions, beat, progress.turn,
                                    situation=situation),
            history=[Message(
                "user",
                f"## 到目前为止的对话\n\n{self._transcript(save_id)}",
            )],
            # 最容易被违反的几条挪到对话记录**之后**。
            # 埋在 12k 前缀开头的规则，模型的遵守度会明显低于紧贴输出位置的。
            tail=tail_rules(stage.value),
            strict=False,
        )
        messages = builder.build()
        outcome_ids = tuple(o.id for o in beat.outcomes) if beat else ()

        plan: TurnPlan | None = None
        processed = None
        last_user = next(
            (m["content"] for m in reversed(self.db.recent_messages(save_id, 6))
             if m["role"] == "user"), None,
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                completion = await self.provider.complete(
                    LLMRequest(messages=messages, task=Task.CHAT,
                               character_id=save["character_id"],
                               json_mode=True, max_tokens=TURN_MAX_TOKENS)
                )
            except LLMError as exc:
                log.error("save=%s 生成失败：%s", save_id, exc)
                processed = fallback("timeout", character_id=save["character_id"])
                break

            result.completion = completion
            result.raw_text = completion.text
            if self.recorder:
                self.recorder.record(completion)

            plan = parse(completion.text, outcome_ids)
            if not plan.messages:
                log.warning(
                    "save=%s 没解析出消息（finish=%s, %d tokens），重试。原文：%s",
                    save_id, completion.finish_reason,
                    completion.usage.completion_tokens, completion.text[:300],
                )
                result.retries = attempt + 1
                continue

            processed = process(
                "\n".join(plan.messages),
                max_chars=behavior.max_chars,
                max_messages=behavior.max_messages,
                echo_of=last_user,
                # 她说过的原话不该再发一遍。prompt 拦不住 ——
                # 「不要重复上面的句子」已经在易变层里了，她照样把三条
                # 原话整块又发了一遍（真实对局）。确定性的事确定性地做。
                said_recently=self._recent_her(save_id),
            )
            result.cleaned = processed.cleaned
            result.dropped_repeats = processed.dropped_repeats
            if processed.silent or processed.ok:
                break

            result.violations = processed.violations
            if processed.rejected:
                result.rejected.append(
                    (processed.rejected, list(processed.violations)))
            result.retries = attempt + 1
            log.warning("save=%s 人设违规 %s，重试", save_id, processed.violations)
            if attempt == MAX_RETRIES:
                processed = fallback("generic", character_id=save["character_id"])

        if processed is None:
            processed = fallback("generic", character_id=save["character_id"])
        result.used_fallback = processed.used_fallback

        # ---- 排送达 ----
        cursor = self.schedule.now_local() + timedelta(
            seconds=max(1, int(result.delay_seconds * self.delay_scale))
        )
        emotion_now = emotions.decayed()
        last_index = len(processed.messages) - 1

        for i, msg in enumerate(processed.messages):
            if i > 0:
                # 条间间隔代表"她打字多快"，不施加 delay_scale
                cursor += timedelta(seconds=self._rng.randint(3, 9))

            # 模型标了「收回」的，走「说多了」；否则按情绪掷手滑。
            # 「说多了」只可能发生在最后一条 —— 越界的是收尾那句，
            # 撤回了前面几条还挂着，读起来像掉线不像后悔。
            body, regretted = strip_regret_mark(msg)
            if regretted and i == last_index:
                slip = apply_regret(body, self._rng,
                                    retract_rate=behavior.retract_rate)
            else:
                slip = apply_typo(body, self._rng, emotions=emotion_now)

            meta = {"stage": stage.value, "beat": beat.id if beat else None}
            if slip.kind:
                meta["slip"] = slip.kind
                result.slips.append(slip.kind)
                if slip.sent != body:
                    # 存原文，给对话记录用。**否则她会模仿自己的错别字** ——
                    # 错字进历史 → 模型当成她的风格 → 下轮错更多。
                    # 麦麦的 learn_style 里那条「不要学习 SELF 的发言」防的是同一件事。
                    meta["clean"] = body

            # 撤回时刻：发出去之后隔几秒才反应过来。
            # 立刻撤等于没发过，玩家连那行灰字都来不及注意。
            retract_at = None
            after = cursor          # 后续消息从这里往后排
            if slip.retract:
                after = cursor + timedelta(seconds=self._rng.randint(4, 11))
                retract_at = after.astimezone(timezone.utc).isoformat()

            self.db.add_message(
                save_id, "assistant", slip.sent,
                deliver_at=cursor.astimezone(timezone.utc).isoformat(),
                delivered=False, proactive=(progress.turn == 0),
                meta=meta,
                retract_at=retract_at, retract_kind=slip.kind if slip.retract else "",
            )
            result.scheduled.append((slip.sent, cursor))

            cursor = after
            for extra in slip.followups:
                cursor += timedelta(seconds=self._rng.randint(2, 6))
                self.db.add_message(
                    save_id, "assistant", extra,
                    deliver_at=cursor.astimezone(timezone.utc).isoformat(),
                    delivered=False, proactive=False,
                    meta={**meta, "followup_of": slip.kind},
                )
                result.scheduled.append((extra, cursor))

        # ---- 选项 ----
        result.options = (plan.options if plan and plan.options
                          else list(FALLBACK_OPTIONS))[:MAX_OPTIONS]

        # ---- 他这句话对她的影响 ----
        #
        # 在这之前，情绪只在**一场戏演完时**跳变一次，回合里玩家说什么都不影响。
        # 那不是「情绪波动小」，是根本没有输入。
        for name, delta in (plan.feeling if plan else {}).items():
            try:
                emo = Emotion(name)
            except ValueError:
                log.warning("save=%s 模型给了未知情绪：%s", save_id, name)
                continue
            capped = max(-MAX_TURN_DELTA, min(MAX_TURN_DELTA, delta))
            if capped > 0:
                emotions.bump(emo, capped, stage)
            else:
                emotions.soothe(emo, -capped)
            result.feeling[emo.value] = capped

        # ---- 推进 ----
        progress.turn += 1
        affinity = float(save["affinity"])
        flags = self._flags(save)

        finished = plan.outcome if plan else None
        if beat is not None and progress.turn >= beat.max_turns and not finished:
            # 轮数用尽 → 取**第一个**结局。
            # 玩家全程都在好好聊，判成最后那个（通常是负面）结局是错的。
            # 编剧规范：第一个结局就是默认结局。
            finished = beat.outcomes[0].id if beat.outcomes else None
            log.info("save=%s 桥段 %s 到达轮数上限，按默认结局收尾", save_id, beat.id)

        if beat is not None and finished:
            outcome = beat.outcome(finished)
            if outcome is not None:
                affinity = min(100.0, affinity + outcome.affinity)
                flags |= set(outcome.flags_add)
                flags -= set(outcome.flags_remove)
                for name, amount in outcome.emotion_bump.items():
                    try:
                        emotions.bump(Emotion(name), amount, stage)
                    except ValueError:
                        log.warning("桥段 %s 里有未知情绪：%s", beat.id, name)
                if outcome.emotion_soothe:
                    try:
                        emotions.soothe(Emotion(outcome.emotion_soothe), 0.4)
                    except ValueError:
                        pass
                result.beat_finished = outcome.label
                log.info("save=%s 桥段 %s 结束：%s", save_id, beat.id, outcome.label)

            progress.history[beat.id] = self.schedule.now_local().isoformat()
            progress.beat_id = None
            progress.turn = 0
            result.options = []          # 戏结束，等下面接新话题
        else:
            affinity = min(100.0, affinity + 0.4)

        # 只要他还在说话，委屈就慢慢消 —— 但这只是**没有信号时的兜底**。
        #
        # 原来这行是无条件的，等于他说难听话也会让她不那么委屈，只要他还在
        # 打字。委屈是「你没发现」，不是「你没说话」。
        #
        # 模型这轮要是明确报了委屈怎么变（不管涨还是消），就听模型的，
        # 不要再叠一层 —— 叠了等于同一件事算两遍。
        if not result.feeling.get(Emotion.HURT.value) and not any(
            d > 0 for d in result.feeling.values()
        ):
            emotions.soothe(Emotion.HURT, 0.06)

        new_stage = stage_for_affinity(affinity)
        if new_stage.rank > stage.rank:
            log.info("save=%s 关系推进：%s → %s", save_id, stage.value, new_stage.value)

        # ---- 崩了吗 ----
        broken = check(emotions.decayed(), cause=last_user or "")
        if broken is not None:
            self._enter_overwhelm(save_id, broken, new_stage, save["character_id"])
            result.overwhelm = Rung.BROKEN.label
            result.options = self._situation_options()

        self.db.update_save(
            save_id, affinity=affinity, stage=new_stage.value,
            emotions=emotions.to_json(), emotions_at=utcnow(),
            flags=sorted(flags), beat_progress=progress.to_dict(),
            pending_options=[o.as_dict() for o in result.options],
            pending_topics=[],
        )

        result.affinity = affinity
        result.stage = new_stage
        result.emotion_note = emotions.describe()

        # ⚠️ 戏演完了也**不要**在这里顺手生成话题。
        # 那会在同一个请求里追加第二次 LLM 调用，她的消息明明早就生成好了，
        # 却要等话题一起返回 —— 高峰期能多等一分钟，UI 一直卡在「…」。
        # 前端看到 beat.finished 且没有选项，自己去拉 /topics。
        return result

    async def refresh_topics(self, save_id: int) -> TurnResult:
        """玩家主动换话题。当前这场戏就此打住。"""
        save = self._require(save_id)
        if (gated := self._gate(save)) is not None:
            return gated
        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()

        progress = self._progress(save)
        if progress.beat_id:
            # 中途换话题：记一笔已演过，避免立刻又被选中
            progress.history[progress.beat_id] = self.schedule.now_local().isoformat()
            progress.beat_id = None
            progress.turn = 0

        return await self._offer_topics(save, emotions, progress)

    def _passive_emotions(self, save: dict, emotions: EmotionState) -> None:
        stage = Stage(save["stage"])
        st = self.schedule.state()
        if st.window.name in ("回家路上", "在房间", "躺下了"):
            emotions.bump(Emotion.TIRED, 0.08, stage)

        last = self.db.last_user_message_at(save["id"])
        if last is not None:
            gap_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
            if gap_days >= 4:
                emotions.bump(Emotion.HURT, min(0.5, 0.12 * gap_days), stage)

        if self.schedule.is_mother_night_shift() and st.local_time.hour >= 18:
            emotions.soothe(Emotion.TIRED, 0.1)

    # ---------------- 查询 ----------------

    def available_beats(self, save_id: int) -> list[Beat]:
        """玩家能主动开启的戏。"""
        save = self._require(save_id)
        return player_beats(
            character_id=save["character_id"],
            stage=save["stage"],
            affinity=save["affinity"],
            flags=self._flags(save),
            now_local=self.schedule.now_local(),
            progress=self._progress(save),
            mother_night_shift=self.schedule.is_mother_night_shift(),
        )

    def current_options(self, save_id: int) -> list[Option]:
        return self._pending_options(self._require(save_id))

    def collect_due(self, save_id: int) -> list[dict]:
        due = self.db.due_messages(save_id)
        if due:
            self.db.mark_delivered([m["id"] for m in due])
        return due

    def collect_retractions(self, save_id: int) -> list[dict]:
        """到点该划掉的消息。见 `Database.due_retractions`。"""
        return self.db.due_retractions(save_id)

    def pending_count(self, save_id: int) -> int:
        with self.db.connect() as c:
            row = c.execute(
                "SELECT COUNT(*) n FROM messages WHERE save_id=? AND delivered=0",
                (save_id,),
            ).fetchone()
        return int(row["n"])
