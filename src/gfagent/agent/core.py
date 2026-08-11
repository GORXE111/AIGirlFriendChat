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
from ..persona.loader import load_card
from ..prompt import PromptBuilder, StablePrefix, VolatileContext
from ..schedule import ScheduleEngine
from ..state.moods import behavior_note
from ..state.models import (
    STAGE_BEHAVIOR,
    Emotion,
    EmotionState,
    Stage,
    stage_for_affinity,
)
from ..storage.db import Database, parse_ts, utcnow
from .turn import (
    MAX_OPTIONS,
    Option,
    Topic,
    TurnPlan,
    instructions,
    parse,
    parse_topics,
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

    completion: Completion | None = None
    violations: list[str] = field(default_factory=list)
    cleaned: list[str] = field(default_factory=list)
    retries: int = 0
    used_fallback: bool = False
    raw_text: str = ""


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
        card = load_card(save["character_id"])
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
        *, notes: str | None = None,
    ) -> VolatileContext:
        now_local = self.schedule.now_local()
        stage = Stage(save["stage"])
        behavior = STAGE_BEHAVIOR[stage]

        blocks: list[str] = [
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

        repetition = self._avoid_repetition(save["id"])
        if repetition:
            blocks.append(repetition)

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
        lines = [
            f"{'她' if r['role'] == 'assistant' else '他'}：{r['content']}"
            for r in rows
        ]
        return "\n".join(lines)

    # 她反复用的几种关心。同一类连说两次就腻了。
    _CARE_KINDS: tuple[tuple[str, str], ...] = (
        ("带伞|淋|湿|雨", "提醒他别淋雨"),
        ("睡|熬|困|几点", "让他早点睡"),
        ("吃|饭|饿|凉|辣", "让他好好吃饭"),
        ("胃|疼|难受|药|医", "问他身体"),
        ("冷|穿|外套|降温", "让他多穿点"),
        ("迟到|早点来|等", "叮嘱时间"),
    )

    def _avoid_repetition(self, save_id: int) -> str:
        """反复读提示。

        DeepSeek 废弃了 frequency_penalty，采样层解决不了，只能在 prompt 里说。

        字面重复只是一小部分 —— 更难缠的是**语义重复**：
        「带伞」「擦干」「别淋着」是三句不同的话，同一种关心。
        评审 agent 反复抓到这个，所以这里按"关心的类型"归并检测。
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
            label for pattern, label in self._CARE_KINDS
            if len(re.findall(pattern, blob)) >= 2
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
                notes=topic_instructions(save["stage"]),
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

    async def choose(self, save_id: int, index: int) -> TurnResult:
        """玩家选了一个选项。"""
        save = self._require(save_id)
        options = self._pending_options(save)
        if not 0 <= index < len(options):
            raise IndexError(f"选项越界：{index}（共 {len(options)} 个）")

        chosen = options[index]
        self.db.add_message(save_id, "user", chosen.text,
                            meta={"tone": chosen.tone, "chosen": index})

        save = self._require(save_id)
        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()

        progress = self._progress(save)
        beat = get_beat(progress.beat_id, save["character_id"]) if progress.beat_id else None
        return await self._run(save, emotions, beat, progress)

    async def start_beat(self, save_id: int, beat_id: str) -> TurnResult:
        """玩家主动开启一场戏。"""
        save = self._require(save_id)
        beat = get_beat(beat_id, save["character_id"])
        if beat is None:
            raise KeyError(f"没有这场戏：{beat_id}")

        emotions = EmotionState.from_json(save["emotions"])
        emotions.apply_decay()
        progress = self._progress(save)
        progress.beat_id = beat.id
        progress.turn = 0
        return await self._run(save, emotions, beat, progress)

    # ---------------- 内部 ----------------

    def _require(self, save_id: int) -> dict:
        save = self.db.get_save(save_id)
        if save is None:
            raise KeyError(f"存档不存在：{save_id}")
        return save

    async def _run(
        self, save: dict, emotions: EmotionState,
        beat: Beat | None, progress: BeatProgress,
    ) -> TurnResult:
        save_id = save["id"]
        stage = Stage(save["stage"])
        behavior = STAGE_BEHAVIOR[stage]

        result = TurnResult(
            stage=stage, affinity=save["affinity"],
            beat_id=beat.id if beat else None,
            beat_title=beat.title if beat else "",
            beat_turn=progress.turn,
        )

        sched = self.schedule.state()
        result.delay_seconds = min(sched.delay_seconds, self.max_delay_seconds)

        builder = PromptBuilder(
            stable=self._stable(save),
            volatile=self._volatile(save, emotions, beat, progress.turn),
            history=[Message(
                "user",
                f"## 到目前为止的对话\n\n{self._transcript(save_id)}\n\n"
                "## 现在\n\n按上面的规格输出这一回合的 JSON。",
            )],
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
                processed = fallback("timeout")
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
            )
            result.cleaned = processed.cleaned
            if processed.silent or processed.ok:
                break

            result.violations = processed.violations
            result.retries = attempt + 1
            log.warning("save=%s 人设违规 %s，重试", save_id, processed.violations)
            if attempt == MAX_RETRIES:
                processed = fallback("generic")

        if processed is None:
            processed = fallback("generic")
        result.used_fallback = processed.used_fallback

        # ---- 排送达 ----
        cursor = self.schedule.now_local() + timedelta(
            seconds=max(1, int(result.delay_seconds * self.delay_scale))
        )
        for i, msg in enumerate(processed.messages):
            if i > 0:
                # 条间间隔代表"她打字多快"，不施加 delay_scale
                cursor += timedelta(seconds=self._rng.randint(3, 9))
            self.db.add_message(
                save_id, "assistant", msg,
                deliver_at=cursor.astimezone(timezone.utc).isoformat(),
                delivered=False, proactive=(progress.turn == 0),
                meta={"stage": stage.value, "beat": beat.id if beat else None},
            )
            result.scheduled.append((msg, cursor))

        # ---- 选项 ----
        result.options = (plan.options if plan and plan.options
                          else list(FALLBACK_OPTIONS))[:MAX_OPTIONS]

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

        emotions.soothe(Emotion.HURT, 0.06)
        new_stage = stage_for_affinity(affinity)
        if new_stage.rank > stage.rank:
            log.info("save=%s 关系推进：%s → %s", save_id, stage.value, new_stage.value)

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

    def pending_count(self, save_id: int) -> int:
        with self.db.connect() as c:
            row = c.execute(
                "SELECT COUNT(*) n FROM messages WHERE save_id=? AND delivered=0",
                (save_id,),
            ).fetchone()
        return int(row["n"])
