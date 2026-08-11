"""慢回路：把对话压缩成事实与情节。

**这是「她记得你上周三说嗓子疼」的实现。** 情节必须带精确日期 ——
她的在意只通过「记得」来表达，记不住日期这个角色就废了。

离线跑，不在玩家等待的链路上。REFLECT 的错误会永久沉淀（抽错一条事实，
她就会一直记错），所以用低温 + JSON 模式 + 校验。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..llm import DeepSeekProvider, LLMError, LLMRequest, Message, Task
from ..metrics import UsageRecorder
from ..storage.db import Database, parse_ts

log = logging.getLogger(__name__)

REFLECT_EVERY = 8
"""累积多少条新消息触发一次归档。"""

SYSTEM = """你在为一个恋爱游戏维护角色记忆。下面是女主与男主的一段聊天记录。

抽取两类信息，用 JSON 输出：

1. facts —— 关于**男主**的稳定事实（他的习惯、喜好、身体状况、生活、在意的事）。
   只记会长期成立的，不记一次性的情绪。每条不超过 25 字。
   宁可多记也不要漏 —— 「他有胃病」「他常熬夜」这类都算。
2. episodes —— 这段对话里发生的**具体事件**，每条必须能对应到记录中的某一天。
   每条不超过 30 字，要具体（「他说胃疼」而不是「聊了健康」）。

**这两类内容会直接读给女主看**，所以一律用「他」指代男主，**绝不出现「男主」「女主」**，
也不要描述女主自己说了什么做了什么 —— 只记他的事。

  正确：「他说胃疼」「他问我几点睡」「他夸了我的耳环」
  错误：「男主说胃不舒服」「他问女主几点睡」「女主关心了他」

规则：
- 只抽取记录里明确出现的内容，绝不推测、绝不补全。
- 不要记录女主自己的设定（那些已经在人设里）。
- 没有可抽取的内容就返回空数组。
- happened_at 用记录中标注的日期，格式 YYYY-MM-DD。

输出 JSON：
{"facts": [{"content": "...", "category": "身体|喜好|生活|学习|其他"}],
 "episodes": [{"summary": "...", "happened_at": "YYYY-MM-DD", "importance": 1}]}
importance 取 1-5，只有真正重要的才给 4-5。"""


INSIGHT_EVERY = 6
"""累积多少条情节触发一次综合。"""

INSIGHT_SYSTEM = """你在为一个恋爱游戏维护角色记忆的**高层部分**。

下面是女主与男主之间发生过的一系列事。请从中**综合**出更高层的结论。

抽取和综合的区别：

  抽取：「他有胃病」          ← 单条事实，已经有了
  综合：「他每次考试前都会胃疼」← 从多条里看出来的规律

**只有综合才是「她懂你」，抽取只是「她记得」。**

产出三类，用 JSON 输出：

1. `him` —— 关于他的模式。他反复做的事、他的习惯、他回避什么。
   例：「他从不主动说自己的事，都要她问」「他嘴上说没事，其实在硬撑」

2. `us` —— 关于他们之间的模式。相处的方式、默契、反复出现的互动。
   例：「他们的对话经常从吃什么开始」「她一说累他就转移话题」

3. `joke` —— **内梗**。反复出现的共同话题、只有他们俩懂的东西。
   例：「那家面馆的辣酱」「他答应过的粥」

规则：

- **必须有依据。** 只从下面的记录里看，一条规律至少要有两次印证。
  看不出来就返回空数组，**不要为了凑数编**。
- 每条不超过 25 字。
- 用「他」指代男主，不要写「男主」。
- 写规律，不要复述单个事件（那是情节的活，已经有了）。
- `joke` 尤其重要 —— 内梗是亲密感的核心。哪怕只有一个也要挑出来。

输出：
{"insights": [{"content": "...", "kind": "him"}]}"""


@dataclass(slots=True)
class ReflectResult:
    facts_added: int = 0
    episodes_added: int = 0
    insights_added: int = 0
    skipped: bool = False
    error: str = ""


class Reflector:
    def __init__(
        self,
        db: Database,
        provider: DeepSeekProvider,
        recorder: UsageRecorder | None = None,
    ) -> None:
        self.db = db
        self.provider = provider
        self.recorder = recorder

    def should_run(self, save_id: int) -> bool:
        mark = self.db.get_reflect_mark(save_id)
        return len(self.db.messages_since(save_id, mark)) >= REFLECT_EVERY

    async def run(self, save_id: int, *, force: bool = False) -> ReflectResult:
        mark = self.db.get_reflect_mark(save_id)
        rows = self.db.messages_since(save_id, mark)
        if not rows or (not force and len(rows) < REFLECT_EVERY):
            return ReflectResult(skipped=True)

        transcript = []
        for r in rows:
            when = parse_ts(r["created_at"]).astimezone()
            who = "男主" if r["role"] == "user" else "女主"
            transcript.append(f"[{when:%Y-%m-%d %H:%M}] {who}：{r['content']}")

        try:
            completion = await self.provider.complete(
                LLMRequest(
                    messages=[
                        Message("system", SYSTEM),
                        Message("user", "\n".join(transcript)),
                    ],
                    task=Task.REFLECT,
                    json_mode=True,
                )
            )
        except LLMError as exc:
            log.error("save=%s 归档失败：%s", save_id, exc)
            return ReflectResult(error=str(exc))

        if self.recorder:
            self.recorder.record(completion)

        try:
            data = json.loads(completion.text)
        except json.JSONDecodeError:
            log.error("save=%s 归档返回非 JSON：%s", save_id, completion.text[:200])
            return ReflectResult(error="非 JSON")

        result = ReflectResult()

        for f in data.get("facts") or []:
            content = str(f.get("content", "")).strip()
            if not content or len(content) > 40:
                continue
            self.db.add_fact(save_id, content, str(f.get("category", "其他")))
            result.facts_added += 1

        for e in data.get("episodes") or []:
            summary = str(e.get("summary", "")).strip()
            happened = str(e.get("happened_at", "")).strip()
            if not summary or len(summary) > 50:
                continue
            try:
                # 校验日期，坏数据宁可丢弃也不能污染记忆
                ts = parse_ts(f"{happened}T12:00:00+08:00")
            except (ValueError, TypeError):
                log.warning("save=%s 丢弃日期非法的情节：%r", save_id, happened)
                continue
            importance = e.get("importance", 1)
            importance = importance if isinstance(importance, int) and 1 <= importance <= 5 else 1
            self.db.add_episode(save_id, summary, ts.isoformat(), importance)
            result.episodes_added += 1

        self.db.set_reflect_mark(save_id, rows[-1]["id"])

        # 综合 —— 情节够多了才做，太少看不出规律
        episodes = self.db.get_episodes(save_id, limit=40)
        if len(episodes) >= INSIGHT_EVERY:
            result.insights_added = await self._synthesize(save_id, episodes)

        log.info(
            "save=%s 归档完成：+%d 事实 +%d 情节 +%d 洞察",
            save_id, result.facts_added, result.episodes_added,
            result.insights_added,
        )
        return result

    async def _synthesize(self, save_id: int, episodes: list[dict]) -> int:
        """从情节里综合出规律与内梗。

        这一步是「她懂你」的来源。抽取只能让她记得发生过什么，
        综合才能让她说出「你每次考完试都这样」。
        """
        lines = []
        for e in reversed(episodes):          # 按时间正序更容易看出规律
            when = parse_ts(e["happened_at"]).astimezone()
            lines.append(f"[{when:%m-%d}] {e['summary']}")

        existing = self.db.get_insights(save_id, limit=30)
        known = ""
        if existing:
            known = ("\n\n已经总结过的（如果下面的记录再次印证，"
                     "原样重复它；有新的就补充）：\n"
                     + "\n".join(f"- {i['content']}" for i in existing))

        try:
            completion = await self.provider.complete(
                LLMRequest(
                    messages=[
                        Message("system", INSIGHT_SYSTEM),
                        Message("user", "\n".join(lines) + known),
                    ],
                    task=Task.REFLECT,
                    json_mode=True,
                )
            )
        except LLMError as exc:
            log.error("save=%s 综合失败：%s", save_id, exc)
            return 0

        if self.recorder:
            self.recorder.record(completion)

        try:
            data = json.loads(completion.text)
        except json.JSONDecodeError:
            log.error("save=%s 综合返回非 JSON", save_id)
            return 0

        added = 0
        for item in data.get("insights") or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            kind = str(item.get("kind", "him")).strip()
            if not content or len(content) > 40 or kind not in ("him", "us", "joke"):
                continue
            self.db.upsert_insight(save_id, content, kind)
            added += 1
        return added
