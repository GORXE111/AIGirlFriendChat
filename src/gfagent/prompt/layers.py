"""按易变度分层的 prompt 组装。

DeepSeek 的上下文缓存是**自动前缀匹配，且要求完全前缀命中，部分匹配不算**。
这条约束决定了 prompt 只能这样排：

    [ 稳定层 ]  人设卡 → 语言指纹 → 已解锁事实池 → 长期记忆摘要
    ------------ 缓存边界大致在这 ------------
    [ 易变层 ]  当前状态（情绪/好感/她在干嘛/现在几点） → 最近对话 → 本条消息

最常见也最致命的错误：把"现在是 2026-08-04 14:32"塞进 system prompt 开头。
那会让每一次调用都完全 miss，输入成本涨 50-120 倍。

`StablePrefix.fingerprint` 就是为了在线上抓这种事 —— 稳定层的指纹只应该在
版本更新或章节推进时变。变得比这频繁，说明有人往里塞了会动的东西。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..llm.types import Message

_SEP = "\n\n"

# 会让缓存失效的典型内容。不是穷举，是抓最常见的几种。
_VOLATILE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("时间戳", re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}")),
    ("时刻", re.compile(r"\d{1,2}:\d{2}")),
    ("UUID", re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}")),
)


class VolatileContentInStableLayer(ValueError):
    """稳定层里出现了看起来会变的内容。"""


@dataclass(frozen=True, slots=True)
class StablePrefix:
    """缓存友好的稳定前缀。内容只应在版本更新 / 章节推进时变化。

    persona:  人设卡 —— 她是谁。几乎永不变。
    lexicon:  语言指纹 + 样本台词 —— 决定"像不像这个人"。版本更新才变。
    facts:    当前章节已解锁的事实池 —— 章节推进时变。
    memory:   长期记忆摘要 —— 由慢回路定期重写（小时级），是稳定层里最活跃的一层，
              所以排在最后，重写时只失效它自己后面的部分。
    """

    persona: str
    lexicon: str = ""
    facts: str = ""
    memory: str = ""

    def render(self) -> str:
        parts = [p.strip() for p in (self.persona, self.lexicon, self.facts, self.memory)]
        return _SEP.join(p for p in parts if p)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()[:16]

    def check_volatile(self) -> list[str]:
        """返回检测到的可疑易变内容。空列表 = 干净。"""
        text = self.render()
        return [
            f"{label}: {m.group(0)!r}"
            for label, pat in _VOLATILE_PATTERNS
            for m in pat.finditer(text)
        ]

    def assert_stable(self) -> None:
        found = self.check_volatile()
        if found:
            raise VolatileContentInStableLayer(
                "稳定层出现易变内容，会击穿缓存（把它挪到 VolatileContext）：\n  "
                + "\n  ".join(found)
            )


@dataclass(slots=True)
class VolatileContext:
    """每次调用都可能变的部分。必须排在稳定前缀之后。

    state:    当前情绪 / 好感度 / 关系阶段 / 她此刻在干嘛
    clock:    现在几点、星期几、天气 —— 共享物理世界那一层
    notes:    本次调用的临时指令（例如"她刚被冷落三天，语气要疏远"）
    """

    state: str = ""
    clock: str = ""
    notes: str = ""

    def render(self) -> str:
        parts = [p.strip() for p in (self.clock, self.state, self.notes)]
        return _SEP.join(p for p in parts if p)


@dataclass(slots=True)
class PromptBuilder:
    stable: StablePrefix
    volatile: VolatileContext = field(default_factory=VolatileContext)
    history: list[Message] = field(default_factory=list)
    strict: bool = True
    """True 时稳定层含易变内容直接抛错。生产建议开着。"""

    def build(self) -> list[Message]:
        if self.strict:
            self.stable.assert_stable()

        messages: list[Message] = [Message("system", self.stable.render())]

        vol = self.volatile.render()
        if vol:
            # 独立成一条 system，而不是拼进上面那条 —— 拼进去会改动稳定前缀的字节，
            # 前缀匹配立刻断掉。
            messages.append(Message("system", vol))

        messages.extend(self.history)
        return messages

    def stable_fingerprint(self) -> str:
        return self.stable.fingerprint()
