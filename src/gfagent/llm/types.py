"""LLM 接入层的核心类型。

这一层刻意不依赖任何 provider —— DeepSeek 只是 provider 之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Task(str, Enum):
    """任务分层。模型路由以此为键 —— 不同任务对延迟、质量、成本的要求差异极大。"""

    CHAT = "chat"
    """快回路：玩家发消息，她回。高频、低延迟、短输出、要语感。thinking 必须关。"""

    REFLECT = "reflect"
    """慢回路：把对话压缩成情节记忆和事实。离线、批量、不在乎延迟。"""

    PLAN = "plan"
    """慢回路：规划主动消息的选题和时机。离线。"""

    MODERATE = "moderate"
    """审核与一致性检查。极高频、极短输出、要便宜。"""

    AUTHOR = "author"
    """离线内容预生成。质量优先，成本无所谓。"""


class Thinking(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class ReasoningEffort(str, Enum):
    LOW = "low"
    HIGH = "high"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率。这是本项目的头号成本指标 —— 掉下来就是有人污染了稳定前缀。"""
        cacheable = self.cache_hit_tokens + self.cache_miss_tokens
        if cacheable <= 0:
            return 0.0
        return self.cache_hit_tokens / cacheable

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_hit_tokens=self.cache_hit_tokens + other.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens + other.cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> Usage:
        if not raw:
            return cls()
        details = raw.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=raw.get("prompt_tokens", 0),
            completion_tokens=raw.get("completion_tokens", 0),
            cache_hit_tokens=raw.get("prompt_cache_hit_tokens", 0),
            cache_miss_tokens=raw.get("prompt_cache_miss_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0) or 0,
            total_tokens=raw.get("total_tokens", 0),
        )


@dataclass(frozen=True, slots=True)
class Cost:
    """人民币。分项保留，便于定位成本大头。"""

    cache_hit_cny: float = 0.0
    cache_miss_cny: float = 0.0
    output_cny: float = 0.0
    peak_multiplier: float = 1.0

    @property
    def total_cny(self) -> float:
        return self.cache_hit_cny + self.cache_miss_cny + self.output_cny

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            cache_hit_cny=self.cache_hit_cny + other.cache_hit_cny,
            cache_miss_cny=self.cache_miss_cny + other.cache_miss_cny,
            output_cny=self.output_cny + other.output_cny,
            # 累加后的倍率没有意义，置 1 表示"混合"
            peak_multiplier=1.0,
        )


@dataclass(slots=True)
class LLMRequest:
    messages: list[Message]
    task: Task = Task.CHAT
    character_id: str | None = None
    """哪个女主。用于模型路由和成本归因 —— 三个女主可以跑不同模型。"""

    model: str | None = None
    """显式指定模型。为 None 时由 router 按 (task, character_id) 决定。"""

    thinking: Thinking | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    json_mode: bool = False
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    usage: Usage
    cost: Cost
    latency_ms: int
    finish_reason: str | None = None
    task: Task = Task.CHAT
    character_id: str | None = None
    reasoning_text: str | None = None
    raw: dict[str, Any] | None = None
