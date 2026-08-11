from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from .types import Completion, LLMRequest, Usage


@dataclass(slots=True)
class StreamEvent:
    """流式增量。

    delta:      正文增量
    reasoning:  思维链增量（thinking 开启时才有；快回路不应出现）
    usage:      仅最后一个 event 携带
    done:       流结束
    """

    delta: str = ""
    reasoning: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    done: bool = False


@runtime_checkable
class Provider(Protocol):
    """LLM provider 抽象。

    DeepSeek 只是其中之一 —— 盲测阶段要能一行配置换到 GPT / Kimi 对照，
    上线后要能按女主、按任务切换和灰度。
    """

    name: str

    async def complete(self, req: LLMRequest) -> Completion: ...

    def stream(self, req: LLMRequest) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...
