"""用量与成本记账。

这一层不是可选的运维装饰品。本项目的成本模型完全建立在"缓存命中率 90%+"这个假设上，
而命中率是会被无声破坏的 —— 有人往人设卡里加了个时间戳，成本就悄悄涨 50 倍，
功能上一点问题都没有，没人会发现。所以命中率必须是一等观测指标。
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..llm.types import Completion, Cost, Task, Usage

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Bucket:
    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: Cost = field(default_factory=Cost)
    latency_ms_total: int = 0

    def add(self, c: Completion) -> None:
        self.calls += 1
        self.usage = self.usage + c.usage
        self.cost = self.cost + c.cost
        self.latency_ms_total += c.latency_ms

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.calls if self.calls else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "reasoning_tokens": self.usage.reasoning_tokens,
            "cache_hit_rate": round(self.usage.cache_hit_rate, 4),
            "cost_cny": round(self.cost.total_cny, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class UsageRecorder:
    """进程内聚合 + 可选 JSONL 落盘。

    落盘格式每行一条调用，方便直接喂给任何分析工具。上线后应换成时序库/数仓，
    但接口不变。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._by_model: dict[str, Bucket] = defaultdict(Bucket)
        self._by_task: dict[Task, Bucket] = defaultdict(Bucket)
        self._by_character: dict[str, Bucket] = defaultdict(Bucket)
        self._total = Bucket()

        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, c: Completion) -> None:
        with self._lock:
            self._total.add(c)
            self._by_model[c.model].add(c)
            self._by_task[c.task].add(c)
            if c.character_id:
                self._by_character[c.character_id].add(c)

        if self._path is not None:
            self._append(c)

    def _append(self, c: Completion) -> None:
        row = {
            "model": c.model,
            "task": c.task.value,
            "character_id": c.character_id,
            "prompt_tokens": c.usage.prompt_tokens,
            "completion_tokens": c.usage.completion_tokens,
            "reasoning_tokens": c.usage.reasoning_tokens,
            "cache_hit_tokens": c.usage.cache_hit_tokens,
            "cache_miss_tokens": c.usage.cache_miss_tokens,
            "cache_hit_rate": round(c.usage.cache_hit_rate, 4),
            "cost_cny": round(c.cost.total_cny, 8),
            "peak_multiplier": c.cost.peak_multiplier,
            "latency_ms": c.latency_ms,
            "finish_reason": c.finish_reason,
        }
        try:
            with self._path.open("a", encoding="utf-8") as f:  # type: ignore[union-attr]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            # 记账失败不能影响主链路
            log.warning("用量落盘失败：%s", exc)

    # ---------------- 读取 ----------------

    @property
    def total(self) -> Bucket:
        return self._total

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "total": self._total.as_dict(),
                "by_model": {k: v.as_dict() for k, v in self._by_model.items()},
                "by_task": {k.value: v.as_dict() for k, v in self._by_task.items()},
                "by_character": {k: v.as_dict() for k, v in self._by_character.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()
            self._by_task.clear()
            self._by_character.clear()
            self._total = Bucket()


def aggregate(completions: Iterable[Completion]) -> Bucket:
    b = Bucket()
    for c in completions:
        b.add(c)
    return b
