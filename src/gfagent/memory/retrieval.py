"""记忆检索：三因子打分。

对标 Stanford《Generative Agents》的检索模型：

    score = recency + importance + relevance

**为什么不能只取最新的。** 原来的 `get_episodes(limit=12)` 按时间倒序，
记忆一多，早期的重要事件就永远拿不到 —— 这是长期玩会"失忆"的机制性原因。
玩家在第 30 天提起第 3 天那件重要的事，她想不起来。

**relevance 用词＋字的重叠，不用 embedding。** 理由：

  - 省一个模型依赖和一次网络往返（检索在快回路上，延迟敏感）
  - 她的世界里名词就那么几十个，比的是「面馆」这种具体词
  - 中文单字信息量大，字级重叠能救「胃病 vs 胃疼」这类
    用词不同但明显相关的情况 —— 这本来是 embedding 的主场

以后要换 embedding，只需替换 `_relevance`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from ..life.daily import _STOP, _keywords

# 三因子权重。Stanford 原文全设为 1；这里略微抬高 relevance ——
# 聊天场景下"跟当前话题有关"比"最近发生"更重要。
W_RECENCY = 1.0
W_IMPORTANCE = 1.0
W_RELEVANCE = 1.5

RECENCY_HALFLIFE_DAYS = 14.0
"""记忆的新鲜度半衰期。两周前的事权重减半，但不会归零。"""


def _recency(happened: datetime, now: datetime) -> float:
    days = max(0.0, (now - happened).total_seconds() / 86400)
    return 0.5 ** (days / RECENCY_HALFLIFE_DAYS)


def _importance(raw: int) -> float:
    return max(0.0, min(1.0, (raw - 1) / 4))     # 1..5 → 0..1


def _content_chars(text: str) -> set[str]:
    """去掉虚词后的单字。

    中文单字信息量大（胃／雨／猫／灯），而且能跨过用词差异：
    「胃病」和「胃疼」的 n-gram 完全不重叠，但都含「胃」。
    """
    return {c for c in text if "一" <= c <= "鿿" and c not in _STOP}


def _relevance(text: str, context_keys: set[str], context_chars: set[str]) -> float:
    if not context_keys and not context_chars:
        return 0.0

    strong = 0.0
    if context_keys:
        hit = len(_keywords(text) & context_keys)
        strong = hit / len(context_keys)

    # 字级重叠是弱信号 —— 单字容易碰巧撞上，但它能救「胃病 vs 胃疼」这类
    weak = 0.0
    if context_chars:
        weak = len(_content_chars(text) & context_chars) / len(context_chars)

    combined = strong + 0.4 * weak
    # 开方压缩：命中一点就有明显分数，多命中收益递减
    return math.sqrt(min(1.0, combined))


def context_keywords(*texts: str) -> set[str]:
    """从当前语境（最近几句话）里抽关键词，作为 relevance 的比对基准。"""
    keys: set[str] = set()
    for t in texts:
        if t:
            keys |= _keywords(t)
    return keys


@dataclass(slots=True)
class Scored:
    row: dict[str, Any]
    score: float
    recency: float
    importance: float
    relevance: float

    @property
    def summary(self) -> str:
        return self.row.get("summary") or self.row.get("content") or ""


def rank_episodes(
    episodes: Sequence[dict[str, Any]],
    *,
    now: datetime,
    context: set[str],
    limit: int = 12,
) -> list[Scored]:
    """按 recency + importance + relevance 排序。

    注意返回的是**打分序**，不是时间序 —— 送进 prompt 前要按时间重排，
    否则她会觉得事情的先后顺序是乱的。
    """
    from ..storage.db import parse_ts

    context_ch = {c for k in context for c in k}
    out: list[Scored] = []
    for row in episodes:
        try:
            happened = parse_ts(row["happened_at"])
        except (ValueError, KeyError, TypeError):
            continue
        summary = row.get("summary", "")
        rec = _recency(happened, now)
        imp = _importance(int(row.get("importance", 1) or 1))
        rel = _relevance(summary, context, context_ch)
        out.append(Scored(
            row=row,
            score=W_RECENCY * rec + W_IMPORTANCE * imp + W_RELEVANCE * rel,
            recency=rec, importance=imp, relevance=rel,
        ))

    out.sort(key=lambda s: -s.score)
    return out[:limit]


def rank_facts(
    facts: Sequence[dict[str, Any]],
    *,
    context: set[str],
    limit: int = 24,
) -> list[dict[str, Any]]:
    """事实没有时间衰减 —— 「他有胃病」不会因为过了一个月就不成立。

    只按 relevance 排，但**永远保留一批**：即使跟当前话题无关，
    她也该记得他的基本情况。
    """
    if len(facts) <= limit:
        return list(facts)

    context_ch = {c for k in context for c in k}
    scored = sorted(
        facts,
        key=lambda f: -_relevance(f.get("content", ""), context, context_ch),
    )
    # 相关的占八成，剩下两成留给最近学到的，保证新事实不会被挤掉
    keep = int(limit * 0.8)
    recent = [f for f in facts if f not in scored[:keep]][:limit - keep]
    return scored[:keep] + recent
