"""模型价目表与成本计算。

单位：人民币元 / 百万 token。

⚠️ 价格会变，且下表来自 2026-08 的公开资料（部分由美元报价换算），**上线前请对着
官方价格页核一遍**。这里刻意做成可改的数据表而不是散落在代码里的常量。

本项目的成本结构很特殊：稳定前缀巨大（人设卡+语言指纹+事实池+记忆摘要 ≈ 3-5k token）、
输出很短（50-200 token）。因此：
  - 缓存命中价是输入侧的主导项
  - **输出价是总成本的主导项**（高命中率下能占 40-55%）
选型时先看输出价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..timewindow import price_multiplier
from .types import Cost, Task, Usage

_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """元 / 百万 token。"""

    cache_miss_in: float
    cache_hit_in: float
    output: float
    peak_pricing: bool = False
    """是否适用峰谷 ×2（DeepSeek 适用）。"""


PRICES: dict[str, ModelPrice] = {
    # DeepSeek 官方以人民币计价
    "deepseek-v4-flash": ModelPrice(
        cache_miss_in=1.00, cache_hit_in=0.02, output=2.00, peak_pricing=True
    ),
    "deepseek-v4-pro": ModelPrice(
        cache_miss_in=3.09, cache_hit_in=0.026, output=6.18, peak_pricing=True
    ),
    # 备选 provider，用于盲测与成本对照（美元报价按 7.1 换算，仅供估算）
    "gpt-5.4-nano": ModelPrice(cache_miss_in=1.42, cache_hit_in=0.14, output=8.88),
    "gpt-5.4-mini": ModelPrice(cache_miss_in=5.33, cache_hit_in=0.53, output=31.95),
    "kimi-k2.5": ModelPrice(cache_miss_in=4.26, cache_hit_in=0.71, output=17.75),
    "kimi-k3": ModelPrice(cache_miss_in=21.30, cache_hit_in=2.13, output=106.50),
}


def price_of(model: str) -> ModelPrice | None:
    return PRICES.get(model)


def compute_cost(model: str, usage: Usage, *, at=None) -> Cost:
    """按 usage 算钱。未知模型返回全零 —— 不猜价，宁可显示 0 也别给个假数字。"""
    price = PRICES.get(model)
    if price is None:
        return Cost()

    mult = price_multiplier(at) if price.peak_pricing else 1.0

    return Cost(
        cache_hit_cny=usage.cache_hit_tokens / _PER_MILLION * price.cache_hit_in * mult,
        cache_miss_cny=usage.cache_miss_tokens / _PER_MILLION * price.cache_miss_in * mult,
        # reasoning tokens 按输出计费，必须算进来，否则误开 thinking 时成本会凭空消失
        output_cny=usage.completion_tokens / _PER_MILLION * price.output * mult,
        peak_multiplier=mult,
    )


def estimate_daily_cny(
    model: str,
    *,
    calls_per_day: int,
    prompt_tokens: int,
    cache_hit_rate: float,
    output_tokens: int,
    peak_share: float = 0.0,
) -> float:
    """单用户日成本估算。选型和容量规划用。

    peak_share: 落在高峰时段的调用占比。我们的活跃在夜间，实际应该接近 0。
    """
    price = PRICES.get(model)
    if price is None:
        raise KeyError(f"未知模型价格：{model}")

    hit = prompt_tokens * cache_hit_rate
    miss = prompt_tokens - hit
    per_call = (
        hit / _PER_MILLION * price.cache_hit_in
        + miss / _PER_MILLION * price.cache_miss_in
        + output_tokens / _PER_MILLION * price.output
    )
    mult = 1.0 + peak_share if price.peak_pricing else 1.0
    return per_call * calls_per_day * mult


@dataclass(frozen=True, slots=True)
class TaskLoad:
    """单用户单日、某个任务的调用画像。用于全盘成本估算。"""

    task: Task
    calls_per_day: float
    prompt_tokens: int
    cache_hit_rate: float
    output_tokens: int
    """含 reasoning token —— PLAN 开着 thinking，别漏算。"""


# 三女主并行下的粗略画像。数字是估的，等线上有真实 usage 后用 metrics 的数据回填。
#
# 关键点：CHAT 只占总成本约三成。慢回路（REFLECT/PLAN）因为缓存命中率天然低
# （每天的对话都是新内容）且输出长，合起来比在线对话还贵。
# 所以"慢回路量小、可以上贵模型"这个直觉是错的。
DEFAULT_WORKLOAD: tuple[TaskLoad, ...] = (
    TaskLoad(Task.CHAT, calls_per_day=90, prompt_tokens=4000,
             cache_hit_rate=0.90, output_tokens=150),
    TaskLoad(Task.REFLECT, calls_per_day=15, prompt_tokens=6000,
             cache_hit_rate=0.40, output_tokens=800),
    TaskLoad(Task.PLAN, calls_per_day=9, prompt_tokens=5000,
             cache_hit_rate=0.50, output_tokens=2000),
    TaskLoad(Task.MODERATE, calls_per_day=90, prompt_tokens=1000,
             cache_hit_rate=0.80, output_tokens=50),
)


def estimate_workload_cny(
    model: str,
    loads: Iterable[TaskLoad] = DEFAULT_WORKLOAD,
    *,
    peak_share: float = 0.0,
) -> dict[Task, float]:
    """按任务拆的单用户日成本。返回 {task: 元}，便于看出钱花在哪。"""
    return {
        load.task: estimate_daily_cny(
            model,
            calls_per_day=int(load.calls_per_day),
            prompt_tokens=load.prompt_tokens,
            cache_hit_rate=load.cache_hit_rate,
            output_tokens=load.output_tokens,
            peak_share=peak_share,
        )
        for load in loads
    }
