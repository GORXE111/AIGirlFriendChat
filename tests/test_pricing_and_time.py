from __future__ import annotations

from datetime import datetime

import pytest

from gfagent.llm.pricing import compute_cost, estimate_daily_cny
from gfagent.llm.types import Usage
from gfagent.timewindow import (
    BEIJING,
    is_peak,
    next_offpeak_window,
    price_multiplier,
    seconds_until_offpeak,
)


def bj(h: int, m: int = 0, day: int = 4) -> datetime:
    return datetime(2026, 8, day, h, m, tzinfo=BEIJING)


# ---------------- 峰谷 ----------------


@pytest.mark.parametrize("hour", [9, 10, 11, 14, 15, 17])
def test_peak_hours(hour: int):
    assert is_peak(bj(hour))
    assert price_multiplier(bj(hour)) == 2.0


@pytest.mark.parametrize("hour", [0, 8, 12, 13, 18, 21, 23])
def test_offpeak_hours(hour: int):
    assert not is_peak(bj(hour))
    assert price_multiplier(bj(hour)) == 1.0


def test_peak_boundaries_are_half_open():
    assert is_peak(bj(9, 0))
    assert not is_peak(bj(12, 0))
    assert is_peak(bj(14, 0))
    assert not is_peak(bj(18, 0))


def test_evening_activity_is_offpeak():
    """产品活跃高峰在夜间 —— 全部落在低谷区，这是我们的成本优势。"""
    assert all(not is_peak(bj(h)) for h in (19, 20, 21, 22, 23))


def test_seconds_until_offpeak():
    assert seconds_until_offpeak(bj(10, 30)) == 90 * 60
    assert seconds_until_offpeak(bj(21, 0)) == 0


def test_next_offpeak_window_from_peak():
    start, end = next_offpeak_window(bj(10, 0))
    assert start == bj(12, 0)
    assert end == bj(14, 0)


def test_next_offpeak_window_spans_to_next_day():
    start, end = next_offpeak_window(bj(20, 0))
    assert start == bj(20, 0)
    assert end == datetime(2026, 8, 5, 9, 0, tzinfo=BEIJING)


# ---------------- 计费 ----------------


def test_cost_offpeak():
    usage = Usage(
        prompt_tokens=4000,
        completion_tokens=150,
        cache_hit_tokens=3600,
        cache_miss_tokens=400,
    )
    cost = compute_cost("deepseek-v4-flash", usage, at=bj(21))
    assert cost.peak_multiplier == 1.0
    assert cost.cache_hit_cny == pytest.approx(3600 / 1e6 * 0.02)
    assert cost.cache_miss_cny == pytest.approx(400 / 1e6 * 1.00)
    assert cost.output_cny == pytest.approx(150 / 1e6 * 2.00)


def test_cost_doubles_at_peak():
    usage = Usage(completion_tokens=1000, cache_miss_tokens=1000)
    off = compute_cost("deepseek-v4-flash", usage, at=bj(21))
    on = compute_cost("deepseek-v4-flash", usage, at=bj(10))
    assert on.total_cny == pytest.approx(off.total_cny * 2)


def test_reasoning_tokens_are_billed_as_output():
    """误开 thinking 时成本不能凭空消失 —— reasoning 计入 completion_tokens。"""
    usage = Usage(completion_tokens=1200, reasoning_tokens=1050)
    cost = compute_cost("deepseek-v4-flash", usage, at=bj(21))
    assert cost.output_cny == pytest.approx(1200 / 1e6 * 2.00)


def test_unknown_model_returns_zero_not_a_guess():
    assert compute_cost("some-future-model", Usage(completion_tokens=999)).total_cny == 0.0


def test_cache_hit_rate():
    assert Usage(cache_hit_tokens=900, cache_miss_tokens=100).cache_hit_rate == 0.9
    assert Usage().cache_hit_rate == 0.0


def test_usage_is_addable():
    total = Usage(prompt_tokens=10, cache_hit_tokens=8) + Usage(
        prompt_tokens=20, cache_hit_tokens=16
    )
    assert total.prompt_tokens == 30
    assert total.cache_hit_tokens == 24


# ---------------- 选型对照 ----------------


def test_flash_is_cheapest_for_our_call_shape():
    """本项目的形状：大固定前缀 + 高命中 + 短输出。Flash 应明确胜出。"""
    shape = dict(
        calls_per_day=90, prompt_tokens=4000, cache_hit_rate=0.9, output_tokens=150
    )
    costs = {
        m: estimate_daily_cny(m, **shape)
        for m in ("deepseek-v4-flash", "deepseek-v4-pro", "gpt-5.4-nano", "kimi-k2.5")
    }
    assert min(costs, key=costs.__getitem__) == "deepseek-v4-flash"
    # 与第二名拉开明显差距
    ordered = sorted(costs.values())
    assert ordered[1] / ordered[0] > 2.5


def test_estimate_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_daily_cny(
            "nope", calls_per_day=1, prompt_tokens=1, cache_hit_rate=0.5, output_tokens=1
        )
