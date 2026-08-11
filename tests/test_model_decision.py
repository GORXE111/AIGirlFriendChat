"""锁定 2026-08-04 的模型选型决策：全线 deepseek-v4-flash，单模型。

这些测试的作用是让"偏离决策"变成一件必须显式做的事 —— 有人改了路由或价目表，
这里会先叫出来，而不是等月底账单。
"""

from __future__ import annotations

import pytest

from gfagent.llm import (
    DEFAULT_ROUTES,
    DEFAULT_WORKLOAD,
    PRIMARY_MODEL,
    LLMRequest,
    Message,
    ModelRouter,
    ModelSpec,
    Task,
    Thinking,
    estimate_workload_cny,
)


def test_primary_model_is_flash():
    assert PRIMARY_MODEL == "deepseek-v4-flash"


def test_all_routes_use_the_single_primary_model():
    """单模型是决策的一部分 —— 只需校准一套 prompt 和一套语感。"""
    offenders = {t: s.model for t, s in DEFAULT_ROUTES.items() if s.model != PRIMARY_MODEL}
    assert not offenders, f"这些任务偏离了主力模型：{offenders}"


def test_every_task_has_a_route():
    assert set(DEFAULT_ROUTES) == set(Task)


def test_thinking_off_everywhere_except_plan():
    """thinking 会让输出更规整更助理味，且 reasoning token 按输出计费。
    只有 PLAN（离线多步选题）值得开。"""
    for task, spec in DEFAULT_ROUTES.items():
        expected = Thinking.ENABLED if task is Task.PLAN else Thinking.DISABLED
        assert spec.thinking is expected, f"{task} 的 thinking 设置不符合决策"


def test_chat_output_is_capped_short():
    assert DEFAULT_ROUTES[Task.CHAT].max_tokens is not None
    assert DEFAULT_ROUTES[Task.CHAT].max_tokens <= 300


def test_override_still_works_for_single_task():
    """决策是"默认单模型"，不是"禁止例外"。REFLECT 质量不够时要能单点换。"""
    router = ModelRouter()
    router.override("lin_wan", Task.REFLECT, ModelSpec(model="deepseek-v4-pro"))

    req = LLMRequest(
        messages=[Message("user", "x")], task=Task.REFLECT, character_id="lin_wan"
    )
    assert router.resolve(req).model == "deepseek-v4-pro"

    other = LLMRequest(
        messages=[Message("user", "x")], task=Task.REFLECT, character_id="gao_leng"
    )
    assert router.resolve(other).model == PRIMARY_MODEL


# ---------------- 成本结构 ----------------


def test_slow_loop_is_not_negligible():
    """曾经的错误直觉：慢回路量小所以可以上贵模型。

    实际上慢回路缓存命中率天然低（每天对话都是新内容）且输出长，
    REFLECT+PLAN 合计比在线对话还贵。这条测试把这个事实钉住。
    """
    by_task = estimate_workload_cny(PRIMARY_MODEL)
    chat = by_task[Task.CHAT]
    slow = by_task[Task.REFLECT] + by_task[Task.PLAN]
    assert slow > chat


def test_chat_is_minority_of_total_cost():
    by_task = estimate_workload_cny(PRIMARY_MODEL)
    total = sum(by_task.values())
    assert by_task[Task.CHAT] / total < 0.40


def test_all_flash_beats_mixed_flash_pro():
    """如果哪天有人想把慢回路挪回 Pro，这条会显示代价。"""
    all_flash = sum(estimate_workload_cny("deepseek-v4-flash").values())

    mixed = 0.0
    for load in DEFAULT_WORKLOAD:
        model = (
            "deepseek-v4-pro" if load.task in (Task.REFLECT, Task.PLAN) else "deepseek-v4-flash"
        )
        mixed += estimate_workload_cny(model, [load])[load.task]

    assert all_flash < mixed
    assert mixed / all_flash > 1.8


def test_workload_total_in_expected_ballpark():
    """粗估单用户日成本。数字变化超出量级说明画像或价目表被改了。"""
    total = sum(estimate_workload_cny(PRIMARY_MODEL).values())
    assert 0.1 < total < 0.5, f"单用户日成本估算 ¥{total:.3f}，超出预期区间"


def test_peak_share_increases_cost():
    off = sum(estimate_workload_cny(PRIMARY_MODEL, peak_share=0.0).values())
    on = sum(estimate_workload_cny(PRIMARY_MODEL, peak_share=1.0).values())
    assert on == pytest.approx(off * 2)
