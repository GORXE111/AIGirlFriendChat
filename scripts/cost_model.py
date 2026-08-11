"""成本模型速算。

    python scripts/cost_model.py [DAU]

按任务拆分单用户日成本，并对照"慢回路改走 Pro"的代价。
画像见 llm/pricing.py 的 DEFAULT_WORKLOAD —— 那是估的，线上有真实 usage 后
用 metrics 的数据回填。
"""

from __future__ import annotations

import sys

from gfagent.llm import (
    DEFAULT_WORKLOAD,
    PRIMARY_MODEL,
    Task,
    estimate_workload_cny,
)

SLOW_LOOP = (Task.REFLECT, Task.PLAN)


def breakdown(model: str) -> float:
    by_task = estimate_workload_cny(model)
    total = sum(by_task.values())
    print(f"\n{model}   ¥{total:.4f} / 人 / 天   （¥{total * 30:.2f} / 人 / 月）")
    for task, cny in sorted(by_task.items(), key=lambda kv: -kv[1]):
        bar = "█" * round(cny / total * 30)
        print(f"   {task.value:<9} ¥{cny:.4f}  {cny / total:>4.0%}  {bar}")
    return total


def mixed_cost() -> float:
    """慢回路走 Pro、其余走 Flash。"""
    total = 0.0
    for load in DEFAULT_WORKLOAD:
        model = "deepseek-v4-pro" if load.task in SLOW_LOOP else PRIMARY_MODEL
        total += estimate_workload_cny(model, [load])[load.task]
    return total


def main() -> int:
    dau = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

    all_flash = breakdown(PRIMARY_MODEL)
    mixed = mixed_cost()

    print(f"\n慢回路(REFLECT+PLAN)改走 Pro：¥{mixed:.4f} / 人 / 天  "
          f"（{mixed / all_flash:.2f}× 全 Flash）")

    print(f"\nDAU {dau:,} 的月度模型成本")
    print(f"   全 Flash   ¥{all_flash * 30 * dau:>12,.0f}")
    print(f"   混搭 Pro   ¥{mixed * 30 * dau:>12,.0f}")
    print(f"   差额       ¥{(mixed - all_flash) * 30 * dau:>12,.0f}")

    print("\n注：以上按低谷时段计价。DeepSeek 高峰（北京 9-12、14-18）×2，")
    print("    产品活跃在夜间，慢回路应主动避峰 —— 见 timewindow.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
