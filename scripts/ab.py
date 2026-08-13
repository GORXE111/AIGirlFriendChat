"""输出契约 A/B —— 改 schema 之前必须跑的那一步。

    python scripts/ab.py --variant tail_rules --n 4
    python scripts/ab.py --variant feeling --n 6 --preset s2
    python scripts/ab.py --list

---

## 为什么需要它

我们的单次调用现在同时产出 **messages + options + outcome + feeling**。
每加一个字段，前面几个都可能变差 —— 模型的注意力是有限的，多一项任务
就少一分给别的。

但我们一直是**凭感觉加字段的**。「加个字段几乎免费」是错觉，而且这个错觉
在这个项目里已经生效过好几次：加了规则效果递减、指令越堆越长。

这个脚本把「改输出契约」变成一件有代价、可测量的事：

    关掉这个字段跑 N 局  vs  开着跑 N 局  →  比评审分数

如果开着更差，那这个字段要么删掉，要么换个位置放。

## 怎么读结果

分数是 LLM 评审给的 1–5，本身有噪声。所以：

- **只看方向和幅度，不要看小数点。** Δ < 0.3 基本等于没差别
- **N 至少 4，最好 6+。** 一局定不了任何事
- **机械指标比评审分数更可信** —— 它们不花钱、不抖动，
  违规数和兜底次数变多就是实打实变差了

## 怎么加新变体

在 `VARIANTS` 里加一条，写清 `off()` 怎么把这个字段关掉。
关闭方式必须是**运行时打补丁**，不要改源码 —— 否则这个脚本自己就成了
需要维护的第二份实现。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from gfagent.agent import Agent
from gfagent.config import get_settings
from gfagent.evals import mechanical, play, review
from gfagent.llm import DeepSeekProvider
from gfagent.metrics import UsageRecorder
from gfagent.schedule import ScheduleEngine
from gfagent.storage import Database

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

# Windows 控制台默认 GBK，中文之外的符号（⚠ ━ ↑）会直接抛
# UnicodeEncodeError 把脚本打死 —— 报告里全是这些。
with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("logs/ab")


# ---------------- 变体 ----------------


@contextlib.contextmanager
def _without_tail() -> Iterator[None]:
    """关掉历史后指令 —— 回到「所有规则都在 12k 前缀里」。"""
    from gfagent.agent import core

    original = core.tail_rules
    core.tail_rules = lambda stage="S0": ""
    try:
        yield
    finally:
        core.tail_rules = original


@contextlib.contextmanager
def _without_feeling() -> Iterator[None]:
    """关掉 feeling 字段 —— 从输出规格里删掉那一段，并丢弃解析结果。

    两处都要动：只删规格模型可能还是会输出；只丢结果的话它照样占了注意力。
    """
    from gfagent.agent import turn

    original_instructions = turn.instructions
    original_parse = turn.parse

    def instructions(**kw):
        text = original_instructions(**kw)
        head, sep, _ = text.partition("## 他这句话对她的影响（feeling）")
        if not sep:
            return text
        # 砍掉 feeling 那一节，保留它后面的内容
        tail = text[text.index(sep) :]
        rest = tail.split("\n\n", 1)
        return head + (rest[1] if len(rest) > 1 else "")

    def parse(text, outcome_ids):
        plan = original_parse(text, outcome_ids)
        plan.feeling = {}
        return plan

    turn.instructions = instructions
    turn.parse = parse
    # core 是 `from .turn import instructions, parse`，得单独打
    from gfagent.agent import core

    core_instructions, core_parse = core.instructions, core.parse
    core.instructions, core.parse = instructions, parse
    try:
        yield
    finally:
        turn.instructions, turn.parse = original_instructions, original_parse
        core.instructions, core.parse = core_instructions, core_parse


@contextlib.contextmanager
def _nothing() -> Iterator[None]:
    yield


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    what: str
    off: Callable[[], contextlib.AbstractContextManager]


VARIANTS: dict[str, Variant] = {
    v.name: v for v in (
        Variant("tail_rules", "历史后指令（PromptBuilder.tail）", _without_tail),
        Variant("feeling", "回合级情绪字段", _without_feeling),
    )
}


# ---------------- 跑 ----------------


@dataclass(slots=True)
class Arm:
    label: str
    averages: list[float] = field(default_factory=list)
    scores: list[dict[str, int]] = field(default_factory=list)
    violations: int = 0
    fallbacks: int = 0
    concrete: list[float] = field(default_factory=list)
    mech_problems: int = 0
    transcripts: list[str] = field(default_factory=list)
    """**必须留档。** 分数只告诉你哪边高，不告诉你为什么。

    第一次用这个脚本就想看对局到底哪不一样，结果没存 —— 白跑一次。
    """

    @property
    def mean(self) -> float:
        return statistics.fmean(self.averages) if self.averages else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.averages) if len(self.averages) > 1 else 0.0

    def by_dimension(self) -> dict[str, float]:
        keys = {k for s in self.scores for k in s}
        return {
            k: statistics.fmean([s[k] for s in self.scores if k in s])
            for k in sorted(keys)
        }


async def _one(db, agent, provider, recorder, preset, style, turns) -> tuple:
    session = await play(db, agent, provider, preset=preset, style=style,
                         turns=turns, recorder=recorder)
    return mechanical(session), await review(session, provider, recorder), session


async def run_arm(label, ctx, *, n, preset, style, turns) -> Arm:
    settings = get_settings()
    arm = Arm(label=label)
    print(f"\n  {label}", end="", flush=True)

    # 每局一个独立库 —— 存档状态（好感、记忆、情绪）会跨局累积，
    # 共用一个库的话第 4 局和第 1 局的起点根本不同，两边就不可比了。
    db = Database(f"data/ab_{label}.db")

    with ctx():
        async with DeepSeekProvider(settings) as provider:
            agent = Agent(db, provider, UsageRecorder(None),
                          ScheduleEngine(tz=settings.story_timezone),
                          delay_scale=0.0, max_delay_seconds=0)
            for _ in range(n):
                mech, rev, session = await _one(
                    db, agent, provider, None, preset, style, turns)

                arm.averages.append(rev.average)
                if rev.scores:
                    arm.scores.append(rev.scores)
                arm.violations += len(session.violations)
                arm.fallbacks += session.fallbacks
                arm.concrete.append(mech.concrete_ratio)
                arm.mech_problems += len(mech.problems())
                arm.transcripts.append(session.transcript())
                print(f" {rev.average:.1f}", end="", flush=True)
    print()
    return arm


def assert_arms_differ(variant: Variant) -> None:
    """开关两边**装出来的 prompt 必须真的不一样**。

    补丁悄悄失效的话，两边跑的是同一个东西，报告会得出「没差别」这个
    看起来很合理、实际上完全错误的结论 —— 而且没有任何迹象。

    不花 token：只装 prompt，不发请求。
    """
    from gfagent.agent import core

    kw = dict(her_max_chars=30, her_max_messages=2, in_beat=False,
              can_finish=False, outcome_ids=(), stage="S2")
    on = core.instructions(**kw) + core.tail_rules("S2")
    with variant.off():
        off = core.instructions(**kw) + core.tail_rules("S2")

    if on == off:
        raise SystemExit(
            f"✗ 变体 {variant.name} 的 off() 没有改变 prompt —— "
            "补丁失效了，跑了也是白跑。"
        )
    print(f"  （补丁生效：prompt 差 {abs(len(on) - len(off))} 字符）")


def report(variant: Variant, on: Arm, off: Arm, *, n: int) -> dict:
    delta = on.mean - off.mean
    print(f"\n{'━' * 62}")
    print(f"  {variant.name} —— {variant.what}")
    print(f"  每边 {n} 局")
    print("━" * 62)
    print(f"\n  评审均分   开 {on.mean:.2f} (σ{on.stdev:.2f})"
          f"   关 {off.mean:.2f} (σ{off.stdev:.2f})   Δ {delta:+.2f}")

    on_dims, off_dims = on.by_dimension(), off.by_dimension()
    if on_dims:
        print("\n  分维度：")
        for k in on_dims:
            d = on_dims[k] - off_dims.get(k, 0)
            mark = "  " if abs(d) < 0.3 else ("↑ " if d > 0 else "↓ ")
            print(f"    {mark}{k:<6} 开 {on_dims[k]:.1f}  关 {off_dims.get(k, 0):.1f}"
                  f"  Δ {d:+.1f}")

    print("\n  机械指标（不花钱、不抖动，比评审分更可信）：")
    print(f"    人设违规    开 {on.violations}   关 {off.violations}")
    print(f"    走兜底      开 {on.fallbacks}   关 {off.fallbacks}")
    print(f"    机械问题    开 {on.mech_problems}   关 {off.mech_problems}")
    print(f"    具体性      开 {statistics.fmean(on.concrete):.0%}"
          f"   关 {statistics.fmean(off.concrete):.0%}")

    # 判读。阈值定在 0.3 —— 评审分本身有噪声，比这小的差别读不出方向。
    if abs(delta) < 0.3:
        verdict = ("评审分看不出差别。看机械指标；如果也没差别，"
                   "这个字段既没帮忙也没添乱 —— 那就要问它值不值那些 token。")
    elif delta > 0:
        verdict = "开着更好。保留。"
    else:
        verdict = ("**开着更差。** 这个字段在抢注意力。"
                   "考虑：删掉／挪到历史后／拆成单独一次调用。")
    print(f"\n  {verdict}\n")

    return {
        "variant": variant.name, "n": n,
        "on": {"mean": round(on.mean, 3), "stdev": round(on.stdev, 3),
               "dims": {k: round(v, 2) for k, v in on_dims.items()},
               "violations": on.violations, "fallbacks": on.fallbacks,
               "mech_problems": on.mech_problems},
        "off": {"mean": round(off.mean, 3), "stdev": round(off.stdev, 3),
                "dims": {k: round(v, 2) for k, v in off_dims.items()},
                "violations": off.violations, "fallbacks": off.fallbacks,
                "mech_problems": off.mech_problems},
        "delta": round(delta, 3),
        "verdict": verdict,
        # 分数只说哪边高，不说为什么。差异要靠读对局。
        "transcripts": {"on": on.transcripts, "off": off.transcripts},
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="输出契约 A/B")
    ap.add_argument("--variant", help="要测哪个字段")
    ap.add_argument("--list", action="store_true", help="列出可测的变体")
    ap.add_argument("--n", type=int, default=4, help="每边跑几局（建议 ≥4）")
    ap.add_argument("--preset", default="s2")
    ap.add_argument("--style", default="normal")
    ap.add_argument("--turns", type=int, default=10)
    args = ap.parse_args()

    if args.list or not args.variant:
        print("\n可测的变体：\n")
        for v in VARIANTS.values():
            print(f"  {v.name:<12} {v.what}")
        print()
        return 0

    variant = VARIANTS.get(args.variant)
    if variant is None:
        print(f"没有这个变体：{args.variant}（--list 看全部）")
        return 2

    if not get_settings().deepseek_api_key:
        print("没配 DEEPSEEK_API_KEY")
        return 2

    if args.n < 3:
        print(f"⚠️  每边只跑 {args.n} 局，评审分的噪声会盖过真实差别。建议 ≥4。")

    print(f"\n跑 {variant.name}：每边 {args.n} 局，{args.preset} / "
          f"{args.style} / {args.turns} 轮")
    assert_arms_differ(variant)

    on = await run_arm("开", _nothing, n=args.n, preset=args.preset,
                       style=args.style, turns=args.turns)
    off = await run_arm("关", variant.off, n=args.n, preset=args.preset,
                        style=args.style, turns=args.turns)

    result = report(variant, on, off, n=args.n)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d-%H%M")
    path = OUT / f"{variant.name}-{stamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"  写到 {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
