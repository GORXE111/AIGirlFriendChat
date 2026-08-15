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
from gfagent.evals import compare_pair, mechanical, play, review, sign_test
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
    """关掉 feeling 字段 —— 从 tail 里删掉那一条，并丢弃解析结果。

    两处都要动：只删规格模型可能还是会输出；只丢结果的话它照样占了注意力。

    ⚠️ feeling 的规格 2026-08 从 `instructions()` 中段挪到了 `tail_rules()`。
    补丁点跟着挪 —— 留在旧位置的话补丁会静默失效，
    `assert_arms_differ` 会拦住，但那时候已经浪费了一次调试。
    """
    from gfagent.agent import core, turn

    original_tail = core.tail_rules
    original_parse = turn.parse

    def tail_rules(stage="S0"):
        text = original_tail(stage)
        # 砍掉第 5 条（feeling）及其续行，保留结尾的「现在输出 JSON」
        lines = [
            ln for ln in text.splitlines()
            if not ln.startswith("5. feeling") and not ln.startswith("   **大多数回合")
        ]
        return "\n".join(lines)

    def parse(text, outcome_ids):
        plan = original_parse(text, outcome_ids)
        plan.feeling = {}
        return plan

    # core 是 `from .turn import ...`，两个模块都要打
    turn.parse = parse
    core_parse = core.parse
    core.parse = parse
    core.tail_rules = tail_rules
    try:
        yield
    finally:
        turn.parse = original_parse
        core.parse = core_parse
        core.tail_rules = original_tail


@contextlib.contextmanager
def _without_stage_gating() -> Iterator[None]:
    """关掉人设卡的阶段门控 —— 回到「所有阶段的样本一起塞」。

    这个改动当初的理由是「不是省 token，是别给错示范」：S0 的语气样本里
    混着 S3 的「你过来」「想见你」，模型会往那个方向偏。理由说得通，
    但一直没有数据。

    ⚠️ 这个变体只在 **S0/S1** 上测得出东西 —— S3 本来就该看到 S3 的样本，
    门控开关对它几乎没区别。
    """
    from gfagent.agent import core

    original = core.load_card
    # 丢掉 stage，永远装全量卡
    core.load_card = lambda character_id="h01", stage="": original(character_id, "")
    try:
        yield
    finally:
        core.load_card = original


@contextlib.contextmanager
def _ungate(heading: str) -> Iterator[None]:
    """只把**一节**样本放回所有阶段，其余门控照旧。

    整包门控在 S0 实测五项指标全部偏负（成对判优 2:4，具体性 17% vs 24%，
    话题跨度 2.3 vs 3.3），但那测的是四道门捆在一起，**说不出是哪一道的锅**。

    我的判断是「九、S3 热恋期」挡得对、「五、试探」砍错了 ——
    后者是她在 S0 唯一的主动样本，砍掉之后她只会接话。两者在互相抵消。
    这个函数就是用来验证这个判断的。
    """
    from gfagent.persona import loader

    original = loader.SAMPLE_SECTIONS
    loader.SAMPLE_SECTIONS = tuple(
        # frozen dataclass，只能重建
        loader.Section(sec.file, sec.headings, sec.flatten_tables, None)
        if heading in sec.headings else sec
        for sec in original
    )
    loader.load_card.cache_clear()      # 不清就永远读到旧卡
    try:
        yield
    finally:
        loader.SAMPLE_SECTIONS = original
        loader.load_card.cache_clear()


def _gate_variant(name: str, heading: str, why: str) -> Variant:
    return Variant(f"gate_{name}", f"门控「{heading}」—— {why}",
                   lambda h=heading: _ungate(h))


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
        Variant("stage_gating", "人设卡按阶段门控样本（整包，建议 --preset s0/s1）",
                _without_stage_gating),
        # 拆开的四道门。整包测出「S0 变差」但分不清是哪道，这四个用来定位。
        # 全部要用 --preset s0 —— 其他阶段这些门本来就是开的，测不出东西。
        # gate_probe 已经结案 —— A/B 证实这道门砍错了，「五、试探」现在
        # 所有阶段都进卡（见 manifest），没有门可以再关。
        _gate_variant("retract", "六、撤回",
                      "小节自注「S1 几乎每次越界后都跟一条」"),
        _gate_variant("flustered", "八、慌（被撩 · S1）",
                      "标题里写着 S1"),
        _gate_variant("s3", "九、S3 · 热恋期",
                      "S3 直球，我认为这道挡得对"),
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


async def run_arm(label, ctx, *, variant, n, preset, style, turns) -> Arm:
    settings = get_settings()
    arm = Arm(label=label)
    print(f"\n  {label}", end="", flush=True)

    # 库名带上变体 —— 否则同时跑两个变体会撞在同一个文件上，
    # 两边的存档互相污染，而且不会报错，只会得出一堆没意义的数。
    db = Database(f"data/ab_{variant}_{label}.db")

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

    ⚠️ 比对的东西必须覆盖**所有变体可能动的地方**。只比 instructions
    会漏掉改人设卡的变体，然后误报「补丁失效」——那比不检查还糟。
    """
    def snapshot(stage: str) -> str:
        from gfagent.agent import core

        kw = dict(her_max_chars=30, her_max_messages=2, in_beat=False,
                  can_finish=False, outcome_ids=(), stage=stage)
        card = core.load_card("h01", stage)
        persona, lexicon = card.stable_text()
        return persona + lexicon + core.instructions(**kw) + core.tail_rules(stage)

    # 逐阶段比 —— 有些变体（阶段门控）只在特定阶段有区别
    diffs = {}
    for stage in ("S0", "S1", "S2", "S3"):
        on = snapshot(stage)
        with variant.off():
            off = snapshot(stage)
        if on != off:
            diffs[stage] = abs(len(on) - len(off))

    if not diffs:
        raise SystemExit(
            f"✗ 变体 {variant.name} 的 off() 在任何阶段都没有改变 prompt —— "
            "补丁失效了，跑了也是白跑。"
        )
    detail = "、".join(f"{s} 差 {n}" for s, n in diffs.items())
    print(f"  （补丁生效：{detail} 字符）")
    if stage_hint := {"S0", "S1", "S2", "S3"} - set(diffs):
        print(f"  （注意：{'/'.join(sorted(stage_hint))} 阶段两边完全一样，"
              f"用这些 preset 测不出东西）")


async def judge_pairs(on: Arm, off: Arm, *, stage: str) -> dict:
    """成对判优 —— A/B 真正能做决策的那一半。

    绝对打分在 n=6 测不出东西（实测 p=0.77 / 0.36）。这里改成把两边的对局
    并排给评审选，每对正反各判一次消除位置偏好。
    """
    settings = get_settings()
    pairs = min(len(on.transcripts), len(off.transcripts))
    if pairs == 0:
        return {}

    results: list[tuple[str, str]] = []
    print("\n  成对判优", end="", flush=True)
    async with DeepSeekProvider(settings) as provider:
        for i in range(pairs):
            verdict, why = await compare_pair(
                on.transcripts[i], off.transcripts[i], provider, stage=stage)
            results.append((verdict, why))
            print({"on": " 开", "off": " 关", "tie": " 平"}[verdict],
                  end="", flush=True)
    print()

    wins = sum(1 for v, _ in results if v == "on")
    losses = sum(1 for v, _ in results if v == "off")
    ties = sum(1 for v, _ in results if v == "tie")
    return {"wins": wins, "losses": losses, "ties": ties,
            "p": sign_test(wins, losses),
            "reasons": [w for v, w in results if v != "tie"][:4]}


def report(variant: Variant, on: Arm, off: Arm, *, n: int,
           pairwise: dict | None = None) -> dict:
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

    if pairwise:
        w, l, t = pairwise["wins"], pairwise["losses"], pairwise["ties"]
        p = pairwise["p"]
        print(f"\n  成对判优（每对正反各判一次，不一致算平）：")
        print(f"    开 {w} 胜　关 {l} 胜　平 {t}　符号检验 p={p:.3f}")
        if w + l > 0 and p >= 0.05:
            need = {2: 6, 4: 6, 6: 6, 8: 8, 10: 9}.get(w + l)
            hint = f"（{w + l} 对里要 {need} 胜才显著）" if need else ""
            print(f"    ⚠ 不显著 {hint}—— 对数不够，或者真没差别")
        for why in pairwise.get("reasons", [])[:2]:
            print(f"    · {why[:110]}")

    print("\n  机械指标（不花钱、不抖动，比评审分更可信）：")
    print(f"    人设违规    开 {on.violations}   关 {off.violations}")
    print(f"    走兜底      开 {on.fallbacks}   关 {off.fallbacks}")
    print(f"    机械问题    开 {on.mech_problems}   关 {off.mech_problems}")
    print(f"    具体性      开 {statistics.fmean(on.concrete):.0%}"
          f"   关 {statistics.fmean(off.concrete):.0%}")

    # 判读优先级：成对判优 > 机械指标 > 绝对打分。
    # 绝对打分排最后是因为实测它在 n=6 就是噪声（p=0.77 / 0.36）。
    if pairwise and pairwise["p"] < 0.05:
        better = "开" if pairwise["wins"] > pairwise["losses"] else "关"
        verdict = f"**成对判优显著：{better}着更好**（p={pairwise['p']:.3f}）。按这个定。"
    elif abs(delta) < 0.3:
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
        "pairwise": pairwise or None,
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
    ap.add_argument("--no-pairwise", action="store_true",
                    help="跳过成对判优（省 2×N 次调用，但基本等于放弃结论）")
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

    on = await run_arm("开", _nothing, variant=variant.name, n=args.n,
                       preset=args.preset, style=args.style, turns=args.turns)
    off = await run_arm("关", variant.off, variant=variant.name, n=args.n,
                        preset=args.preset, style=args.style, turns=args.turns)

    pairwise = {} if args.no_pairwise else await judge_pairs(
        on, off, stage=args.preset.upper())
    result = report(variant, on, off, n=args.n, pairwise=pairwise)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d-%H%M")
    path = OUT / f"{variant.name}-{stamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"  写到 {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
