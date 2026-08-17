"""全量基准 —— 现在到底什么水平。

    python scripts/benchmark.py                  # 4 阶段 × 3 画像 × 2 局
    python scripts/benchmark.py --games 1        # 快跑一遍
    python scripts/benchmark.py --stages s2 s3   # 只看后期

跟 `ab.py` 的分工：

    ab.py         比较两个变体，回答「这个改动有没有用」
    benchmark.py  刻画当前系统，回答「现在什么水平、哪里最弱」

---

## 为什么需要它

我们攒了很多机制（手滑、撤回、情绪崩溃、恢复阶梯、重话、阶段门控、
历史后指令），每一个都单独验过，但**从来没有一次把它们放在一起
看整体是什么样**。

单项 A/B 回答不了三个问题：

1. **哪个阶段最弱？** S0 的对局读起来跟 S3 完全不是一个东西，
   但我们的指标一直是混在一起看的
2. **机制在真实对局里触发吗？** 手滑是概率的、崩溃要连着两轮伤害、
   重话要选项里真出现那种话。**一次都不触发的机制等于不存在**
3. **不同玩家画像下表现一致吗？** 「懂套路的」和「没耐心的」
   会把她逼到完全不同的地方

## 怎么读

**机械指标看绝对值，评审分只看跨格对比。** 绝对打分在 n=10 就是噪声
（实测 p=0.77），拿它比 S0 和 S3 谁高没有意义 —— 但同一个评审在
同一批对局上给的相对高低还有点信息。

**最后要人读对局。** 这个脚本产出的是「哪里值得读」，不是结论。
情绪价值没有指标，只能看。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gfagent.agent import Agent
from gfagent.config import get_settings
from gfagent.evals import mechanical, play, review
from gfagent.llm import DeepSeekProvider
from gfagent.metrics import UsageRecorder
from gfagent.schedule import ScheduleEngine
from gfagent.storage import Database

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("logs/benchmark")

STAGES = ("s0", "s1", "s2", "s3")

PROFILES = ("experienced", "galgamer", "casual")
"""三个画像覆盖三种压力：

- `experienced` —— **对「不像真人」最敏感**。太懂事、太周到、回应得太完美都会被他抓到
- `galgamer` —— 懂套路，会主动往边界上试
- `casual` —— 没耐心，连着几轮无聊就不投入了。**最能测出「有没有勾住人」**
"""


@dataclass(slots=True)
class Cell:
    """矩阵里的一格：一个阶段 × 一个画像。"""

    stage: str
    profile: str
    scores: list[float] = field(default_factory=list)
    dims: list[dict[str, int]] = field(default_factory=list)
    review_failures: int = 0

    concrete: list[float] = field(default_factory=list)
    subjectless: list[float] = field(default_factory=list)
    comma: list[float] = field(default_factory=list)
    len_stdev: list[float] = field(default_factory=list)
    topic_spread: list[int] = field(default_factory=list)
    opt_variety: list[float] = field(default_factory=list)
    her_len: list[float] = field(default_factory=list)

    problems: Counter = field(default_factory=Counter)
    violations: int = 0
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)
    """被判违规丢掉的原文。**分辨真阳性和误报的唯一依据。**"""

    violation_kinds: Counter = field(default_factory=Counter)
    """违规的**类型**，不只是数量。

    只记数量的话，违规一涨就只能猜是哪条规则被踩了 —— 我已经猜错过两次
    （先怪 S3 内容改动，再怪复读检测；第二次才对了一半）。记下来就不用猜。
    """

    fallbacks: int = 0
    retries: int = 0

    slips: Counter = field(default_factory=Counter)
    crises: Counter = field(default_factory=Counter)
    overwhelms: Counter = field(default_factory=Counter)
    feelings: int = 0

    transcripts: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0


def _avg(xs) -> float:
    return statistics.fmean(xs) if xs else 0.0


async def run_cell(db, agent, provider, stage, profile, games, turns) -> Cell:
    cell = Cell(stage=stage, profile=profile)
    for _ in range(games):
        session = await play(db, agent, provider, preset=stage,
                             style=profile, turns=turns)
        m = mechanical(session)
        rev = await review(session, provider)

        if rev.ok:
            cell.scores.append(rev.average)
            cell.dims.append(rev.scores)
        else:
            cell.review_failures += 1

        cell.concrete.append(m.concrete_ratio)
        cell.subjectless.append(m.subjectless_ratio)
        cell.comma.append(m.comma_ratio)
        cell.len_stdev.append(m.len_stdev)
        cell.topic_spread.append(m.topic_spread)
        cell.opt_variety.append(m.option_text_variety)
        cell.her_len.append(m.avg_len)

        # 只留问题的**类型**，不留具体数字 —— 要看的是哪类问题反复出现
        for p in m.problems():
            cell.problems[p.split("：")[0].split("（")[0][:12]] += 1
        cell.violations += len(session.violations)
        cell.violation_kinds.update(session.violations)
        cell.rejected.extend(session.rejected)
        cell.fallbacks += session.fallbacks
        cell.retries += session.retries

        cell.slips.update(session.slips)
        cell.crises.update(session.crises)
        cell.overwhelms.update(session.overwhelms)
        cell.feelings += len(session.feelings)
        cell.transcripts.append(session.transcript())

        print("·" if rev.ok else "✗", end="", flush=True)
    return cell


def report(cells: list[Cell]) -> dict:
    print(f"\n\n{'━' * 74}")
    print("  逐格：评审分 / 具体性 / 省主语 / 话题跨度 / 选项差异 / 问题数")
    print("━" * 74)
    header = f"  {'':6}"
    for p in PROFILES:
        header += f"{p:>22}"
    print(header)

    by_stage: dict[str, list[Cell]] = {}
    for c in cells:
        by_stage.setdefault(c.stage, []).append(c)

    for stage in STAGES:
        row = by_stage.get(stage)
        if not row:
            continue
        line = f"  {stage.upper():6}"
        for p in PROFILES:
            c = next((x for x in row if x.profile == p), None)
            line += (f"{c.mean:>6.2f}{_avg(c.concrete):>7.0%}"
                     f"{_avg(c.topic_spread):>5.1f}"
                     f"{sum(c.problems.values()):>4}" if c else f"{'—':>22}")
        print(line)

    print(f"\n{'━' * 74}")
    print("  语言指纹（她的坐标：省主语 72%、带逗号约两成）")
    print("━" * 74)
    for stage in STAGES:
        row = by_stage.get(stage) or []
        if not row:
            continue
        sub = _avg([x for c in row for x in c.subjectless])
        com = _avg([x for c in row for x in c.comma])
        sd = _avg([x for c in row for x in c.len_stdev])
        ln = _avg([x for c in row for x in c.her_len])
        flag = "" if 0.45 <= sub else "  ← 偏离"
        print(f"  {stage.upper():4} 省主语 {sub:>5.0%}   带逗号 {com:>5.0%}   "
              f"句长 {ln:>4.1f}±{sd:<4.1f}{flag}")

    print(f"\n{'━' * 74}")
    print("  机制触发（**一次都不触发的机制等于不存在**）")
    print("━" * 74)
    tot_slips = Counter()
    tot_crisis = Counter()
    tot_ovw = Counter()
    tot_feel = games = 0
    for c in cells:
        tot_slips.update(c.slips)
        tot_crisis.update(c.crises)
        tot_ovw.update(c.overwhelms)
        tot_feel += c.feelings
        games += len(c.transcripts)
    print(f"  共 {games} 局")
    print(f"    手滑／撤回   {dict(tot_slips) or '一次都没有'}")
    print(f"    情绪变化     {tot_feel} 次（feeling 非空的回合）")
    print(f"    情绪崩溃     {dict(tot_ovw) or '一次都没有'}")
    print(f"    他说重话     {dict(tot_crisis) or '一次都没有'}")

    print(f"\n{'━' * 74}")
    print("  最常见的机械问题")
    print("━" * 74)
    allp = Counter()
    for c in cells:
        allp.update(c.problems)
    for name, n in allp.most_common(8):
        print(f"    {n:>3}×  {name}")

    v = sum(c.violations for c in cells)
    f = sum(c.fallbacks for c in cells)
    r = sum(c.retries for c in cells)
    rf = sum(c.review_failures for c in cells)
    kinds = Counter()
    for c in cells:
        kinds.update(c.violation_kinds)
    print(f"\n  人设违规 {v}　走兜底 {f}　重试 {r}　评审失败 {rf}")
    if kinds:
        print("    " + "　".join(f"{k}×{n}" for k, n in kinds.most_common()))
        # 原文必须打出来 —— 只看数量分不出真阳性和误报，
        # 而复读那个检测器已经改错过两次。
        for c in cells:
            for text, why in c.rejected[:2]:
                print(f"      [{'/'.join(why)}] {text[:56]}")

    print(f"\n{'━' * 74}")
    print("  评审分维度（跨格平均。**绝对值别当真，看相对高低**）")
    print("━" * 74)
    dims: dict[str, list[int]] = {}
    for c in cells:
        for d in c.dims:
            for k, val in d.items():
                dims.setdefault(k, []).append(val)
    for k, vals in sorted(dims.items(), key=lambda kv: _avg(kv[1])):
        bar = "█" * int(round(_avg(vals) * 4))
        print(f"    {k:<6} {_avg(vals):.2f}  {bar}")

    return {
        "at": datetime.now().isoformat(timespec="seconds"),
        "games": games,
        "cells": [
            {
                "stage": c.stage, "profile": c.profile,
                "review_mean": round(c.mean, 3),
                "review_failures": c.review_failures,
                "concrete": round(_avg(c.concrete), 3),
                "subjectless": round(_avg(c.subjectless), 3),
                "comma": round(_avg(c.comma), 3),
                "len_stdev": round(_avg(c.len_stdev), 2),
                "her_len": round(_avg(c.her_len), 1),
                "topic_spread": round(_avg(c.topic_spread), 1),
                "option_variety": round(_avg(c.opt_variety), 3),
                "problems": dict(c.problems),
                "violations": c.violations,
             "violation_kinds": dict(c.violation_kinds),
             "rejected": c.rejected,
             "fallbacks": c.fallbacks,
                "slips": dict(c.slips), "crises": dict(c.crises),
                "overwhelms": dict(c.overwhelms), "feelings": c.feelings,
                "transcripts": c.transcripts,
            }
            for c in cells
        ],
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="全量基准")
    ap.add_argument("--games", type=int, default=2, help="每格几局")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--profiles", nargs="*", default=list(PROFILES))
    args = ap.parse_args()

    s = get_settings()
    if not s.deepseek_api_key:
        print("没配 DEEPSEEK_API_KEY")
        return 2

    total = len(args.stages) * len(args.profiles) * args.games
    print(f"\n全量基准：{len(args.stages)} 阶段 × {len(args.profiles)} 画像 "
          f"× {args.games} 局 = {total} 局\n")

    db = Database("data/benchmark.db")
    cells: list[Cell] = []
    async with DeepSeekProvider(s) as provider:
        agent = Agent(db, provider, UsageRecorder(None),
                      ScheduleEngine(tz=s.story_timezone),
                      delay_scale=0.0, max_delay_seconds=0)
        for stage in args.stages:
            for profile in args.profiles:
                print(f"  {stage.upper():4} {profile:<12}", end="", flush=True)
                cells.append(await run_cell(
                    db, agent, provider, stage, profile,
                    args.games, args.turns))
                print()

    result = report(cells)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{datetime.now():%m%d-%H%M}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n  对局全文写到 {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
