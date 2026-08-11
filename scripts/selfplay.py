"""自动对局 + 复盘。

    python scripts/selfplay.py                      # 热恋期，有恋爱经验的玩家，12 轮
    python scripts/selfplay.py --preset s0 --turns 10
    python scripts/selfplay.py --all                # 四个阶段各跑一局
    python scripts/selfplay.py --style galgamer     # 换玩家画像
    python scripts/selfplay.py --personas           # 同一阶段，六种玩家各跑一局

改完设定或 prompt 之后跑一遍，看分数和问题清单有没有变化。
比人肉玩快，而且**每次的评审标准一致**。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from gfagent.agent import Agent
from gfagent.config import get_settings
from gfagent.evals import PLAYER_PROFILES, mechanical, play, review
from gfagent.llm import DeepSeekProvider
from gfagent.metrics import UsageRecorder
from gfagent.schedule import ScheduleEngine
from gfagent.storage import Database

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

OUT = Path("logs/selfplay")


async def run_one(db, agent, provider, recorder, preset: str, style: str,
                  turns: int, show: bool) -> dict:
    print(f"\n{'━' * 62}")
    print(f"  {preset.upper()}　玩家：{style}　{turns} 轮")
    print("━" * 62)

    session = await play(db, agent, provider, preset=preset, style=style,
                         turns=turns, recorder=recorder)

    if show:
        print()
        for ln in session.lines:
            print("  " + ln.render())

    mech = mechanical(session)
    rev = await review(session, provider, recorder)

    print(f"\n  ── 机械检查 ──")
    print(f"  她说了 {mech.her_count} 条，平均 {mech.avg_len:.0f} 字"
          f"　具体性 {mech.concrete_ratio:.0%}　关系话 {mech.relational_ratio:.0%}"
          f"　选项语气种类 {mech.option_tone_variety:.1f}")
    problems = mech.problems()
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("  ✓ 没有机械问题")
    if session.violations:
        print(f"  ✗ 人设违规：{session.violations}")
    if session.fallbacks:
        print(f"  ✗ 走了 {session.fallbacks} 次兜底")

    print(f"\n  ── 评审 ──")
    if rev.scores:
        line = "　".join(f"{k} {v}" for k, v in rev.scores.items())
        print(f"  {line}　（均分 {rev.average:.1f}）")
    if rev.worst:
        print(f"\n  最严重：{rev.worst}")
    for p in rev.problems:
        print(f"\n  ✗ 「{p.get('quote','')}」")
        print(f"     问题：{p.get('issue','')}")
        print(f"     建议：{p.get('fix','')}")
    if rev.good:
        print()
        for g in rev.good:
            print(f"  ✓ {g}")
    if rev.verdict:
        print(f"\n  {rev.verdict}")

    print(f"\n  好感 {session.affinity_start:.0f} → {session.affinity_end:.0f}"
          f"　{session.stage_start} → {session.stage_end}"
          f"　桥段：{session.beats_played or '无'}")

    return {
        "preset": preset, "style": style, "turns": session.turns,
        "transcript": session.transcript(),
        "mechanical": {
            "her_count": mech.her_count,
            "avg_len": round(mech.avg_len, 1),
            "concrete_ratio": round(mech.concrete_ratio, 3),
            "relational_ratio": round(mech.relational_ratio, 3),
            "option_tone_variety": round(mech.option_tone_variety, 2),
            "problems": problems,
        },
        "violations": session.violations,
        "fallbacks": session.fallbacks,
        "scores": rev.scores,
        "average": round(rev.average, 2),
        "worst": rev.worst,
        "problems": rev.problems,
        "good": rev.good,
        "verdict": rev.verdict,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="s3", choices=["s0", "s1", "s2", "s3"])
    ap.add_argument("--style", default="experienced",
                    choices=list(PLAYER_PROFILES),
                    help="玩家画像：他是谁，不是他怎么选")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--all", action="store_true", help="四个阶段各跑一局")
    ap.add_argument("--personas", action="store_true",
                    help="同一阶段，换不同玩家画像各跑一局")
    ap.add_argument("--quiet", action="store_true", help="不打印对话全文")
    args = ap.parse_args()

    s = get_settings()
    if not s.deepseek_api_key:
        print("✗ 未配置 DEEPSEEK_API_KEY")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    db = Database("data/selfplay.db")
    recorder = UsageRecorder(None)

    async with DeepSeekProvider(s) as provider:
        agent = Agent(db, provider, recorder, ScheduleEngine(tz=s.story_timezone),
                      delay_scale=0.0, max_delay_seconds=0)
        if args.all:
            combos = [(p, args.style) for p in ("s0", "s1", "s2", "s3")]
        elif args.personas:
            combos = [(args.preset, st) for st in PLAYER_PROFILES]
        else:
            combos = [(args.preset, args.style)]
        reports = [
            await run_one(db, agent, provider, recorder, p, st,
                          args.turns, not args.quiet)
            for p, st in combos
        ]

    stamp = datetime.now().strftime("%m%d-%H%M")
    path = OUT / f"{stamp}.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print(f"\n{'━' * 62}")
    for r in reports:
        print(f"  {r['preset'].upper():4s} {r['style']:12s} 均分 {r['average']:.1f}"
              f"　具体性 {r['mechanical']['concrete_ratio']:.0%}"
              f"　机械 {len(r['mechanical']['problems'])}"
              f"　评审 {len(r['problems'])}")
    total = recorder.total
    print(f"\n  {total.calls} 次调用　¥{total.cost.total_cny:.4f}"
          f"　缓存 {total.usage.cache_hit_rate:.0%}")
    print(f"  报告：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
