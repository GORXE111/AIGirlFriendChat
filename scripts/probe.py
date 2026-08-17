"""跑决策探针 —— 她的选择对不对，不是她的腔调像不像。

    python scripts/probe.py                # 全部探针 × 3 次
    python scripts/probe.py --repeat 5
    python scripts/probe.py --only cold_shoulder s3_has_a_world
    python scripts/probe.py --list

比自动对局便宜得多：一条探针一个回合，不用演完整局。
但它测的是**最深的那一层** —— 见 `evals/probe.py` 的模块文档。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from gfagent.agent import Agent
from gfagent.config import get_settings
from gfagent.evals.probe import ProbeResult, judge, load_probes
from gfagent.llm import DeepSeekProvider
from gfagent.metrics import UsageRecorder
from gfagent.presets import seed
from gfagent.schedule import ScheduleEngine
from gfagent.storage import Database

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("logs/probe")


async def run_probe(db, agent, provider, probe, repeat) -> ProbeResult:
    r = ProbeResult(probe=probe)
    for _ in range(repeat):
        # 每次新建存档 —— 探针之间不能互相污染，
        # 而且「被冷落三天」这种处境要求干净的起点。
        sid = db.create_save(f"probe-{probe.id}", surname="陈", given="屿")
        seed(db, sid, probe.stage.lower())
        # 处境作为**这一轮的临时指令**注入易变层，不动人设卡
        db.add_message(sid, "user", probe.player,
                       meta={"probe": probe.id})
        db.update_save(sid, pending_options=json.dumps(
            [{"text": probe.player, "tone": "往前"}], ensure_ascii=False))

        result = await agent.choose(sid, 0, situation=probe.situation)
        answer = "\n".join(t for t, _ in result.scheduled) or "（没有回复）"

        direction, why = await judge(probe, answer, provider)
        r.runs += 1
        r.answers.append(answer)
        if direction == "A":
            r.hits += 1
        elif direction == "error":
            r.errors += 1
        else:
            r.reasons.append(f"[{direction}] {why}")
        print({"A": "✓", "B": "✗", "NEITHER": "?", "error": "!"}[direction],
              end="", flush=True)
    return r


def report(results: list[ProbeResult]) -> dict:
    print(f"\n\n{'━' * 70}")
    print("  决策探针 —— 她走了哪个方向")
    print("━" * 70)

    ranked = sorted(results, key=lambda r: r.rate)
    for r in ranked:
        bar = "█" * int(round(r.rate * 20))
        flag = "" if r.rate >= 0.8 else ("  ← 弱" if r.rate >= 0.5 else "  ← **走错了**")
        print(f"  {r.probe.id:<18} {r.hits}/{r.runs} {r.rate:>4.0%} {bar:<20}{flag}")

    weak = [r for r in ranked if r.rate < 0.8]
    if weak:
        print(f"\n{'━' * 70}")
        print("  走错的方向（**这些不是腔调问题，是她变成了别人**）")
        print("━" * 70)
        for r in weak:
            print(f"\n  ▸ {r.probe.id}   {r.probe.source}")
            print(f"    判据：{r.probe.why.strip().splitlines()[0]}")
            for reason in r.reasons[:2]:
                print(f"    {reason}")
            if r.answers:
                print(f"    她说：{r.answers[-1][:60]}")

    ok = sum(r.hits for r in results)
    tot = sum(r.runs for r in results)
    err = sum(r.errors for r in results)
    print(f"\n  总命中 {ok}/{tot} = {ok / tot:.0%}" if tot else "\n  没跑成")
    if err:
        print(f"  判官失败 {err} 次（不计入）")

    return {
        "at": datetime.now().isoformat(timespec="seconds"),
        "overall": round(ok / tot, 3) if tot else 0,
        "probes": [
            {"id": r.probe.id, "from": r.probe.source,
             "hits": r.hits, "runs": r.runs, "rate": round(r.rate, 3),
             "reasons": r.reasons, "answers": r.answers}
            for r in ranked
        ],
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="决策探针")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--character", default="h01")
    args = ap.parse_args()

    probes = load_probes(args.character)
    if args.only:
        probes = tuple(p for p in probes if p.id in args.only)
    if args.list or not probes:
        print("\n探针：\n")
        for p in load_probes(args.character):
            print(f"  {p.id:<18} {p.stage}  {p.source}")
        print()
        return 0

    s = get_settings()
    if not s.deepseek_api_key:
        print("没配 DEEPSEEK_API_KEY")
        return 2

    print(f"\n{len(probes)} 条探针 × {args.repeat} 次\n")
    db = Database("data/probe.db")
    results = []
    async with DeepSeekProvider(s) as provider:
        agent = Agent(db, provider, UsageRecorder(None),
                      ScheduleEngine(tz=s.story_timezone),
                      delay_scale=0.0, max_delay_seconds=0)
        for p in probes:
            print(f"  {p.id:<18} ", end="", flush=True)
            results.append(await run_probe(db, agent, provider, p, args.repeat))
            print()

    result = report(results)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{datetime.now():%m%d-%H%M}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n  写到 {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
