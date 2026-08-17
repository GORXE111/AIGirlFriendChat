"""跑玩家行为预测 —— 模拟玩家像不像那种玩家。

    python scripts/player_probe.py              # 全部题 × 3 次
    python scripts/player_probe.py --repeat 5
    python scripts/player_probe.py --no-motive  # 只测选择，省一半调用

**先看分离度，再看准确率。** 分离度低的话准确率高低都没意义 ——
那说明六个画像其实是同一个玩家，benchmark 里的跨画像对比全部作废。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

from gfagent.config import get_settings
from gfagent.evals.autoplay import AutoPlayer
from gfagent.evals.player_probe import ItemResult, judge_motive, load_items
from gfagent.llm import DeepSeekProvider
from gfagent.metrics import UsageRecorder
from gfagent.agent.turn import Option

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("logs/player_probe")


async def run_item(item, provider, repeat, check_motive) -> ItemResult:
    r = ItemResult(item=item)
    options = [Option(text=t, tone=tone) for t, tone in item.options]

    for style in item.expect:
        player = AutoPlayer(provider, style=style)
        r.picks[style] = []
        r.motives[style] = []
        r.motive_ok[style] = 0
        for _ in range(repeat):
            idx, why = await player.pick(item.transcript, options, "你可以说")
            r.picks[style].append(idx)
            r.motives[style].append(why)
            if check_motive and idx == item.expect[style].pick:
                if await judge_motive(item.options[idx][0],
                                      item.expect[style].motive, why, provider):
                    r.motive_ok[style] += 1
        hit = r.hits(style)
        print("✓" if hit == repeat else ("~" if hit else "✗"), end="", flush=True)
    return r


def report(results: list[ItemResult], repeat: int, check_motive: bool) -> dict:
    print(f"\n\n{'━' * 72}")
    print("  ① 画像分离度 —— **不依赖我写的答案对不对**")
    print("━" * 72)
    print("  低分离度 = 六个画像其实是同一个玩家，跨画像的 benchmark 全部作废\n")

    for r in results:
        got, want = r.separation(), r.expected_separation()
        modes = {s: max(set(p), key=p.count) for s, p in r.picks.items() if p}
        flag = "" if got >= want * 0.6 else "  ← **画像没起作用**"
        print(f"  {r.item.id:<20} 实际 {got:.2f} / 预期 {want:.2f}{flag}")
        print(f"    {'  '.join(f'{s}={i}' for s, i in modes.items())}")

    overall_got = statistics.fmean([r.separation() for r in results])
    overall_want = statistics.fmean([r.expected_separation() for r in results])
    print(f"\n  平均分离度 实际 {overall_got:.2f} / 预期 {overall_want:.2f}")

    print(f"\n{'━' * 72}")
    print("  ② 行为预测准确率（依赖我写的答案，答案本身可能有偏）")
    print("━" * 72)
    styles = sorted({s for r in results for s in r.item.expect})
    for style in styles:
        rows = [r for r in results if style in r.item.expect]
        hits = sum(r.hits(style) for r in rows)
        runs = len(rows) * repeat
        bar = "█" * int(round(hits / runs * 20)) if runs else ""
        line = f"  {style:<12} {hits:>2}/{runs:<3} {hits / runs:>4.0%} {bar}"
        if check_motive:
            mo = sum(r.motive_ok.get(style, 0) for r in rows)
            line += f"   动机对 {mo}/{hits}" if hits else "   动机对 —"
        print(line)

    print(f"\n{'━' * 72}")
    print("  ③ 分歧最大的题（模拟玩家跟预期差最远）")
    print("━" * 72)
    worst = sorted(results,
                   key=lambda r: sum(r.hits(s) for s in r.item.expect))[:3]
    for r in worst:
        print(f"\n  ▸ {r.item.id}   {r.item.why_discriminating}")
        for style, e in r.item.expect.items():
            got = r.picks.get(style, [])
            mode = max(set(got), key=got.count) if got else "?"
            mark = "" if mode == e.pick else "  ✗"
            print(f"      {style:<12} 预期 {e.pick} 实际 {mode}{mark}")
        if r.motives:
            s0 = next(iter(r.motives))
            if r.motives[s0]:
                print(f"      {s0} 的理由：{r.motives[s0][-1][:60]}")

    return {
        "at": datetime.now().isoformat(timespec="seconds"),
        "repeat": repeat,
        "separation": {"actual": round(overall_got, 3),
                       "expected": round(overall_want, 3)},
        "accuracy": {
            s: {
                "hits": sum(r.hits(s) for r in results if s in r.item.expect),
                "runs": len([r for r in results if s in r.item.expect]) * repeat,
                "motive_ok": sum(r.motive_ok.get(s, 0) for r in results),
            } for s in styles
        },
        "items": [
            {"id": r.item.id,
             "separation": round(r.separation(), 3),
             "expected_separation": round(r.expected_separation(), 3),
             "picks": r.picks, "motives": r.motives,
             "expect": {s: {"pick": e.pick, "motive": e.motive}
                        for s, e in r.item.expect.items()}}
            for r in results
        ],
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="玩家行为预测")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-motive", action="store_true")
    args = ap.parse_args()

    items = load_items()
    if not items:
        print("没有题库")
        return 2

    dull = [i.id for i in items if not i.discriminating]
    if dull:
        print(f"⚠️  这些题所有画像选的一样，没有分辨力：{dull}")

    s = get_settings()
    if not s.deepseek_api_key:
        print("没配 DEEPSEEK_API_KEY")
        return 2

    check_motive = not args.no_motive
    print(f"\n{len(items)} 道题 × {args.repeat} 次"
          f"{'（含动机核对）' if check_motive else ''}\n")

    results = []
    async with DeepSeekProvider(s) as provider:
        for it in items:
            print(f"  {it.id:<20} ", end="", flush=True)
            results.append(await run_item(it, provider, args.repeat, check_motive))
            print()

    result = report(results, args.repeat, check_motive)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{datetime.now():%m%d-%H%M}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n  写到 {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
