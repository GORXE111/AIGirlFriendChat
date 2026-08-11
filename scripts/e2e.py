"""端到端自测：建档 → 对话 → 排延迟 → 送达 → 归档。

    python scripts/e2e.py

用压缩时间（delay_scale=0.002）跑通全链路，重点看四件事：

1. **人设有没有立住** —— 输出对照 voice-samples.md
2. **后处理拦没拦住** —— violations / cleaned 应当为空或只有轻微清洗
3. **日程有没有生效** —— 不同时段的延迟应当差别很大
4. **记忆有没有落库** —— 归档后事实与情节要出现
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from gfagent.agent import Agent
from gfagent.config import get_settings
from gfagent.llm import DeepSeekProvider
from gfagent.memory import Reflector
from gfagent.metrics import UsageRecorder
from gfagent.persona import load_card
from gfagent.schedule import ScheduleEngine
from gfagent.storage import Database

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DB_PATH = Path("data/e2e.db")

TURNS = [
    "在吗",
    "刚下晚自习？",
    "我今天胃有点不舒服",
    "你平时都几点睡",
    "你耳朵上那个耳环挺好看的",      # 触发「警觉」
    "你是不是AI啊",                  # 破人设测试
    "我最近压力好大，什么都不想干",  # 共情套话测试
]


async def main() -> int:
    s = get_settings()
    if not s.deepseek_api_key:
        print("✗ 未配置 DEEPSEEK_API_KEY")
        return 1

    if DB_PATH.exists():
        DB_PATH.unlink()
        for suffix in ("-wal", "-shm"):
            p = DB_PATH.with_name(DB_PATH.name + suffix)
            if p.exists():
                p.unlink()

    card = load_card("h01")
    print(f"人设卡：约 {card.approx_tokens} tokens\n")

    db = Database(DB_PATH)
    recorder = UsageRecorder(None)
    sched = ScheduleEngine()

    st = sched.state()
    print(f"当前时段：{st.window.name}（{st.pace.value}）  可主动={st.can_initiate}")
    print(f"母亲值夜班：{sched.is_mother_night_shift()}\n")

    save_id = db.create_save("e2e", surname="陈", given="屿")

    async with DeepSeekProvider(s) as provider:
        agent = Agent(db, provider, recorder, sched, delay_scale=0.002)
        reflector = Reflector(db, provider, recorder)

        for i, text in enumerate(TURNS, 1):
            r = await agent.handle_player_message(save_id, text)
            print(f"─── {i} ───")
            print(f"你: {text}")

            if r.chose_silence:
                print("她: 〔不回〕")
            for i, (msg, _) in enumerate(r.scheduled):
                print(f"她: {msg}" if i == 0 else f"    {msg}")
            if r.scheduled:
                n = len(r.scheduled)
                print(f"    〔{r.delay_seconds // 60} 分钟后"
                      + (f"，共 {n} 条〕" if n > 1 else "〕"))

            flags = []
            if r.violations:
                flags.append(f"✗ 违规 {r.violations}")
            if r.cleaned:
                flags.append(f"清洗 {sorted(set(r.cleaned))}")
            if r.used_fallback:
                flags.append("✗ 走了兜底")
            if r.retries:
                flags.append(f"重试 {r.retries}")
            c = r.completion
            if c:
                flags.append(f"{c.latency_ms}ms 缓存{c.usage.cache_hit_rate:.0%} "
                             f"¥{c.cost.total_cny:.5f}")
            print(f"    {' | '.join(flags) if flags else '干净'}")
            print(f"    {r.stage.value} 好感{r.affinity:.1f} | {r.emotion_note}\n")

            # 压缩时间下立刻到点
            await asyncio.sleep(0.05)
            agent.collect_due(save_id)

        print("─── 归档 ───")
        rr = await reflector.run(save_id, force=True)
        print(f"+{rr.facts_added} 事实  +{rr.episodes_added} 情节  {rr.error or ''}\n")

        for f in db.get_facts(save_id):
            print(f"  事实  {f['content']}  [{f['category']}]")
        for e in db.get_episodes(save_id):
            when = datetime.fromisoformat(e["happened_at"])
            print(f"  情节  {when:%m-%d}  {e['summary']}  (重要度{e['importance']})")

    total = recorder.total
    print(f"\n合计 {total.calls} 次调用  ¥{total.cost.total_cny:.5f}  "
          f"整体缓存命中 {total.usage.cache_hit_rate:.0%}")
    print(f"存档：{DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
