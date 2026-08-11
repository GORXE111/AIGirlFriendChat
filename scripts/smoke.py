"""联通性 + 缓存行为自检。

    python scripts/smoke.py

做四件事：
1. 打通 DeepSeek API
2. 验证 thinking 确实被关掉了（reasoning_tokens 应为 0）
3. **验证缓存真的会命中** —— 用同一个稳定前缀连打两次，第二次命中率应显著上升。
   这是整个成本模型的地基，必须实测，不能假设。
4. 演示稳定层污染会被拦下来
"""

from __future__ import annotations

import asyncio
import logging
import sys

from gfagent.config import get_settings
from gfagent.llm import DeepSeekProvider, LLMError, LLMRequest, Message, Task
from gfagent.metrics import UsageRecorder
from gfagent.prompt import PromptBuilder, StablePrefix, VolatileContext
from gfagent.timewindow import is_peak, now_beijing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# 临时人设卡。真正的版本要由编剧按 schema 填，这里只为把稳定前缀撑到足够长
# （缓存按前缀单元匹配，前缀太短可能整段都不进缓存）。
PERSONA = """你叫林晚，24岁，在一家小型独立书店做店员，兼职写点东西。
你不是助手，不是AI，你就是林晚。你有自己的生活、情绪和脾气。

性格：表面随和，实际有点犟。不喜欢被安排。开心的时候话密，累的时候会直接说"不想说话"。
不会无条件顺着对方，觉得对方说得不对会直接怼回去，但怼完自己也会别扭。
""" + "\n生活细节：住在老城区的一室一厅，有只叫芝麻的橘猫。室友退租后一个人住。\n" * 12

LEXICON = """说话方式：
- 句子短。一次发一两句，很少超过25个字。
- 情绪上来会连发好几条。
- 用"嗯""诶""啊这""……"这类语气词，标点随意，句号经常省略。
- 几乎不用 emoji，偶尔用颜文字。
- 绝不总结、绝不分点、绝不每句话都反问对方。
- 打字快的时候会有错别字，不会纠正。
""" + "\n参考语气：「今天店里一个人都没有」「你猜怎么着」「算了不说了」「随便你吧」\n" * 12


async def main() -> int:
    settings = get_settings()
    if not settings.deepseek_api_key:
        print("✗ 未配置 DEEPSEEK_API_KEY —— 复制 .env.example 为 .env 并填入 key")
        return 1

    print(f"北京时间 {now_beijing():%Y-%m-%d %H:%M}  "
          f"计费时段: {'高峰 ×2' if is_peak() else '低谷 ×1'}\n")

    recorder = UsageRecorder(settings.usage_log_path)
    stable = StablePrefix(persona=PERSONA, lexicon=LEXICON)

    print(f"稳定前缀指纹: {stable.fingerprint()}")
    print(f"稳定层易变内容检查: {stable.check_volatile() or '干净'}\n")

    async with DeepSeekProvider(settings) as provider:
        turns = ["在干嘛", "今天店里人多吗"]
        history: list[Message] = []

        for i, text in enumerate(turns, 1):
            history.append(Message("user", text))
            builder = PromptBuilder(
                stable=stable,
                volatile=VolatileContext(
                    clock=f"现在是周三晚上 21:40，外面在下雨。",
                    state="心情：有点累，但不排斥聊天。好感度 32/100，关系阶段：普通朋友。",
                ),
                history=history,
            )

            try:
                c = await provider.complete(
                    LLMRequest(
                        messages=builder.build(),
                        task=Task.CHAT,
                        character_id="lin_wan",
                    )
                )
            except LLMError as exc:
                print(f"✗ 第 {i} 轮失败: {type(exc).__name__}: {exc}")
                return 1

            recorder.record(c)
            history.append(Message("assistant", c.text))

            print(f"--- 第 {i} 轮 ---")
            print(f"你: {text}")
            print(f"她: {c.text}")
            print(
                f"    {c.model} | {c.latency_ms}ms | "
                f"prompt {c.usage.prompt_tokens} "
                f"(命中 {c.usage.cache_hit_tokens} / 未命中 {c.usage.cache_miss_tokens}, "
                f"{c.usage.cache_hit_rate:.0%}) | "
                f"输出 {c.usage.completion_tokens} | "
                f"reasoning {c.usage.reasoning_tokens} | "
                f"¥{c.cost.total_cny:.6f}\n"
            )

            if c.usage.reasoning_tokens > 0:
                print("    ⚠ thinking 应该是关的，却产生了 reasoning tokens\n")

    total = recorder.total
    print("=" * 56)
    print(f"合计 {total.calls} 次调用，¥{total.cost.total_cny:.6f}，"
          f"整体缓存命中率 {total.usage.cache_hit_rate:.0%}")

    if total.usage.cache_hit_tokens == 0:
        print("\n⚠ 全程零缓存命中。可能是稳定前缀太短（未达缓存单元），")
        print("  也可能是首次调用尚未建立缓存 —— 再跑一次看第二次是否命中。")

    # 演示污染拦截
    print("\n--- 稳定层污染拦截演示 ---")
    bad = StablePrefix(persona=f"{PERSONA}\n现在时间是 2026-08-04 14:32。")
    try:
        PromptBuilder(stable=bad).build()
        print("✗ 没拦住，检查 assert_stable")
    except ValueError as exc:
        print(f"✓ 已拦截:\n{exc}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
