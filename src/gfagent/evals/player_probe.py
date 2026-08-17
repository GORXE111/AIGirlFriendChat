"""玩家行为预测 —— 模拟玩家像不像那种玩家。

arXiv:2404.18231 第四层（Decision-making）的多选题范式，**用在玩家侧**。

---

## 为什么这一层在我们这里比测她还基础

benchmark 和 A/B 的每一个数字都建立在一个没验过的假设上：
**模拟玩家是那种玩家的可信替身。**

它跟她有同一个病 —— 一个 LLM 被告知「你是玩过五十部 galgame 的人」，
它的自述会很像，它的**选择**未必像。

如果 galgamer 和 casual 实际选得一样，那六个画像就是装饰，
benchmark 里每格「不同画像」的数据都是重复的。

## 两个指标，第二个更硬

**行为预测准确率** —— 选的 == 该画像该选的。依赖我写的答案对不对。

**画像分离度** —— 不同画像在同一题上选得有多不同。
**这个不依赖答案对错，也不用 LLM。** 就算我的 ground truth 全写歪了，
分离度低也直接说明画像没起作用。

分离度用的是「一题里出现了几个不同选项 / 该题有几个画像作答」，
1.0 表示所有画像各选各的，0 附近表示画像之间没有区别。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..llm import DeepSeekProvider, LLMError, LLMRequest, Message, Task
from ..metrics import UsageRecorder
from ..persona.loader import content_root

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Expectation:
    pick: int
    motive: str


@dataclass(frozen=True, slots=True)
class PlayerItem:
    id: str
    transcript: str
    options: tuple[tuple[str, str], ...]
    """(text, tone)"""

    expect: dict[str, Expectation]
    why_discriminating: str = ""

    @property
    def discriminating(self) -> bool:
        """至少两个画像选不同的选项，否则这道题没有分辨力。"""
        return len({e.pick for e in self.expect.values()}) >= 2


@lru_cache(maxsize=2)
def load_items() -> tuple[PlayerItem, ...]:
    path: Path = content_root() / "players" / "probes.yaml"
    if not path.exists():
        log.warning("没有 %s，跳过玩家行为预测", path)
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for it in raw.get("items") or []:
        if not it.get("id"):
            continue
        opts = tuple(
            (str(o.get("text", "")), str(o.get("tone", "")))
            for o in it.get("options") or []
        )
        exp = {}
        for style, e in (it.get("expect") or {}).items():
            if isinstance(e, dict) and "pick" in e:
                idx = int(e["pick"])
                if 0 <= idx < len(opts):
                    exp[str(style)] = Expectation(
                        idx, str(e.get("motive", "")).strip())
        if opts and exp:
            out.append(PlayerItem(
                id=str(it["id"]), transcript=str(it.get("transcript", "")).strip(),
                options=opts, expect=exp,
                why_discriminating=str(it.get("why_discriminating", "")).strip()))
    return tuple(out)


@dataclass(slots=True)
class ItemResult:
    item: PlayerItem
    picks: dict[str, list[int]] = field(default_factory=dict)
    """画像 → 每次跑选了什么。"""

    motives: dict[str, list[str]] = field(default_factory=dict)
    motive_ok: dict[str, int] = field(default_factory=dict)

    def hits(self, style: str) -> int:
        want = self.item.expect[style].pick
        return sum(1 for p in self.picks.get(style, []) if p == want)

    def separation(self) -> float:
        """画像分离度 0–1。**不依赖 ground truth。**

        每个画像取它自己的众数选择，看这些众数里有几个不同的值。
        """
        modes = []
        for style, ps in self.picks.items():
            if ps:
                modes.append(max(set(ps), key=ps.count))
        if len(modes) < 2:
            return 0.0
        return (len(set(modes)) - 1) / (len(modes) - 1)

    def expected_separation(self) -> float:
        """我的答案本身有多大分辨力 —— 用来对比，判断是模拟不行还是题不行。"""
        picks = [e.pick for e in self.item.expect.values()]
        if len(picks) < 2:
            return 0.0
        return (len(set(picks)) - 1) / (len(picks) - 1)


MOTIVE_JUDGE = """判断两段「选择理由」说的是不是同一个动机。

不比文笔，不比长短。**只看动机是否一致。**

情景里他选了：{choice}

预期动机：{want}

他实际给的理由：{got}

⚠️ 选对了但理由完全不同，说明他是**蒙对的** —— 那要判 false。

严格输出 JSON：{{"same": true 或 false, "why": "一句话"}}"""


async def judge_motive(
    choice: str,
    want: str,
    got: str,
    provider: DeepSeekProvider,
    recorder: UsageRecorder | None = None,
) -> bool:
    """动机对不对。选对了但理由是错的，说明它是蒙对的。"""
    prompt = (MOTIVE_JUDGE.replace("{choice}", choice)
              .replace("{want}", want).replace("{got}", got))
    try:
        completion = await provider.complete(LLMRequest(
            messages=[Message("system", prompt)],
            task=Task.REFLECT, json_mode=True, max_tokens=200,
        ))
    except LLMError:
        return False
    if recorder:
        recorder.record(completion)
    try:
        return bool(json.loads(completion.text).get("same"))
    except json.JSONDecodeError:
        return False
