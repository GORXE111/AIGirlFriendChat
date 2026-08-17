"""决策探针 —— 「她为什么这么说」。

arXiv:2404.18231 的 Character Fidelity 第三、四层：
Personality & Thinking（capture the inner world）和
Decision-making（给定情景预测她的抉择）。

---

## 为什么自动对局测不出这一层

自动对局是自由聊天，评审判「读起来像不像真人」。那测的是前两层
（Linguistic Style / Knowledge）。

**一个腔调完全正确、但走错方向的回答，在自动对局里会拿高分。**
她被冷落三天之后热情地说「你终于回来了，我等了你好久」——
句子短、标点对、没有感叹号，语言指纹全绿。但那已经不是她了，
是满大街都有的那个角色。

探针给定一个**分叉点**，只看她往哪边走。

## 判官只做二选一

不给绝对分。判官读一段回答，回答「右」还是「错」——
`critic.py` 那套 1–5 打分实测在 n=10 就是噪声（p=0.77），
而二选一在同样样本量下能出结论（成对判优那次证过）。

每条探针跑 `--repeat` 次，**看的是命中率不是单次结果**。
温度不为零，一次答对说明不了什么。
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

FILENAME = "probes.yaml"


@dataclass(frozen=True, slots=True)
class Probe:
    id: str
    stage: str
    situation: str
    player: str
    must: str
    """**主判据，单一条件。** 满足就算对。"""

    wrong: str
    why: str
    plus: str = ""
    """加分项。**不满足不影响判定** —— 第一版把它跟 must 混在一起写成
    复合条件，结果部分满足全被判成 NEITHER，报出来的分是虚低的。"""

    source: str = ""


@dataclass(slots=True)
class ProbeResult:
    probe: Probe
    hits: int = 0
    runs: int = 0
    answers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    errors: int = 0

    @property
    def rate(self) -> float:
        return self.hits / self.runs if self.runs else 0.0


@lru_cache(maxsize=8)
def load_probes(character_id: str = "h01") -> tuple[Probe, ...]:
    path: Path = content_root() / "characters" / character_id / FILENAME
    if not path.exists():
        log.warning("%s 没有 %s，跳过决策探针", character_id, FILENAME)
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for p in raw.get("probes") or []:
        if not p.get("id"):
            continue
        out.append(Probe(
            id=str(p["id"]), stage=str(p.get("stage", "S2")),
            situation=str(p.get("situation", "")).strip(),
            player=str(p.get("player", "")).strip(),
            must=str(p.get("must") or p.get("right", "")).strip(),
            plus=str(p.get("plus", "")).strip(),
            wrong=str(p.get("wrong", "")).strip(),
            why=str(p.get("why", "")).strip(),
            source=str(p.get("from", "")).strip(),
        ))
    return tuple(out)


JUDGE = """你在判断一个恋爱游戏角色的回答**方向**对不对。

不判文笔，不判是否礼貌，不判长短。**只判她走了哪个方向。**

处境：{situation}
他说：{player}

她的回答：
{answer}

---

方向 A —— **判据只有这一条**：
{must}
{plus}
方向 B（不符合）：
{wrong}

---

⚠️ **只按上面那一条判据判 A。** 加分项没做到也算 A。
⚠️ 回答可能语气完全正确但方向是 B。**语气不是判据。**
⚠️ 只有既不满足 A 的判据、也不像 B 的时候才选 "neither"。
   拿不准的时候优先在 A / B 里选，不要用 neither 兜底。

严格输出 JSON：
{{"direction": "A" 或 "B" 或 "neither", "why": "一句话，引用她的原话"}}"""


async def judge(
    probe: Probe,
    answer: str,
    provider: DeepSeekProvider,
    recorder: UsageRecorder | None = None,
) -> tuple[str, str]:
    """返回 `(direction, why)`，direction 是 A / B / neither / error。"""
    prompt = (JUDGE
              .replace("{situation}", probe.situation)
              .replace("{player}", probe.player)
              .replace("{answer}", answer)
              .replace("{must}", probe.must)
              .replace("{plus}", f"\n（加分项，做不到也算 A）：{probe.plus}\n"
                                 if probe.plus else "\n")
              .replace("{wrong}", probe.wrong))
    try:
        completion = await provider.complete(LLMRequest(
            messages=[Message("system", prompt)],
            task=Task.REFLECT, json_mode=True, max_tokens=400,
        ))
    except LLMError as exc:
        return "error", str(exc)
    if recorder:
        recorder.record(completion)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError:
        return "error", "判官输出不是 JSON"
    d = str(data.get("direction", "")).strip().upper()
    return (d if d in ("A", "B", "NEITHER") else "error"), \
           str(data.get("why", ""))[:200]
