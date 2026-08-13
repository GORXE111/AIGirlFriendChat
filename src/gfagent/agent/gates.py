"""闸门层 —— 这一轮该不该正常走。

标准 agent 是 **reactive** 的：调用即执行。玩家点一下，模型跑一次，一定产出回复。

活人不是这样。真人的「不理你」大部分时候不是想过之后决定不理，是**根本没进入
到「想」这一步**。麦麦（`study/maibot-modules.md` 第③层）为此在 Planner 之前
放了四道闸：focus 槽位、空闲退避、必要性评分、频率阈值 —— 全是纯规则，
**一个 token 都不花**。

这里是同一个位置，只是我们的闸不同（选项制下「完全不回」会被玩家当成故障，
见 `turn.SITUATION_OPTIONS`）。

---

## 为什么要单独一层

崩溃短路一开始是写在 `choose()` 里的一个 `if`。结果 `open_chat` /
`choose_topic` / `start_beat` / `refresh_topics` 四个入口全都绕过了它 ——
她崩溃期间玩家点「换个话题」，会完全正常地演完一场戏。

一个特例可以用 `if`；第二个特例就会变成散在五个入口里的五个 `if`，
而且每个都要记得改。后面还要加的闸不止一个（延后回复、存在感配额、
注意力漂移），所以现在就把位置留出来。

**入口只问一句「这一轮怎么处理」，不关心为什么。**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..state.overwhelm import Overwhelm, Rung


class Disposition(str, Enum):
    """这一轮的处理方式。"""

    NORMAL = "normal"
    """照常走：装 prompt、调模型、出台词和选项。"""

    SITUATION = "situation"
    """她崩着。不调模型，只回一句极短的，选项换成局面处置。"""


@dataclass(frozen=True, slots=True)
class GateResult:
    disposition: Disposition
    overwhelm: Overwhelm | None = None
    """当前的崩溃记录。`NORMAL` 时也可能非空 —— 那是恢复期，照常走但更慢。"""

    reason: str = ""
    """给日志和观测用。"""

    @property
    def normal(self) -> bool:
        return self.disposition is Disposition.NORMAL

    @property
    def rung(self) -> Rung | None:
        """恢复到第几级。没崩过就是 None。"""
        return self.overwhelm.rung() if self.overwhelm is not None else None


def evaluate(save: dict, *, now: datetime | None = None) -> GateResult:
    """所有入口的唯一问句。

    **不产生副作用** —— 不写库、不改状态。清账由 `Agent._run` 做，
    因为那里才知道这一轮真的要跑。
    """
    broken = Overwhelm.from_json(save.get("overwhelm"))
    if broken is None:
        return GateResult(Disposition.NORMAL)

    if broken.recovered(now):
        # 爬完阶梯了。照常走 —— 但**必须把记录带出去**，
        # 否则调用方拿不到东西可清，旧记录会永远留在库里挡住下一次崩溃。
        return GateResult(Disposition.NORMAL, overwhelm=broken,
                          reason="崩溃已恢复，待清账")

    rung = broken.rung(now)
    if rung is Rung.BROKEN:
        return GateResult(
            Disposition.SITUATION, overwhelm=broken,
            reason=f"崩溃中（{broken.emo.value}）",
        )

    # 缓过来了但还没缓透 —— 照常调模型，但慢，而且行为受约束
    # （`overwhelm.behavior_note` 注入易变层）。
    return GateResult(
        Disposition.NORMAL, overwhelm=broken,
        reason=f"恢复中（{rung.label}）",
    )
