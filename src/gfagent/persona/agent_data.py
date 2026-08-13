"""角色的**代码侧**数据 —— `content/characters/<id>/agent.yaml`。

跟 `loader.py` 的分工：

    loader.py     抽 .md 的小节 → 进 prompt，给模型读
    agent_data.py 读 agent.yaml → 进代码，给逻辑读

两边都以 `content/` 为源，谁都不复制内容。

## 为什么需要这个

崩溃期台词、恢复开场白、兜底话术、关心类型、角色名 —— 这些以前散在
`turn.py` / `core.py` / `postprocess.py` / `overwhelm.py` / `critic.py`。
做第二个女主得改 Python。

`manifest.py` 开头就写着「保持 content 是单一事实源，不复制内容」，
但一直没有承载**非 prompt 数据**的地方，所以它们只能落进源码。这就是那个地方。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .loader import content_root

log = logging.getLogger(__name__)

FILENAME = "agent.yaml"


@dataclass(frozen=True, slots=True)
class CareKind:
    pattern: str
    label: str


@dataclass(frozen=True, slots=True)
class AgentData:
    character_id: str
    name: str
    critic_brief: str = ""
    fallbacks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    broken_lines: dict[str, tuple[str, ...]] = field(default_factory=dict)
    recovery_openers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    care_kinds: tuple[CareKind, ...] = ()

    # ---- 取值 ----

    def fallback_pool(self, kind: str = "generic") -> tuple[str, ...]:
        return self.fallbacks.get(kind) or self.fallbacks.get("generic") or ("……",)

    def broken_pool(self, emotion: str) -> tuple[str, ...]:
        return (
            self.broken_lines.get(emotion)
            or self.broken_lines.get("_default")
            or ("……",)
        )

    def recovery_pool(self, stage: str) -> tuple[str, ...]:
        """崩溃只可能发生在 S1 以后，所以 S0 落到 S1 的池子。"""
        return (
            self.recovery_openers.get(stage)
            or self.recovery_openers.get("S1")
            or ("在干嘛。",)
        )


def _lines(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    return ()


def _pools(raw: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): _lines(v) for k, v in raw.items() if _lines(v)}


@lru_cache(maxsize=8)
def load_agent_data(character_id: str = "h01") -> AgentData:
    """读 agent.yaml。缺文件不报错 —— 退回一份空数据。

    **故意不硬失败。** 新角色刚建目录时还没有这份文件，那时候应该能跑起来
    看到别的东西缺什么，而不是卡在这一步。缺什么由各处的 `or` 兜底，
    并在这里 warn 一次。
    """
    path: Path = content_root() / "characters" / character_id / FILENAME
    if not path.exists():
        log.warning("%s 缺少 %s，代码侧角色数据将使用兜底值", character_id, FILENAME)
        return AgentData(character_id=character_id, name=character_id)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} 不是合法 YAML：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 顶层应该是一个映射")

    care = []
    for item in raw.get("care_kinds") or []:
        if isinstance(item, dict) and item.get("pattern") and item.get("label"):
            care.append(CareKind(str(item["pattern"]), str(item["label"])))

    return AgentData(
        character_id=character_id,
        name=str(raw.get("name") or character_id).strip(),
        critic_brief=str(raw.get("critic_brief") or "").strip(),
        fallbacks=_pools(raw.get("fallbacks")),
        broken_lines=_pools(raw.get("broken_lines")),
        recovery_openers=_pools(raw.get("recovery_openers")),
        care_kinds=tuple(care),
    )
