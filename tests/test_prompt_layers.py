from __future__ import annotations

import pytest

from gfagent.llm.types import Message
from gfagent.prompt import (
    PromptBuilder,
    StablePrefix,
    VolatileContentInStableLayer,
    VolatileContext,
)


def test_stable_prefix_is_first_message():
    b = PromptBuilder(
        stable=StablePrefix(persona="我是林晚"),
        volatile=VolatileContext(clock="现在 21:40"),
        history=[Message("user", "在吗")],
    )
    msgs = b.build()
    assert msgs[0].role == "system"
    assert msgs[0].content == "我是林晚"


def test_volatile_goes_after_stable_in_its_own_message():
    """易变内容必须独立成条。拼进稳定层那条会改动字节，前缀匹配立刻断掉。"""
    b = PromptBuilder(
        stable=StablePrefix(persona="我是林晚"),
        volatile=VolatileContext(state="心情：累"),
        history=[Message("user", "在吗")],
    )
    msgs = b.build()
    assert len(msgs) == 3
    assert msgs[0].content == "我是林晚"
    assert msgs[1].content == "心情：累"
    assert msgs[2].role == "user"


def test_no_volatile_message_when_empty():
    b = PromptBuilder(stable=StablePrefix(persona="我是林晚"))
    assert len(b.build()) == 1


def test_fingerprint_stable_across_volatile_changes():
    stable = StablePrefix(persona="我是林晚", lexicon="句子短")
    fp = stable.fingerprint()

    a = PromptBuilder(stable=stable, volatile=VolatileContext(clock="21:40"))
    b = PromptBuilder(stable=stable, volatile=VolatileContext(clock="09:15"))
    assert a.stable_fingerprint() == b.stable_fingerprint() == fp


def test_fingerprint_changes_when_stable_changes():
    a = StablePrefix(persona="我是林晚").fingerprint()
    b = StablePrefix(persona="我是林晚", facts="她有只猫叫芝麻").fingerprint()
    assert a != b


@pytest.mark.parametrize(
    "bad",
    ["现在是 2026-08-04", "当前时间 14:32", "会话 3f2a1b4c-9d8e-4f11-a000-000000000000"],
)
def test_volatile_content_detected(bad: str):
    assert StablePrefix(persona=f"我是林晚。{bad}").check_volatile()


def test_strict_mode_raises_on_volatile():
    b = PromptBuilder(stable=StablePrefix(persona="我是林晚。现在 14:32"))
    with pytest.raises(VolatileContentInStableLayer):
        b.build()


def test_non_strict_mode_allows_volatile():
    b = PromptBuilder(stable=StablePrefix(persona="我是林晚。现在 14:32"), strict=False)
    assert b.build()


def test_clean_persona_passes():
    assert StablePrefix(persona="我是林晚，24岁，在书店工作").check_volatile() == []
