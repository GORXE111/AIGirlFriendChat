"""A/B 台架本身。

猴子补丁最容易出的事是**泄漏** —— 补丁没还原干净，后面所有测试都在
被污染的状态下跑，而且症状会出现在完全无关的地方。这里就是钉这个。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ab  # noqa: E402

from gfagent.agent import core, turn  # noqa: E402

KW = dict(her_max_chars=30, her_max_messages=2, in_beat=False,
          can_finish=False, outcome_ids=(), stage="S2")

SAMPLE = '{"messages":["刚到家。"],"options":[{"text":"嗯","tone":"守住"}],' \
         '"feeling":{"难过":0.4}}'


def test_variants_are_registered():
    assert set(ab.VARIANTS) == {
        "tail_rules", "feeling", "stage_gating",
        # 整包门控在 S0 测出「变差」但分不清是哪道门 —— 这四个用来定位
        "gate_retract", "gate_flustered", "gate_s3",
        # 新加的内容也要能被关掉验证 —— 不然就是凭论证改东西
        "initiative",
    }
    for v in ab.VARIANTS.values():
        assert v.what and callable(v.off)


def test_removing_a_section_leaves_the_rest_intact():
    """摘掉一节不能把同一个 Section 里打包的其他节一起带走。"""
    from gfagent.persona.loader import load_card

    base = load_card("h01", "S0").samples
    with ab.VARIANTS["initiative"].off():
        off = load_card("h01", "S0").samples
        assert "四点五、她主动说的" not in off
        assert "记得吃饭。" in off, "把同包的其他节一起摘掉了"
        assert "十五、禁用对照" in off
        assert "五、试探" in off
    assert load_card("h01", "S0").samples == base, "没还原（lru_cache 没清？）"


@pytest.mark.parametrize("name", ["gate_retract", "gate_flustered", "gate_s3"])
def test_each_gate_variant_isolates_one_section(name):
    """一个变体只能放回**一节**，否则测出来的还是混合效果。"""
    from gfagent.persona.loader import load_card

    base = load_card("h01", "S0").samples
    with ab.VARIANTS[name].off():
        opened = load_card("h01", "S0").samples
    assert len(opened) > len(base), "没放回任何东西"
    assert load_card("h01", "S0").samples == base, "补丁没还原（lru_cache 没清？）"


# ---------------- tail_rules ----------------

def test_without_tail_removes_it_and_restores():
    before = core.tail_rules("S2")
    assert before, "基线就没有 tail，这个变体测不出东西"

    with ab._without_tail():
        assert core.tail_rules("S2") == ""

    assert core.tail_rules("S2") == before


def test_without_tail_restores_on_exception():
    before = core.tail_rules("S2")
    with pytest.raises(RuntimeError):
        with ab._without_tail():
            raise RuntimeError("boom")
    assert core.tail_rules("S2") == before


# ---------------- feeling ----------------

def test_feeling_spec_lives_in_the_tail_not_mid_prompt():
    """A/B 显示 feeling 放在 instructions 中段会拖累其他输出。

    规格挪到 tail 之后，中段不该再有它 —— 两处都写等于白挪。
    """
    assert "feeling" not in core.instructions(**KW).replace('"feeling": {}', "")
    assert "feeling" in core.tail_rules("S2")


def test_without_feeling_strips_the_spec():
    base = core.tail_rules("S2")
    assert "feeling" in base

    with ab._without_feeling():
        off = core.tail_rules("S2")
        assert "feeling" not in off
        assert len(off) < len(base)
        assert "现在输出 JSON" in off, "别把结尾一起删了"

    assert core.tail_rules("S2") == base


def test_without_feeling_also_discards_parsed_values():
    """只删规格不够 —— 模型可能照样输出，那它还是占了注意力和 token。"""
    with ab._without_feeling():
        plan = core.parse(SAMPLE, ())
        assert plan.feeling == {}
        assert plan.messages == ["刚到家。"], "别把别的字段一起弄没了"
        assert plan.options, "别把别的字段一起弄没了"


def test_without_feeling_patches_both_modules():
    """core 是 `from .turn import parse`，只补 turn 不够。"""
    with ab._without_feeling():
        assert core.parse(SAMPLE, ()).feeling == {}
        assert turn.parse(SAMPLE, ()).feeling == {}


def test_without_feeling_restores_everything():
    orig_turn_parse, orig_core_parse = turn.parse, core.parse
    orig_turn_inst, orig_core_inst = turn.instructions, core.instructions

    with ab._without_feeling():
        pass

    assert turn.parse is orig_turn_parse
    assert core.parse is orig_core_parse
    assert turn.instructions is orig_turn_inst
    assert core.instructions is orig_core_inst
    assert core.parse(SAMPLE, ()).feeling == {"难过": pytest.approx(0.4)}


def test_without_feeling_restores_on_exception():
    with pytest.raises(RuntimeError):
        with ab._without_feeling():
            raise RuntimeError("boom")
    assert core.parse(SAMPLE, ()).feeling == {"难过": pytest.approx(0.4)}


# ---------------- 汇总 ----------------

def test_arm_stats():
    arm = ab.Arm(label="开")
    arm.averages = [3.0, 4.0, 3.5]
    arm.scores = [{"活人感": 3, "出戏": 4}, {"活人感": 5, "出戏": 4}]
    assert arm.mean == pytest.approx(3.5)
    assert arm.stdev > 0
    assert arm.by_dimension() == {"活人感": pytest.approx(4.0),
                                  "出戏": pytest.approx(4.0)}


def test_empty_arm_does_not_divide_by_zero():
    arm = ab.Arm(label="关")
    assert arm.mean == 0.0 and arm.stdev == 0.0 and arm.by_dimension() == {}
