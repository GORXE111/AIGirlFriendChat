"""角色的代码侧数据：agent.yaml。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gfagent.persona.agent_data import load_agent_data

SRC = Path(__file__).resolve().parents[1] / "src" / "gfagent"


def test_loads_h01():
    d = load_agent_data("h01")
    assert d.name == "林静姝"
    assert d.critic_brief
    assert d.care_kinds and all(k.pattern and k.label for k in d.care_kinds)


def test_missing_file_does_not_explode():
    """新角色刚建目录时还没这份文件，那时候应该能跑起来看别的缺什么。"""
    d = load_agent_data("h99_不存在")
    assert d.fallback_pool() == ("……",)
    assert d.broken_pool("难过") == ("……",)
    assert d.recovery_pool("S2") == ("在干嘛。",)
    assert d.care_kinds == ()


def test_pools_fall_back_sensibly():
    d = load_agent_data("h01")
    assert d.broken_pool("慌") == d.broken_pool("_default")   # 没定义 → 默认池
    assert d.recovery_pool("S0") == d.recovery_pool("S1")     # S0 不会崩到这一步
    assert d.fallback_pool("不存在的类型") == d.fallback_pool("generic")


def test_care_patterns_compile():
    for kind in load_agent_data("h01").care_kinds:
        re.compile(kind.pattern)


def test_broken_lines_stay_short():
    """界面上那一句必须极短 —— 崩溃期她说不出完整的话。"""
    d = load_agent_data("h01")
    for pool in d.broken_lines.values():
        for line in pool:
            assert len(line) <= 8, line


def test_recovery_openers_never_apologise():
    """她不说对不起（edge-cases.md）。服软的方式是主动说件别的事。"""
    for pool in load_agent_data("h01").recovery_openers.values():
        for line in pool:
            assert "对不起" not in line and "抱歉" not in line


def test_no_character_lines_left_in_source():
    """**这条测试是 agent.yaml 存在的理由。**

    崩溃台词、恢复开场白、兜底话术、关心类型、角色名以前散在五个 py 文件里，
    做第二个女主得改 Python。抽走之后要防止它们回流。

    只查角色名出现在**代码**里的情况；注释里说明设计理由是可以的。
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue                      # 注释可以提她
            if "林静姝" in line:
                offenders.append(f"{path.relative_to(SRC)}:{lineno}")

    # slips.py 的模块 docstring 在解释「为什么不用拼音库」时引了她的人设，
    # 那是设计理由不是数据。
    offenders = [o for o in offenders if not o.startswith("output\\slips.py")
                 and not o.startswith("output/slips.py")]
    assert not offenders, "角色名回流到代码里了：\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("module,symbol", [
    ("gfagent.agent.core", "_RECOVERY_OPENERS"),
    ("gfagent.state.overwhelm", "_BROKEN_LINES"),
    ("gfagent.output.postprocess", "FALLBACKS"),
])
def test_old_hardcoded_tables_are_gone(module, symbol):
    import importlib

    mod = importlib.import_module(module)
    assert not hasattr(mod, symbol), f"{module}.{symbol} 又回来了"
