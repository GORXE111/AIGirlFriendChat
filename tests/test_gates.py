"""闸门层：所有入口统一，不能有人绕过。"""

from __future__ import annotations

import json
import random
from datetime import timedelta

import pytest

from gfagent.agent import gates
from gfagent.agent.core import Agent
from gfagent.agent.gates import Disposition
from gfagent.agent.turn import SITUATION_TONE
from gfagent.llm.types import Completion, Cost, Task, Usage
from gfagent.state.models import Emotion, Stage
from gfagent.state.overwhelm import RUNG_MINUTES, Overwhelm, Rung
from gfagent.storage.db import Database


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, req):            # noqa: ANN001
        self.calls += 1
        return Completion(
            text=json.dumps({
                "messages": ["在写作业。"],
                "options": [{"text": "嗯", "tone": "守住"}],
                "topics": [{"title": "随便", "opener": "在干嘛"}],
            }, ensure_ascii=False),
            model="fake", usage=Usage(), cost=Cost(),
            latency_ms=1, finish_reason="stop", task=Task.CHAT,
        )


def _agent(tmp_path):
    db = Database(tmp_path / "t.db")
    save_id = db.create_save("t", surname="李", given="明")
    db.update_save(save_id, stage=Stage.S2.value, affinity=55.0)
    return Agent(db, FakeProvider(), rng=random.Random(0), delay_scale=0.0), save_id


def _break(agent, save_id, *, minutes_ago: float = 0.0) -> Overwhelm:
    b = Overwhelm(
        at=agent.schedule.now_local() - timedelta(minutes=minutes_ago),
        emo=Emotion.SAD, peak=0.9, cause="他说了很重的话",
    )
    agent.db.update_save(save_id, overwhelm=b.to_json())
    return b


# ---------------- 纯判定 ----------------

def test_no_breakdown_is_normal():
    r = gates.evaluate({"overwhelm": ""})
    assert r.normal and r.overwhelm is None and r.rung is None


def test_broken_is_situation():
    b = Overwhelm(at=__import__("datetime").datetime.now(
        __import__("datetime").timezone.utc), emo=Emotion.SAD, peak=0.9, cause="x")
    r = gates.evaluate({"overwhelm": b.to_json()})
    assert r.disposition is Disposition.SITUATION
    assert r.rung is Rung.BROKEN


def test_recovering_is_normal_but_carries_the_record():
    """缓过来了照常调模型，但行为受约束、延迟更长 —— 记录必须带出去。"""
    import datetime as dt
    at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        minutes=RUNG_MINUTES[Rung.BROKEN] + 5)
    b = Overwhelm(at=at, emo=Emotion.SAD, peak=0.9, cause="x")
    r = gates.evaluate({"overwhelm": b.to_json()})
    assert r.normal
    assert r.overwhelm is not None and r.rung is Rung.LENGTH


def test_recovered_still_carries_the_record_for_cleanup():
    """不带出去的话调用方无从清账，旧记录会挡住下一次崩溃。"""
    import datetime as dt
    at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    b = Overwhelm(at=at, emo=Emotion.SAD, peak=0.9, cause="x")
    r = gates.evaluate({"overwhelm": b.to_json()})
    assert r.normal
    assert r.overwhelm is not None and r.overwhelm.recovered()


def test_evaluate_has_no_side_effects():
    """闸不写库。清账由真的要跑这一轮的地方做。"""
    save = {"overwhelm": Overwhelm(
        at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc),
        emo=Emotion.SAD, peak=0.9, cause="x").to_json()}
    before = dict(save)
    gates.evaluate(save)
    assert save == before


# ---------------- 五个入口一个都不能漏 ----------------

ENTRIES = ("open_chat", "choose_topic", "start_beat", "refresh_topics", "choose")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_every_entry_is_gated(tmp_path, entry):
    """这条测试是这一层存在的理由。

    崩溃短路原来只写在 choose() 里，另外四个入口全绕过 ——
    她崩溃期间玩家点「换个话题」会完全正常地演完一场戏。
    """
    agent, save_id = _agent(tmp_path)
    agent.db.update_save(save_id, pending_options=json.dumps(
        [{"text": "你别这样", "tone": "往前"}], ensure_ascii=False))
    agent.db.update_save(save_id, pending_topics=json.dumps(
        [{"title": "随便", "opener": "在干嘛"}], ensure_ascii=False))
    _break(agent, save_id)

    before = agent.provider.calls
    if entry == "start_beat":
        beats = agent.available_beats(save_id)
        if not beats:
            pytest.skip("这个存档没有可开启的戏")
        r = await agent.start_beat(save_id, beats[0].id)
    elif entry in ("choose", "choose_topic"):
        r = await getattr(agent, entry)(save_id, 0)
    else:
        r = await getattr(agent, entry)(save_id)

    assert r.overwhelm == Rung.BROKEN.label, f"{entry} 绕过了闸"
    assert agent.provider.calls == before, f"{entry} 在崩溃期还调了模型"
    assert r.options and all(o.tone == SITUATION_TONE for o in r.options)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("open_chat", "refresh_topics"))
async def test_peeking_does_not_consume_an_action(tmp_path, entry):
    """打开聊天不是处置局面。每点一下就多「……」一条会像刷屏。"""
    agent, save_id = _agent(tmp_path)
    _break(agent, save_id)

    r = await getattr(agent, entry)(save_id)
    assert r.situation == "peek"
    assert r.scheduled == [], "只是看一眼，她不该又发一条"

    msgs = agent.db.recent_messages(save_id, limit=20, delivered_only=False)
    assert msgs == [], "只是看一眼，不该记玩家的话"


@pytest.mark.asyncio
async def test_choosing_a_situation_option_does_consume_it(tmp_path):
    agent, save_id = _agent(tmp_path)
    _break(agent, save_id)
    await agent.open_chat(save_id)              # 摆出局面

    r = await agent.choose(save_id, 0)          # 等一会
    assert r.situation == "wait"
    assert r.scheduled, "处置了局面，她该回一句"


@pytest.mark.asyncio
async def test_gate_reopens_after_recovery(tmp_path):
    """爬完阶梯之后一切恢复正常，且旧记录被清掉。"""
    agent, save_id = _agent(tmp_path)
    agent.db.update_save(save_id, pending_topics=json.dumps(
        [{"title": "随便", "opener": "在干嘛"}], ensure_ascii=False))
    _break(agent, save_id, minutes_ago=10_000)

    r = await agent.choose_topic(save_id, 0)
    assert not r.overwhelm
    assert agent.db.get_save(save_id)["overwhelm"] == ""
    assert agent.provider.calls > 0, "恢复之后应该照常调模型"
