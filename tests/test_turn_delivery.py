"""排送达：手滑／撤回真的落到 messages 表上。

`test_slips.py` 测的是纯函数；这里测的是 `Agent._run_turn` 那段排期逻辑 ——
撤回时刻在送达之后、补发排在撤回之后、撤回的消息仍然进对话记录。
"""

from __future__ import annotations

import json
import random

import pytest

from gfagent.agent.core import Agent
from gfagent.llm.types import Completion, Cost, Task, Usage
from gfagent.storage.db import Database, parse_ts
from gfagent.state.models import Stage


class FakeProvider:
    """按脚本吐 JSON，不联网。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def complete(self, req):            # noqa: ANN001
        self.calls += 1
        return Completion(
            text=json.dumps(self.payload, ensure_ascii=False),
            model="fake", usage=Usage(), cost=Cost(),
            latency_ms=1, finish_reason="stop", task=Task.CHAT,
        )


def _agent(tmp_path, payload: dict, seed: int = 0) -> tuple[Agent, int]:
    db = Database(tmp_path / "t.db")
    save_id = db.create_save("t", surname="李", given="明")
    agent = Agent(db, FakeProvider(payload), rng=random.Random(seed),
                  delay_scale=0.0)
    return agent, save_id


def _arm(agent: Agent, save_id: int, options: list[str]) -> None:
    """给存档塞上待选项，好让 choose() 能走下去。"""
    agent.db.update_save(
        save_id,
        pending_options=json.dumps([{"text": o, "tone": "往前"} for o in options],
                                   ensure_ascii=False),
    )


def _messages(agent: Agent, save_id: int) -> list[dict]:
    return agent.db.recent_messages(save_id, limit=50, delivered_only=False)


# ---------------- 手滑 ----------------

@pytest.mark.asyncio
async def test_typo_never_fires_at_base_rate_in_a_short_run(tmp_path):
    """平静状态错字率 2%，一轮两条不该常态触发。"""
    agent, save_id = _agent(tmp_path, {
        "messages": ["刚到家。", "在写作业。"],
        "options": [{"text": "吃饭了吗", "tone": "往前"}],
    })
    _arm(agent, save_id, ["在吗"])
    result = await agent.choose(save_id, 0)
    assert result.slips == []


@pytest.mark.asyncio
async def test_regret_mark_produces_a_retraction(tmp_path):
    """模型标了 [收回]，S1 的 retract_rate=0.75 下应当能撤回。"""
    agent, save_id = _agent(tmp_path, {
        "messages": ["我妈今天值夜班。[收回]"],
        "options": [{"text": "那你一个人？", "tone": "往前"}],
    }, seed=3)
    agent.db.update_save(save_id, stage=Stage.S1.value)
    _arm(agent, save_id, ["在吗"])

    for _ in range(12):                     # 概率性，多试几次
        await agent.choose(save_id, 0)
        _arm(agent, save_id, ["在吗"])
        rows = [m for m in _messages(agent, save_id) if m["retract_at"]]
        if rows:
            break
    else:
        pytest.fail("12 轮都没撤回，retract_rate 没接上")

    row = rows[0]
    assert row["retract_kind"] == "说多了"
    assert "[收回]" not in row["content"], "标记必须被剥掉，不能发给玩家"
    assert row["content"] == "我妈今天值夜班。"


@pytest.mark.asyncio
async def test_retract_happens_after_delivery(tmp_path):
    """立刻撤等于没发过 —— 玩家连那行灰字都来不及注意。"""
    agent, save_id = _agent(tmp_path, {
        "messages": ["我妈今天值夜班。[收回]"],
        "options": [{"text": "哦", "tone": "守住"}],
    }, seed=3)
    agent.db.update_save(save_id, stage=Stage.S1.value)

    for _ in range(12):
        _arm(agent, save_id, ["在吗"])
        await agent.choose(save_id, 0)
        rows = [m for m in _messages(agent, save_id) if m["retract_at"]]
        if rows:
            break
    else:
        pytest.fail("没抽到撤回")

    row = rows[0]
    assert parse_ts(row["retract_at"]) > parse_ts(row["deliver_at"])


@pytest.mark.asyncio
async def test_followup_is_scheduled_after_the_retraction(tmp_path):
    """找补那句要排在撤回之后，不然玩家先看到「算了。」再看到消息消失。"""
    agent, save_id = _agent(tmp_path, {
        "messages": ["我妈今天值夜班。[收回]"],
        "options": [{"text": "哦", "tone": "守住"}],
    }, seed=3)
    agent.db.update_save(save_id, stage=Stage.S1.value)

    for _ in range(12):
        _arm(agent, save_id, ["在吗"])
        await agent.choose(save_id, 0)
        msgs = [m for m in _messages(agent, save_id) if m["role"] == "assistant"]
        retracted = [m for m in msgs if m["retract_at"]]
        if retracted:
            break
    else:
        pytest.fail("没抽到撤回")

    row = retracted[0]
    after = [m for m in msgs if m["id"] > row["id"]]
    assert after, "撤回之后应该有一句找补"
    assert parse_ts(after[0]["deliver_at"]) >= parse_ts(row["retract_at"])
    assert after[0]["content"] in ("……没什么。", "当我没说。", "算了。", "没事，你忙。")


# ---------------- 撤回后的记忆与投递 ----------------

@pytest.mark.asyncio
async def test_retracted_message_stays_in_her_transcript(tmp_path):
    """她记得自己差点说了什么，否则下一轮会若无其事再说一遍。"""
    agent, save_id = _agent(tmp_path, {
        "messages": ["我妈今天值夜班。[收回]"],
        "options": [{"text": "哦", "tone": "守住"}],
    }, seed=3)
    agent.db.update_save(save_id, stage=Stage.S1.value)

    for _ in range(12):
        _arm(agent, save_id, ["在吗"])
        await agent.choose(save_id, 0)
        if any(m["retract_at"] for m in _messages(agent, save_id)):
            break
    else:
        pytest.fail("没抽到撤回")

    transcript = agent._transcript(save_id)
    assert "我妈今天值夜班。" in transcript
    assert "撤回" in transcript, "记录里要标出这条被收回了"


@pytest.mark.asyncio
async def test_retraction_only_reported_after_delivery(tmp_path):
    """还没送达的消息不该出现在撤回列表里 —— 玩家都没收到，划掉什么。"""
    agent, save_id = _agent(tmp_path, {
        "messages": ["我妈今天值夜班。[收回]"],
        "options": [{"text": "哦", "tone": "守住"}],
    }, seed=3)
    agent.db.update_save(save_id, stage=Stage.S1.value)

    for _ in range(12):
        _arm(agent, save_id, ["在吗"])
        await agent.choose(save_id, 0)
        if any(m["retract_at"] for m in _messages(agent, save_id)):
            break
    else:
        pytest.fail("没抽到撤回")

    far_future = "2099-01-01T00:00:00+00:00"

    # 一条都还没送达 —— 即便撤回时刻早过了，也不该报
    assert agent.db.due_retractions(save_id, now=far_future) == []

    agent.db.mark_delivered([m["id"] for m in _messages(agent, save_id)])
    later = agent.db.due_retractions(save_id, now=far_future)
    assert later and later[0]["retract_kind"] == "说多了"


@pytest.mark.asyncio
async def test_she_never_sees_her_own_typos(tmp_path):
    """错字进历史 → 模型当成她的风格 → 下轮错更多。

    麦麦的 learn_style 明写「不要学习 SELF 的发言」防的就是这种自我强化。
    """
    agent, save_id = _agent(tmp_path, {
        "messages": ["我妈今天值夜班。"],
        "options": [{"text": "哦", "tone": "守住"}],
    }, seed=1)

    # 直接构造一条手滑记录，不靠概率
    agent.db.add_message(
        save_id, "assistant", "我妈今天直夜班。",
        meta={"slip": "手滑", "clean": "我妈今天值夜班。"},
    )
    transcript = agent._transcript(save_id)
    assert "我妈今天值夜班。" in transcript
    assert "直夜班" not in transcript, "错字漏进了给模型看的记录"


@pytest.mark.asyncio
async def test_clean_turn_writes_no_retract_columns(tmp_path):
    agent, save_id = _agent(tmp_path, {
        "messages": ["刚到家。"],
        "options": [{"text": "累不累", "tone": "往前"}],
    })
    _arm(agent, save_id, ["在吗"])
    await agent.choose(save_id, 0)

    for m in _messages(agent, save_id):
        assert m["retract_at"] is None
        assert m["retract_kind"] == ""
