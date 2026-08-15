"""崩溃在真实回合里跑通：触发 → 局面选项 → 恢复 → 主动。"""

from __future__ import annotations

import json
import random
from datetime import timedelta

import pytest

from gfagent.agent.core import Agent
from gfagent.agent.turn import SITUATION_TONE
from gfagent.llm.types import Completion, Cost, Task, Usage
from gfagent.state.models import Emotion, EmotionState, Stage
from gfagent.state.overwhelm import RUNG_MINUTES, Overwhelm, Rung
from gfagent.storage.db import Database, parse_ts


class FakeProvider:
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


def _setup(tmp_path, payload, *, stage=Stage.S2, seed=0):
    db = Database(tmp_path / "t.db")
    save_id = db.create_save("t", surname="李", given="明")
    db.update_save(save_id, stage=stage.value, affinity=55.0)
    agent = Agent(db, FakeProvider(payload), rng=random.Random(seed),
                  delay_scale=0.0)
    return agent, save_id


def _arm(agent, save_id, text="你别这样"):
    agent.db.update_save(save_id, pending_options=json.dumps(
        [{"text": text, "tone": "往前"}], ensure_ascii=False))


def _hurt(agent, save_id, delta: float):
    """让下一轮模型报出这么大的伤害。"""
    agent.provider.payload["feeling"] = {"难过": delta}


# ---------------- 回合级情绪 ----------------

@pytest.mark.asyncio
async def test_feeling_moves_her_emotions(tmp_path):
    """在这之前，回合里玩家说什么对情绪的影响是零。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["嗯。"], "options": [{"text": "好", "tone": "守住"}],
        "feeling": {"难过": 0.4},
    })
    _arm(agent, save_id)
    r = await agent.choose(save_id, 0)

    assert r.feeling == {"难过": pytest.approx(0.4)}
    saved = EmotionState.from_json(agent.db.get_save(save_id)["emotions"])
    assert saved.decayed()[Emotion.SAD] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_feeling_delta_is_capped(tmp_path):
    """模型报 0.95 也只能涨 MAX_TURN_DELTA —— 一句话崩不了。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "对不起", "tone": "守住"}],
        "feeling": {"难过": 0.95},
    })
    _arm(agent, save_id)
    r = await agent.choose(save_id, 0)
    assert r.feeling["难过"] == pytest.approx(0.5)
    assert not r.overwhelm, "一轮就崩了，累积机制没生效"


@pytest.mark.asyncio
async def test_negative_delta_soothes(tmp_path):
    agent, save_id = _setup(tmp_path, {
        "messages": ["嗯。"], "options": [{"text": "我在", "tone": "守住"}],
        "feeling": {"委屈": -0.3},
    })
    st = EmotionState()
    st.bump(Emotion.HURT, 0.6, Stage.S2)
    agent.db.update_save(save_id, emotions=st.to_json())

    _arm(agent, save_id)
    await agent.choose(save_id, 0)
    after = EmotionState.from_json(agent.db.get_save(save_id)["emotions"])
    assert after.decayed().get(Emotion.HURT, 0) == pytest.approx(0.3, abs=0.02)


@pytest.mark.asyncio
async def test_hurt_no_longer_heals_while_he_is_hurting_her(tmp_path):
    """原来每回合无条件 soothe(委屈, 0.06) —— 他说难听话也会让她不那么委屈。

    委屈是「你没发现」，不是「你没说话」。
    """
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "随便", "tone": "后退"}],
        "feeling": {"委屈": 0.2},
    })
    st = EmotionState()
    st.bump(Emotion.HURT, 0.5, Stage.S2)
    agent.db.update_save(save_id, emotions=st.to_json())

    _arm(agent, save_id)
    await agent.choose(save_id, 0)
    after = EmotionState.from_json(agent.db.get_save(save_id)["emotions"])
    # 涨了 0.2，且**没有**被那条无条件消解抵掉 0.06
    assert after.decayed()[Emotion.HURT] == pytest.approx(0.7, abs=0.02)


@pytest.mark.asyncio
async def test_unknown_emotion_is_ignored_not_fatal(tmp_path):
    agent, save_id = _setup(tmp_path, {
        "messages": ["嗯。"], "options": [{"text": "好", "tone": "守住"}],
        "feeling": {"忧郁": 0.4, "难过": 0.2},
    })
    _arm(agent, save_id)
    r = await agent.choose(save_id, 0)
    assert r.feeling == {"难过": pytest.approx(0.2)}


# ---------------- 崩溃 ----------------

@pytest.mark.asyncio
async def test_two_bad_turns_break_her(tmp_path):
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "你什么意思", "tone": "越界"}],
        "feeling": {"难过": 0.5},
    })
    _arm(agent, save_id)
    first = await agent.choose(save_id, 0)
    assert not first.overwhelm

    _arm(agent, save_id)
    second = await agent.choose(save_id, 0)
    assert second.overwhelm == Rung.BROKEN.label

    saved = Overwhelm.from_json(agent.db.get_save(save_id)["overwhelm"])
    assert saved is not None and saved.emo is Emotion.SAD
    assert saved.cause, "崩溃必须记下起因"


@pytest.mark.asyncio
async def test_breakdown_switches_to_situation_options(tmp_path):
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        r = await agent.choose(save_id, 0)

    assert r.overwhelm
    assert r.options and all(o.tone == SITUATION_TONE for o in r.options)
    assert [o.text for o in r.options] == ["等一会。", "再说一句。", "明天再找她。"]


@pytest.mark.asyncio
async def test_apology_is_scheduled_for_the_future(tmp_path):
    """她的道歉＝恢复主动。复用 deliver_at 排期，不需要额外定时器。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        await agent.choose(save_id, 0)

    rows = agent.db.recent_messages(save_id, limit=50, delivered_only=False)
    apology = [m for m in rows if json.loads(m["meta"]).get("apology")]
    assert len(apology) == 1

    broken = Overwhelm.from_json(agent.db.get_save(save_id)["overwhelm"])
    assert parse_ts(apology[0]["deliver_at"]) >= broken.recovers_at() - timedelta(seconds=2)
    assert bool(apology[0]["proactive"])
    # 她不说对不起
    assert "对不起" not in apology[0]["content"]


# ---------------- 局面处置 ----------------

@pytest.mark.asyncio
async def test_situation_never_calls_the_model(tmp_path):
    """崩溃期玩家可能连点好几次，每次都调一遍纯属烧钱。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        await agent.choose(save_id, 0)

    before = agent.provider.calls
    for _ in range(4):
        await agent.choose(save_id, 0)          # 等一会
    assert agent.provider.calls == before


@pytest.mark.asyncio
async def test_she_always_says_something(tmp_path):
    """界面不能是空的 —— 点了没反应，玩家以为卡了。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        await agent.choose(save_id, 0)

    r = await agent.choose(save_id, 0)
    assert r.scheduled, "崩溃期一条消息都没发"
    assert len(r.scheduled[0][0]) <= 8


@pytest.mark.asyncio
async def test_pushing_while_broken_backfires(tmp_path):
    """moods.md：反复追问「你怎么了」→ 更僵。追问对她是压力。

    **惩罚必须落在恢复时间上。** 她崩的时候情绪常常已经顶到 1.0
    （阈值 0.85，一轮最多涨 0.5），再 bump 会被上限整个吃掉 ——
    这条测试原来断言的是情绪递增，实际上两边都是 1.0，
    靠读取时衰减产生的浮点尾数（差 1e-9）通过的，等于没测。
    """
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        await agent.choose(save_id, 0)

    before = Overwhelm.from_json(agent.db.get_save(save_id)["overwhelm"])
    r = await agent.choose(save_id, 1)          # 再说一句
    after = Overwhelm.from_json(agent.db.get_save(save_id)["overwhelm"])

    assert r.situation == "push_backfired"
    assert after.credit_minutes < before.credit_minutes, "戳了她一下但什么都没发生"
    assert after.recovers_at() > before.recovers_at(), "恢复没有被推后"


@pytest.mark.asyncio
async def test_saturated_emotion_does_not_swallow_the_penalty(tmp_path):
    """情绪顶到 1.0 时，惩罚仍然要生效 —— 这就是它不能只靠 bump 的原因。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        await agent.choose(save_id, 0)

    st = EmotionState.from_json(agent.db.get_save(save_id)["emotions"])
    assert st.decayed()[Emotion.SAD] == pytest.approx(1.0), "前提没成立，这条测试没意义"

    before = Overwhelm.from_json(agent.db.get_save(save_id)["overwhelm"]).recovers_at()
    await agent.choose(save_id, 1)
    after = Overwhelm.from_json(agent.db.get_save(save_id)["overwhelm"]).recovers_at()
    assert after > before


@pytest.mark.asyncio
async def test_leaving_ends_the_session_without_punishment(tmp_path):
    agent, save_id = _setup(tmp_path, {
        "messages": ["……"], "options": [{"text": "算了", "tone": "后退"}],
        "feeling": {"难过": 0.5},
    })
    for _ in range(2):
        _arm(agent, save_id)
        await agent.choose(save_id, 0)

    before = EmotionState.from_json(
        agent.db.get_save(save_id)["emotions"]).decayed()[Emotion.SAD]
    r = await agent.choose(save_id, 2)          # 明天再找她
    after = EmotionState.from_json(
        agent.db.get_save(save_id)["emotions"]).decayed()[Emotion.SAD]

    assert r.situation == "leave"
    assert r.options == []
    assert after == pytest.approx(before, abs=0.01)


# ---------------- 恢复后收尾 ----------------

@pytest.mark.asyncio
async def test_recovered_state_is_cleared(tmp_path):
    """不清账的话恢复期约束永远挂着，下一次崩溃也会被旧记录挡住。"""
    agent, save_id = _setup(tmp_path, {
        "messages": ["在写作业。"], "options": [{"text": "在干嘛", "tone": "守住"}],
    })
    old = Overwhelm(
        at=agent.schedule.now_local() - timedelta(days=1),
        emo=Emotion.SAD, peak=0.9, cause="旧账",
    )
    assert old.recovered()
    agent.db.update_save(save_id, overwhelm=old.to_json())

    _arm(agent, save_id)
    r = await agent.choose(save_id, 0)

    assert agent.db.get_save(save_id)["overwhelm"] == ""
    assert not r.overwhelm
    assert all(o.tone != SITUATION_TONE for o in r.options)


@pytest.mark.asyncio
async def test_recovery_rung_slows_her_down_and_shapes_behavior(tmp_path):
    agent, save_id = _setup(tmp_path, {
        "messages": ["在写作业"], "options": [{"text": "嗯", "tone": "守住"}],
    })
    mid = agent.schedule.now_local() - timedelta(
        minutes=RUNG_MINUTES[Rung.BROKEN] + 5)
    agent.db.update_save(save_id, overwhelm=Overwhelm(
        at=mid, emo=Emotion.SAD, peak=0.9, cause="他说了很重的话").to_json())

    _arm(agent, save_id)
    r = await agent.choose(save_id, 0)

    assert r.overwhelm == Rung.LENGTH.label
    # 恢复期的行为约束要真的进 prompt，否则阶梯只存在于代码里
    save = agent.db.get_save(save_id)
    volatile = agent._volatile(save, EmotionState.from_json(save["emotions"]),
                               None, 0)
    assert "标点还掉着" in volatile.render()
