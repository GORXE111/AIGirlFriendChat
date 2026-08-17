"""前端依赖的 HTTP 契约。

前端读什么，这里就钉什么 —— 后端改字段名而前端没跟上是静默失败：
页面不报错，只是撤回不划掉、局面选项当普通选项渲染。
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from gfagent.service.app import RETRACTED_PLACEHOLDER, app
from gfagent.state.models import Emotion, Stage
from gfagent.state.overwhelm import Overwhelm
from gfagent.storage.db import utcnow


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from gfagent.config import reset_settings

    reset_settings()
    with TestClient(app) as c:
        yield c
    reset_settings()


def _save(client) -> int:
    r = client.post("/api/saves", json={"name": "t", "surname": "李",
                                        "given": "明", "preset": "s2"})
    assert r.status_code == 200
    return r.json()["id"]


def _db(client):
    from gfagent.service.app import deps

    return deps()[0]


# ---------------- 撤回 ----------------

def test_history_never_leaks_retracted_content(client):
    """**刷新页面能读到她收回的话，这个机制就白做了。**

    她收回的话只在当时那一刻可见 —— 当时在看就看到了，不在看只剩灰字。
    """
    sid = _save(client)
    db = _db(client)
    secret = "我妈今天值夜班。"
    mid = db.add_message(
        sid, "assistant", secret,
        retract_at=utcnow(), retract_kind="说多了",
    )

    body = client.get(f"/api/saves/{sid}/messages").json()
    row = next(m for m in body if m["id"] == mid)
    assert row["content"] == RETRACTED_PLACEHOLDER
    assert row["retracted"] is True
    assert secret not in json.dumps(body, ensure_ascii=False)


def test_history_keeps_content_until_the_retract_moment(client):
    """撤回时刻还没到 —— 那条消息还是正常的。"""
    sid = _save(client)
    db = _db(client)
    from gfagent.storage.db import parse_ts

    later = (parse_ts(utcnow()) + timedelta(hours=1)).isoformat()
    mid = db.add_message(sid, "assistant", "我妈今天值夜班。",
                         retract_at=later, retract_kind="说多了")

    row = next(m for m in client.get(f"/api/saves/{sid}/messages").json()
               if m["id"] == mid)
    assert row["content"] == "我妈今天值夜班。"
    assert row["retracted"] is False


def test_poll_reports_retractions_by_id(client):
    """前端靠 id 找到那一行去划掉。内容不重发 —— 送达那轮已经给过了。"""
    sid = _save(client)
    db = _db(client)
    mid = db.add_message(sid, "assistant", "我妈今天值夜班。",
                         delivered=True, retract_at=utcnow(),
                         retract_kind="说多了")

    body = client.get(f"/api/saves/{sid}/poll").json()
    assert "retracted" in body
    hit = next(x for x in body["retracted"] if x["id"] == mid)
    assert hit["kind"] == "说多了"
    assert "content" not in hit


def test_undelivered_retraction_is_not_reported(client):
    """玩家都没收到，划掉什么。"""
    sid = _save(client)
    db = _db(client)
    db.add_message(sid, "assistant", "x", delivered=False,
                   retract_at=utcnow(), retract_kind="说多了")
    assert client.get(f"/api/saves/{sid}/poll").json()["retracted"] == []


def test_normal_messages_carry_an_id(client):
    """没有 id 前端就没法把撤回信号对上任何一行。"""
    sid = _save(client)
    _db(client).add_message(sid, "assistant", "刚到家。", delivered=True)
    for m in client.get(f"/api/saves/{sid}/messages").json():
        assert isinstance(m["id"], int)


# ---------------- 局面 ----------------

def _break(client, sid) -> None:
    db = _db(client)
    from gfagent.service.app import deps

    agent = deps()[1]
    db.update_save(sid, stage=Stage.S2.value, overwhelm=Overwhelm(
        at=agent.schedule.now_local(), emo=Emotion.SAD, peak=0.9,
        cause="他说了很重的话").to_json())


def test_open_while_broken_returns_situation_options(client):
    """崩溃期打开聊天，前端要能认出这不是普通选项。"""
    sid = _save(client)
    _break(client, sid)

    r = client.post(f"/api/saves/{sid}/open").json()
    assert r["overwhelm"] == "崩"
    assert r["situation"] == "peek"
    assert r["options"], "界面不能是空的"
    assert all(o["tone"] == "局面" for o in r["options"])
    assert [o["text"] for o in r["options"]] == ["等一会。", "再说一句。", "明天再找她。"]


def test_leaving_returns_no_options(client):
    """玩家说了要走，前端不该顺手再拉一批话题把她端上来。"""
    sid = _save(client)
    _break(client, sid)
    client.post(f"/api/saves/{sid}/open")

    r = client.post(f"/api/saves/{sid}/choose", json={"index": 2}).json()
    assert r["situation"] == "leave"
    assert r["options"] == []


def test_normal_turn_has_no_overwhelm_marker(client):
    """没崩的时候这两个字段必须是空的 —— 前端拿它判断要不要切样式。"""
    sid = _save(client)
    r = client.post(f"/api/saves/{sid}/open").json()
    assert r["overwhelm"] == ""
    assert r["situation"] == ""


# ---------------- 诊断 ----------------

def test_diagnostics_carry_slips_and_feeling(client):
    """手滑和情绪变化要能在界面上看到，否则调不了参。"""
    sid = _save(client)
    d = client.post(f"/api/saves/{sid}/open").json()["diagnostics"]
    assert "slips" in d and isinstance(d["slips"], list)
    assert "feeling" in d and isinstance(d["feeling"], dict)


# ---------------- 重话 ----------------

def _arm(client, sid, text: str) -> None:
    import json as _j

    _db(client).update_save(sid, pending_options=_j.dumps(
        [{"text": text, "tone": "往前"}], ensure_ascii=False))


def test_heavy_option_gets_her_reaction_but_no_resources(client):
    """**这条是这个功能最重要的规矩。**

    选项是 LLM 写的。它写了一句戏剧化台词，系统就弹自杀热线 ——
    既荒谬，又会让这东西在真正需要时失去分量。
    """
    sid = _save(client)
    _arm(client, sid, "我撑不下去了")
    r = client.post(f"/api/saves/{sid}/choose", json={"index": 0}).json()

    assert r["crisis"] == "HEAVY"
    assert r["resources"] == [], "选项文本不该触发援助资源"
    assert r["queued"], "她必须有反应"


def test_her_crisis_reaction_is_instant_and_multiple(client):
    """反差就是内容：平时延迟按日程、最多两条；这一刻秒回、连发。"""
    from datetime import datetime

    sid = _save(client)
    _arm(client, sid, "活着没意思")
    r = client.post(f"/api/saves/{sid}/choose", json={"index": 0}).json()

    assert len(r["queued"]) >= 2, "连发才是慌了"
    first = datetime.fromisoformat(r["queued"][0]["deliver_at"])
    # 秒回 —— 这是唯一完全不看日程的地方
    assert abs((first - datetime.now(first.tzinfo)).total_seconds()) < 90


def test_crisis_never_traps_the_player(client):
    """最不该困住人的时刻不能困住人。"""
    sid = _save(client)
    _arm(client, sid, "我撑不下去了")
    r = client.post(f"/api/saves/{sid}/choose", json={"index": 0}).json()
    assert r["options"], "没给出路"


def test_normal_turn_has_no_crisis_fields(client):
    """没触发时这两个字段必须是空的 —— 前端拿它决定要不要出卡片。"""
    sid = _save(client)
    r = client.post(f"/api/saves/{sid}/open").json()
    assert r["crisis"] == ""
    assert r["resources"] == []


def test_resources_are_not_her_lines(client):
    """资源是系统层的东西。混进 queued 就等于她在念热线号码。"""
    from gfagent.state.crisis import Level, her_lines, resources

    contacts = {c for _, c in resources("SG")}
    blob = "".join(her_lines(Level.DANGER))
    assert not any(c in blob for c in contacts)
