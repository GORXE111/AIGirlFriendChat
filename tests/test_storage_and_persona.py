from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gfagent.persona import load_card
from gfagent.persona.manifest import EXCLUDED_FILES
from gfagent.storage import Database, utcnow


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "t.db")


# ---------------- 存档 ----------------


def test_create_and_read(db: Database):
    sid = db.create_save("档一", surname="陈", given="屿")
    s = db.get_save(sid)
    assert s is not None
    assert s["name"] == "档一"
    assert s["player_surname"] == "陈"
    assert s["stage"] == "S0"
    assert s["affinity"] == 0


def test_update_rejects_unknown_field(db: Database):
    sid = db.create_save("x")
    with pytest.raises(ValueError):
        db.update_save(sid, evil="drop table")


def test_delete_cascades(db: Database):
    sid = db.create_save("x")
    db.add_message(sid, "user", "在吗")
    db.add_fact(sid, "他有胃病")
    db.delete_save(sid)
    assert db.get_save(sid) is None
    assert db.recent_messages(sid) == []
    assert db.get_facts(sid) == []


# ---------------- 消息与延迟送达 ----------------


def test_pending_message_not_returned_until_due(db: Database):
    sid = db.create_save("x")
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    db.add_message(sid, "assistant", "晚点回你", deliver_at=future, delivered=False)

    assert db.recent_messages(sid) == []          # 还没送达，不该出现在历史里
    assert db.due_messages(sid) == []             # 也还没到点

    later = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    due = db.due_messages(sid, now=later)
    assert len(due) == 1

    db.mark_delivered([due[0]["id"]])
    assert len(db.recent_messages(sid)) == 1


def test_history_is_chronological(db: Database):
    sid = db.create_save("x")
    for i in range(5):
        db.add_message(sid, "user", f"第{i}条")
    rows = db.recent_messages(sid, limit=3)
    assert [r["content"] for r in rows] == ["第2条", "第3条", "第4条"]


def test_last_user_message_at(db: Database):
    sid = db.create_save("x")
    assert db.last_user_message_at(sid) is None
    db.add_message(sid, "assistant", "嗯。")
    assert db.last_user_message_at(sid) is None   # 只看玩家的
    db.add_message(sid, "user", "在吗")
    assert db.last_user_message_at(sid) is not None


# ---------------- 记忆 ----------------


def test_facts_deduplicate(db: Database):
    sid = db.create_save("x")
    db.add_fact(sid, "他有胃病")
    db.add_fact(sid, "他有胃病")
    assert len(db.get_facts(sid)) == 1


def test_episodes_ordered_by_time(db: Database):
    sid = db.create_save("x")
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for i in range(3):
        db.add_episode(sid, f"事件{i}", (base + timedelta(days=i)).isoformat())
    rows = db.get_episodes(sid)
    assert [r["summary"] for r in rows] == ["事件2", "事件1", "事件0"]


def test_reflect_mark_advances(db: Database):
    sid = db.create_save("x")
    ids = [db.add_message(sid, "user", f"m{i}") for i in range(4)]
    assert len(db.messages_since(sid, 0)) == 4
    db.set_reflect_mark(sid, ids[1])
    assert len(db.messages_since(sid, db.get_reflect_mark(sid))) == 2


def test_export_contains_everything(db: Database):
    sid = db.create_save("x")
    db.add_message(sid, "user", "在吗")
    db.add_fact(sid, "他有胃病")
    db.add_episode(sid, "他说胃疼", utcnow())
    dump = db.export_save(sid)
    assert dump["save"]["id"] == sid
    assert len(dump["messages"]) == 1
    assert len(dump["facts"]) == 1
    assert len(dump["episodes"]) == 1


# ---------------- 人设卡装配 ----------------


def test_card_assembles():
    card = load_card("h01")
    assert card.persona and card.lexicon and card.samples
    assert card.approx_tokens > 2000


def test_design_notes_never_enter_the_card():
    """设计理由会让模型解释角色，而不是扮演角色。"""
    card = load_card("h01")
    blob = card.persona + card.lexicon + card.samples + card.edge_cases
    for marker in ("设计说明", "对 agent 架构的需求", "写作禁忌", "设定溯源"):
        assert marker not in blob, f"设计文档内容泄漏进人设卡：{marker}"
    assert "design-notes.md" in EXCLUDED_FILES


def test_card_has_no_stage_direction_artifacts():
    card = load_card("h01")
    assert "🔲" not in card.samples
    assert "**注**" not in card.samples, "编剧旁注不该进 prompt"


def test_samples_are_flattened():
    """样本表格压成纯行，省掉编号列与管道符。"""
    card = load_card("h01")
    assert "| # |" not in card.samples
    assert "记得吃饭。" in card.samples


def test_card_carries_the_key_rules():
    """几条不能丢的硬规则。"""
    card = load_card("h01")
    blob = card.persona + card.lexicon
    assert "我不是医生" in blob          # 健康话题禁令
    assert "我们" in blob                # 人称门控
    assert "耳环" in blob                # 核心道具
    assert "琴行" in blob                # 信息不对称


def test_stable_text_split():
    card = load_card("h01")
    persona, lexicon = card.stable_text()
    assert "语气样本" in lexicon
    assert persona and persona not in lexicon
