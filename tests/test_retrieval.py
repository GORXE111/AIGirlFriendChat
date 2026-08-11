from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gfagent.memory import context_keywords, rank_episodes, rank_facts
from gfagent.storage import Database

NOW = datetime(2026, 8, 7, 20, tzinfo=timezone.utc)


def ep(days_ago: int, summary: str, importance: int = 1) -> dict:
    return {
        "summary": summary,
        "happened_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "importance": importance,
    }


def test_relevance_beats_recency():
    """核心诉求：玩家第 30 天提起第 3 天那件事，她必须想得起来。

    只按时间倒序的话，早期的重要事件永远拿不到 —— 这是长期玩会「失忆」的机制性原因。
    """
    episodes = [ep(i, f"闲聊{i}") for i in range(1, 12)]
    episodes.append(ep(30, "他第一次说起他爸的事", 5))

    ctx = context_keywords("你上次说你爸")
    top = rank_episodes(episodes, now=NOW, context=ctx, limit=3)
    assert "他爸" in top[0].summary


def test_importance_matters_when_context_is_empty():
    episodes = [ep(2, "他说食堂菜一般", 1), ep(2, "他说了他家里的事", 5)]
    top = rank_episodes(episodes, now=NOW, context=set(), limit=1)
    assert "家里" in top[0].summary


def test_recency_matters_when_all_else_equal():
    episodes = [ep(1, "他说风扇坏了", 2), ep(60, "他说风扇坏了很久以前", 2)]
    top = rank_episodes(episodes, now=NOW, context=set(), limit=1)
    assert "很久以前" not in top[0].summary


def test_recency_decays_but_never_zero():
    """一年前的事权重很低，但不该彻底消失。"""
    from gfagent.memory.retrieval import _recency

    assert _recency(NOW - timedelta(days=365), NOW) > 0
    assert _recency(NOW - timedelta(days=14), NOW) == pytest.approx(0.5, abs=0.01)


def test_limit_respected():
    episodes = [ep(i, f"事件{i}") for i in range(50)]
    assert len(rank_episodes(episodes, now=NOW, context=set(), limit=8)) == 8


def test_bad_timestamps_skipped_not_crash():
    episodes = [ep(1, "正常"), {"summary": "坏的", "happened_at": "不是时间"}]
    out = rank_episodes(episodes, now=NOW, context=set(), limit=5)
    assert len(out) == 1


def test_facts_all_returned_when_few():
    facts = [{"content": f"事实{i}"} for i in range(10)]
    assert len(rank_facts(facts, context=set(), limit=24)) == 10


def test_facts_ranked_by_relevance_when_many():
    facts = [{"content": f"无关事实{i}"} for i in range(40)]
    facts.append({"content": "他有胃病不能吃凉的"})
    out = rank_facts(facts, context=context_keywords("我胃疼"), limit=10)
    assert any("胃病" in f["content"] for f in out)


# ---------------- 洞察 ----------------


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "t.db")


def test_insight_upsert_increments_weight(db: Database):
    """同一条规律被再次印证时加权 —— weight 就是「出现过几次」。"""
    sid = db.create_save("x")
    db.upsert_insight(sid, "他考试前会胃疼", "him")
    db.upsert_insight(sid, "他考试前会胃疼", "him")
    rows = db.get_insights(sid)
    assert len(rows) == 1
    assert rows[0]["weight"] == 2


def test_insights_sorted_by_weight(db: Database):
    sid = db.create_save("x")
    db.upsert_insight(sid, "弱规律", "him")
    for _ in range(3):
        db.upsert_insight(sid, "强规律", "him")
    assert db.get_insights(sid)[0]["content"] == "强规律"


def test_insights_filtered_by_kind(db: Database):
    sid = db.create_save("x")
    db.upsert_insight(sid, "他总在硬撑", "him")
    db.upsert_insight(sid, "那家面馆的辣酱", "joke")
    assert len(db.get_insights(sid, kind="joke")) == 1


def test_insights_cascade_on_delete(db: Database):
    sid = db.create_save("x")
    db.upsert_insight(sid, "某规律", "him")
    db.delete_save(sid)
    assert db.get_insights(sid) == []
