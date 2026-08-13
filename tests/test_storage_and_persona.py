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


# ---------------- 样本的阶段门控 ----------------


def test_s3_lines_absent_before_s3():
    """S3 的直球摆在 S0 的语气样本里，模型会往那个方向偏。

    这不是省 token —— 是易变层的 STAGE_BEHAVIOR 不该被迫去跟稳定前缀里的
    范例对抗。
    """
    for stage in ("S0", "S1", "S2"):
        samples = load_card("h01", stage).samples
        assert "九、S3" not in samples, f"{stage} 混进了热恋期样本"
        assert "想见你。" not in samples, f"{stage} 混进了 S3 直球"
    assert "想见你。" in load_card("h01", "S3").samples


def test_probing_starts_at_s1():
    """S0 她刚加上好友，不会主动扔钩子。"""
    assert "五、试探" not in load_card("h01", "S0").samples
    for stage in ("S1", "S2", "S3"):
        assert "五、试探" in load_card("h01", stage).samples


def test_retract_samples_match_their_own_note():
    """小节自注：「S1 阶段几乎每次越界后都跟一条。S3 基本消失。」"""
    assert "六、撤回" not in load_card("h01", "S0").samples
    assert "六、撤回" in load_card("h01", "S1").samples
    assert "六、撤回" in load_card("h01", "S2").samples
    assert "六、撤回" not in load_card("h01", "S3").samples


def test_universal_sections_survive_every_stage():
    """底色、接话能力、禁令在任何阶段都不能掉。"""
    for stage in ("S0", "S1", "S2", "S3"):
        samples = load_card("h01", stage).samples
        assert "记得吃饭。" in samples          # 关心（医生侧）
        assert "十五、禁用对照" in samples      # 硬禁
        assert "十二点五、把话接住" in samples  # 接话能力


def test_s3_prohibitions_survive_gating_out_section_nine():
    """「九、S3」被门控掉时，里面那条「S3 也绝不会说的」不能跟着丢。

    波浪号／感叹号／叠字的硬禁在 lexicon 和「十五、禁用对照」里各有一份，
    所以门控是安全的 —— 这条测试就是钉住这个前提。
    """
    for stage in ("S0", "S1", "S2"):
        card = load_card("h01", stage)
        blob = card.lexicon + card.samples
        assert "波浪号" in blob
        assert "不想说啦" in blob     # 禁用对照里的反例


def test_gating_is_monotonic_and_cheap():
    """S0 最省，且每个阶段都仍是完整可用的卡。"""
    sizes = {s: len(load_card("h01", s).samples) for s in ("S0", "S1", "S2", "S3")}
    assert sizes["S0"] < sizes["S1"] <= sizes["S2"] < sizes["S3"]
    assert all(n > 800 for n in sizes.values()), sizes
    # 全量（不门控）应该是所有阶段的上界
    assert len(load_card("h01").samples) >= max(sizes.values())


def test_gating_yields_three_stable_prefixes():
    """前缀缓存要完整匹配，所以变体数量直接决定命中率。

    S1 和 S2 的**样本集合故意相同** —— 两个阶段用的是同一类台词，
    区别只是频率（`STAGE_BEHAVIOR.retract_rate` 0.75 → 0.35）。
    频率属于易变层，不该靠换一份前缀来表达。

    于是四个阶段只落成三份前缀，缓存变体更少。
    """
    cards = {s: load_card("h01", s) for s in ("S0", "S1", "S2", "S3")}
    texts = {s: c.stable_text()[1] for s, c in cards.items()}
    assert texts["S1"] == texts["S2"]
    assert len({texts["S0"], texts["S1"], texts["S3"]}) == 3

    # 同参数必须命中同一个缓存对象，否则每回合重新装配
    assert load_card("h01", "S1") is cards["S1"]
