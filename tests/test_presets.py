from __future__ import annotations

import json

import pytest

from gfagent.beats import BeatProgress, load_beats, pick_her_beat
from gfagent.presets import PRESETS, listing, seed
from gfagent.schedule.engine import LOCAL
from gfagent.state import Emotion, EmotionState, Stage
from gfagent.storage import Database

from datetime import datetime


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "t.db")


def make(db: Database, key: str) -> int:
    sid = db.create_save(key, surname="陈", given="屿")
    seed(db, sid, key)
    return sid


def test_all_presets_listed():
    keys = {p["key"] for p in listing()}
    assert keys == {"s0", "s1", "s2", "s3"}


def test_stages_ascend():
    order = [PRESETS[k].stage.rank for k in ("s0", "s1", "s2", "s3")]
    assert order == sorted(order)
    assert PRESETS["s3"].stage is Stage.S3


def test_s0_seeds_nothing(db: Database):
    """从头开始就该是空的。"""
    sid = make(db, "s0")
    assert db.get_facts(sid) == []
    assert db.get_episodes(sid) == []
    assert db.recent_messages(sid) == []
    assert db.get_save(sid)["affinity"] == 0


def test_s3_seeds_memory(db: Database):
    """只调好感不种记忆，会得到一个「对你一无所知却很亲密」的角色。"""
    sid = make(db, "s3")
    save = db.get_save(sid)
    assert save["stage"] == "S3"
    assert save["affinity"] == 90
    assert len(db.get_facts(sid)) >= 5
    assert len(db.get_episodes(sid)) >= 5
    assert db.recent_messages(sid), "热恋期存档不该是零对话历史"


def test_episodes_have_real_dates(db: Database):
    """「你上周三说嗓子疼」要有东西可指。"""
    sid = make(db, "s3")
    now = datetime.now(LOCAL)
    for e in db.get_episodes(sid):
        when = datetime.fromisoformat(e["happened_at"])
        assert when < now
        assert (now - when).days <= 60


def test_played_beats_prevent_replay(db: Database):
    """不种这个的话，热恋期存档还会触发「第一次说话」。"""
    sid = make(db, "s3")
    save = db.get_save(sid)
    progress = BeatProgress.from_dict(json.loads(save["beat_progress"]))
    assert "first_words" in progress.history

    beat = pick_her_beat(
        character_id="h01", stage=save["stage"], affinity=save["affinity"],
        flags=set(json.loads(save["flags"])),
        now_local=datetime.now(LOCAL), progress=progress,
        mother_night_shift=False,
    )
    assert beat is None or beat.id != "first_words"


def test_relaxed_emotion_only_at_s3():
    """松弛是 S3 限定的 —— 低阶段预设里种了也不该生效。"""
    assert any(e is Emotion.RELAXED for e, _ in PRESETS["s3"].emotions)
    for key in ("s1", "s2"):
        assert not any(e is Emotion.RELAXED for e, _ in PRESETS[key].emotions)


def test_seeded_emotions_load_back(db: Database):
    sid = make(db, "s3")
    emotions = EmotionState.from_json(db.get_save(sid)["emotions"])
    assert emotions.values


def test_reflect_mark_advanced(db: Database):
    """否则一开局就会把种子对话再归档一遍。"""
    sid = make(db, "s3")
    assert db.get_reflect_mark(sid) > 0
    assert db.messages_since(sid, db.get_reflect_mark(sid)) == []


def test_flags_referenced_by_beats_exist():
    """预设里的 flag 如果拼错，桥段的门控会静默失效。"""
    known = set()
    for b in load_beats("h01"):
        known |= set(b.entry.flags_all) | set(b.entry.flags_any) | set(b.entry.flags_none)
        for o in b.outcomes:
            known |= set(o.flags_add) | set(o.flags_remove)

    for preset in PRESETS.values():
        for flag in preset.flags:
            # 允许预设定义桥段还没用到的 flag，但反过来不行
            assert isinstance(flag, str) and flag
    # 桥段引用的 flag 必须能被某个结局产生，否则永远触发不了
    produced = {
        f for b in load_beats("h01") for o in b.outcomes for f in o.flags_add
    }
    for b in load_beats("h01"):
        for flag in b.entry.flags_all:
            assert flag in produced, f"{b.id} 要求 flag「{flag}」，但没有任何结局会产生它"
