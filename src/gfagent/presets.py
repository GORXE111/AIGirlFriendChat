"""测试存档预设。

直接开一个热恋期存档来玩，不用先刷 90 点好感。

**关键是连记忆一起种。** 只把好感调到 90 而记忆是空的，会得到一个
「对你一无所知却很亲密」的角色 —— 比 S0 还假。所以每个预设都带：

  - 关于他的事实（她记得的）
  - 带日期的情节（「你上周三说嗓子疼」要有东西可指）
  - 已演过的桥段（否则热恋期还会触发「第一次说话」）
  - 一小段像样的历史对话

⚠️ 仅供开发测试。正式流程只应该从 S0 开始。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .schedule.engine import LOCAL
from .state.models import Emotion, EmotionState, Stage
from .storage.db import Database


@dataclass(frozen=True, slots=True)
class Preset:
    key: str
    label: str
    hint: str
    affinity: float
    flags: tuple[str, ...] = ()
    played_beats: tuple[str, ...] = ()
    facts: tuple[tuple[str, str], ...] = ()
    """(内容, 分类)"""
    episodes: tuple[tuple[int, str, int], ...] = ()
    """(几天前, 摘要, 重要度)"""
    history: tuple[tuple[str, str], ...] = ()
    """(role, 内容)。最近的一小段对话，让开局不至于凭空开始。"""

    threads: tuple[tuple[str, str, str], ...] = ()
    """(title, kind, owner)。还悬着的共同事 —— 对话的引力中心。"""
    emotions: tuple[tuple[Emotion, float], ...] = ()

    @property
    def stage(self) -> Stage:
        from .state.models import stage_for_affinity
        return stage_for_affinity(self.affinity)


_COMMON_FACTS = (
    ("他有胃病，不能吃太凉的", "身体"),
    ("他晚上睡得晚", "生活"),
    ("他也在高二，不同班", "生活"),
)

PRESETS: dict[str, Preset] = {
    "s0": Preset(
        key="s0", label="从头开始", hint="好友躺了两周，谁也没说过话",
        affinity=0,
    ),
    "s1": Preset(
        key="s1", label="试探期", hint="说得上话了，她开始越界又撤回",
        affinity=25,
        flags=("first_conversation_done",),
        played_beats=("first_words",),
        facts=_COMMON_FACTS,
        episodes=(
            (6, "他说考场那次是他借的笔", 2),
            (3, "他说胃疼，她让他去看", 3),
            (1, "他问她几点睡", 1),
        ),
        history=(
            ("user", "你还没睡？"),
            ("assistant", "在写作业。"),
            ("assistant", "你也很晚。"),
        ),
        emotions=((Emotion.TIRED, 0.3),),
        threads=(("他借了她一支笔还没还", "物件", "him"),),
    ),
    "s2": Preset(
        key="s2", label="确认期", hint="互相知道了，她开始说「睡了」",
        affinity=62,
        flags=("first_conversation_done", "she_admitted_tired"),
        played_beats=("first_words", "post_exam_night"),
        facts=_COMMON_FACTS + (
            ("他会记得她说过的小事", "其他"),
            ("他不追问她家里的事", "其他"),
        ),
        episodes=(
            (14, "他说考场那次是他借的笔", 2),
            (9, "月考后她说今天有点烦，没说为什么", 4),
            (5, "他胃疼，她让他别吃凉的", 3),
            (2, "她说她妈值夜班，家里就她一个人", 3),
            (1, "他说他也睡不好", 2),
        ),
        history=(
            ("user", "你今天弹琴了吗"),
            ("assistant", "弹了一个小时。"),
            ("assistant", "手有点僵。"),
        ),
        emotions=((Emotion.TIRED, 0.25),),
        threads=(
            ("他说要请她喝奶茶", "亏欠", "him"),
            ("她想知道他为什么找她说话", "悬念", "her"),
        ),
    ),
    "s3": Preset(
        key="s3", label="热恋期", hint="她不撤回了，开始说「我们」",
        affinity=90,
        flags=(
            "first_conversation_done", "she_admitted_tired",
            "she_opened_up", "knows_about_earring",
        ),
        played_beats=(
            "first_words", "post_exam_night",
            "late_night_awake", "mother_night_shift",
        ),
        facts=_COMMON_FACTS + (
            ("他会记得她说过的小事", "其他"),
            ("他不追问她家里的事", "其他"),
            ("他知道她戴耳环的事", "其他"),
            ("他答应过带她去吃面", "其他"),
        ),
        episodes=(
            (32, "他说考场那次是他借的笔", 2),
            (24, "月考后她说今天有点烦，第一次松口", 5),
            (18, "她妈值夜班那晚，她说她在弹别的曲子", 4),
            (11, "他发现她戴耳环，她没否认", 5),
            (6, "深夜她说睡不着，聊到两点", 4),
            (3, "他胃疼，她说别熬了", 3),
            (1, "她说明天要考试", 2),
        ),
        history=(
            ("user", "考完了吗"),
            ("assistant", "刚考完。"),
            ("assistant", "最后一道大题没写完。"),
            ("user", "那也没关系"),
            ("assistant", "嗯。"),
            ("assistant", "我们周末还去那家面馆吗。"),
        ),
        emotions=((Emotion.TIRED, 0.35), (Emotion.RELAXED, 0.5)),
        threads=(
            ("周末去那家面馆", "约定", "both"),
            ("他答应带她吃回本", "亏欠", "him"),
            ("她落在他那的伞", "物件", "her"),
        ),
    ),
}


def seed(db: Database, save_id: int, preset_key: str, *, now: datetime | None = None) -> None:
    """把预设内容写进存档。"""
    preset = PRESETS.get(preset_key)
    if preset is None or preset.key == "s0":
        return

    now = now or datetime.now(LOCAL)

    for content, category in preset.facts:
        db.add_fact(save_id, content, category)

    for title, kind, owner in preset.threads:
        db.open_thread(save_id, title, kind, owner)

    for days_ago, summary, importance in preset.episodes:
        when = now - timedelta(days=days_ago)
        db.add_episode(save_id, summary, when.isoformat(), importance)

    # 历史对话按时间顺序排在过去几十分钟内，全部标记为已送达
    base = now - timedelta(minutes=40)
    for i, (role, content) in enumerate(preset.history):
        db.add_message(save_id, role, content, delivered=True)
        _ = base + timedelta(minutes=i * 3)

    emotions = EmotionState()
    for emo, value in preset.emotions:
        emotions.bump(emo, value, preset.stage)

    # 已演过的桥段 —— 不种这个的话，热恋期存档还会触发「第一次说话」
    beat_progress = {
        "beat_id": None,
        "turn": 0,
        "history": {
            bid: (now - timedelta(days=30 + i)).isoformat()
            for i, bid in enumerate(preset.played_beats)
        },
    }

    db.update_save(
        save_id,
        affinity=preset.affinity,
        stage=preset.stage.value,
        flags=sorted(preset.flags),
        emotions=emotions.to_json(),
        beat_progress=beat_progress,
    )

    # 慢回路水位推到最新，避免一开局就把种子对话再归档一遍
    rows = db.recent_messages(save_id, limit=1000, delivered_only=False)
    if rows:
        db.set_reflect_mark(save_id, rows[-1]["id"])


def listing() -> list[dict[str, object]]:
    return [
        {
            "key": p.key, "label": p.label, "hint": p.hint,
            "stage": p.stage.value, "affinity": p.affinity,
        }
        for p in PRESETS.values()
    ]
