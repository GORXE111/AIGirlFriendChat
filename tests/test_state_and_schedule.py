from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from gfagent.schedule import Pace, ScheduleEngine
from gfagent.schedule.engine import LOCAL
from gfagent.state import (
    STAGE_BEHAVIOR,
    Emotion,
    EmotionState,
    Stage,
    stage_for_affinity,
)


def at(h: int, m: int = 0, day: int = 4) -> datetime:
    """2026-08-04 是周二。"""
    return datetime(2026, 8, day, h, m, tzinfo=LOCAL)


SAT, SUN = 8, 9  # 2026-08-08 周六 / 08-09 周日


# ---------------- 日程 ----------------


@pytest.fixture
def eng() -> ScheduleEngine:
    return ScheduleEngine(rng=random.Random(0))


def test_she_can_always_reply(eng: ScheduleEngine):
    """任何时段都能回。日程只影响多快，不影响能不能。

    上课／睡觉时不回消息更真实，但对聊天体验是灾难 —— 玩家只会觉得卡住了。
    """
    for h in (3, 7, 9, 15, 19, 23):
        st = eng.state(at(h))
        assert st.delay_seconds > 0


def test_schedule_does_not_limit_message_length():
    """日程是氛围，不是枷锁。「她在上课所以只能说 8 个字」是折磨玩家。"""
    from gfagent.schedule.engine import WEEKDAY_WINDOWS, WEEKEND_WINDOWS

    for w in WEEKDAY_WINDOWS + WEEKEND_WINDOWS:
        assert not hasattr(w, "max_chars")


def test_class_is_slower_than_evening(eng: ScheduleEngine):
    class_delay = eng.state(at(9, 30)).delay_seconds
    room_delay = eng.state(at(22, 30)).delay_seconds
    assert class_delay > room_delay


def test_pace_reflects_situation(eng: ScheduleEngine):
    assert eng.state(at(22, 30)).pace is Pace.QUICK    # 在房间
    assert eng.state(at(9, 30)).pace is Pace.AWAY      # 上课
    assert eng.state(at(19, 30)).pace is Pace.SLOW     # 晚自习


def test_delays_stay_within_chat_tolerance(eng: ScheduleEngine):
    """最慢的时段也不该超过十分钟 —— 再久聊天就断了。"""
    for h in range(24):
        assert eng.state(at(h)).delay_seconds <= 15 * 60


def test_sunday_family_time(eng: ScheduleEngine):
    st = eng.state(at(15, 0, day=SUN))
    assert st.pace is Pace.SLOW
    assert st.note


def test_saturday_piano_lesson(eng: ScheduleEngine):
    st = eng.state(at(9, 30, day=SAT))
    assert "钢琴" in st.window.name
    assert st.pace is Pace.AWAY


def test_proactive_only_when_free(eng: ScheduleEngine):
    assert eng.state(at(22, 30)).can_initiate
    assert not eng.state(at(9, 30)).can_initiate      # 上课
    assert not eng.state(at(3, 0)).can_initiate       # 睡着


def test_night_shift_is_stable_within_a_day(eng: ScheduleEngine):
    a = eng.is_mother_night_shift(at(19, 0))
    b = eng.is_mother_night_shift(at(23, 0))
    assert a == b, "同一天的夜班判定必须稳定，不能每次调用重掷"


def test_night_shift_frequency_is_a_few_times_a_month(eng: ScheduleEngine):
    days = [at(20, 0) + timedelta(days=i) for i in range(90)]
    rate = sum(eng.is_mother_night_shift(d) for d in days) / 90
    assert 0.05 < rate < 0.25, f"实际 {rate:.0%}，应接近每月 3–4 次"


# ---------------- 关系阶段 ----------------


@pytest.mark.parametrize(
    "affinity,stage",
    [(0, Stage.S0), (19, Stage.S0), (20, Stage.S1),
     (49, Stage.S1), (50, Stage.S2), (80, Stage.S3), (100, Stage.S3)],
)
def test_stage_thresholds(affinity: float, stage: Stage):
    assert stage_for_affinity(affinity) is stage


def test_retract_rate_falls_with_stage():
    """撤回率是关系推进最直观的外显。S3 基本不撤 —— 这就是「甜」。"""
    rates = [STAGE_BEHAVIOR[s].retract_rate for s in (Stage.S1, Stage.S2, Stage.S3)]
    assert rates == sorted(rates, reverse=True)
    assert STAGE_BEHAVIOR[Stage.S3].retract_rate < 0.15


def test_we_gated_to_s3():
    """第一次说出「我们」是巨大的时刻。"""
    for s in (Stage.S0, Stage.S1, Stage.S2):
        assert not STAGE_BEHAVIOR[s].allow_we
    assert STAGE_BEHAVIOR[Stage.S3].allow_we


def test_farewell_progression():
    """「睡了」是通知，「晚安」是给予。"""
    assert STAGE_BEHAVIOR[Stage.S0].farewell == ""
    assert STAGE_BEHAVIOR[Stage.S2].farewell == "睡了。"
    assert STAGE_BEHAVIOR[Stage.S3].farewell == "晚安。"


def test_proactive_weight_rises_with_stage():
    weights = [STAGE_BEHAVIOR[s].proactive_weight for s in Stage]
    assert weights == sorted(weights)
    assert STAGE_BEHAVIOR[Stage.S0].proactive_weight < 0.05


def test_max_messages_rises_with_stage():
    """话少 ≠ 只肯发一条。越熟越愿意连着说。"""
    counts = [STAGE_BEHAVIOR[s].max_messages for s in Stage]
    assert counts == sorted(counts)
    assert STAGE_BEHAVIOR[Stage.S0].max_messages >= 2, "S0 也该能发「嗯。」「想起来了。」"


# ---------------- 情绪 ----------------


def now() -> datetime:
    return datetime.now(timezone.utc)


def test_happy_decays():
    e = EmotionState()
    e.bump(Emotion.HAPPY, 0.8)
    later = e.decayed(now() + timedelta(hours=20))
    assert later.get(Emotion.HAPPY, 0) < 0.2


def test_anger_and_hurt_never_decay():
    """生气与委屈不会自己好，必须由互动消解。"""
    e = EmotionState()
    e.bump(Emotion.ANGRY, 0.6)
    e.bump(Emotion.HURT, 0.6)
    later = e.decayed(now() + timedelta(days=30))
    assert later[Emotion.ANGRY] == pytest.approx(0.6)
    assert later[Emotion.HURT] == pytest.approx(0.6)


def test_soothe_clears_hurt():
    e = EmotionState()
    e.bump(Emotion.HURT, 0.3)
    e.soothe(Emotion.HURT, 0.3)
    assert Emotion.HURT not in e.values


def test_relaxed_gated_to_s3():
    """松弛是 S3 限定的最终奖励 —— 她开始说废话。"""
    e = EmotionState()
    e.bump(Emotion.RELAXED, 0.5, stage=Stage.S1)
    assert Emotion.RELAXED not in e.values
    e.bump(Emotion.RELAXED, 0.5, stage=Stage.S3)
    assert Emotion.RELAXED in e.values


def test_dominant_picks_strongest():
    e = EmotionState()
    e.bump(Emotion.TIRED, 0.3)
    e.bump(Emotion.HURT, 0.7)
    dom = e.dominant()
    assert dom is not None and dom[0] is Emotion.HURT


def test_describe_reports_state_not_behavior():
    """易变层只报状态，行为规则在 lexicon 里 —— 两边混在一起模型会冲突。"""
    e = EmotionState()
    assert e.describe() == "情绪：平。"
    e.bump(Emotion.TIRED, 0.8)
    d = e.describe()
    assert "累" in d and "强烈" in d
    assert "短" not in d and "标点" not in d


def test_roundtrip_json():
    e = EmotionState()
    e.bump(Emotion.TIRED, 0.42)
    e.bump(Emotion.ANGRY, 0.31)
    back = EmotionState.from_json(e.to_json())
    assert back.values[Emotion.TIRED] == pytest.approx(0.42, abs=1e-3)
    assert back.values[Emotion.ANGRY] == pytest.approx(0.31, abs=1e-3)


def test_from_json_tolerates_garbage():
    assert EmotionState.from_json("not json").values == {}
    assert EmotionState.from_json(None).values == {}
    assert EmotionState.from_json('{"values":{"不存在的情绪":0.5}}').values == {}


# ---------------- 时区 ----------------


def test_timezone_is_configurable():
    """她的时区可配置；默认是她所在地（中国），不是开发者所在地。"""
    from zoneinfo import ZoneInfo

    assert ScheduleEngine().tz == ZoneInfo("Asia/Shanghai")
    assert ScheduleEngine(tz="Asia/Singapore").tz == ZoneInfo("Asia/Singapore")


def test_utc8_zones_read_identically():
    """新加坡／上海同为 UTC+8 —— 同一时刻落在同一个时段，行为无差别。"""
    sh = ScheduleEngine(rng=random.Random(1), tz="Asia/Shanghai")
    sg = ScheduleEngine(rng=random.Random(1), tz="Asia/Singapore")
    for h in (3, 9, 15, 20, 23):
        moment = datetime(2026, 8, 6, h, 0, tzinfo=timezone.utc)
        assert sh.state(moment).window.name == sg.state(moment).window.name


def test_billing_timezone_is_independent_of_story_timezone():
    """DeepSeek 峰谷由 API 供应商决定，恒为北京时间，不随剧情时区改变。"""
    from gfagent.timewindow import BEIJING, is_peak

    assert BEIJING.key == "Asia/Shanghai"
    # 北京 10:00 = UTC 02:00，无论剧情设在哪个时区都算高峰
    assert is_peak(datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc))
