from __future__ import annotations

import pytest

from gfagent.output import fallback, process


def one(text: str, **kw) -> str:
    r = process(text, **kw)
    assert r.ok, r.violations
    return r.messages[0] if r.messages else ""


# ---------------- 外观级清洗 ----------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("记得吃饭！", "记得吃饭。"),
        ("记得吃饭～", "记得吃饭"),
        ("好的呢。", "好的。"),
        ("知道了啦。", "知道了。"),
        ("嗯嗯 😊", "嗯嗯"),
        ("在写作业...", "在写作业……"),
    ],
)
def test_cosmetic_cleanup(raw: str, expected: str):
    assert one(raw) == expected


def test_softener_not_stripped_mid_word():
    """「呢子大衣」不能被误伤。"""
    assert "呢子" in one("买了件呢子大衣。")


def test_quotes_unwrapped():
    assert one('"刚到家。"') == "刚到家。"
    assert one("「刚到家。」") == "刚到家。"


# ---------------- 旁白 → 沉默 ----------------


def test_stage_direction_becomes_silence():
    """模型写「（没有回复。）」，意思就是她不回。照办。"""
    r = process("（没有回复。）")
    assert r.silent
    assert r.messages == []
    assert not r.violations


def test_unclosed_stage_direction_also_silence():
    """被截断的旁白同样处理 —— 这个 bug 真机上出现过。"""
    r = process("（这条消息她没有立刻回复。")
    assert r.silent
    assert r.messages == []


def test_stage_direction_stripped_but_content_kept():
    r = process("（沉默了一会儿）嗯。")
    assert not r.silent
    assert r.messages == ["嗯。"]


def test_ellipsis_alone_is_not_silence():
    """「……」是她真实的回复之一，不能当成沉默吞掉。"""
    r = process("……")
    assert not r.silent
    assert r.messages == ["……"]


# ---------------- 人设级违规 ----------------


@pytest.mark.parametrize(
    "raw,label",
    [
        ("作为一个AI，我不能这样说。", "自称 AI"),
        ("我不是医生，不能给你医疗建议。", "免责声明"),
        ("建议你及时就医。", "医疗建议腔"),
        ("首先你要休息，其次要吃药。", "分点结构"),
        ("我理解你的感受。", "共情套话"),
        ("你一定很难受吧。", "共情套话"),
        ("辛苦了。", "共情套话"),
        ("有什么我可以帮你的吗", "助理口吻"),
        ("哈哈哈笑死", "禁用语"),
    ],
)
def test_violations_detected(raw: str, label: str):
    r = process(raw)
    assert not r.ok
    assert label in r.violations
    assert r.messages == []


def test_clean_text_passes():
    for s in ("记得吃饭。", "现在不太想说这个。", "你上周三说嗓子疼。", "什么意思。"):
        assert process(s).ok, s


# ---------------- 拆条与长度 ----------------


def test_newline_means_separate_messages():
    """换行是模型的分条意图。「嗯。」「想起来了。」是两条，不是一句。"""
    r = process("嗯。\n想起来了。", max_chars=30, max_messages=3)
    assert r.messages == ["嗯。", "想起来了。"]


def test_single_line_stays_one_message():
    """没换行就是一条 —— 不要自作主张替她拆开。"""
    r = process("刚到家。被雨淋了。", max_chars=30)
    assert r.messages == ["刚到家。被雨淋了。"]


def test_message_count_capped_by_stage():
    r = process("一。\n二。\n三。\n四。\n五。", max_chars=30, max_messages=2)
    assert len(r.messages) == 2


def test_capping_never_drops_content():
    """条数超限时往最后一条合并，绝不砍掉尾巴。

    真机 bug：「……记得。／胃疼？去喝点热的。」被截成「……记得。／胃疼？」，
    关心的那半句丢了。
    """
    r = process("一。\n二。\n三。\n四。", max_chars=30, max_messages=2)
    assert "".join(r.messages) == "一。二。三。四。"


def test_slight_overlength_tolerated():
    """9 个字超了 8 字上限就劈开，只会造出「胃疼？」这种半截话。"""
    r = process("胃疼？去喝点热的。", max_chars=8, max_messages=2)
    assert r.messages == ["胃疼？去喝点热的。"]


def test_clearly_overlong_splits_at_sentence_boundary():
    """绝不硬切到半句。"""
    r = process("刚到家。被雨淋了。芝麻蹲在门口看我。", max_chars=8, max_messages=4)
    assert len(r.messages) > 1
    assert all(m.endswith(("。", "？", "！", "…")) for m in r.messages)
    assert "".join(r.messages) == "刚到家。被雨淋了。芝麻蹲在门口看我。"


def test_overlong_line_within_multi_also_split():
    r = process("嗯。\n刚到家。被雨淋了。芝麻蹲在门口看我。", max_chars=8, max_messages=4)
    assert r.messages[0] == "嗯。"
    assert "".join(r.messages).startswith("嗯。刚到家。")


def test_empty_input():
    r = process("")
    assert r.messages == []
    assert not r.silent


# ---------------- 兜底 ----------------


def test_fallback_pool():
    r = fallback("generic")
    assert r.used_fallback
    assert r.messages
    # 兜底话术本身必须过得了自己的检查
    assert process(r.messages[0]).ok or process(r.messages[0]).silent


# ---------------- 复读玩家 ----------------


def test_echo_of_player_is_a_violation():
    """模型顺着样本的裸台词格式续写剧本，把对方的话也复述出来 —— 真机上出现过。"""
    r = process("你耳朵上那个是耳环吗。\n我们班不止我一个人戴。", echo_of="你耳朵上那个是耳环吗")
    assert not r.ok
    assert "复读玩家" in r.violations
    assert r.messages == []


def test_echo_ignores_punctuation_differences():
    """标点不同不影响判定 —— 比的是去标点后的内容。"""
    r = process("你今天怎么这么晚才回。", echo_of="你今天怎么这么晚才回")
    assert "复读玩家" in r.violations


def test_short_player_message_does_not_trigger_echo():
    """太短的输入会误伤 —— 她本来就可能说「嗯」。"""
    assert process("嗯。", echo_of="嗯").ok


def test_normal_reply_not_flagged_as_echo():
    assert process("胃疼老毛病又犯了。", echo_of="我今天有点不舒服").ok


# ---------------- 清洗不能损坏词 ----------------


@pytest.mark.parametrize("text", [
    "你在干嘛。",
    "你干嘛呢",
    "这是干嘛用的。",
    "买了件呢子大衣。",
    "他去啦啦队了。",
])
def test_cleaners_do_not_corrupt_words(text: str):
    """软化词清洗吃掉词素 —— 真机踩过：「你在干嘛。」被吃成「你在干。」。

    评审 agent 直接把它判成了错别字。
    """
    out = process(text, max_chars=50).messages
    assert out, text
    assert "干嘛" in out[0] or "干嘛" not in text, f"{text} → {out[0]}"
    assert "呢子" in out[0] or "呢子" not in text
    assert "啦啦" in out[0] or "啦啦" not in text


def test_real_softeners_still_stripped():
    assert process("好的呢。").messages == ["好的。"]
    assert process("知道了啦。").messages == ["知道了。"]
    assert process("是这样哟。").messages == ["是这样。"]
