"""输出后处理。

**prompt 约束压不住的，在这里强制执行。**

两级处理：

  - **外观级**（emoji、感叹号、波浪号、软化尾词、分点符号）→ 静默清洗
  - **人设级**（助理腔、破人设、医疗建议）→ 判定为违规，交给上层重试；
    重试仍失败则走兜底话术池

第二级不能靠清洗解决 —— 一句「建议你及时就医」删掉「建议」还是错的，
整句都不该存在。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------- 外观级 ----------------

_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # 符号与象形
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"   # 杂项符号与装饰
    "\U0001F1E6-\U0001F1FF"   # 旗帜
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"   # 变体选择符
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)

_KAOMOJI = re.compile(r"[（(][^）)]{0,10}[･ω・´｀°∀ㅠㅜ゜^><≧≦][^）)]{0,10}[）)]")

# 括号里的一律是旁白／动作描写。模型表达「她没有回复」时会写成
# 「（没有回复。）」「（长久的沉默）」—— 删掉之后如果整条空了，
# 那就**真的让她不回**。模型的舞台指示由此变成真实的沉默。
_STAGE_DIRECTION = re.compile(r"[（(][^（）()]{0,40}[）)]")

# 未闭合的开括号 —— 模型被 max_tokens 截断，或者本来就没写完。
# 从开括号一路吃到结尾。
_UNCLOSED_PAREN = re.compile(r"[（(][^（）()]*$")

_TILDE = re.compile(r"[~～]+")
_EXCLAM = re.compile(r"[!！]+")
_REPEAT_Q = re.compile(r"[?？]{2,}")

# 句尾软化词。只在句末或标点前删，避免误伤「呢子大衣」「啦啦队」。
#
# ⚠️ 前面的负向环视是必须的：这些字在某些词里是词素，不是语气词。
# 真机踩过 ——「你在干嘛。」被吃成「你在干。」，评审直接判成错别字。
_SOFTENERS = re.compile(
    r"(?<![干做搞弄])"          # 干嘛／做嘛
    r"(?<![哪那这什])"          # 哪呢／那呢
    r"(?<=[一-鿿])"
    r"(呢|啦|嘛|哟|咯|喔|呀)"
    r"(?=[。，、？…\s]|$)"
)

# 分点／条列标记
_BULLETS = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|第[一二三四五六七八九十]+[点条])\s*",
                      re.MULTILINE)

_MULTI_DOT = re.compile(r"\.{3,}|。{2,}")
_SPACES = re.compile(r"[ \t]{2,}")

# ---------------- 人设级 ----------------

ASSISTANT_TELLS: tuple[tuple[str, str], ...] = (
    (r"作为(一个)?(AI|人工智能|语言模型|助手)", "自称 AI"),
    (r"我(只)?是(一个)?(AI|人工智能|程序|语言模型)", "自称 AI"),
    (r"我不是(医生|专业人士)", "免责声明"),
    (r"不能(给你|提供).{0,6}(医疗|专业)建议", "免责声明"),
    (r"建议你(及时|尽快|去)?(就医|就诊|看医生)", "医疗建议腔"),
    (r"^(首先|第一点)", "分点结构"),
    (r"(其次|最后|总之|综上)", "分点结构"),
    (r"我(完全)?(理解|明白)你的(感受|心情)", "共情套话"),
    (r"你(一定|肯定)(很|非常)(难受|辛苦|不容易)", "共情套话"),
    (r"辛苦(你)?了", "共情套话"),
    (r"有什么(我)?可以(帮|为你).{0,4}的", "助理口吻"),
    (r"希望(能|可以)?(帮到你|对你有帮助)", "助理口吻"),
    (r"需要我.{0,8}吗", "助理口吻"),
    (r"[❤️💕😊😭🥺]", "表情符号"),
    (r"(哈哈哈|笑死|xswl|233)", "禁用语"),
)

_ASSISTANT_RE = tuple((re.compile(p), label) for p, label in ASSISTANT_TELLS)

# 兜底话术池（edge-cases.md §9）。宁可沉默，也不要破人设。
#
# 台词在 `content/characters/<id>/agent.yaml` 的 fallbacks。这里只留一份
# **不认得任何角色时**的兜底 —— 三个点对谁都成立，不会破任何人的人设。
LAST_RESORT: tuple[str, ...] = ("……",)


@dataclass(slots=True)
class ProcessResult:
    messages: list[str] = field(default_factory=list)
    """拆条后的消息。空列表 ＝ 不发送（沉默）。"""

    silent: bool = False
    """模型输出的全部是旁白 —— 它的意思就是她不回。照办。"""

    violations: list[str] = field(default_factory=list)
    """人设级违规。非空则应重试。"""

    cleaned: list[str] = field(default_factory=list)
    """做过的外观级清洗，用于观测。"""

    dropped_repeats: list[str] = field(default_factory=list)
    """丢掉的逐字重复。

    **她说过的原话不该再发一遍**，而这个用 prompt 拦不住 ——
    真实对局里「你最近几条说的是…不要重复上面的句子」已经在 prompt 里了，
    她照样把三条原话整块又发了一遍。确定性的事在这里做。
    """

    rejected: str = ""
    """判违规而被丢掉的原文。

    **不留档就只能猜。** 复读检测的违规数在基准里从 0 涨到唯一来源，
    但分不出是真阳性还是又一次误报 —— 而这个检测器我已经改错两次了。
    被丢的文本本来就在手里，存一份是零成本的。
    """

    used_fallback: bool = False

    @property
    def ok(self) -> bool:
        return not self.violations


def _strip_cosmetic(text: str, log: list[str]) -> str:
    def sub(pattern: re.Pattern[str], repl: str, label: str, s: str) -> str:
        new = pattern.sub(repl, s)
        if new != s:
            log.append(label)
        return new

    text = sub(_EMOJI, "", "emoji", text)
    text = sub(_KAOMOJI, "", "颜文字", text)
    text = sub(_STAGE_DIRECTION, "", "旁白", text)
    text = sub(_UNCLOSED_PAREN, "", "残缺旁白", text)
    text = sub(_TILDE, "", "波浪号", text)
    text = sub(_EXCLAM, "。", "感叹号", text)
    text = sub(_REPEAT_Q, "？", "叠问号", text)
    text = sub(_SOFTENERS, "", "软化尾词", text)
    text = sub(_BULLETS, "", "分点标记", text)
    text = sub(_MULTI_DOT, "……", "省略号", text)
    text = sub(_SPACES, " ", "多余空格", text)

    # 清洗软化词后可能留下「。。」或空句
    text = re.sub(r"。{2,}", "。", text)
    return text.strip()


def _detect_violations(text: str) -> list[str]:
    found: list[str] = []
    for pattern, label in _ASSISTANT_RE:
        if pattern.search(text):
            found.append(label)
    return sorted(set(found))


LENGTH_TOLERANCE = 1.6
"""超出字数上限多少倍才真的动手拆。

字数上限的**主要执行者是 prompt**，后处理只是安全网。9 个字超了 8 字的限
就把句子劈开，反而制造出「胃疼？」这种半截话 —— 比超一个字糟糕得多。
"""


def _split_messages(text: str, max_chars: int, max_messages: int) -> list[str]:
    """拆条。

    **换行是模型的分条意图，优先尊重。** 真人在 IM 里想到什么发什么 ——
    「嗯。」「想起来了。」是两条，不是一句。

    三条原则：
      1. 模型换行 → 分条
      2. 只有明显超长才按句边界拆，绝不硬切到半句
      3. **绝不丢内容** —— 条数超了就往最后一条合并，不是砍掉
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    limit = int(max_chars * LENGTH_TOLERANCE)
    out: list[str] = []
    for ln in lines:
        out.extend(_fit(ln, max_chars, limit))

    if len(out) <= max_messages:
        return out

    # 超出条数：把多出来的并进最后一条，而不是丢掉
    head = out[: max_messages - 1]
    tail = "".join(out[max_messages - 1:])
    return head + [tail]


def _fit(line: str, max_chars: int, limit: int) -> list[str]:
    """一行明显超长时按句切开；否则原样放行。"""
    if len(line) <= limit:
        return [line]
    parts = [s.strip() for s in re.split(r"(?<=[。？！…])\s*", line) if s.strip()]
    out: list[str] = []
    buf = ""
    for s in parts:
        if buf and len(buf) + len(s) > max_chars:
            out.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        out.append(buf)
    return out or [line]


_PUNCT = re.compile(r"[\s。，、？！,.?!…～~「」“”\"']+")


def _normalize(s: str) -> str:
    return _PUNCT.sub("", s)


# 虚词。判「说的是不是同一件事」时不算数 ——
# 「我今天看到学校门口修路了」和「校门口修路」共享的实词才是重点。
_FUNCTION_WORDS = frozenset("我你他她的了是在有和就都也很个着过吗呢吧啊今天")

ECHO_THRESHOLD = 0.75
"""她这句里的实词有多少来自他刚说的那句，就算复读。

0.75 是折中：接话本来就会重复对方的词（他说「胃疼」她说「还疼吗」，
共享「疼」是正常的）。要抓的是**整句实词几乎全来自他**的情况。
"""


def _content_chars(s: str) -> set[str]:
    return {c for c in _normalize(s) if c not in _FUNCTION_WORDS}


# 他在提问。**问句里复用他的词是回答，不是复读。**
#
# 「你作业写完了吗」→「作业写完了。」实词完全重合，但那是标准应答。
# 少了这个判断，检测器会把正常回答当违规 —— 12 局基准实测违规从 2 涨到 11、
# 兜底 4 次，她被逼着说「……」。
def _asks(text: str) -> bool:
    t = text.strip()
    return bool(t.endswith(("吗", "吗。", "吗？", "呢", "呢。", "呢？", "没", "没？",
                            "?", "？", "吧", "吧。", "吧？"))
                or re.search(r"什么|怎么|哪个|哪儿|几点|多久|是不是|有没有", t))


MIN_ECHO_CHARS = 6
"""他那句要有这么多实词，复读才算数。

短句本来就容易整句重合（「明天考试吧」→「明天考试。」）。
要抓的是**他讲了一件事、她原样倒回去**，那种句子不会短。
"""


def _echoes(text: str, echo_of: str) -> bool:
    """她是不是在复读他刚说的话。

    ⚠️ 这个检查改过两次，两次都错了，记下来免得再犯。

    **第一版方向反了**：`if 他的整句 in 她的这一行` —— 要求他说的话完整
    出现在她的话里。她的话更短就永远抓不到，而真实失败模式恰恰是
    **用更少的字重复同一件事**。这个版本从上线到基准复盘没起过作用。

    **第二版误伤了正常应答**：改成「她的实词有多少来自他」之后，
    「你作业写完了吗」→「作业写完了。」被判违规 —— 那是标准回答。
    12 局基准实测违规 2→11、兜底 0→4，她被逼着说「……」。

    现在两个前提都要满足：

      1. **他不是在提问** —— 回答问题时复用他的词是应该的
      2. **他那句够长** —— 短句整句重合是常态，不是复读
    """
    if _asks(echo_of):
        return False
    src = _content_chars(echo_of)
    if len(src) < MIN_ECHO_CHARS:
        return False
    for line in text.splitlines():
        mine = _content_chars(line)
        if len(mine) < 3:
            continue        # 「嗯。」这种短应答不算复读
        if len(mine & src) / len(mine) >= ECHO_THRESHOLD:
            return True
    return False


def process(
    text: str,
    *,
    max_chars: int = 30,
    max_messages: int = 2,
    echo_of: str | None = None,
    said_recently: list[str] | None = None,
) -> ProcessResult:
    """清洗 → 校验 → 拆条。

    echo_of: 玩家刚说的那句。模型有时会顺着样本的裸台词格式续写剧本，
             把对方的话也复述出来 —— 真机上出现过。
    """
    result = ProcessResult()

    text = (text or "").strip()
    if not text:
        return result

    # 模型偶尔会把台词包在引号或「」里
    text = text.strip('"“”').strip()
    if text.startswith("「") and text.endswith("」"):
        text = text[1:-1].strip()

    text = _strip_cosmetic(text, result.cleaned)

    # 沉默的判定必须是「**剥掉旁白之后**才变空」。
    # 「……」本身是她的真实回复（慌／生气／委屈都用它），不能当沉默吞掉。
    stripped_narration = {"旁白", "残缺旁白"} & set(result.cleaned)
    if stripped_narration and not re.search(r"[\w一-鿿]", text):
        result.silent = True
        return result

    result.violations = _detect_violations(text)

    if echo_of and _echoes(text, echo_of):
        result.violations.append("复读玩家")

    if result.violations:
        result.rejected = text
        result.messages = []
        return result

    result.messages = _split_messages(text, max_chars, max_messages)

    # 逐字重复直接丢。**不重试** —— 重试要多花一次调用，而这件事
    # 确定性可判、确定性可修：她说过的话，删掉就是了。
    #
    # 全丢光的话就让她沉默 —— 一句原话都没重复的余地时，
    # 不说话比复读好。
    if said_recently:
        seen = {_normalize(x) for x in said_recently if len(x) > 3}
        kept = []
        for m in result.messages:
            if len(m) > 3 and _normalize(m) in seen:
                result.dropped_repeats.append(m)
            else:
                kept.append(m)
        result.messages = kept

    return result


def fallback(
    kind: str = "generic", index: int = 0, character_id: str = "h01",
) -> ProcessResult:
    """人设级违规重试仍失败时的兜底。宁可沉默，也不要破人设。"""
    from ..persona.agent_data import load_agent_data

    pool = load_agent_data(character_id).fallback_pool(kind) or LAST_RESORT
    return ProcessResult(messages=[pool[index % len(pool)]], used_fallback=True)
