"""从 content/ 装配人设卡。

流程：读 md → 按二级标题抽取 → 清洗（去引用块注释、去 🔲 标记）→ 拼成稳定前缀。

清洗很重要：设定文档里的 `>` 引用块常常是给人看的旁白（「注：这一条最重要」），
喂给模型会让它开始**评论角色**而不是**扮演角色**。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .manifest import (
    EDGE_SECTIONS,
    EXCLUDED_FILES,
    LEXICON_SECTIONS,
    PERSONA_SECTIONS,
    SAMPLE_SECTIONS,
    Section,
)

log = logging.getLogger(__name__)

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# 引用块**保留内容，只去掉 `>` 记号**。
# 之前是整行删除，理由是"引用块多半是给人看的旁白"—— 那是错的：
# 设定文档里 `>` 恰恰用来强调题眼（「她的撒娇是命令句」「她不知道他知道」），
# 删掉等于把最重要的规则从 prompt 里抹了。
# 真正的编者旁注由 _EDITOR_NOTE 单独处理。
_QUOTE_MARK = re.compile(r"^>[ \t]?", re.MULTILINE)
_TODO_MARK = re.compile(r"🔲\s*[^|\n]*")
_MULTI_BLANK = re.compile(r"\n{3,}")


def content_root() -> Path:
    """content/ 目录。相对包位置向上找，兼容 editable 安装与源码运行。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "content"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("找不到 content/ 目录")


def _split_sections(text: str) -> dict[str, str]:
    """按二级标题切分。返回 {标题: 正文}。"""
    out: dict[str, str] = {}
    matches = list(_H2.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text[start:end].strip()
    return out


def _clean(text: str) -> str:
    """去掉给人看的旁白，保留设定本身。"""
    text = _QUOTE_MARK.sub("", text)       # 去掉 > 记号，保留内容
    text = _TODO_MARK.sub("", text)        # 待填标记
    text = text.replace("⚠️", "").replace("⭐", "")
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_PURE_INDEX = re.compile(r"^(#|\d+)$")
# 整段吃掉（注释常跨行），直到空行为止
_EDITOR_NOTE = re.compile(r"^[ \t]*\*\*注\*\*[：:][\s\S]*?(?=\n[ \t]*\n|\Z)",
                          re.MULTILINE)
_HR = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)


def _flatten_tables(text: str) -> str:
    """markdown 表格 → 纯行。

    `| 1 | 刚到家。 |` → `刚到家。`
    `| ❌ 绝不 | ✅ 她会说 |` → `❌ 绝不 → ✅ 她会说`

    编号列和管道符不携带信息，但要占掉整份样本约三分之一的 token。

    同时删掉 `**注**：…` —— 那是给编剧的旁注，而且常常按编号引用具体条目
    （「3–7 这几条」），扁平化之后编号已经不存在，留着只会让模型困惑。
    """
    text = _EDITOR_NOTE.sub("", text)
    text = _HR.sub("", text)
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _TABLE_SEP.match(stripped):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            cells = [c for c in cells if c and not _PURE_INDEX.match(c)]
            if not cells:
                continue
            if cells in (["台词"], ["台词", "备注"]):   # 表头
                continue
            out.append(" → ".join(cells) if len(cells) > 1 else cells[0])
        else:
            out.append(line)
    return _MULTI_BLANK.sub("\n\n", "\n".join(out)).strip()


def _render(sections: tuple[Section, ...], char_dir: Path, stage: str = "") -> str:
    parts: list[str] = []
    for sec in sections:
        if sec.stages is not None and stage and stage not in sec.stages:
            continue
        if sec.file in EXCLUDED_FILES:
            raise ValueError(f"{sec.file} 在排除清单里，不应进人设卡")
        path = char_dir / sec.file
        if not path.exists():
            log.warning("人设卡缺少文件：%s", path)
            continue

        raw = path.read_text(encoding="utf-8")
        post = _flatten_tables if sec.flatten_tables else (lambda s: s)

        if not sec.headings:
            body = post(_clean(raw))
            if body:
                parts.append(body)
            continue

        found = _split_sections(raw)
        for heading in sec.headings:
            if heading not in found:
                log.warning("%s 中找不到小节「%s」—— 设定文档可能改过标题",
                            sec.file, heading)
                continue
            body = post(_clean(found[heading]))
            if body:
                parts.append(f"## {heading}\n\n{body}")

    return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class PersonaCard:
    character_id: str
    persona: str
    lexicon: str
    samples: str
    edge_cases: str

    @property
    def approx_tokens(self) -> int:
        """粗估。中文约 1.5 字/token，英文和标点另算 —— 只用于体量告警。"""
        total = len(self.persona) + len(self.lexicon) + len(self.samples) + len(self.edge_cases)
        return int(total / 1.5)

    def stable_text(self) -> tuple[str, str]:
        """返回 (persona 层, lexicon 层)。samples 与 edge 并入 lexicon 层尾部。"""
        lex = "\n\n".join(
            p for p in (
                self.lexicon,
                f"# 语气样本\n\n{self.samples}" if self.samples else "",
                f"# 越界与破防处理\n\n{self.edge_cases}" if self.edge_cases else "",
            ) if p
        )
        return self.persona, lex


@lru_cache(maxsize=32)
def load_card(character_id: str = "h01", stage: str = "") -> PersonaCard:
    """装配人设卡。结果缓存 —— 稳定前缀在进程生命周期内不该变。

    `stage` 为 `"S0"`–`"S3"` 时，只装配该阶段适用的样本小节（见 manifest 的
    `Section.stages`）。空串 ＝ 不门控，装全部；用于工具脚本和体量检查。

    每个阶段是一份独立但稳定的前缀，缓存各自命中。

    改了 content/ 需要重启，或调用 `load_card.cache_clear()`。
    """
    char_dir = content_root() / "characters" / character_id
    if not char_dir.is_dir():
        raise FileNotFoundError(f"角色目录不存在：{char_dir}")

    card = PersonaCard(
        character_id=character_id,
        persona=_render(PERSONA_SECTIONS, char_dir, stage),
        lexicon=_render(LEXICON_SECTIONS, char_dir, stage),
        samples=_render(SAMPLE_SECTIONS, char_dir, stage),
        edge_cases=_render(EDGE_SECTIONS, char_dir, stage),
    )

    # 阈值定在 12k：稳定前缀在所有玩家之间相同，命中缓存后每次只花
    # ~¥0.02/M，10k 前缀 ≈ ¥0.0002/次，可以接受。真正的代价不是钱，
    # 是**注意力稀释** —— 系统提示越长，模型对具体格式规则的遵守越差。
    # 超过这个数就该问：哪一段是模型真的需要的，哪一段只是给人看的。
    if card.approx_tokens > 12000:
        log.warning(
            "人设卡约 %d tokens，过大。规则会被稀释，模型对格式约束的遵守会下降。"
            "考虑裁剪 manifest。", card.approx_tokens,
        )
    return card
