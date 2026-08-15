"""用我们自己的指标去量一份参考语料，得到有依据的目标区间。

    python scripts/analyze_corpus.py reference/amakano.txt
    python scripts/analyze_corpus.py reference/*.txt --compare s3

**为什么要这个。** `critic.py` 里的阈值（具体性 >30%、关系话 <40%、
句长 ≤40）全是拍脑袋估的。拿真实的 galgame 对白量一遍，就能知道
成品作品的实际分布长什么样，阈值才有依据。

⚠️ **只量统计特征，不复制表达。**
输出的是数字和分布，不是台词。参考语料自己准备，本仓库不附带。

输入格式：一行一句台词。可以带说话人前缀（`角色名：台词`），会自动剥掉。
只想量某个角色就先用 `--speaker` 过滤。
"""

from __future__ import annotations

import argparse
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gfagent.evals.critic import _ASSISTANT, _BANNED, _CONCRETE, _RELATIONAL  # noqa: E402

_SPEAKER = re.compile(r"^\s*[\[【]?([^\]】：:]{1,12})[\]】]?\s*[：:]\s*")
_PUNCT_SOFT = re.compile(r"[呢啦嘛哟咯喔呀]")
_EXCLAM = re.compile(r"[!！]")
_TILDE = re.compile(r"[~～]")
_ELLIPSIS = re.compile(r"…|\.\.\.")
_QUESTION = re.compile(r"[?？]")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

# ── 中文角色声音的六个维度 ──
# 日语靠一人称／語尾／敬语区分角色，中文没有这套工具（见
# content/craft/chinese-character-voice.md）。这些是中文实际能用的。

_PERIOD = re.compile(r"[。！？!?]")
_COMMA = re.compile(r"[，,、]")
_SUBJECT = re.compile(r"^(我|你|他|她|咱)")
"""句首主语。中文可以大量省略主语，省略程度直接反映亲密度与性格。"""

_REASON = re.compile(r"因为|所以|不然|要不然|毕竟|反正.{0,6}[，。]|得|要")
"""给理由 vs 不解释 —— 比任何语气词都能区分性格。"""

_FORMAL = re.compile(r"确实|的确|不过|或许|并[没不]|然而|因此|是否|无法|已经")
_COLLOQUIAL = re.compile(r"反正|就是|其实|那个|挺|咋|干嘛|呗|啥")
_NETSPEAK = re.compile(r"绝了|典|蚌埠|栓Q|xswl|233|绝绝子|笑死|哈哈哈")

_BACKCHANNEL = re.compile(r"^(嗯|哦|啊|噢|唔|额)[。，、！？!?]?$")
"""只回一个字 —— 沉默式回应。"""


def load(path: Path, speaker: str | None) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith(("#", "//", ";")):
            continue
        m = _SPEAKER.match(raw)
        who, text = (m.group(1), raw[m.end():]) if m else (None, raw)
        if speaker and who != speaker:
            continue
        text = text.strip().strip("「」『』\"“”")
        if text:
            lines.append(text)
    return lines


def analyze(lines: list[str], label: str) -> dict[str, float]:
    if not lines:
        return {}
    n = len(lines)
    lengths = [len(s) for s in lines]

    def ratio(pattern) -> float:
        return sum(bool(pattern.search(s)) for s in lines) / n

    stats = {
        "条数": n,
        "平均字数": mean(lengths),
        "中位字数": median(lengths),
        "p90 字数": sorted(lengths)[int(n * 0.9)],
        "最长": max(lengths),
        "≤10 字占比": sum(l <= 10 for l in lengths) / n,
        "≤20 字占比": sum(l <= 20 for l in lengths) / n,
        ">40 字占比": sum(l > 40 for l in lengths) / n,
        "具体性": ratio(_CONCRETE),
        "关系话": ratio(_RELATIONAL),
        "感叹号": ratio(_EXCLAM),
        "波浪号": ratio(_TILDE),
        "省略号": ratio(_ELLIPSIS),
        "问号": ratio(_QUESTION),
        "语气词": ratio(_PUNCT_SOFT),
        "emoji": ratio(_EMOJI),
        "助理腔": ratio(_ASSISTANT),
        "禁用符号": ratio(_BANNED),
    }

    dupes = Counter(lines)
    stats["重复句占比"] = sum(c for s, c in dupes.items() if c > 1 and len(s) > 3) / n

    # ── 六个维度 ──
    chars = sum(lengths)
    stats["—— 六维 ——"] = 0.0
    stats["句号密度"] = len(_PERIOD.findall("".join(lines))) / max(1, chars) * 100
    stats["逗号密度"] = len(_COMMA.findall("".join(lines))) / max(1, chars) * 100
    stats["带主语开头"] = ratio(_SUBJECT)
    stats["给理由"] = ratio(_REASON)
    stats["书面词"] = ratio(_FORMAL)
    stats["口语词"] = ratio(_COLLOQUIAL)
    stats["网络词"] = ratio(_NETSPEAK)
    stats["单字回应"] = ratio(_BACKCHANNEL)
    return stats


def show(name: str, stats: dict[str, float]) -> None:
    if not stats:
        print(f"  {name}: 没有可分析的内容")
        return
    ratios = {
        "具体性", "关系话", "感叹号", "波浪号", "省略号", "问号",
        "语气词", "emoji", "助理腔", "禁用符号",
        "带主语开头", "给理由", "书面词", "口语词", "网络词", "单字回应",
    }
    print(f"\n  {name}")
    print("  " + "─" * 46)
    for k, v in stats.items():
        if k.startswith("——"):
            print(f"\n  {k}")
        elif k == "条数":
            print(f"  {k:<12} {int(v)}")
        elif "占比" in k or k in ratios:
            print(f"  {k:<12} {v:>6.1%}")
        elif "密度" in k:
            print(f"  {k:<12} {v:>6.1f} 个/百字")
        else:
            print(f"  {k:<12} {v:>6.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="语料文件，一行一句")
    ap.add_argument("--speaker", help="只分析某个角色（按 `名字：` 前缀过滤）")
    ap.add_argument("--compare", metavar="PRESET",
                    help="拿最近一次 selfplay 的同阶段结果对照")
    args = ap.parse_args()

    print("\n用我们自己的指标量参考语料。**只统计，不复制表达。**")

    all_lines: list[str] = []
    for pattern in args.paths:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if not path.exists():
                print(f"  ✗ 找不到 {path}")
                continue
            lines = load(path, args.speaker)
            all_lines += lines
            show(f"{path.name}（{args.speaker or '全部角色'}）",
                 analyze(lines, path.name))

    if len(args.paths) > 1 and all_lines:
        show("合计", analyze(all_lines, "合计"))

    if args.compare:
        import json
        reports = sorted(Path("logs/selfplay").glob("*.json"))
        if not reports:
            print("\n  没有 selfplay 报告可对照")
        else:
            data = json.loads(reports[-1].read_text(encoding="utf-8"))
            rows = [r for r in data if r["preset"] == args.compare]
            if rows:
                m = rows[-1]["mechanical"]
                print(f"\n  林静姝（{args.compare}，{reports[-1].name}）")
                print("  " + "─" * 46)
                print(f"  {'平均字数':<12} {m['avg_len']:>6.1f}")
                print(f"  {'具体性':<12} {m['concrete_ratio']:>6.1%}")
                print(f"  {'关系话':<12} {m['relational_ratio']:>6.1%}")

    print("\n  怎么读：")
    print("  · 平均／中位字数 → 校准 STAGE_BEHAVIOR 的 max_chars")
    print("  · 具体性、关系话 → 校准 critic 的阈值（现在是拍脑袋定的）")
    print("  · 感叹号／波浪号／语气词 → 印证「她不用这些」是不是过头了")
    print("  · **六维** → 这才是中文区分角色的工具，"
          "见 content/craft/chinese-character-voice.md")
    print("  · 设计 H02／H03 时，至少要在三维以上跟林静姝拉开\n")
    print("  ! 拆维度，不要抄句子。性格标签是结果，维度才是原因；")
    print("     抄来的句子换个角色就废了。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
