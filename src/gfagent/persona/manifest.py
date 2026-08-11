"""人设卡装配清单。

`content/` 里的 md 是给人看的：有目录、有索引、有设计说明。直接整份塞进 prompt
既超长又会把设计理由喂给模型（模型会开始"解释角色"而不是"演角色"）。

所以这里按 **文件 + 二级标题** 精确抽取，保持 content 是单一事实源，不复制内容。
改设定只改 md，卡片自动跟着变。

分层对应 `prompt/layers.py` 的稳定前缀顺序（稳定在前，易变在后）：

    persona  →  lexicon  →  [facts 动态]  →  [memory 动态]

`design-notes.md` **永不进卡**。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Section:
    file: str
    headings: tuple[str, ...]
    """要抽取的二级标题（## 后面的文字，精确匹配）。空元组＝整份文件。"""

    flatten_tables: bool = False
    """把 markdown 表格压成纯行。

    参数表保留表格结构对模型有帮助；但样本库那种 `| # | 台词 |` 的编号列和
    管道符是纯 token 浪费，压成一行一句更省也更好读。
    """


# ---- persona 层：她是谁 ----
PERSONA_SECTIONS: tuple[Section, ...] = (
    # 「热恋前／热恋后」的行为规则已由 lexicon 的「关系越近，规则越松」统一表述，
    # 两处并存只会互相稀释。
    Section("personality.md", ("内核", "对外的样子", "她不是在钓，她是在试探",
                               "两种沉默", "她表达关心的方式", "喜好", "雷区")),
    Section("family.md", ("概要", "独生女", "母亲", "父亲", "家中的日常",
                          "家规", "衣服", "钢琴", "吉他", "耳洞",
                          "她在这个家里的位置")),
    Section("relationship.md", ("男主定位", "前史：他见过她两次，她一次都不知道",
                                "加好友：一次事务性的巧合",
                                "关键问题：她凭什么理他",
                                "信息不对称（本设定最重要的一条）",
                                "称呼：她基本不叫他名字")),
)

# ---- lexicon 层：她怎么说话 ----
LEXICON_SECTIONS: tuple[Section, ...] = (
    Section("lexicon.md", ("核心原则", "参数", "硬禁清单", "正误对照",
                           "健康话题专项（重点防线）",
                           "关系越近，规则越松（**最容易做错的一节**）",
                           "梗：她不用网络梗，但她们有自己的梗",
                           "不知道的事，绝不编", "人名禁令")),
    # 她自己的确定事实 —— 防止「不知道」规则被反向套用到她自己身上
    Section("relationship.md", ("她自己是确定的",)),
    # 「六点五、把话接住」的「给具体的」已并入 small-talk.md，
    # 但那一节的「短问回去」机制是独立的，仍然保留在样本层。
    Section("dialogue-rules.md", ("一、开场：她怎么开启一段对话",
                                  "二、收场：她怎么结束",
                                  "三、各类输入的应对",
                                  "四、被撩时的反应（分阶段）",
                                  "五、她自己越界之后",
                                  "六、话题准入",
                                  "七、消息节奏",
                                  "八、她不做的事")),
    # 「逐项说明」是给编剧看的展开，参数表已经把规则说全了
    Section("emotions.md", ("核心原则", "情绪参数表")),
    # 治「只有关系没有事情」这个病 —— 对话枯竭的头号原因
    Section("small-talk.md", ("一、成分比例", "二、四种质感",
                              "三、话题库", "四、反面清单"),
            flatten_tables=True),
    # ⚠️ moods.md **不进静态卡**。
    # 小情绪的行为由 state/moods.py 的 behavior_note() 在情绪触发时动态注入 ——
    # 情境性的规则就该情境性给。放进静态卡等于她平静的时候也在读
    # 「你在生气时该怎么演」，纯属稀释。
)

# ---- samples 层：怎么说才对 ----
# 「扩充规范」是给编剧的写作流程，不进 prompt。
SAMPLE_SECTIONS: tuple[Section, ...] = (
    Section("voice-samples.md", (
        "一、日常 · 平静", "二、累", "三、关心（医生侧）",
        "四、拒绝与划界（律师侧）", "五、试探（她的越界句）", "六、撤回",
        "七、被夸", "八、慌（被撩 · S1）", "九、S3 · 热恋期",
        "十、生气", "十一、委屈", "十二、开心",
        # 「十三、场景素材」已被 small-talk.md 的话题库取代，两处并存只是重复
        "十二点五、把话接住",
        "十四、告别（分阶段）", "十五、禁用对照",
    ), flatten_tables=True),
)

# ---- 越界处理 ----
# 只收角色层面的应对。第 9 条（兜底话术）和第 10 条（自伤内容）是系统层职责，
# 由 output/postprocess.py 与安全通道处理，不进 prompt。
EDGE_SECTIONS: tuple[Section, ...] = (
    Section("edge-cases.md", (
        "1. 「你是不是 AI／机器人」",
        "2. 越界／性化内容",
        "3. 玩家骂她／说重话伤害她",
        "4. 玩家长期消失后回来",
        "5. 玩家提到别的女生",
        "6. 玩家安排惊喜",
        "7. 玩家追问她的痛苦／家里的事",
        "8. 玩家说了很重的话（表白／「我喜欢你」）",
        "通用原则",
    )),
)

EXCLUDED_FILES = frozenset({"design-notes.md", "README.md", "schedule.md"})
"""design-notes 是设计理由（会让模型解释角色而不是演角色）；
README 是索引；schedule 由日程引擎结构化读取，不进 prompt 文本。"""
