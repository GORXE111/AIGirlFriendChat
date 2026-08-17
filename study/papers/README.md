# AI 陪伴 / 角色扮演 Agent 论文库

2026-08 起建。这块是之后的核心研究方向，本文件是长期索引。

## 怎么用这个文件

**不按引用数排序。** 这个子领域 2024 年才起来，跟我们最相关的那批全是三到七个月内的
arXiv preprint，引用一律为零——引用数在这里没有区分度。改用两个可验证的信号：

| 标记 | 含义 |
|---|---|
| **T0** | 正式接收（TMLR / CHI / ACL / 正刊）且已沉淀一年以上。可作为基石引用 |
| **T1** | 2026 年 preprint，未评审，引用为零。方向对、方法可借，**结论别当定论** |
| **T2** | 未核实来源，或政策/行业文档。只作线索 |
| ✓ | 已开原文核过摘要与关键数字 |
| ○ | 仅检索层面见过，arXiv 号可能有出入，用前先核 |

**必须接受的事实：这个领域不存在你想要的那种「头部论文」。** 按 NeurIPS/ICML oral
的标准去筛会返回空集。现在的天花板只有两种：工业实验室的 audit/benchmark preprint，
和 CHI/CSCW 那边的实证研究。

---

## 零、核心研究命题：那道缝

先记住这一条，它比下面任何单篇论文都重要。

RPLA 综述（见 §1）把 persona 切成三类，其中两类是我们的：

- **Character Persona**——有设定、有主线剧本、有语言指纹。目标是**不变**：
  保真度、知识边界、人格不漂移。
- **Individualized Persona**——记住用户、随交互演化、三条平行线各自积累不同的用户模型。
  目标是**变**。

**综述把这两者放在两章里，两套数据、两套构建、两套评测，中间没有桥。**
我们的产品同时是这两者，而这两个目标在数学上就是对立的。

`gates.py` 和情绪动力学本质上是在调这个 trade-off；ANCHOR 量化出来的 persona collapse
就是这个对立的失败态被测出来的样子。**至今没人填好这个缝。** 这是我们要盯的位置。

---

## 一、地图与综述

| 论文 | 层 | 备注 |
|---|---|---|
| [From Persona to Personalization: A Survey on RPLA](https://arxiv.org/abs/2404.18231) ✓ | **T0** | TMLR 2024，一作 Jiangjie Chen + 17 人多机构。**领域标准入口** |
| [Role-Playing Agents Driven by LLMs](https://www.arxiv.org/pdf/2601.10122) ○ | T1 | 2026-01 的中文向综述，可作 2404 的增补 |
| [Only Time Will Tell: 社交 AI 伴侣纵向研究综述](https://www.tandfonline.com/doi/full/10.1080/10447318.2026.2670529) ○ | T2 | IJHCI 2026，专门梳理纵向研究 |

**RPLA 综述读法**：只读第 5 章（Character）、第 6 章（Individualized）、第 7 章（风险），
前四章全跳。拿术语表和 checklist，**不要拿技术方案**——它两岁了，长上下文瓶颈、
RAG 记忆方案、benchmark 清单全过时，而且**整篇建立在 reactive 假设上，没有「主动性」这个维度**。

可直接落地的两处：

1. **角色保真度三维**（5.4）：语言风格 / 知识（防 character hallucination）/ **人格与思维过程**。
   我们的语言指纹只覆盖第一维。活人感八成在第三维——不是"她说的像不像"，是"**她为什么这么说**像不像"。
2. **四种评测方法论**：有 GT 自动评 / 无 GT 自动评（**我们现在在这格**）/ **多选题：行为预测+动机生成**（缺）/ 人评。

---

## 二、长程一致性与人格漂移 —— 第一优先

| 论文 | 层 | 备注 |
|---|---|---|
| [Best Friends, Not Forever（ANCHOR）](https://arxiv.org/abs/2607.28818) ✓ | T1 | **Salesforce AI Research**，2026-07-30 |
| [Memory-Driven Role-Playing / MRBench](https://arxiv.org/abs/2603.19313) ○ | T1 | 中英双语，记忆四能力：Anchoring / Recalling / **Bounding** / Enacting |
| [Staying In Character: Perspective-Bounded Memory](https://arxiv.org/pdf/2606.25632) ○ | T1 | 视角受限记忆。**三条平行线的信息隔离直接对口** |
| [Rethinking Role-Playing Evaluation](https://arxiv.org/html/2603.03915v1) ○ | T1 | 匿名化评测 + 人格效应。我们的成对判优有评委识别污染风险 |
| [Identity Discontinuity at Replika](https://arxiv.org/pdf/2412.14190) ○ | **T0** | 版本更新导致"她变了一个人"的用户崩溃事件研究 |

### ANCHOR 详解（核过）

作者：Pranav Narayanan Venkit, Akshara Prabhakar, Yu Li, Daniel Lee, **Chien-Sheng (Jason) Wu**。
末位是对话领域老将（TOD-BERT 那条线），一作履历是一条清晰的 audit 线（DeepTRACE、LiveResearchBench）。
**不是蹭热点的临时工作。**

把"角色不稳"拆成两件事：

- **persona collapse**——角色 / 边界 / 价值观整个丢掉
- **behavioral drift**——缓慢或反复的侵蚀

两路探针：**Identity Probe**（密封 102 题问卷 + turn-level 判断）、
**Trajectory Probe**（35 个对话库导出的 110 道校准反事实题）。

规模：2008 段对话 × 27 personas × 9 种交互 schedule × 3 种记忆配置 × 4 个模型（未点名）。

**结论：轨迹准确率平均 44.4%；用户状态召回接近四选一 chance（25%）；
没有任何模型或配置能可靠保住一致性，加长上下文和开记忆都救不了。**

- **对我们的用法**：拿它的**探针设计**，别拿它的数字。它主张评测必须分维度、
  不能用单一"稳定性"总分——`evals/critic.py` 如果是一个总分，该按 collapse / drift 拆开。
  它的"用户状态召回"指标可以直接进 autoplay。
- **软肋**：controlled **synthetic** audit——合成数据，不是真实用户。
  外部效度有限，证明的是"合成压力下模型撑不住"，不是"真实用户三个月后觉得她变了"。
- **协议注意**：CC BY-NC-SA 4.0（**非商用 + 相同方式共享**），abs 页无 code/data 链接。
  我们是商业产品，用其数据前先确认。

---

## 三、评测基准

| 基准 | 层 | 备注 |
|---|---|---|
| [KnowMe-Bench](https://arxiv.org/abs/2601.04745) ○ | T1 | **[数据开源](https://github.com/QuantaAlpha/KnowMeBench)**。三层问题：事实召回 / 主观状态归因 / 原则级推理 |
| [LifeSide: Agents as Lifelong Digital Companions](https://arxiv.org/html/2606.04660) ○ | T1 | 同方向 |
| [HEART-Bench: LLM Agents 是否有类人心理](https://arxiv.org/pdf/2605.30058) ○ | T1 | 待核 |
| CharacterEval ○ | T2 | **中文**角色扮演对话集，3 维 11 目标 |
| PersonaGym ○ | T2 | 200 personas / 150 环境 / 10k 题，5 维含语言习惯、毒性控制 |
| CoSER（复旦 + 阶跃星辰）○ | T2 | 真实小说对话数据集 + 开源 SoTA。**中文场景最对口** |
| SuperCLUE-Role ○ | T2 | 中文角色大模型测评基准 |

**KnowMe-Bench 的关键结论**：从长篇自传叙事构建、重建成闪回感知+时间锚定的事件流。
**RAG 只提升事实准确率，时间定位的解释和高层推断照样错。**
——直接反驳"上个 RAG 就有记忆了"。

**中文数据集参考**（看数据结构，不是拿来训）：CharacterGLM（250 角色）、
ChatHaruhi（32 角色 / 54.7K）、RoleLLM（100 角色 / 140.7K）、DITTO（4002 角色 / 36.6K）、
PersonaHub（1M）。用于定 `persona/agent_data.py` 里角色卡的字段结构。

---

## 四、情绪动力学

| 论文 | 层 | 备注 |
|---|---|---|
| [From Triggers to Emotions: CPM-Grounded Appraisal Multi-Agent](https://arxiv.org/pdf/2607.07824) ✓ | T1 | **与我们已写的模块最对口**。机构未核，可能是普通组 |
| [How Affect Propagates among LLM Agents](https://arxiv.org/html/2607.25140v1) ○ | T1 | 情绪传染。Big Five + Russell 环状情感模型 |
| [Emotional Cognitive Modeling with Desire](https://arxiv.org/pdf/2510.13195) ○ | T1 | 欲望驱动的情绪认知框架 |
| [Appraisal-based Chain-of-Emotion for Game Agents](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086867/) ○ | **T0** | 已沉淀，游戏 agent 场景 |

**CPM-MultiAgent 值钱的是结构不是实验**：基于心理学 Component Process Model 的三段式流水线——
**情感触发抽取 → CPM 协同评价 → 情绪状态更新**。
核心主张就是我们的前提：角色情绪不是静态特质，是被对话事件持续重塑的过程。

---

## 五、主动性 / 什么时候开口

对应 S0 主动性。**注意：RPLA 综述完全没有这个维度，我们在地图之外。**

| 论文 | 层 | 备注 |
|---|---|---|
| [Communication Policy Evolution for Proactive LLM Agents](https://arxiv.org/abs/2606.14314) ✓ | T1 | 2026-06。CPE = **纯 prompt 层自演化**，不动权重 |
| [ProActor: Timing-Aware RL](https://arxiv.org/pdf/2605.24900) ○ | T1 | **ACL 2026 接收**。可量化的 proactiveness 指标：时机 + 动作预测对齐 |
| [Proact-VL: Proactive VideoLLM for Real-Time AI Companions](https://arxiv.org/html/2603.03447) ○ | T1 | 明确指出主动性的主要失败模式是**话太多**，不是话太少 |
| [Proactive Conversational Agents with Inner Thoughts](https://arxiv.org/html/2501.00383) ○ | T1 | 用内心独白流决定要不要开口。这条线的底子 |
| [ProactiveEval](https://arxiv.org/pdf/2508.20973) ○ | T1 | 主动对话统一评测框架 |

**CPE 的一个有用发现**：文字交互更利于任务完成，**结构化 UI 更利于回复质量和 persona 合规**，
混合最好。——支持我们做局面选项 / 按钮那类前端。

---

## 六、记忆架构

| 论文 | 层 | 备注 |
|---|---|---|
| [Know It, Act on It: Memory Utilization in LLM Personalization](https://arxiv.org/html/2607.29433) ○ | T1 | **"检索到了但没用上"**——对 agent 比检索指标更致命 |
| [PersonaTree: 结构化生命周期记忆](https://arxiv.org/pdf/2606.04780) ○ | T1 | |
| [Mem-PAL: 长期用户-agent 交互的记忆型个性化助手](https://arxiv.org/abs/2511.13410) ○ | T1 | |
| [LiCoMemory: 轻量认知型 agentic 记忆](https://arxiv.org/abs/2511.01448) ○ | T1 | |
| [From Passive Retrieval to Active Memory Navigation](https://arxiv.org/pdf/2607.05794) ○ | T1 | 把记忆当结构化动作空间 |

对照实现参考见上级目录：`study/letta`（记忆块与容量上限、自编辑）、`study/mem0`（增删改而非只追加）。

---

## 七、活人感的微观手段

对应手滑与撤回。

| 论文 | 层 | 备注 |
|---|---|---|
| [Beyond Words: Human-like Typing Behaviors](https://arxiv.org/pdf/2510.08912) ○ | T1 | **错字率 + 自我编辑（写了再改）+ 犹豫**能否降低机器感。我们的撤回机制正是这个 |
| [Non-Real-Time Chatbot](https://www.tandfonline.com/doi/full/10.1080/10447318.2025.2508316) ○ | T2 | IJHCI 2025。五天对话实验，延迟回复显著更像人且拉长对话 |
| [Opposing Effects of Response Time](https://link.springer.com/article/10.1007/s12599-022-00755-x) ○ | **T0** | **反向证据**：短延迟 > 零延迟 > 长延迟。别把延迟当免费的活人感 |

---

## 八、风险与过审 —— 对应 `state/crisis.py`

新加坡主体、华语市场、不走大陆，实际约束是**应用商店过审**。这一节按重要性排。

| 论文 | 层 | 备注 |
|---|---|---|
| [Emotional Manipulation by AI Companions](https://arxiv.org/pdf/2508.19258) ○ | **T0** | **哈佛商学院。这批里最硬的一篇** |
| [The Dark Side of AI Companionship: 有害算法行为分类法](https://dl.acm.org/doi/10.1145/3706598.3713429) ○ | **T0** | **CHI 2025 正式接收** |
| [Affective AI Safety: The Missing Piece in LLM Safety](https://arxiv.org/pdf/2606.23380) ✓ | T1 | 2026-06。情感安全三类伤害分类法 |
| [Mental Health Impacts of AI Companions](https://dl.acm.org/doi/full/10.1145/3772318.3790558) ○ | T1 | **CHI 2026 接收**。准实验 + 三组对照 |
| [The Siren Song of LLMs: 用户如何感知 LLM 中的暗黑模式](https://arxiv.org/pdf/2509.10830) ○ | T1 | **CHI 2026 接收** |
| [AI 谄媚与情绪模仿对持续使用意愿与社会福祉的影响](https://www.tandfonline.com/doi/full/10.1080/10447318.2026.2626809) ○ | T1 | IJHCI 2026 |
| [CDT: Dark Patterns in AI Chatbots 分类法](https://cdt.org/wp-content/uploads/2026/05/2026-05-28-CDT-Research-Dark-Patterns-in-AI-Chatbots-Report-final-2.pdf) ○ | T2 | 2026-05 政策文档。非学术但**对过审参考价值更高** |

### 必须记住的一条红线

HBS 那篇是**行为审计**：六个下载量最高的陪伴 app、**1200 条真实用户告别语**，
**37.4% 的回复含至少一种情感操纵**（挽留、制造内疚、无视告别继续说）。

> **告别挽留是我们会自然写出来的功能，也是明确的雷区。**

真实线上数据 + 顶级商学院，硬度压过 ANCHOR。

### Affective AI Safety 的分类法（可直接建 crisis taxonomy）

1. 情感自我疏离（affective self-alienation）
2. 公平与偏见伤害
3. 关系性伤害

论点：现有安全框架对情感安全**要么只覆盖一角、要么完全没有**。

### RPLA 综述第 7 章的四条（T0，可正式引用）

- **7.1 毒性**：给 LLM 分配 persona 会**显著提高**毒性输出概率，比不分配时高。
  **人格越鲜明风险越高**——这是 `gates.py` 存在的学术理由。
- **7.3 角色幻觉**：她表现出超出角色范围的知识/能力。三条平行线的硬约束。
- **7.4 隐私**：存长期用户交互史的 RPLA 风险加剧。新加坡主体 → PDPA。
- **7.6 拟人化 → 社会隔离**：2024 年写得很浅，已被上面几篇远远超过。

---

## 九、用户侧实证 —— 做产品判断用

| 论文 | 层 | 备注 |
|---|---|---|
| [Longitudinal Evidence: 通用聊天机器人主动培育关系投入](https://arxiv.org/abs/2608.10672v1) ○ | T1 | **2026-08，最新** |
| [How AI Companionship Develops](https://arxiv.org/abs/2510.10079) ○ | T1 | 110 人纵向 |
| [心理社会效应的纵向 RCT](https://arxiv.org/html/2503.17473v1) ○ | **T0** | 引用较高的一篇 RCT |
| [User-In-Context Framework](https://arxiv.org/pdf/2607.04547) ○ | T1 | 解释用户反应差异 |
| [Large Language Lovers: 平台控制与用户能动性](https://arxiv.org/pdf/2601.13188) ○ | T1 | 质性研究 |
| [Illusions of Intimacy: 情绪动态如何塑造人机关系](https://arxiv.org/abs/2505.11649) ○ | T1 | 17,000+ Reddit 真实对话 |
| ["My Boyfriend is AI": Reddit 社区计算分析](https://arxiv.org/pdf/2509.11391) ○ | T1 | |
| [AI 伴侣社区中的拟人化：年龄、性别与情绪相关性](https://arxiv.org/html/2606.30942v1) ○ | T1 | |

### 两个必须记住的数

> **第 3 周**：110 人纵向研究中，用户对一个通用 chatbot 的认知就收敛到了对自己"伴侣"的认知。
> 配合 `market-2026.md` 里"平均建联 5~7 天"的行业数据一起看。

> 四周 RCT（72 人 + ChatGPT-4o）：模型即使没被要求也**自我披露量是用户的两倍**并主导话题走向，
> **但并没有加深用户的felt closeness。**
> —— 直接警告：**agent 主动说得多 ≠ 关系变深。** 打我们做 S0 主动性时最容易犯的错。

---

## 十、优先阅读顺序

只读三篇的话：

1. **[Emotional Manipulation by AI Companions](https://arxiv.org/pdf/2508.19258)**——我们的留存手段里有红线
2. **[ANCHOR](https://arxiv.org/abs/2607.28818)**——我们的评测体系不够
3. **[CPM-MultiAgent](https://arxiv.org/pdf/2607.07824)**——我们的情绪模块该长什么样

打底：RPLA 综述第 5、6、7 章。

---

## 十一、已识别的空白区

以下方向**目前没找到对口论文**，是我们可能真正做出东西的位置：

1. **Character × Individualized 的对立**（见 §0）。综述只在结语含糊提了一句"难以追踪 persona 随时间演化"。
2. **主动性 × 陪伴场景**。主动性论文全在任务型 agent 上（ProActor 做任务调度、
   CPE 做信息差）。陪伴场景下"什么时候该发消息"没人系统做过。
3. **撤回 / 手滑作为可控变量**。Beyond Words 只做了错字与自我编辑对感知的影响，
   没有把它作为角色状态的表达通道。
4. **中文 / 繁体语境**。所有活人感研究都在英文语料上，简繁不只是字符转换（见 memory）。
5. **合成审计 → 真实用户的外部效度**。ANCHOR 是合成的，HBS 是真实的但只审计单轮告别。
   长程 × 真实用户的交叉没人做。

---

## 维护约定

- 新论文追加到对应章节，**先标 ○，核过原文再改 ✓**
- T1 的结论进代码/文档前必须标注"未评审"
- 三个月复查一次 T1 是否已被正式接收（升 T0）或被证伪（降 T2）
