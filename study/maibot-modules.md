# 麦麦的模块划分与 agent 工程定位

`src/` 约 21.6 万行。按**运行时职责**分，是十二层。
下面每层写三件事：**有哪些文件 / 干什么 / 在 agent 工程里是什么位置**。

---

## 总图

```
                    ┌─────────────────────────────────────┐
   QQ/CLI/…  ──①──> │  platform_io   传输适配              │
                    └──────────────┬──────────────────────┘
                                   ↓
                    ┌─────────────────────────────────────┐
                    │ ② chat/message_receive  会话装配      │
                    └──────────────┬──────────────────────┘
                                   ↓
    ╔══════════════════════════════════════════════════════╗
    ║ ③ 唤醒层  focus / turn_gates / reply_necessity        ║  ← 0 token
    ║          idle_backoff / turn_scheduler                ║     麦麦独有
    ╚══════════════════════════╤═══════════════════════════╝
                               ↓ （过闸了才花钱）
    ┌──────────────────────────────────────────────────────┐
    │ ④ reasoning_engine   Planner：ReAct 多轮 + 工具        │
    │      ├─ query_memory ──────────→ ⑦ A_memorix         │
    │      ├─ wait / send_emoji / fetch_history            │
    │      └─ reply ─────────┐                             │
    └────────────────────────┼─────────────────────────────┘
                             ↓
    ┌──────────────────────────────────────────────────────┐
    │ ⑤ chat/replyer   表达层：先检索表达库，再生成台词        │
    └────────────────────────┬─────────────────────────────┘
                             ↓
    ┌──────────────────────────────────────────────────────┐
    │ ⑥ send_service   拟真输出：拆句 / 打字延迟 / 错别字      │
    └────────────────────────┬─────────────────────────────┘
                             ↓ 发出去
                    ┌────────┴────────┐
                    ↓                 ↓
         ⑨ reply_effect 观察      ⑩ learners 慢回路学习
            （打分 ASI）      ←──    （行为 / 表达 / 黑话）

    旁挂：⑦ A_memorix（独立进程）  ⑧ 中期记忆与印象
          ⑪ plugin_runtime / mcp   ⑫ webui / 统计 / 配置
```

---

## ① platform_io —— 传输适配

`manager.py 655` `adapter_policy.py 453` `outbound_tracker.py 286`
`routing.py` `dedupe.py` `route_key_factory.py` `registry.py`

**干什么**：把 QQ / CLI / 各种 adapter 抽象成统一的
`InboundMessageEnvelope` → `DeliveryReceipt`。带路由键、去重、投递追踪、
`DeliveryStatus` 状态机。

**agent 工程定位**：**Perception / Actuation I/O 边界**。
标准的 ports-and-adapters（六边形架构）。核心逻辑对「消息从哪来、发到哪去」完全无知。

> 这层跟我们「传输无关」的既定约束是同一个想法，
> 但他们把**投递回执和去重**也放进来了——这两件事我们还没有。
> 一旦上真实 IM，「发出去了吗」「重复发了吗」会立刻变成问题。

---

## ② chat/message_receive —— 会话装配

`bot.py 805` `message.py 544` `chat_manager.py 542` `uni_message_sender.py 372`

**干什么**：把裸消息装配成 `SessionMessage`（带 `is_at` / `is_mentioned` /
`processed_plain_text` / 用户身份），并按 `session_id` 分派到对应的运行时实例。
`chat_manager` 维护「一个会话 = 一个 agent 实例」的映射。

**agent 工程定位**：**Session / Episode 管理**。
每个会话是一个独立的、有生命周期的 agent 实例，不是无状态函数。

---

## ③ 唤醒层 —— 麦麦真正的特色

`focus/manager.py 442` `focus/runtime_mixin.py 741`
`turn_scheduler.py 132` `turn_gates.py 196` `reply_necessity.py 277`
`idle_backoff.py 95` `mode_policy.py 23`

**干什么**：决定这条消息**该不该唤醒思考**。四道闸：focus 槽位（=1）、
指数空闲退避、必要性打分（≥80）、以及频率阈值 / 空窗补偿。全部是纯规则，
**一个 token 都不花**。

**agent 工程定位**：标准 agent 架构里**没有这一层**。
标准 agent 是「被调用即执行」——你调 `agent.run()`，它就一定跑。
麦麦在 invoke 之前插了一个 **arousal / attention gate**。

用控制论的话说：这是把 agent 从
**reactive（有输入必有输出）** 改成了 **thresholded（输入要够强才有输出）**。

> **这是整个仓库对我们最有价值的架构启示。**
> 我们的 `agent/core.py` 是 reactive 的：`choose()` 进来，一定产出回复。
> 「她不回你」这个行为在我们的架构里**根本没有位置可放**——
> 不是没实现，是没有那个层。

---

## ④ reasoning_engine —— Planner

`reasoning_engine.py 2101` `runtime.py 2008` `chat_loop_service.py 1251`
`context/*.py`（history / messages / planner_messages / post_processor）
`builtin_tool/*.py` `core/tooling.py 419`

**干什么**：经典 ReAct 循环。多内部轮次，每轮 LLM 输出「分析 + 工具调用」，
执行工具、把结果塞回上下文、继续下一轮，直到不调工具或调了 `reply`。
可被新消息打断（`ReqAbortException`）后带新上下文重来。

工具集：`reply` `wait` `send_emoji` `send_image` `query_memory`
`query_person_profile` `fetch_history` `switch_chat` `view_forward_message` `tool_search`

`core/tooling.py` 是标准工具抽象：`ToolSpec` / `ToolInvocation` /
`ToolExecutionResult` / `ToolProvider`(Protocol) / `ToolRegistry`。

**agent 工程定位**：**Planner / Controller**，教科书式的 tool-calling agent loop。
唯一特别的是 prompt 里那句「你不是本人，不要替她发言」——
把 Planner 限定成**导演视角**，说话必须走 `reply` 工具。

> 注意 `tool_search`：工具不是全部常驻，而是 deferred，需要时先搜出来。
> 跟 World Info 的按需注入是同一个思路，只不过作用在工具上。

---

## ⑤ chat/replyer —— 表达层

`expression_vector_index.py 1683` `maisaka_generator_base.py 1280`
`maisaka_expression_selector.py 680` `replyer_manager.py`

**干什么**：`reply` 工具触发后，**先**从学来的表达库里按语境选 N 条
（`expression_select.prompt` 出 `{"selected_situations":[2,3,5]}`），
**再**用 `maisaka_replyer.prompt` 生成台词。表达库有向量索引。

**agent 工程定位**：**Decision / Execution 分离里的 Execution 半边**。
更准确说是**风格的 RAG**——检索增强的对象不是知识，是**说话方式**。

> 一般 RAG 检索事实，这里检索「在这种场合可以怎么说」。
> 我们的 147 条 voice-samples 是全量常驻，等价于「不检索，全塞」。

---

## ⑥ send_service —— 拟真输出

`services/send_service.py 1376` `chat/utils/typo_generator.py 477`
`chat/utils/utils.py:697 calculate_typing_time`

**干什么**：拆句、按字数算打字延迟（中文 0.3 s/字，单字 ×3，emoji 固定 1 s）、
**故意打错别字**。

错别字生成器是基于**拼音 + 字频**的：

```python
ChineseTypoGenerator(
    error_rate=0.3,        # 单字替换概率
    tone_error_rate=0.2,   # 声调错误概率
    word_replace_rate=0.3, # 整词替换概率
    max_freq_diff=200,     # 只替换成字频接近的字
)
```

用 `jieba` 分词 + `pypinyin` 取音，然后在**同音、且字频接近**的字里挑一个换掉。
`max_freq_diff` 那个参数是精髓：真人的错别字是输入法选词错误，
所以错成的字一定是**常见字**——不会把「在」打成某个生僻同音字。

**agent 工程定位**：**Actuation shaping**，纯拟真层，跟能力完全无关。
这层的存在本身说明一个判断：**活人感有相当一部分是在模型之外做出来的。**

---

## ⑦ A_memorix —— 记忆系统（独立进程）

**6.2 万行，占整个仓库 29%。** 通过 `a_memorix_host_service.invoke()` 调用，
是**独立启动的服务**，不是普通 import。

```
storage/     metadata_fact 1303     结构化事实账本 + 确定性状态机
             metadata_episode 1545  情节
             graph_store 1733       实体-关系图谱
             vector_store 2089      向量
             metadata_store 4497    元数据总store
retrieval/   dual_path 3013         双路检索：段落向量 ⊕ 关系图谱，加权融合
strategies/  embedding/             切分与嵌入
runtime/     sdk_memory_kernel 1755
             services/feedback_correction 1523   记忆纠错
             services/delete_admin 1888          删除治理
             graph_admin / vector_runtime / correction_admin
utils/       person_profile_service 1242   人物画像
             retrieval_tuning_manager 2369  检索调参
             web_import_manager 4345
```

**双路检索**（`RetrievalStrategy`）：

```
PARA_ONLY   仅段落（向量）
REL_ONLY    仅关系（图谱）
DUAL_PATH   双路融合  ← 推荐
```

图谱路带**可靠性估计**：一条关系的两个实体都在查询里锚定 → ×0.78，
只锚定一个 → ×0.55，只有谓词 → ×0.4，完全没锚定 → ×0.3。
再叠加支持度（几条来源佐证）和一致性权重。

`metadata_fact` 的注释写明它是「结构化事实账本 + **确定性状态机**」，
有 `supersedes`（新事实推翻旧事实）——就是 mem0 那套 ADD/UPDATE/DELETE。

**agent 工程定位**：**Long-term memory as a service**。
关键架构选择是**进程隔离**：记忆有自己的生命周期、自己的 admin 接口、
自己的调参器，主 agent 只通过 `invoke(component, args)` 说话。

> 这个规模对我们**严重过量**。他们要处理的是「一个 bot 混几十个群、
> 记几百号人」；我们是**一个玩家、一条关系线、一个存档**。
> 值得学的是**分层**（fact 账本 / episode / graph / vector 各司其职）
> 和 **fact 带状态机**，不是这个体量。

---

## ⑧ 中期记忆与印象

`maisaka/memory/mid_term.py 986` `heuristic_injector.py 406` `person_profile.py 271`

**干什么**：
- `mid_term` —— 短期上下文被裁掉时压成「聊天回想」，
  同时生成 `recall_cues`（写入时就想好「什么情景会用到」）
- `heuristic_injector` —— 自动生成「当前聊天印象」，用它去语义召回长期记忆，
  **不用 Planner 显式调 `query_memory`**
- `person_profile` —— 把对话里出现的人的画像自动注入 Planner 上下文

**agent 工程定位**：**Memory 的自动召回通道**，与 ④ 的 `query_memory`（显式检索）
并行。一条是 agent 主动查，一条是系统按语境推。

> 双通道这个设计值得记：显式检索精准但要花一轮 tool call，
> 自动注入便宜但可能不相关。两条都要。

---

## ⑨ reply_effect —— 在线评估

`tracker.py 271` `scoring.py 262` `models.py 166` `judge.py 116` `storage.py`

**干什么**：每条 reply 登记为 pending，观察后续用户消息，然后结算 ASI：

```
ASI = 0.45×行为满意度 + 0.35×感知质量 + 0.20×(1−摩擦)
```

行为满意度是**纯规则**（继续 2 轮 / 情绪 / 展开程度 / 无纠正 / 无中止），
感知质量是 **LLM rubric**（social_presence / warmth / competence /
appropriateness / **uncanny_risk**）。

**agent 工程定位**：**Reward model + online eval**。
这是 ⑩ 学习层的信号源。

> 拆成「规则信号 + LLM 评分」两半很聪明：
> 规则那半不花钱且客观，LLM 那半评规则测不出来的东西。

---

## ⑩ learners —— 慢回路学习

`behavior_learner.py 1525` `behavior_scene_cluster_store.py 1217`
`expression_learner.py 956` `jargon_learner.py 866` `jargon_miner.py 895`
`behavior_pattern_store.py 692` `behavior_generic_tags.py 532`
`behavior_scenario.py 411` `behavior_selector.py 358`
`behavior_pattern_maintenance.py 326` `expression_review_store.py 216`

**干什么**：三条学习线，都是异步慢回路。

| 线 | 学什么 | 闭环 |
|---|---|---|
| **表达** | 「当 A 时可以 B」的说话方式 | learn_style → expression_evaluation → 库 |
| **行为** | 「场景-行为-结果」策略 | learn_behavior → evaluate_feedback → consolidate |
| **黑话** | 群内梗和缩写 | learn_jargon → 上下文推断 / 纯内容推断 |

行为线用了 RL 的语言：`score_delta` 明写「类似**奖励预测误差**」，
success +0.5~1.0 / partial +0.1~0.35 / failed −0.4~−1.0。

**agent 工程定位**：**Offline policy learning**，写回 ⑤ 的表达库和
`behavior_pattern_store`，下次由 `behavior_selector` 检索出来喂给 Planner。

> 完整的 **感知 → 决策 → 执行 → 评估 → 学习 → 回写决策依据** 闭环。
> 这是麦麦架构上最完整的地方，也是绝大多数同类项目缺的一环。

---

## ⑪ 扩展层

`plugin_runtime/` （integration 1873 / runner_main 2526 / supervisor 2054 /
component_query 1100 / manifest_validator 1505 / transport / protocol）
`mcp_module/` （manager 660 / connection 688 / host_llm_bridge 590）
`core/event_bus.py 225`

**干什么**：插件跑在**独立子进程**里（有 supervisor、transport、manifest 校验），
不是同进程 import。MCP 接外部工具。event_bus 做进程内解耦，
支持 `intercept`（拦截式 handler，可改消息）。

**agent 工程定位**：**Capability extension + 故障隔离**。
第三方插件崩了不能带崩 agent——所以要子进程 + supervisor。

> 我们不需要插件生态。但 `event_bus` 的 **intercept 型 handler**
> 这个模式值得记：一条消息在到达 agent 前可以被链式改写。
> 我们的 postprocess 现在是硬编码的一串函数，改成可插拔的拦截链会更干净。

---

## ⑫ 观测与配置

`webui/`（routers: memory 3533 / config 2537 / expression 2284 /
reasoning_process 1949 / chat 1479+1474 / jargon 1319 / system 1784）
`services/llm_cache_stats.py 1521` `chat/utils/statistic.py 3192`
`llm_models/request_snapshot.py 920` `common/logger.py 999`
`config/official_configs.py 6118`

**干什么**：
- **webui** —— 记忆管理、表达库编辑、**推理过程回放**、配置界面。
  注意 `chat` 路由是**只读历史浏览**，不能对话（所以我才要写那个测试壳）
- **llm_cache_stats 1521 行** —— 他们非常在意 prefix cache 命中
- **request_snapshot** —— 每次 LLM 调用落盘，报错时给出
  `replay_llm_request.py <file>` 让你原样重放（我们排 401 那次就是靠这个）
- **official_configs 6118 行** —— 配置项自带 UI 元数据：
  `x-widget: slider`、三语 label、`advanced: True`

**agent 工程定位**：**Observability + 配置即产品界面**。

> 6118 行配置里塞 UI 元数据这件事，说明他们把**配置面板当成产品的一部分**，
> 而不是开发者的调试入口。这是 to-C 的做法。
> 我们如果要给策划改人设参数，迟早也要走这条路。

---

# 对照我们

| # | 层 | 麦麦 | 我们 (`src/gfagent/`) |
|---|---|---|---|
| ① | 传输适配 | platform_io | 无（有意为之，但缺**投递回执/去重**） |
| ② | 会话装配 | chat/message_receive | `agent/core.py` + 存档 |
| ③ | **唤醒闸** | **四道，0 token** | **整层缺失** ← 最大结构差距 |
| ④ | Planner | ReAct 多轮 + 10 工具 | 一次调用直出，无工具 |
| ⑤ | 表达层 | 检索式 few-shot + 向量索引 | 147 条样本全量常驻 |
| ⑥ | 拟真输出 | 打字延迟 + 错别字 | `output/postprocess.py`，延迟与长度无关 |
| ⑦ | 长期记忆 | 6.2 万行独立服务 | `memory/` 三因子检索（对我们够用） |
| ⑧ | 中期记忆 | mid_term + 印象自动召回 | threads 覆盖一部分，无 recall_cues |
| ⑨ | 在线评估 | ASI + uncanny_risk | `evals/critic.py`，全正向维度 |
| ⑩ | 学习回写 | 三条线 + RPE | **无**（样本库是死的） |
| ⑪ | 扩展 | 子进程插件 + MCP | 不需要 |
| ⑫ | 观测 | webui + 快照重放 | audit + 自动对局 |

**我们有而他们没有的**：关系阶段（S0–S3）、好感度、剧情主线、
**选项制**。最后一个是我们的结构性优势——玩家选了哪个选项，
是比群聊「有没有人接话」干净得多的学习信号，⑩ 那一层我们做起来比他们容易。

**结构性差距只有一处**：③。
④⑤⑥⑧⑨⑩ 都是「做了但不如他们细」，可以增量补。
③ 是**架构里没有这个位置**——我们的 `choose()` 一进来就必然产出回复，
「她此刻不想理你」这件事没地方表达。

补 ③ 在选项制下不能照搬（见 `maibot-internals.md` 末尾），
应该做成**回复体量与时机的调节器**，而不是「回/不回」的开关。
