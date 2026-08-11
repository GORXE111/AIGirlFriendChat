# AIGirlFriendChat

一个**选项制**的 AI 女友对话 agent。玩家不打字，从三个选项里选 ——
她的台词和玩家的选项由同一次调用一起生成。

不是聊天机器人套壳。这里的赌注是**活人感**：她有自己的今天、会记得你说过的话、
会闹小情绪、会因为在上课而回得慢。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
copy .env.example .env      # 填入 DEEPSEEK_API_KEY

.\.venv\Scripts\python.exe -m pytest                 # 235 passed
.\.venv\Scripts\python.exe -m uvicorn gfagent.service:app
```

打开 `http://127.0.0.1:8000`。左栏存档（含 S0–S3 一键预设），中间聊天，
右栏实时状态与生成诊断。

---

## 它是怎么运作的

```
玩家打开聊天
  ↓
导演挑一场戏（桥段）— 或者给「今天聊什么」的话题
  ↓
装配 prompt：人设卡 + 记忆 + 情绪 + 她的今天 + 桥段骨架
  ↓
一次调用生成：她的消息 ＋ 玩家的三个选项 ＋ 结局判定
  ↓
后处理：格式强制、人设违规拦截、拆条、兜底
  ↓
按日程排延迟落库 —— 她说完之前不给选项
```

## 分层

| 层 | 位置 | 职责 |
|---|---|---|
| 内容 | `content/characters/h01/` | 设定文档，人写人读 |
| 桥段 | `content/beats/` | 编剧写骨架，AI 填肉 |
| 装配 | `persona/` | 按 manifest 抽取小节 → 稳定前缀 |
| 状态 | `state/` | 情绪（持久＋衰减）、关系阶段、好感、**小情绪行为** |
| 记忆 | `storage/` `memory/` | 事实、情节、洞察、三因子检索 |
| 生活 | `life/` | 她的今天 —— 具体性的来源 |
| 情境 | `schedule/` | 多快回、能不能主动 |
| 生成 | `llm/` `prompt/` | provider、路由、缓存友好编排 |
| 约束 | `output/` | 格式强制、人设违规拦截、兜底 |
| 编排 | `agent/` | 全链路 |
| 自评 | `evals/` | 自动对局 ＋ 复盘 |
| 接口 | `service/` | HTTP ＋ 聊天窗口 |

**`content/` 是单一事实源。** `persona/manifest.py` 声明哪些文件的哪些小节进 prompt，
改设定只改 md。`design-notes.md` 永不进卡 ——
设计理由会让模型**解释角色**而不是**扮演角色**。

---

## 几条踩出来的设计约束

### `thinking` 必须默认关

DeepSeek 默认 `thinking=enabled` + `reasoning_effort=high`，对本项目是最坏组合：
慢、贵（reasoning token 按输出计费，而输出是成本主导项）、
且思维链会让输出更规整更「助理味」。

只有 `Task.PLAN`（离线选题）开着。客户端会审计：thinking 关闭却出现
reasoning_tokens 直接打 ERROR。

### 稳定前缀不能被污染

DeepSeek 的上下文缓存是**自动前缀匹配，要求完全前缀命中**。实测缓存粒度
**64 token**，命中率稳定在 88–99%。

最常见的破坏方式：往 system prompt 开头塞时间戳。**功能上毫无异常，
没人会发现**，但输入成本涨 50–120 倍。`prompt/layers.py` 强制分层并检测污染。

### JSON 模式下，历史不能放进 assistant 轮

把她之前的回复放进 `assistant` 轮，模型看到「前面几轮都在说大白话」
却被 `response_format` 要求输出 JSON，两个信号打架，**直接返回空白**。
对话记录要作为**文本**放进最后一条 user 消息。

### 玩家还没读到 ≠ 她没说过

她的回复带延迟，落库时 `delivered=0`。如果历史按已送达过滤，
玩家连发几条时模型眼里是「对方说了四句，我一句没回」，于是反复回答旧问题。

### 后处理不能损坏词

软化尾词清洗吃掉词素 ——「你在干**嘛**。」被吃成「你在干。」。
清洗器需要负向环视，且**绝不能丢内容**（超条数往最后一条合并，不砍尾巴）。

---

## 自评闭环

```bash
python scripts/selfplay.py --preset s3            # 一局 + 复盘
python scripts/selfplay.py --personas             # 六种玩家画像各一局
python scripts/analyze_corpus.py reference/x.txt  # 量参考语料的统计特征
```

**玩家画像写的是「他是谁」，不是「他怎么选」** ——
galgamer 会算好感度，otaku 容易用力过猛，experienced 对不自然最敏感。
行为指令模拟不出玩家，那是在演标签。

复盘分两层：**机械检查**（重复、具体性、关系话比例、选项语气多样度，零成本无争议）
＋ **评审 agent**（活人感／阶段感／内梗／出戏，必须引用原文）。

这套 harness 抓到过我自己写的正则 bug —— 它读的是玩家看到的东西。

---

## 中文角色声音

`content/craft/chinese-character-voice.md`

日语靠一人称／語尾／敬语区分角色（**役割語**），中文没有这套工具。
照搬日语方法是中文角色声音同质化的机制性原因。

中文实际能用的六维：**句长节奏 / 标点密度 / 信息组织 / 省略程度 / 词汇层级 / 回应策略**。

`analyze_corpus.py` 能把其中几维量出来。当前角色实测：
句号逗号比 **11:1**，**七成句子省主语**，给理由只占 **7.5%**。

---

## 状态

已完成：人设卡装配、持久化、记忆（事实／情节／洞察／三因子检索）、
情绪状态机与小情绪行为、日程、桥段引擎、选项制回合、后处理、
存档预设、自动对局与复盘、聊天前端。

未做：主动性引擎（她不会自己找你）、撤回与未送达、
中期记忆、跨角色调度、H02／H03。

> ⚠️ `config.py` 的 `max_delay_seconds` 当前是 **5 秒的测试值** ——
> 等于关掉了延迟机制。正式跑改回 `600`。
