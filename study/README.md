# 同类项目研究

浅克隆的第三方仓库放在本目录（已 gitignore，只有 md 入库）。

```bash
cd study
git clone --depth 1 https://github.com/SillyTavern/SillyTavern.git
git clone --depth 1 https://github.com/letta-ai/letta.git
git clone --depth 1 https://github.com/mem0ai/mem0.git
```

| 项目 | 看什么 |
|---|---|
| **SillyTavern** | 角色卡 V2 规格、World Info 关键词触发、prompt 分层注入 |
| **letta**（原 MemGPT） | 记忆块与容量上限、自编辑记忆 |
| **mem0** | 记忆的增删改，而不是只追加 |

其余同类（Live2D／语音向，跟我们的问题关系不大）：
Open-LLM-VTuber、Amica、Soul of Waifu。

分析见 `findings.md`。

---

其余研究材料：

| 文件 | 内容 |
|---|---|
| `findings.md` | 同类项目的分析结论 |
| `market-2026.md` | 国内 AI 情感陪伴品类的市场调研 |
| `papers/README.md` | **论文库**——AI 陪伴 / 角色扮演 agent，长期索引 |
