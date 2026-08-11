# 参考语料

**本目录不进版本库**（见 .gitignore）。

## 用途

用 scripts/analyze_corpus.py 量统计特征，校准 critic.py 的阈值和
STAGE_BEHAVIOR 的 max_chars。

    python scripts/analyze_corpus.py reference/xxx.txt --speaker 角色名
    python scripts/analyze_corpus.py reference/*.txt --compare s3

## 使用边界

公开的 galgame 语料（HuggingFace 上的 alpindale/visual-novels、
joujiboi/Galgame-VisualNovel-Reupload 等）**都是从商业游戏提取的**。

| 用途 | |
|---|---|
| 人看，学手艺，写自己的台词 | 正常创作实践 |
| 跑统计，得出长度／标点／话题分布 | 产出的是数字，不是受保护的表达 |
| **直接做 few-shot 或微调语料进产品** | **不要**。来源不清是实打实的合规问题，尤其要过备案 |

而且实际上抄别的角色的台词对我们也没用 —— 那些是给那些角色写的，
套到林静姝身上一句都不合适。**我们要的是分布，不是句子。**

## 格式

一行一句。可带说话人前缀，会自动剥掉：

    角色名：台词内容
    【角色名】台词内容
    台词内容
