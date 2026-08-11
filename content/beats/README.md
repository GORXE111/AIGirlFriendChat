# 桥段（Beat）写作规范

一个桥段 = 一场小戏。编剧写骨架，AI 在骨架内生成台词和选项。

**你写的是这场戏要干什么，不是每句话说什么。**

---

## 文件格式

`content/beats/h01/<id>.md`，YAML 头 + Markdown 正文。

```markdown
---
id: post_exam_night
title: 月考后的晚上
kind: her            # her（她发起）| player（玩家发起）| both
priority: 60         # 0-100，同时满足条件时高的先演
once: true           # 只演一次
entry:
  stage_min: S1
  affinity_min: 20
  time_of_day: [evening, night]
  flags_all: [exam_just_ended]
  flags_none: [knows_about_ranking]
  cooldown_days: 0
turns: [3, 6]        # 最少 3 轮，最多 6 轮
outcomes:
  - id: opened
    label: 她松口了
    affinity: 6
    flags_add: [she_opened_up]
    emotion_soothe: 委屈
  - id: closed
    label: 她关上了
    affinity: 1
    emotion_bump: {委屈: 0.4}
---

## 场景

月考成绩出来了。她名次掉了三名，晚上被她妈叫去谈话，谈了四十分钟。

## 她的状态

累。还有一种说不上来的不对劲——不是难过，是被说服之后的空。

## 她不会说的

- 名次掉了
- 被谈话了
- 谈话的内容是"你自己想清楚"

她不会主动提这些。**除非玩家察觉到不对劲并且不追问**，她才可能松口一点。

## 这场戏在赌什么

玩家会不会发现她不对劲。

追问 → 她关上。
若无其事地陪着 → 她松口。

## 收尾

**她松口**：她说一句「今天有点烦」，然后转开话题。不会说细节。
**她关上**：她说「没事。睡了。」
</markdown>
```

---

## 写作要点

### 「她不会说的」是最重要的一栏

比「她会说什么」重要得多。这一栏定义了**戏的张力**——玩家在猜，她在藏。

没有这一栏，AI 会把所有事直白说出来，戏就没了。

### 不要写具体台词

写「她转开话题」，不要写「她说：不聊这个了」。
具体台词由 AI 根据当时的情绪、好感、记忆生成，每次都不一样。

**例外**：如果某句话是这场戏的题眼，必须一字不差，写在「收尾」里并注明「原句」。

### 结局只写两三个，**第一个是默认结局**

不要写六个分支。玩家感知不到细微差别，而且成本翻倍。
一般就是「她打开了一点」和「她关上了」。

⚠️ **顺序有意义**：戏演到轮数上限而 AI 还没给结局时，系统取**第一个**。
所以第一个应该是「正常演完」的那个，负面结局往后放。
把「她关上了」写在第一位，会让好好聊完的玩家莫名其妙吃个坏结局。

### 选项由 AI 生成，但你可以约束

在「这场戏在赌什么」里写清楚玩家的选择空间，AI 会照着生成。
不用列具体选项。

---

## 触发条件

| 字段 | 说明 |
|---|---|
| `stage_min` / `stage_max` | 关系阶段 S0–S3 |
| `affinity_min` / `affinity_max` | 好感 0–100 |
| `time_of_day` | `morning` `noon` `afternoon` `evening` `night` `late` |
| `weekday` | `[0-6]`，0 是周一 |
| `flags_all` | 必须全部存在的 flag |
| `flags_any` | 至少存在一个 |
| `flags_none` | 必须都不存在 |
| `cooldown_days` | 演过之后多少天内不再演 |
| `mother_night_shift` | `true` 时只在她妈值夜班的晚上触发 |

`kind: her` 的桥段由她主动发起；`kind: player` 出现在玩家的开场选项里；
`both` 两边都行。

---

## 目录

按解锁顺序放，文件名用 `NN_id.md` 便于排序。
