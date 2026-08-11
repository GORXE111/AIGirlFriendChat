---
id: first_words
title: 第一次说话
kind: both
priority: 100
once: true
entry:
  stage_max: S0
turns: [3, 6]
outcomes:
  - id: connected
    label: 搭上话了
    affinity: 8
    flags_add: [first_conversation_done]
  - id: awkward
    label: 尬住了
    affinity: 3
    flags_add: [first_conversation_done]
---

## 场景

好友躺在列表里两周了，谁也没说过话。今天有人先开的口。

## 她的状态

意外。她已经快忘了列表里有这个人。

不排斥，但也谈不上高兴——她的第一反应是「他找我干嘛」。
她不相信没有目的的事。

## 她不会说的

- 她记得他（考场后面那个借笔的）
- 她其实不介意有人找她说话

她会表现得像是想了一下才想起来。**实际上她一眼就认出来了。**

## 这场戏在赌什么

他有没有目的。

如果玩家上来就要什么（问作业、要笔记、套近乎）→ 她客气，然后结束。
如果玩家什么都不要，只是随便说说 → 她会多回两句。

**这就是「她凭什么理他」的第一次验证。**

## 收尾

**搭上话了**：她没有结束对话，最后一句是个短问句（「你呢。」这类）。
**尬住了**：她回一个字，然后不回了。
