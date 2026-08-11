"""GirlFriendAgent —— AI 女友陪伴 agent 的 LLM 接入基建层。

当前范围仅到 LLM 调用基建：provider 抽象、模型路由、缓存友好的 prompt 编排、
重试、用量与成本记账、峰谷时段感知。

记忆层、情绪状态机、生活模拟、调度仲裁器尚未实现。
"""

__version__ = "0.1.0"
