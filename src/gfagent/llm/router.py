"""模型路由：(task, character_id) → 模型 + 采样参数。

两件事：

1. **按任务分层**。快回路要便宜快，慢回路可以贵可以慢，审核要最小最便宜。
2. **按女主覆盖**。盲测很可能发现某个模型更适合某种性格（活泼 vs 高冷），
   允许三个女主跑不同模型是完全合理的做法，也是灰度和 A/B 的抓手。

thinking 默认全关。DeepSeek 的默认是 enabled + reasoning_effort=high，对本项目是
最坏组合：慢、贵（reasoning token 按输出计费，而输出是我们的主导成本项）、且思维链
会让输出更规整更"助理化"。只有确实需要多步推理的离线任务才开。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import LLMConfigError
from .types import LLMRequest, ReasoningEffort, Task, Thinking


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model: str
    thinking: Thinking = Thinking.DISABLED
    reasoning_effort: ReasoningEffort | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None


PRIMARY_MODEL = "deepseek-v4-flash"
"""项目选定的主力模型（2026-08-04 决策）。

单模型而非混搭，理由不只是便宜：

  - 慢回路（REFLECT/PLAN）的缓存命中率天然低 —— 每天的对话内容都是新的，
    缓存不上 —— 且输出长。放 Pro 上反而会成为总成本的大头（实测口径见 README）。
  - 单模型意味着只需校准一套 prompt 和一套语感。三个女主的语言指纹本来就要靠
    prompt 拉开差异，再叠加模型差异只会让归因变难。

改这一个常量就能整体换模型；`DEFAULT_ROUTES` 全部引用它。
需要单点例外时用 `ModelRouter.override()`，不要在这里散落硬编码。
"""


DEFAULT_ROUTES: dict[Task, ModelSpec] = {
    # 在线对话。短输出上限是硬约束 —— 她说的是短句，不是小作文。
    # 温度偏高换语感的活性；真正的风格控制交给语言指纹和后处理层。
    Task.CHAT: ModelSpec(
        model=PRIMARY_MODEL,
        thinking=Thinking.DISABLED,
        temperature=1.1,
        top_p=0.95,
        max_tokens=300,
    ),
    # 记忆归档：要准，不要发挥。低温。
    #
    # ⚠️ 这里的错误会永久沉淀：抽错一条事实，她就会一直"记错"，
    # 而这正是最伤活人感的破绽之一，且不会自己暴露。
    # 全 Flash 方案下这是最该盯的一环 —— 靠结构化输出 + 校验 + 重试兜底，
    # 若线上发现抽取质量不够，这是第一个该单独 override 到 Pro 的任务。
    Task.REFLECT: ModelSpec(
        model=PRIMARY_MODEL,
        thinking=Thinking.DISABLED,
        temperature=0.3,
        max_tokens=1500,
    ),
    # 主动话题规划：要把剧情余韵、下章伏笔、她的日程揉在一起选题，是真正的多步推理，
    # 也是唯一值得开 thinking 的任务。离线跑、频次低（每人每天个位数），
    # reasoning token 的开销在总盘子里可以忽略。
    Task.PLAN: ModelSpec(
        model=PRIMARY_MODEL,
        thinking=Thinking.ENABLED,
        reasoning_effort=ReasoningEffort.LOW,
        temperature=0.8,
        max_tokens=2000,
    ),
    # 审核：极高频、极短输出。
    Task.MODERATE: ModelSpec(
        model=PRIMARY_MODEL,
        thinking=Thinking.DISABLED,
        temperature=0.0,
        max_tokens=200,
    ),
    # 离线预生成。
    Task.AUTHOR: ModelSpec(
        model=PRIMARY_MODEL,
        thinking=Thinking.DISABLED,
        temperature=1.0,
        max_tokens=2000,
    ),
}


class ModelRouter:
    def __init__(
        self,
        routes: dict[Task, ModelSpec] | None = None,
        character_overrides: dict[tuple[str, Task], ModelSpec] | None = None,
    ) -> None:
        self._routes = dict(routes or DEFAULT_ROUTES)
        self._overrides = dict(character_overrides or {})

    def override(self, character_id: str, task: Task, spec: ModelSpec) -> None:
        self._overrides[(character_id, task)] = spec

    def resolve(self, req: LLMRequest) -> ModelSpec:
        spec = None
        if req.character_id is not None:
            spec = self._overrides.get((req.character_id, req.task))
        if spec is None:
            spec = self._routes.get(req.task)
        if spec is None:
            raise LLMConfigError(f"任务 {req.task} 没有配置路由")

        # 请求级参数覆盖路由默认值
        patch: dict[str, object] = {}
        if req.model is not None:
            patch["model"] = req.model
        if req.thinking is not None:
            patch["thinking"] = req.thinking
        if req.reasoning_effort is not None:
            patch["reasoning_effort"] = req.reasoning_effort
        if req.temperature is not None:
            patch["temperature"] = req.temperature
        if req.top_p is not None:
            patch["top_p"] = req.top_p
        if req.max_tokens is not None:
            patch["max_tokens"] = req.max_tokens

        return replace(spec, **patch) if patch else spec
