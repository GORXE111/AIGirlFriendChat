from __future__ import annotations


class LLMError(Exception):
    """LLM 调用层的基类。"""


class LLMConfigError(LLMError):
    """配置问题（缺 key、未知模型、路由缺失）。不可重试。"""


class LLMRequestError(LLMError):
    """请求被拒（4xx，非限流）。不可重试 —— 重试只会再错一次。"""

    def __init__(self, message: str, status_code: int, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LLMRateLimitError(LLMError):
    """429。可重试。"""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMServerError(LLMError):
    """5xx。可重试。"""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """超时或连接失败。可重试。"""


class LLMResponseError(LLMError):
    """响应结构不符合预期（缺字段、JSON 解析失败）。可重试一次。"""
