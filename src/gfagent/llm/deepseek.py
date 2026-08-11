"""DeepSeek provider。

用 httpx 直连而不是 openai SDK：`thinking` 这类非标参数直接控制更干净，也少一个依赖。
DeepSeek 同时提供 OpenAI 与 Anthropic 兼容格式，这里走 OpenAI 格式
（POST {base_url}/chat/completions）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator

import httpx

from ..config import Settings, get_settings
from .base import StreamEvent
from .errors import (
    LLMConfigError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
)
from .pricing import compute_cost
from .router import ModelRouter, ModelSpec
from .types import Completion, LLMRequest, Thinking, Usage

log = logging.getLogger(__name__)

_RETRYABLE = (LLMRateLimitError, LLMServerError, LLMTimeoutError, LLMResponseError)


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        settings: Settings | None = None,
        router: ModelRouter | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._router = router or ModelRouter()

        if not self._settings.deepseek_api_key and client is None:
            raise LLMConfigError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置")

        self._client = client or httpx.AsyncClient(
            base_url=self._settings.deepseek_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                self._settings.llm_timeout_seconds,
                connect=self._settings.llm_connect_timeout_seconds,
            ),
        )
        self._sem = asyncio.Semaphore(self._settings.llm_max_concurrency)

    # ---------------- payload ----------------

    def _payload(self, req: LLMRequest, spec: ModelSpec, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": [m.as_dict() for m in req.messages],
            # 显式写死。DeepSeek 默认 enabled + effort=high，不显式关就会默默烧钱变慢。
            "thinking": {"type": spec.thinking.value},
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "stream": stream,
        }
        if spec.thinking is Thinking.ENABLED and spec.reasoning_effort is not None:
            payload["reasoning_effort"] = spec.reasoning_effort.value
        if spec.max_tokens is not None:
            payload["max_tokens"] = spec.max_tokens
        if req.stop:
            payload["stop"] = req.stop[:16]
        if req.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            # 不加这个，流式拿不到 usage，成本统计会整段丢失
            payload["stream_options"] = {"include_usage": True}
        return payload

    # ---------------- 非流式 ----------------

    async def complete(self, req: LLMRequest) -> Completion:
        spec = self._router.resolve(req)
        payload = self._payload(req, spec, stream=False)
        started = time.perf_counter()

        data = await self._post_with_retry("/chat/completions", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"响应结构异常：{data!r}") from exc

        usage = Usage.from_api(data.get("usage"))
        self._audit(spec, usage, req)

        return Completion(
            text=message.get("content") or "",
            reasoning_text=message.get("reasoning_content"),
            model=data.get("model", spec.model),
            usage=usage,
            cost=compute_cost(spec.model, usage),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            task=req.task,
            character_id=req.character_id,
            raw=data,
        )

    # ---------------- 流式 ----------------

    async def stream(self, req: LLMRequest) -> AsyncIterator[StreamEvent]:
        spec = self._router.resolve(req)
        payload = self._payload(req, spec, stream=True)

        async with self._sem:
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        self._raise_for_status(resp.status_code, body, resp.headers)

                    usage: Usage | None = None
                    finish_reason: str | None = None

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if not chunk:
                            continue
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            log.warning("跳过无法解析的 SSE 分片: %r", chunk[:200])
                            continue

                        if obj.get("usage"):
                            usage = Usage.from_api(obj["usage"])

                        for ch in obj.get("choices") or []:
                            delta = ch.get("delta") or {}
                            if ch.get("finish_reason"):
                                finish_reason = ch["finish_reason"]
                            if delta.get("content") or delta.get("reasoning_content"):
                                yield StreamEvent(
                                    delta=delta.get("content") or "",
                                    reasoning=delta.get("reasoning_content") or "",
                                )

                    if usage is not None:
                        self._audit(spec, usage, req)
                    yield StreamEvent(
                        usage=usage, finish_reason=finish_reason, done=True
                    )

            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"流式请求超时：{exc}") from exc
            except httpx.TransportError as exc:
                raise LLMTimeoutError(f"流式连接失败：{exc}") from exc

    # ---------------- 传输 ----------------

    async def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self._settings.llm_max_retries + 1
        last: Exception | None = None

        for attempt in range(attempts):
            try:
                async with self._sem:
                    resp = await self._client.post(path, json=payload)
                if resp.status_code >= 400:
                    self._raise_for_status(resp.status_code, resp.text, resp.headers)
                try:
                    return resp.json()
                except json.JSONDecodeError as exc:
                    raise LLMResponseError(f"响应非 JSON：{resp.text[:300]}") from exc

            except httpx.TimeoutException as exc:
                last = LLMTimeoutError(f"请求超时：{exc}")
            except httpx.TransportError as exc:
                last = LLMTimeoutError(f"连接失败：{exc}")
            except _RETRYABLE as exc:
                last = exc
            # LLMRequestError / LLMConfigError 不在 _RETRYABLE 里，会直接抛出

            if attempt < attempts - 1:
                delay = self._backoff(attempt, last)
                log.warning(
                    "DeepSeek 调用失败（第 %d/%d 次），%.2fs 后重试：%s",
                    attempt + 1, attempts, delay, last,
                )
                await asyncio.sleep(delay)

        assert last is not None
        raise last

    def _backoff(self, attempt: int, err: Exception | None) -> float:
        if isinstance(err, LLMRateLimitError) and err.retry_after:
            return min(err.retry_after, 30.0)
        # 指数退避 + 抖动，避免三个女主的请求同时重试造成二次冲击
        return min(0.5 * (2**attempt), 8.0) * (0.7 + 0.6 * random.random())

    @staticmethod
    def _raise_for_status(status: int, body: str, headers: httpx.Headers) -> None:
        snippet = body[:500]
        if status == 429:
            retry_after = headers.get("retry-after")
            raise LLMRateLimitError(
                f"限流：{snippet}",
                retry_after=float(retry_after) if retry_after else None,
            )
        if status in (401, 403):
            raise LLMConfigError(f"鉴权失败（{status}）：{snippet}")
        if 400 <= status < 500:
            raise LLMRequestError(f"请求被拒（{status}）：{snippet}", status, body)
        raise LLMServerError(f"服务端错误（{status}）：{snippet}", status)

    # ---------------- 观测 ----------------

    def _audit(self, spec: ModelSpec, usage: Usage, req: LLMRequest) -> None:
        """线上抓两类事故。"""
        # 1. 误开 thinking。reasoning token 按输出计费，输出又是我们的成本大头，
        #    而且思维链会让回复变"助理味"。快回路上出现就是 bug。
        if spec.thinking is Thinking.DISABLED and usage.reasoning_tokens > 0:
            log.error(
                "thinking 已关闭但返回了 %d reasoning tokens（task=%s model=%s）"
                "—— 检查 payload 是否被覆盖",
                usage.reasoning_tokens, req.task.value, spec.model,
            )

        # 2. 缓存命中率异常。稳定前缀被污染时这里会先叫。
        cacheable = usage.cache_hit_tokens + usage.cache_miss_tokens
        if cacheable >= 1000 and usage.cache_hit_rate < 0.5:
            log.warning(
                "缓存命中率仅 %.0f%%（task=%s character=%s prompt=%d tokens）"
                "—— 稳定前缀可能被易变内容污染",
                usage.cache_hit_rate * 100, req.task.value,
                req.character_id, usage.prompt_tokens,
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> DeepSeekProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
