from __future__ import annotations

import httpx
import pytest
import respx

from gfagent.config import Settings
from gfagent.llm import (
    DeepSeekProvider,
    LLMConfigError,
    LLMRequest,
    LLMRequestError,
    LLMServerError,
    Message,
    ModelRouter,
    ModelSpec,
    Task,
    Thinking,
)

BASE = "https://api.deepseek.com"
URL = f"{BASE}/chat/completions"


def settings(**kw) -> Settings:
    return Settings(
        deepseek_api_key="sk-test",
        deepseek_base_url=BASE,
        llm_max_retries=kw.pop("retries", 2),
        usage_log_path=None,
        **kw,
    )


def reply(text: str = "嗯，刚下班", **usage) -> dict:
    u = {
        "prompt_tokens": 4000,
        "completion_tokens": 12,
        "prompt_cache_hit_tokens": 3968,
        "prompt_cache_miss_tokens": 32,
        "total_tokens": 4012,
    }
    u.update(usage)
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": u,
    }


def req(**kw) -> LLMRequest:
    kw.setdefault("messages", [Message("user", "在干嘛")])
    return LLMRequest(**kw)


# ---------------- payload ----------------


@respx.mock
async def test_thinking_disabled_by_default_on_chat():
    """DeepSeek 默认 thinking=enabled。快回路必须显式关掉，否则慢、贵、且更助理味。"""
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=reply()))
    async with DeepSeekProvider(settings()) as p:
        await p.complete(req(task=Task.CHAT))

    import json as _json

    body = _json.loads(route.calls[0].request.content)
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


@respx.mock
async def test_reasoning_effort_only_sent_when_thinking_enabled():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=reply()))
    async with DeepSeekProvider(settings()) as p:
        await p.complete(req(task=Task.PLAN))

    import json as _json

    body = _json.loads(route.calls[0].request.content)
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "low"


@respx.mock
async def test_chat_caps_output_length():
    """她说的是短句不是小作文 —— max_tokens 是硬约束。"""
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=reply()))
    async with DeepSeekProvider(settings()) as p:
        await p.complete(req(task=Task.CHAT))

    import json as _json

    assert _json.loads(route.calls[0].request.content)["max_tokens"] == 300


@respx.mock
async def test_json_mode():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=reply("{}")))
    async with DeepSeekProvider(settings()) as p:
        await p.complete(req(task=Task.REFLECT, json_mode=True))

    import json as _json

    body = _json.loads(route.calls[0].request.content)
    assert body["response_format"] == {"type": "json_object"}


# ---------------- 路由 ----------------


@respx.mock
async def test_per_character_model_override():
    """盲测若发现某个模型更适合某种性格，按女主换模型必须可行。"""
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=reply()))
    router = ModelRouter()
    router.override("gao_leng", Task.CHAT, ModelSpec(model="deepseek-v4-pro"))

    async with DeepSeekProvider(settings(), router=router) as p:
        await p.complete(req(task=Task.CHAT, character_id="gao_leng"))
        await p.complete(req(task=Task.CHAT, character_id="huo_po"))

    import json as _json

    assert _json.loads(route.calls[0].request.content)["model"] == "deepseek-v4-pro"
    assert _json.loads(route.calls[1].request.content)["model"] == "deepseek-v4-flash"


@respx.mock
async def test_request_level_model_override_wins():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=reply()))
    async with DeepSeekProvider(settings()) as p:
        await p.complete(req(task=Task.CHAT, model="kimi-k2.5"))

    import json as _json

    assert _json.loads(route.calls[0].request.content)["model"] == "kimi-k2.5"


# ---------------- 响应解析 ----------------


@respx.mock
async def test_usage_and_cost_parsed():
    respx.post(URL).mock(return_value=httpx.Response(200, json=reply()))
    async with DeepSeekProvider(settings()) as p:
        c = await p.complete(req(character_id="lin_wan"))

    assert c.text == "嗯，刚下班"
    assert c.usage.cache_hit_tokens == 3968
    assert c.usage.cache_hit_rate == pytest.approx(0.992)
    assert c.cost.total_cny > 0
    assert c.character_id == "lin_wan"
    assert c.latency_ms >= 0


@respx.mock
async def test_reasoning_tokens_surfaced():
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json=reply(completion_tokens=800, **{"completion_tokens_details": {"reasoning_tokens": 780}}),
        )
    )
    async with DeepSeekProvider(settings()) as p:
        c = await p.complete(req())
    assert c.usage.reasoning_tokens == 780


# ---------------- 错误与重试 ----------------


@respx.mock
async def test_retries_on_500_then_succeeds():
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(200, json=reply()),
        ]
    )
    async with DeepSeekProvider(settings()) as p:
        c = await p.complete(req())
    assert c.text == "嗯，刚下班"
    assert route.call_count == 2


@respx.mock
async def test_retries_on_429():
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, text="rate limited", headers={"retry-after": "0"}),
            httpx.Response(200, json=reply()),
        ]
    )
    async with DeepSeekProvider(settings()) as p:
        await p.complete(req())
    assert route.call_count == 2


@respx.mock
async def test_gives_up_after_max_retries():
    route = respx.post(URL).mock(return_value=httpx.Response(503, text="down"))
    async with DeepSeekProvider(settings(retries=2)) as p:
        with pytest.raises(LLMServerError):
            await p.complete(req())
    assert route.call_count == 3


@respx.mock
async def test_400_is_not_retried():
    """请求本身有问题，重试只会再错一次。"""
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad param"))
    async with DeepSeekProvider(settings()) as p:
        with pytest.raises(LLMRequestError):
            await p.complete(req())
    assert route.call_count == 1


@respx.mock
async def test_401_is_config_error_not_retried():
    route = respx.post(URL).mock(return_value=httpx.Response(401, text="bad key"))
    async with DeepSeekProvider(settings()) as p:
        with pytest.raises(LLMConfigError):
            await p.complete(req())
    assert route.call_count == 1


def test_missing_api_key_fails_fast():
    with pytest.raises(LLMConfigError):
        DeepSeekProvider(Settings(deepseek_api_key="", usage_log_path=None))


# ---------------- 流式 ----------------


@respx.mock
async def test_stream_yields_deltas_and_final_usage():
    sse = (
        'data: {"choices":[{"delta":{"content":"嗯"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"，刚下班"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4000,"completion_tokens":12,'
        '"prompt_cache_hit_tokens":3968,"prompt_cache_miss_tokens":32}}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(URL).mock(
        return_value=httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )
    )

    async with DeepSeekProvider(settings()) as p:
        events = [e async for e in p.stream(req(stream=True))]

    assert "".join(e.delta for e in events) == "嗯，刚下班"
    final = events[-1]
    assert final.done
    assert final.finish_reason == "stop"
    assert final.usage is not None
    assert final.usage.cache_hit_tokens == 3968


@respx.mock
async def test_stream_requests_usage():
    """不加 stream_options.include_usage，流式会整段丢失成本数据。"""
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, text="data: [DONE]\n\n")
    )
    async with DeepSeekProvider(settings()) as p:
        _ = [e async for e in p.stream(req(stream=True))]

    import json as _json

    body = _json.loads(route.calls[0].request.content)
    assert body["stream_options"] == {"include_usage": True}
    assert body["stream"] is True
