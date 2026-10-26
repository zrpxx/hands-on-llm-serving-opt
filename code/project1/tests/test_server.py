"""HTTP 接口测试：流式与非流式结果一致，并发请求互不串线。"""
import asyncio
import json

import httpx
import pytest

from miniserve import EngineConfig
from miniserve.server import create_app


@pytest.fixture(scope="module")
def app(tiny_model):
    return create_app(EngineConfig(model=tiny_model, device="cpu", num_gpu_blocks=128, block_size=4,
                                   max_model_len=256, max_num_batched_tokens=64))


async def complete(client, prompt, stream, n=12):
    body = {"prompt": prompt, "max_tokens": n, "stream": stream, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    if not stream:
        r = await client.post("/v1/completions", json=body)
        return r.json()["choices"][0]["text"], r.json()["usage"]["completion_tokens"]
    text, usage = "", None
    async with client.stream("POST", "/v1/completions", json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])
                if d["choices"]:
                    text += d["choices"][0]["text"]
                else:
                    usage = d["usage"]["completion_tokens"]
    return text, usage


def test_stream_matches_non_stream_under_concurrency(app):
    async def main():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60) as c:
            prompts = [[10 + i, 20 + i, 30 + i, 40 + i] * (i + 1) for i in range(6)]
            plain = await asyncio.gather(*[complete(c, p, False) for p in prompts])
            streamed = await asyncio.gather(*[complete(c, p, True) for p in prompts])
            assert plain == streamed
            assert all(n == 12 for _, n in plain)
            r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                          "max_tokens": 4, "ignore_eos": True})
            assert r.json()["usage"]["completion_tokens"] == 4
    asyncio.run(asyncio.wait_for(main(), timeout=60))   # 后台线程出错时请求会一直等待，用超时让测试失败
    eng = app.state.engine
    assert eng.allocator.num_free == eng.num_blocks
