"""A tiny OpenAI-compatible mock of the vLLM endpoints the harness uses. TEST ONLY.

Tokens are characters. Streaming emits `tokens_per_chunk` tokens per chunk with a fixed delay, and continuous usage
stats like vLLM. /metrics exposes counters and one histogram. Nothing produced with this server is data: tests write
to temporary directories and tests/test_no_fake_data.py checks that no mock output lands under data/.
"""
from __future__ import annotations

import asyncio
import json

from aiohttp import web

MODEL = "mock-model"


class Mock:
    def __init__(self, ttft_s: float = 0.01, chunk_delay_s: float = 0.002, tokens_per_chunk: int = 2):
        self.ttft_s, self.chunk_delay_s, self.tpc = ttft_s, chunk_delay_s, tokens_per_chunk
        self.requests = 0
        self.gen_tokens = 0
        self.ttft_obs: list[float] = []
        self.bodies: list[dict] = []

    async def models(self, _req):
        return web.json_response({"data": [{"id": MODEL}]})

    async def tokenize(self, req):
        d = await req.json()
        return web.json_response({"tokens": [ord(c) for c in d["prompt"]], "count": len(d["prompt"])})

    async def detokenize(self, req):
        d = await req.json()
        return web.json_response({"prompt": "".join(chr(t) for t in d["tokens"])})

    async def reset(self, _req):
        return web.Response(text="ok")

    async def metrics(self, _req):
        b = [0.01, 0.05, 0.1, 1.0]
        lines = [f"vllm:generation_tokens_total {self.gen_tokens}", "vllm:num_requests_running 0",
                 "vllm:num_requests_waiting 0", "vllm:kv_cache_usage_perc 0.1"]
        for le in b:
            lines.append(f'vllm:time_to_first_token_seconds_bucket{{le="{le}"}} '
                         f'{sum(1 for x in self.ttft_obs if x <= le)}')
        lines.append(f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {len(self.ttft_obs)}')
        lines.append(f"vllm:time_to_first_token_seconds_sum {sum(self.ttft_obs)}")
        lines.append(f"vllm:time_to_first_token_seconds_count {len(self.ttft_obs)}")
        return web.Response(text="\n".join(lines) + "\n")

    async def chat(self, req):
        d = await req.json()
        self.bodies.append(d)
        self.requests += 1
        prompt_len = sum(len(m["content"]) for m in d["messages"])
        n = int(d.get("max_tokens", 16))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        await asyncio.sleep(self.ttft_s)
        self.ttft_obs.append(self.ttft_s)
        sent = 0
        while sent < n:
            k = min(self.tpc, n - sent)
            sent += k
            chunk = {"choices": [{"delta": {"content": "x" * k}, "finish_reason": "length" if sent >= n else None}],
                     "usage": {"prompt_tokens": prompt_len, "completion_tokens": sent,
                               "prompt_tokens_details": {"cached_tokens": 0}}}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(self.chunk_delay_s)
        self.gen_tokens += sent
        await resp.write(b"data: [DONE]\n\n")
        return resp

    def app(self) -> web.Application:
        a = web.Application()
        a.router.add_get("/v1/models", self.models)
        a.router.add_post("/tokenize", self.tokenize)
        a.router.add_post("/detokenize", self.detokenize)
        a.router.add_post("/reset_prefix_cache", self.reset)
        a.router.add_get("/metrics", self.metrics)
        a.router.add_post("/v1/chat/completions", self.chat)
        return a
