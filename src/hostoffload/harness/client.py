"""Streaming OpenAI-compatible client with per-chunk timing and per-chunk token counts.

With speculative decoding one streamed chunk can carry several tokens, so per-chunk inter-arrival times are not
inter-token latencies. We request `stream_options.continuous_usage_stats`, which makes vLLM attach the cumulative
`completion_tokens` to every chunk; each chunk then records (arrival time, cumulative tokens). TTFT, TPOT and the
per-token latency distribution are derived from that list in `hostoffload.analysis.metrics`.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    id: str
    scenario: str
    level: str
    repetition: int
    user: int | None
    scheduled_t: float | None          # intended send time (open loop), seconds from the level start
    sent_wall: float = 0.0             # time.time() at send
    sent_mono: float = 0.0             # time.monotonic() at send
    first_token_mono: float | None = None
    last_token_mono: float | None = None
    end_mono: float | None = None
    chunks: list[tuple[float, int]] = field(default_factory=list)  # (monotonic t, cumulative completion tokens)
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int = 0
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None
    text_chars: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def parse_sse_line(line: str) -> dict | None | str:
    """One SSE line -> parsed JSON payload, the string "DONE", or None for non-data lines."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "DONE"
    return json.loads(payload)


def apply_chunk(rec: RequestRecord, d: dict, now: float) -> None:
    """Update a record with one parsed chunk observed at monotonic time `now`."""
    usage = d.get("usage") or None
    has_text = False
    for c in d.get("choices", []):
        delta = c.get("delta") or {}
        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
        if piece:
            has_text = True
            rec.text_chars += len(piece)
            head = rec.meta.get("text_head", "")
            if len(head) < 300:  # the first characters, to spot degenerate output (NaN rows, repetition)
                rec.meta["text_head"] = (head + piece)[:300]
        if c.get("finish_reason"):
            rec.finish_reason = c["finish_reason"]
    if usage:
        rec.prompt_tokens = usage.get("prompt_tokens", rec.prompt_tokens)
        rec.completion_tokens = usage.get("completion_tokens", rec.completion_tokens) or 0
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            rec.cached_tokens = details["cached_tokens"]
    if has_text:
        if rec.first_token_mono is None:
            rec.first_token_mono = now
        rec.last_token_mono = now
        rec.chunks.append((now, rec.completion_tokens if usage else -1))


def chat_body(model: str, messages: list[dict], max_tokens: int, temperature: float, seed: int | None,
              ignore_eos: bool, thinking: bool | None, top_p: float | None = None) -> dict:
    body: dict[str, Any] = {
        "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
        "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "ignore_eos": ignore_eos,
    }
    if seed is not None:
        body["seed"] = seed
    if top_p is not None:
        body["top_p"] = top_p
    if thinking is not None:
        body["chat_template_kwargs"] = {"thinking": thinking}
    return body


async def stream_request(session, base_url: str, headers: dict, body: dict, rec: RequestRecord,
                         timeout_s: float) -> RequestRecord:
    """Send one streaming chat completion and fill `rec`. `session` is an aiohttp.ClientSession."""
    import aiohttp

    rec.sent_wall = time.time()
    rec.sent_mono = time.monotonic()
    try:
        async with session.post(base_url + "/v1/chat/completions", json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            rec.http_status = r.status
            if r.status != 200:
                rec.error = f"HTTP {r.status}: {(await r.text())[:300]}"
                return rec
            async for raw in r.content:
                parsed = parse_sse_line(raw.decode(errors="replace"))
                if parsed is None:
                    continue
                if parsed == "DONE":
                    break
                apply_chunk(rec, parsed, time.monotonic())
    except asyncio.TimeoutError:
        rec.error = "timeout"
    except Exception as e:  # network errors are data, not crashes
        rec.error = repr(e)[:300]
    finally:
        rec.end_mono = time.monotonic()
    if rec.error is None and rec.first_token_mono is None:
        rec.error = "no tokens"
    return rec
