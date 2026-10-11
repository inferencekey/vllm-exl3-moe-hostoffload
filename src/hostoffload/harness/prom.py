"""Parse the Prometheus text exposition of vLLM's /metrics, including histograms.

Server-side histograms separate what the client cannot: queueing, prefill and decode time per request
(`vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`),
TTFT and inter-token latency. A level's distribution is the difference of two cumulative snapshots.
"""
from __future__ import annotations

import math
import re

_LINE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text: str, prefix: str = "vllm:") -> dict[str, list[tuple[dict[str, str], float]]]:
    """name -> [(labels, value)] for every sample whose name starts with `prefix`."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        labels = dict(_LABEL.findall(m.group("labels") or ""))
        out.setdefault(m.group("name"), []).append((labels, value))
    return out


def scalar(samples: dict, *names: str, drop_labels: tuple[str, ...] = ()) -> float | None:
    """Sum of all samples of the first metric name present (labels summed out)."""
    for n in names:
        if n in samples:
            return sum(v for _, v in samples[n])
    return None


def histogram(samples: dict, base: str) -> dict | None:
    """Cumulative histogram {le: count} plus sum and count, summed over label sets other than `le`."""
    buckets_key, sum_key, count_key = base + "_bucket", base + "_sum", base + "_count"
    if buckets_key not in samples:
        return None
    buckets: dict[float, float] = {}
    for labels, v in samples[buckets_key]:
        le = labels.get("le")
        if le is None:
            continue
        b = math.inf if le in ("+Inf", "inf") else float(le)
        buckets[b] = buckets.get(b, 0.0) + v
    # buckets as sorted [le, cumulative count] pairs (JSON-safe; le may be inf, written as Infinity)
    return {"buckets": [[le, c] for le, c in sorted(buckets.items())], "sum": scalar(samples, sum_key) or 0.0,
            "count": scalar(samples, count_key) or 0.0}


def histogram_delta(h0: dict | None, h1: dict | None) -> dict | None:
    if h0 is None or h1 is None:
        return None
    b0 = {float(le): c for le, c in h0["buckets"]}
    b = [[float(le), c - b0.get(float(le), 0.0)] for le, c in h1["buckets"]]
    return {"buckets": b, "sum": h1["sum"] - h0["sum"], "count": h1["count"] - h0["count"]}


def histogram_quantile(h: dict | None, q: float) -> float | None:
    """Prometheus-style linear interpolation within the bucket that holds quantile q (0..1)."""
    if not h or h["count"] <= 0:
        return None
    rank = q * h["count"]
    prev_le, prev_c = 0.0, 0.0
    for le, c in sorted((float(le), c) for le, c in h["buckets"]):
        if c >= rank:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le


SERVER_HISTOGRAMS = [
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
]

SERVER_COUNTERS = {
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    "spec_accepted": ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"),
    "spec_drafts": ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"),
    "spec_draft_tokens": ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"),
    "prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
    "generation_tokens": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
}

SERVER_GAUGES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}


def snapshot(text: str) -> dict:
    """Compact snapshot: gauges, counters and server histograms."""
    s = parse(text)
    return {
        "gauges": {k: scalar(s, *v) for k, v in SERVER_GAUGES.items()},
        "counters": {k: scalar(s, *v) for k, v in SERVER_COUNTERS.items()},
        "histograms": {h: histogram(s, h) for h in SERVER_HISTOGRAMS if histogram(s, h) is not None},
    }
