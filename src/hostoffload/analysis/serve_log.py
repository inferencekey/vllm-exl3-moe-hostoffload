"""Parse what the server reports about memory placement at load time (vLLM + vllm-exl3 log lines).

The patterns match the log format of vLLM 0.30.1rc1.dev312+g36768d1bf with the patched vllm-exl3 plugin, as seen
in data/exploratory/.../logs/serve.log. Each extractor returns None when its line is absent, so a log from a
different configuration (for example no host experts) parses without errors.
"""
from __future__ import annotations

import re
from pathlib import Path

GiB = 2 ** 30

_HOST_EXPERTS = re.compile(r"\((Worker_TP(\d+)) pid=\d+\).*EXL3 host experts: (\S+)\.ffn\.experts keeps (\d+)/(\d+)")
_ENGRAM = re.compile(r"\((Worker_TP(\d+)) pid=\d+\).*Engram table offloaded to pinned host memory: (\d+) rows x (\d+), "
                     r"([\d.]+) GiB per rank")
_MODEL_LOAD = re.compile(r"\((Worker_TP(\d+)) pid=\d+\).*Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds")
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_MAX_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_GRAPH = re.compile(r"\((Worker_TP(\d+)) pid=\d+\).*Graph capturing finished in (\d+) secs, took ([\d.-]+) GiB")
_VERSION = re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\)")
_NONDEFAULT = re.compile(r"non-default args: (\{.*\})")


def parse(text: str) -> dict:
    host: dict[int, dict[int, tuple[int, int]]] = {}
    for m in _HOST_EXPERTS.finditer(text):
        rank, name, kept, total = int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        layer = int(re.search(r"layers\.(\d+)", name).group(1))
        host.setdefault(rank, {})[layer] = (kept, total)
    engram = [{"rank": int(m.group(2)), "rows": int(m.group(3)), "dim": int(m.group(4)), "gib": float(m.group(5))}
              for m in _ENGRAM.finditer(text)]
    loads = {int(m.group(2)): {"gib": float(m.group(3)), "seconds": float(m.group(4))}
             for m in _MODEL_LOAD.finditer(text)}
    graphs = {int(m.group(2)): {"seconds": int(m.group(3)), "gib": float(m.group(4))} for m in _GRAPH.finditer(text)}
    kv = _KV_TOKENS.search(text)
    conc = _MAX_CONC.search(text)
    ver = _VERSION.search(text)
    nd = _NONDEFAULT.search(text)
    return {
        "vllm_version": ver.group(1) if ver else None,
        "non_default_args_raw": nd.group(1) if nd else None,
        "host_experts_by_rank": {r: {L: list(v) for L, v in sorted(d.items())} for r, d in sorted(host.items())},
        "engram_tables": engram,
        "engram_pinned_gib_total": sum(e["gib"] for e in engram) if engram else None,
        "model_loading_gib_by_rank": {r: v["gib"] for r, v in sorted(loads.items())} or None,
        "model_loading_s_by_rank": {r: v["seconds"] for r, v in sorted(loads.items())} or None,
        "cuda_graph_gib_by_rank": {r: v["gib"] for r, v in sorted(graphs.items())} or None,
        "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "max_concurrency_at_len": ({"len": int(conc.group(1).replace(",", "")), "x": float(conc.group(2))}
                                   if conc else None),
    }


def parse_file(path: Path) -> dict:
    return parse(Path(path).read_text(errors="replace"))


def host_counts(parsed: dict, rank: int = 0) -> dict[int, int]:
    return {L: v[0] for L, v in parsed["host_experts_by_rank"].get(rank, {}).items()}


def identify_plan(parsed: dict, plans: dict[str, dict[int, set[int]]]) -> list[str]:
    """Names of the plans whose per-layer host-expert counts equal the counts the server logged on every rank.
    Counts identify a plan only up to permutations with equal per-layer sizes; the paper says so."""
    out = []
    ranks = parsed["host_experts_by_rank"]
    if not ranks:
        return out
    for name, plan in plans.items():
        sizes = {L: len(ids) for L, ids in plan.items()}
        if all(all(sizes.get(L) == kept for L, (kept, _t) in layers.items()) and len(layers) == len(sizes)
               for layers in ranks.values()):
            out.append(name)
    return out
