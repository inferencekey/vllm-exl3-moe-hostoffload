#!/usr/bin/env python3
"""Closed-loop concurrency load generator for an OpenAI-compatible /v1/chat/completions server (vLLM).

Runs on the pod against 127.0.0.1. For each concurrency level N it keeps N requests in flight until
R = max(REQ_MULT * N, MIN_REQ) requests have been sent, streaming every answer and timing it per request:
  ttft_s      = first content token arrival - request start
  decode_tok_s = (completion_tokens - 1) / (last token arrival - first token arrival)
While a level runs, /metrics and nvidia-smi are sampled every second (running/waiting requests, KV cache usage,
prefix cache hits, DSpark acceptance, VRAM).

Scenarios:
  A  every request is a different ~4K-token slice of real source code (no shared prefix beyond the chat template)
  B  every request is the same ~16K-token code prefix plus a different short question at the end

Usage: API_KEY=... python3 loadgen.py A|B [levels]   (levels default "1 2 4 8 16 32 64")
Writes OUT_DIR/scenario_<X>.json after every level.
"""
import asyncio, glob, json, os, re, statistics, sys, time

import aiohttp

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
KEY = os.environ["API_KEY"]
MODEL = os.environ.get("SERVED", "deepseek-v4.1-flash")
OUT_DIR = os.environ.get("OUT_DIR", "/root/results")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
REQ_MULT = int(os.environ.get("REQ_MULT", "2"))
MIN_REQ = int(os.environ.get("MIN_REQ", "8"))
LEVEL_CAP_S = float(os.environ.get("LEVEL_CAP_S", "360"))      # stop dispatching new requests after this
REQ_TIMEOUT_S = float(os.environ.get("REQ_TIMEOUT_S", "600"))
STOP_TTFT_P95_S = float(os.environ.get("STOP_TTFT_P95_S", "60"))
CODE_GLOB = os.environ.get("CODE_GLOB", "/usr/local/lib/python3.12/dist-packages/vllm/**/*.py")
CHARS_PER_TOKEN_4K = 4.0   # /tokenize on this corpus: 14,240 chars -> 2,769..3,946 tokens depending on the region
CHARS_PER_TOKEN_16K = 3.55  # /tokenize: 55,200 chars of the scenario B region -> 15,552 tokens
HEADERS = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def load_corpus():
    parts = []
    for f in sorted(glob.glob(CODE_GLOB, recursive=True)):
        try:
            parts.append(f"\n# ==== file: {f.split('dist-packages/')[-1]} ====\n" + open(f).read())
        except Exception:
            pass
    return "".join(parts)


def pct(values, p):
    """Nearest-rank percentile (p in 0..100) of a non-empty list."""
    v = sorted(values)
    if not v:
        return None
    k = max(0, min(len(v) - 1, int(round(p / 100 * len(v) + 0.5)) - 1))
    return v[k]


def parse_metrics(text):
    """Sum every vllm:* sample by metric name (labels dropped)."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            name_part, value = line.rsplit(" ", 1)
            name = name_part.split("{", 1)[0]
            if name.endswith("_created"):
                continue
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out


def pick(m, *names):
    for n in names:
        if n in m:
            return m[n]
    return None


async def fetch_metrics(session):
    try:
        async with session.get(BASE + "/metrics", headers=HEADERS, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return parse_metrics(await r.text())
    except Exception as e:
        return {"error": repr(e)}


async def nvidia_smi():
    try:
        p = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
        out, _ = await p.communicate()
        return [[float(x) for x in l.split(",")] for l in out.decode().strip().splitlines()]
    except Exception:
        return None


async def sampler(session, samples, stop):
    i = 0
    while not stop.is_set():
        m = await fetch_metrics(session)
        s = {"t": time.time(),
             "running": pick(m, "vllm:num_requests_running"),
             "waiting": pick(m, "vllm:num_requests_waiting"),
             "kv_usage": pick(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")}
        if i % 5 == 0:
            s["gpus"] = await nvidia_smi()
        samples.append(s)
        i += 1
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


async def one_request(session, messages, rid):
    body = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": False,
            "chat_template_kwargs": {"thinking": False}}
    rec = {"id": rid, "start": time.time(), "ttft_s": None, "decode_tok_s": None, "completion_tokens": 0,
           "prompt_tokens": None, "cached_tokens": None, "error": None, "finish_reason": None, "chunks": 0}
    t_first = t_last = None
    usage = None
    text_len = 0
    try:
        async with session.post(BASE + "/v1/chat/completions", json=body, headers=HEADERS,
                                timeout=aiohttp.ClientTimeout(total=REQ_TIMEOUT_S)) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                return rec
            async for raw in r.content:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                for c in d.get("choices", []):
                    delta = c.get("delta", {})
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                    if piece:
                        now = time.time()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        text_len += len(piece)
                        rec["chunks"] += 1
                    if c.get("finish_reason"):
                        rec["finish_reason"] = c["finish_reason"]
    except asyncio.TimeoutError:
        rec["error"] = "timeout"
    except Exception as e:
        rec["error"] = repr(e)[:300]
    rec["end"] = time.time()
    rec["total_s"] = round(rec["end"] - rec["start"], 3)
    rec["text_chars"] = text_len
    if usage:
        rec["completion_tokens"] = usage.get("completion_tokens", 0)
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if t_first:
        rec["ttft_s"] = round(t_first - rec["start"], 3)
        ct = rec["completion_tokens"]
        if ct > 1 and t_last > t_first:
            rec["decode_tok_s"] = round((ct - 1) / (t_last - t_first), 2)
    if rec["error"] is None and t_first is None:
        rec["error"] = "no tokens"
    return rec


async def run_level(session, n, make_messages):
    total = max(REQ_MULT * n, MIN_REQ)
    queue = list(range(total))
    results, samples = [], []
    m0 = await fetch_metrics(session)
    stop = asyncio.Event()
    samp = asyncio.create_task(sampler(session, samples, stop))
    t0 = time.time()

    async def worker():
        while queue and time.time() - t0 < LEVEL_CAP_S:
            i = queue.pop(0)
            results.append(await one_request(session, make_messages(), f"N{n}-{i}"))

    await asyncio.gather(*[worker() for _ in range(n)])
    wall = time.time() - t0
    stop.set()
    await samp
    m1 = await fetch_metrics(session)

    ok = [r for r in results if r["error"] is None]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    dec = [r["decode_tok_s"] for r in ok if r["decode_tok_s"] is not None and r["completion_tokens"] >= 32]
    out_tokens = sum(r["completion_tokens"] for r in ok)

    def delta(*names):
        a, b = pick(m0, *names), pick(m1, *names)
        return None if a is None or b is None else b - a

    hits = delta("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    queries = delta("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
    acc = delta("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens")
    drafts = delta("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts")
    gpu_samples = [s["gpus"] for s in samples if s.get("gpus")]
    vram_peak = None
    if gpu_samples:
        vram_peak = [max(g[i][1] for g in gpu_samples) for i in range(len(gpu_samples[0]))]

    def mx(key):
        v = [s[key] for s in samples if s.get(key) is not None]
        return max(v) if v else None

    summary = {
        "N": n, "requests_planned": total, "requests_done": len(results), "ok": len(ok),
        "errors": len(results) - len(ok), "error_samples": [r["error"] for r in results if r["error"]][:5],
        "wall_s": round(wall, 2), "capped": len(results) < total,
        "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95), "ttft_max_s": max(ttft) if ttft else None,
        "decode_p50_tok_s": pct(dec, 50), "decode_p5_tok_s": pct(dec, 5), "decode_min_tok_s": min(dec) if dec else None,
        "aggregate_out_tok_s": round(out_tokens / wall, 2) if wall else None,
        "out_tokens": out_tokens,
        "mean_completion_tokens": round(out_tokens / len(ok), 1) if ok else None,
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in ok if r["prompt_tokens"]), 1) if ok else None,
        "mean_cached_tokens": round(statistics.mean((r["cached_tokens"] or 0) for r in ok), 1) if ok else None,
        "server_running_max": mx("running"), "server_waiting_max": mx("waiting"),
        "server_kv_usage_max_pct": round(100 * mx("kv_usage"), 1) if mx("kv_usage") is not None else None,
        "prefix_cache_hits_tokens": hits, "prefix_cache_queries_tokens": queries,
        "prefix_cache_hit_rate_pct": round(100 * hits / queries, 1) if hits is not None and queries else None,
        "spec_mean_accept_len": round(1 + acc / drafts, 2) if acc is not None and drafts else None,
        "vram_peak_mib_per_gpu": vram_peak,
    }
    return summary, results, samples


async def main():
    scenario = sys.argv[1].upper()
    levels = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1 2 4 8 16 32 64").split()]
    corpus = load_corpus()
    print(f"corpus {len(corpus):,} chars", flush=True)
    offset = [0]

    def next_slice(nchars):
        s = corpus[offset[0]:offset[0] + nchars]
        offset[0] += nchars
        if offset[0] > len(corpus):
            raise RuntimeError("corpus exhausted")
        return s

    if scenario == "A":
        nchars = int(4000 * CHARS_PER_TOKEN_4K)
        offset[0] = int(os.environ.get("A_OFFSET", "0"))  # warmup runs use a different region

        def make_messages():
            code = next_slice(nchars)
            return [{"role": "user", "content": "Here is part of a Python code base:\n\n```python\n" + code +
                     "\n```\n\nReview this code: explain briefly what it does and point out one possible bug or improvement."}]
    else:
        n16 = int(16000 * CHARS_PER_TOKEN_16K)
        prefix_code = corpus[len(corpus) - n16 - 1000:len(corpus) - 1000]  # far from scenario A slices
        prefix = ("You are a coding assistant. This is the code the user is working on:\n\n```python\n" + prefix_code +
                  "\n```\n\nAnswer the user's question about this code.\n\nQuestion: ")
        names = list(dict.fromkeys(re.findall(r"def (\w+)\(", prefix_code) + re.findall(r"class (\w+)", prefix_code)))
        templates = ["What does `{}` do, and what are its inputs and outputs?",
                     "How would you write a unit test for `{}`?",
                     "Is there any edge case `{}` does not handle?",
                     "Explain where `{}` is used and why it exists."]
        questions = [t.format(nm) for t in templates for nm in names]
        print(f"scenario B: {len(names)} names, {len(questions)} distinct questions", flush=True)
        counter = [0]

        def make_messages():
            q = questions[counter[0] % len(questions)] + f" (request {counter[0]})"
            counter[0] += 1
            return [{"role": "user", "content": prefix + q}]

    out_path = os.path.join(OUT_DIR, f"scenario_{scenario}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"scenario": scenario, "started": time.strftime("%F %T"), "max_tokens": MAX_TOKENS,
              "temperature": TEMPERATURE, "thinking": False, "req_mult": REQ_MULT, "min_req": MIN_REQ,
              "levels": [], "requests": {}, "samples": {}}
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        if scenario == "B":
            # one request puts the shared prefix in the cache, as an assistant session would already have it
            w = await one_request(session, make_messages(), "B-prefill")
            report["prefix_warm_request"] = w
            print("B prefix warm:", {k: w[k] for k in ("ttft_s", "prompt_tokens", "cached_tokens", "decode_tok_s", "error")}, flush=True)
        for n in levels:
            summary, results, samples = await run_level(session, n, make_messages)
            report["levels"].append(summary)
            report["requests"][str(n)] = results
            report["samples"][str(n)] = samples
            json.dump(report, open(out_path, "w"), indent=1)
            print(json.dumps(summary), flush=True)
            if summary["errors"] or (summary["ttft_p95_s"] or 0) > STOP_TTFT_P95_S:
                print(f"stopping after N={n}: errors={summary['errors']} ttft_p95={summary['ttft_p95_s']}", flush=True)
                break
    report["finished"] = time.strftime("%F %T")
    json.dump(report, open(out_path, "w"), indent=1)
    print("LOADGEN DONE", scenario, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
