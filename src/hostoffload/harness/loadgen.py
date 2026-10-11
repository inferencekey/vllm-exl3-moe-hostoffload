"""Controlled load generator for an OpenAI-compatible vLLM server.

Reads a workload file (experiments/workloads/*.yaml), builds every prompt before measuring (exact token lengths via
the server's /tokenize and /detokenize), then runs each scenario at each load point for R repetitions and writes
raw data only:

  <out>/run.json               resolved config, seeds, start/end, status, harness git revision
  <out>/requests.jsonl         one record per request (timestamps, per-chunk token counts, usage, errors)
  <out>/levels.jsonl           per (scenario, load point, repetition): send/end window and quick summary
  <out>/server_metrics.jsonl   full /metrics snapshots (gauges, counters, histograms) before and after each level

Analysis (`make analyze`) recomputes every summary from requests.jsonl; the quick summaries are for monitoring.

Scenario kinds:
  chat     closed-loop or Poisson arrivals, prompts of a fixed or lognormal token length, optional shared prefix
  prefill  isolation probe: unique prompts, max_tokens 1, one at a time (TTFT is prefill plus a near-empty queue)
  decode   isolation probe: short unique prompts, long fixed outputs, N concurrent streams (TPOT)

  python3 -m hostoffload.harness.loadgen --workload experiments/workloads/concurrency.yaml --out <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ..analysis.metrics import level_summary
from . import prom
from .client import RequestRecord, chat_body, stream_request
from .workload import ClosedLoop, CorpusSpec, Poisson, SliceAllocator, lognormal_lengths, load_corpus

ROOT = Path(__file__).resolve().parents[3]

TEMPLATES = {
    "review": ("Here is part of a Python code base:\n\n```python\n{text}\n```\n\n"
               "Review this code: explain briefly what it does and point out one possible bug or improvement."),
    "continue_prose": "Continue the following text in the same style.\n\n{text}",
    "summarize": "Summarize the following text in a few paragraphs.\n\n{text}",
    "qa_prefix": ("You are a coding assistant. This is the code the user is working on:\n\n```python\n{text}\n```\n\n"
                  "Answer the user's question about this code.\n\nQuestion: "),
    "raw": "{text}",
    "agent": ("You are a coding agent working in this repository. Context so far (files, tool output, history):\n\n"
              "```\n{text}\n```\n\nPropose the next change and write the code for it."),
}
QUESTIONS = ["What does the function or class named in this question do: {k}?",
             "How would you write a unit test for the code near line {k}?",
             "Is there an edge case that the code around item {k} does not handle?",
             "Explain where the code around item {k} is used and why it exists."]


def git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


async def retry_unmeasured(call, attempts: int = 3):
    """Retry a helper call that is not a measurement (tokenize, metrics) on a dropped connection. A measured request
    is never retried: its error is data."""
    import aiohttp
    for i in range(attempts):
        try:
            return await call()
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
            if i == attempts - 1:
                raise
            await asyncio.sleep(0.5)


class Server:
    def __init__(self, session, base_url: str, model: str, api_key: str | None):
        self.s, self.base, self.model = session, base_url.rstrip("/"), model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def tokenize(self, text: str) -> list[int]:
        async def call():
            async with self.s.post(self.base + "/tokenize", json={"model": self.model, "prompt": text,
                                                                    "add_special_tokens": False},
                                   headers=self.headers) as r:
                return (await r.json())["tokens"]
        return await retry_unmeasured(call)

    async def detokenize(self, ids: list[int]) -> str:
        async with self.s.post(self.base + "/detokenize", json={"model": self.model, "tokens": ids},
                               headers=self.headers) as r:
            return (await r.json())["prompt"]

    async def metrics(self) -> dict:
        async def call():
            async with self.s.get(self.base + "/metrics", headers=self.headers) as r:
                return prom.snapshot(await r.text())
        return await retry_unmeasured(call)

    async def reset_prefix_cache(self) -> bool:
        try:
            async with self.s.post(self.base + "/reset_prefix_cache", headers=self.headers) as r:
                return r.status == 200
        except Exception:
            return False

    async def wait_ready(self, timeout_s: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                async with self.s.get(self.base + "/v1/models", headers=self.headers) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(5)
        raise TimeoutError("server not ready")


async def make_text(server: Server, corpus: str, alloc: SliceAllocator, tokens: int) -> tuple[str, int]:
    """A unique corpus region cut to `tokens` tokens."""
    a, b = alloc.next()
    ids = await server.tokenize(corpus[a:b])
    if len(ids) < tokens:
        raise ValueError(f"corpus region of {b - a} chars has {len(ids)} tokens < {tokens}; raise chars_per_token")
    text = await server.detokenize(ids[:tokens])
    return text, tokens


async def build_prompts(server: Server, sc: dict, corpora: dict[str, str], n: int, seed: int) -> list[list[dict]]:
    """All prompts (chat messages) of one (scenario, load point, repetition), built before timing starts."""
    p = sc["prompt"]
    corpus = corpora[p.get("corpus", "code")]
    lo, hi = p.get("region", [0.0, 1.0])  # benchmark and calibration workloads use disjoint corpus halves
    corpus = corpus[int(lo * len(corpus)):int(hi * len(corpus))]
    cpt = float(p.get("chars_per_token", 6.0))  # generous: regions are cut down to the exact token count
    if p.get("length_dist") == "lognormal":
        lengths = lognormal_lengths(n, p["tokens"], p.get("sigma", 0.6), p.get("min_tokens", 128),
                                    p.get("max_tokens", 4 * p["tokens"]), seed)
    else:
        lengths = [p["tokens"]] * n
    shared = p.get("shared_prefix_tokens", 0)
    template = TEMPLATES[p.get("template", "raw")]
    reserved = None
    prefix_text = ""
    if shared:
        # the shared prefix comes from a region that is never handed out to unique prompts
        span = int(shared * cpt)
        start = random.Random(seed ^ 0x5EED).randrange(0, max(1, len(corpus) - span))
        reserved = (start, start + span)
        ids = await server.tokenize(corpus[start:start + span])
        prefix_text = await server.detokenize(ids[:shared])
    region = int(max(lengths) * cpt) if not shared else int(max(64, max(lengths) - shared) * cpt) + 1
    alloc = SliceAllocator(len(corpus), max(region, 256), seed, reserved)
    prompts = []
    for i, L in enumerate(lengths):
        if shared:
            q = QUESTIONS[i % len(QUESTIONS)].format(k=i)
            suffix_tokens = max(0, L - shared - 64)
            extra = (await make_text(server, corpus, alloc, suffix_tokens))[0] if suffix_tokens else ""
            content = template.format(text=prefix_text) + q + (("\n\nContext:\n" + extra) if extra else "") + \
                f" (request {seed}-{i})"
        else:
            text, _ = await make_text(server, corpus, alloc, L)
            content = template.format(text=text)
        prompts.append([{"role": "user", "content": content}])
    return prompts


async def build_agent_prompts(server: Server, sc: dict, corpora: dict[str, str], users: int,
                              seed: int) -> list[list[dict]]:
    """Coding-agent sessions (E13): user u owns a unique base context of `base_tokens`; turn 0 sends it alone (the
    cache write that opens the session, recorded as warm), turn t >= 1 sends base + t appended unique chunks of
    `append_tokens` (new tool output and edits). Each turn's prompt extends the previous one, so vLLM's prefix cache
    serves the earlier turns and only the appended chunk is new prefill. Index of (u, t) is u * (turns + 1) + t."""
    ag, p = sc["agent"], sc["prompt"]
    corpus = corpora[p.get("corpus", "code")]
    lo, hi = p.get("region", [0.0, 1.0])
    corpus = corpus[int(lo * len(corpus)):int(hi * len(corpus))]
    cpt = float(p.get("chars_per_token", 6.0))
    base, app, turns = int(ag["base_tokens"]), int(ag["append_tokens"]), int(ag["turns"])
    template = TEMPLATES[p.get("template", "agent")]
    alloc = SliceAllocator(len(corpus), int(max(base, app) * cpt), seed)
    prompts: list[list[dict]] = []
    for _ in range(users):
        text, _n = await make_text(server, corpus, alloc, base)
        chunks = [(await make_text(server, corpus, alloc, app))[0] for _t in range(turns)]
        for t in range(turns + 1):
            prompts.append([{"role": "user", "content": template.format(text=text + "".join(
                f"\n# --- turn {k + 1} ---\n" + c for k, c in enumerate(chunks[:t])))}])
    return prompts


async def run_level(server: Server, sc: dict, point: dict, rep: int, seed: int, prompts: list[list[dict]],
                    out_f, timeout_s: float, user_queues: list[list[int]] | None = None) -> list[dict]:
    out_cfg, samp = sc.get("output", {}), sc.get("sampling", {})
    level_name = f"{point['mode']}:{point['value']}"
    rng = random.Random(seed)
    records: list[dict] = []

    def body(i: int) -> dict:
        return chat_body(server.model, prompts[i], out_cfg.get("max_tokens", 256), samp.get("temperature", 0.6),
                         seed=rng.randrange(2**31) if samp.get("seeded", True) else None,
                         ignore_eos=out_cfg.get("ignore_eos", False), thinking=sc.get("thinking", False),
                         top_p=samp.get("top_p"))

    async def one(i: int, user: int | None, scheduled: float | None) -> None:
        rec = RequestRecord(id=f"{sc['name']}|{level_name}|r{rep}|{i}", scenario=sc["name"], level=level_name,
                            repetition=rep, user=user, scheduled_t=scheduled)
        await stream_request(server.s, server.base, server.headers, body(i), rec, timeout_s)
        d = json.loads(rec.to_json())
        records.append(d)
        out_f.write(rec.to_json() + "\n")
        out_f.flush()

    t0 = time.monotonic()
    if point["mode"] == "closed":
        cl = ClosedLoop(users=point["value"], requests=len(prompts), stagger_s=sc["arrivals"].get("stagger_s", 0.0),
                        think_s=sc["arrivals"].get("think_s", 0.0), think_exp=sc["arrivals"].get("think_exp", False),
                        seed=seed)
        queue = list(range(len(prompts)))
        thinks = cl.think_times()

        async def user(u: int, offset: float) -> None:
            await asyncio.sleep(offset)
            own = user_queues[u] if user_queues is not None else queue  # agent sessions: each user its own turns
            while own:
                i = own.pop(0)
                await one(i, u, None)
                th = next(thinks)
                if th:
                    await asyncio.sleep(th)

        await asyncio.gather(*[user(u, off) for u, off in enumerate(cl.start_offsets())])
    elif point["mode"] == "poisson":
        times = Poisson(rate_rps=point["value"], requests=len(prompts), seed=seed).arrival_times()
        tasks = []
        for i, at in enumerate(times):
            delay = at - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i, None, at)))
        await asyncio.gather(*tasks)
    else:
        raise ValueError(point["mode"])
    return records


def load_points(sc: dict) -> list[dict]:
    a = sc["arrivals"]
    if a["mode"] == "closed":
        return [{"mode": "closed", "value": int(n)} for n in a["levels"]]
    if a["mode"] == "poisson":
        return [{"mode": "poisson", "value": float(r)} for r in a["rates"]]
    raise ValueError(a["mode"])


def n_requests(sc: dict, point: dict) -> int:
    a = sc["arrivals"]
    if sc.get("agent"):  # one warm turn plus `turns` measured turns per user
        return point["value"] * (int(sc["agent"]["turns"]) + 1)
    if point["mode"] == "closed":
        return max(int(a.get("requests_per_user", 4)) * point["value"], int(a.get("min_requests", 16)))
    return int(a.get("requests", 64))


def expand_probe(sc: dict) -> list[dict]:
    """Prefill and decode probes are expressed as chat scenarios with fixed shapes."""
    if sc.get("kind") == "prefill":
        out = []
        for L in sc["lengths"]:
            out.append({**sc, "kind": "chat", "name": f"{sc['name']}-{L}",
                        "prompt": {**sc.get("prompt", {}), "tokens": L},
                        "output": {"max_tokens": 1, "ignore_eos": True},
                        "arrivals": {"mode": "closed", "levels": [1], "requests_per_user": sc.get("requests", 5),
                                     "min_requests": sc.get("requests", 5)}})
        return out
    if sc.get("kind") == "decode":
        return [{**sc, "kind": "chat", "prompt": {**sc.get("prompt", {}), "tokens": sc.get("prompt_tokens", 128)},
                 "output": {"max_tokens": sc.get("output_tokens", 512), "ignore_eos": True}}]
    if sc.get("kind") == "agent":
        return [{**sc, "kind": "chat", "prompt": {**sc.get("prompt", {}), "template": sc.get("prompt", {}).get(
            "template", "agent")}, "output": {"max_tokens": sc.get("output_tokens", 400), "ignore_eos": True}}]
    return [sc]


async def main_async(args) -> int:
    import aiohttp

    wl = yaml.safe_load(Path(args.workload).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_seed = int(wl.get("seed", 20261006))
    reps = int(args.repetitions or wl.get("repetitions", 3))
    corpora_file = Path(os.environ.get("HOB_CORPORA_FILE", ROOT / "experiments" / "corpora.yaml"))
    corpora_cfg = yaml.safe_load(corpora_file.read_text())
    cache = Path(os.environ.get("HOB_CACHE", ROOT / ".cache" / "corpora"))
    needed = {expand_probe(s)[0]["prompt"].get("corpus", "code") for s in wl["scenarios"]}
    corpora = {}
    for name in needed:
        c = corpora_cfg[name]
        corpora[name] = load_corpus(CorpusSpec(name=c["name"], urls=tuple(c["urls"]), text_sha256=c["text_sha256"],
                                               kind=c["kind"], suffix=c.get("suffix", ".py")), cache)
    run = {"workload_file": str(args.workload), "workload": wl, "base_seed": base_seed, "repetitions": reps,
           "harness_git": git_rev(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "status": "running", "server": {"base_url": args.base_url, "model": args.model}}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    scenarios = [x for s in wl["scenarios"] for x in expand_probe(s)]
    if args.only:
        scenarios = [s for s in scenarios if s["name"] in args.only.split(",")]
    # uvicorn closes idle keep-alive connections after 5 s; closing them here after 2 s means a request never goes out
    # on a socket the server is closing, which caused the client-side "Server disconnected" errors of 2026-10-06
    conn = aiohttp.TCPConnector(limit=0, keepalive_timeout=2.0)
    async with aiohttp.ClientSession(connector=conn) as session:
        server = Server(session, args.base_url, args.model, os.environ.get(args.api_key_env))
        await server.wait_ready(args.ready_timeout)
        with open(out / "requests.jsonl", "a") as rf, open(out / "levels.jsonl", "a") as lf, \
                open(out / "server_metrics.jsonl", "a") as mf:
            for rep in range(reps):
                for si, sc in enumerate(scenarios):
                    stop_rest = False
                    slo_fails = 0
                    for pi, point in enumerate(load_points(sc)):
                        if stop_rest:
                            break
                        seed = base_seed + 1_000_000 * rep + 10_000 * si + pi
                        n = n_requests(sc, point)
                        if sc.get("agent"):
                            prompts = await build_agent_prompts(server, sc, corpora, point["value"], seed)
                        else:
                            prompts = await build_prompts(server, sc, corpora, n, seed)
                        if sc.get("reset_prefix_cache", True):
                            await server.reset_prefix_cache()
                        warm = None
                        if sc["prompt"].get("shared_prefix_tokens"):
                            # one request puts the shared prefix in the cache; recorded, excluded from the level
                            wrec = RequestRecord(id=f"{sc['name']}|warm|r{rep}|{pi}", scenario=sc["name"],
                                                 level="warm", repetition=rep, user=None, scheduled_t=None)
                            await stream_request(session, server.base, server.headers, chat_body(
                                server.model, prompts[-1], 8, 0.0, seed, False, sc.get("thinking", False)),
                                wrec, args.timeout)
                            rf.write(wrec.to_json() + "\n")
                            warm = wrec.id
                        queues = None
                        if sc.get("agent"):
                            # every user opens its session (turn 0, the cache write); recorded, excluded from the level
                            stride = int(sc["agent"]["turns"]) + 1
                            wrecs = []
                            for u in range(point["value"]):
                                wrec = RequestRecord(id=f"{sc['name']}|warm|r{rep}|{pi}|u{u}", scenario=sc["name"],
                                                     level="warm", repetition=rep, user=u, scheduled_t=None)
                                wrecs.append(wrec)
                            await asyncio.gather(*[stream_request(session, server.base, server.headers, chat_body(
                                server.model, prompts[u * stride], 8, 0.0, seed, False, sc.get("thinking", False)),
                                w, args.timeout) for u, w in enumerate(wrecs)])
                            for w in wrecs:
                                rf.write(w.to_json() + "\n")
                            warm = ",".join(w.id for w in wrecs)
                            queues = [[u * stride + t for t in range(1, stride)] for u in range(point["value"])]
                        m0 = await server.metrics()
                        t_start = time.time()
                        recs = await run_level(server, sc, point, rep, seed, prompts, rf, args.timeout, queues)
                        m1 = await server.metrics()
                        summ = level_summary(recs, warmup_s=float(sc.get("warmup_s", 0.0)))
                        mf.write(json.dumps({"scenario": sc["name"], "point": point, "rep": rep, "when": "before",
                                             "t_wall": t_start, **m0}) + "\n")
                        mf.write(json.dumps({"scenario": sc["name"], "point": point, "rep": rep, "when": "after",
                                             "t_wall": time.time(), **m1}) + "\n")
                        lf.write(json.dumps({"scenario": sc["name"], "point": point, "rep": rep, "seed": seed,
                                             "n_planned": n, "t_start_wall": t_start, "t_end_wall": time.time(),
                                             "warm_request": warm, "summary": summ}) + "\n")
                        lf.flush()
                        mf.flush()
                        ttft95 = (summ.get("ttft_s") or {}).get("p95") or 0
                        print(json.dumps({"scenario": sc["name"], "point": point, "rep": rep, "ok": summ.get("ok"),
                                          "err": summ.get("errors"), "ttft_p95": ttft95,
                                          "tpot_p50": (summ.get("tpot_s") or {}).get("p50")}), flush=True)
                        errors = summ.get("errors", 0)
                        err_rate = errors / max(1, summ.get("n", 1))
                        # a single error among few requests is not saturation: require stop_min_errors errors too
                        # (added after the 2026-10-06 run, where one disconnect among 8 requests ended a sweep)
                        if (errors >= int(wl.get("stop_min_errors", 2))
                                and err_rate > float(wl.get("stop_error_rate", 0.05))) or ttft95 > float(
                                wl.get("stop_ttft_p95_s", 120)):
                            print(f"saturated at {point}: stopping this scenario's sweep", flush=True)
                            stop_rest = True
                        # per-stream SLO (E13): stop after `stop_after_slo_failures` consecutive levels whose median
                        # per-request decode rate is below `slo_stream_tok_s`; every level run is kept
                        slo = wl.get("slo_stream_tok_s")
                        if slo is not None:
                            med = (summ.get("decode_tok_s") or {}).get("p50")
                            slo_fails = slo_fails + 1 if (med is None or med < float(slo)) else 0
                            if slo_fails >= int(wl.get("stop_after_slo_failures", 2)):
                                print(f"per-stream SLO failed {slo_fails}x at {point}: stopping this sweep", flush=True)
                                stop_rest = True
    run.update(status="complete", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "run.json").write_text(json.dumps(run, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="controlled vLLM load generator")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("SERVED", "deepseek-v4.1-flash"))
    ap.add_argument("--api-key-env", default="API_KEY")
    ap.add_argument("--repetitions", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--ready-timeout", type=float, default=3600.0)
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
