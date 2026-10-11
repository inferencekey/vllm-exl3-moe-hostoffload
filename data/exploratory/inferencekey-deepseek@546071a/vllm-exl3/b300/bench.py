"""Benchmark an OpenAI-compatible vLLM server. Run on the pod: python bench.py [base_url] > results.json
Measures decode tok/s (120/480 tokens), TTFT at ~4K/~16K code prompts, prefix-cache reuse on the 16K prompt,
aggregate tok/s at concurrency 4 and one coding answer."""
import json, sys, time, glob, os, re, threading, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "deepseek-v41-exl3"
OUT = {}

def post(path, body, timeout=1800):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)

def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    m = {}
    for line in t.splitlines():
        if line.startswith("#"): continue
        for key in ("prefix_cache_hits", "prefix_cache_queries", "prompt_tokens_total", "generation_tokens_total"):
            if re.match(r"vllm:%s(_total)?\{" % key, line) or line.startswith("vllm:%s{" % key):
                m[key] = m.get(key, 0) + float(line.rsplit(" ", 1)[1])
    return m

def chat(messages, max_tokens, force_len=True, temperature=0.0, extra=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    if force_len: body.update({"ignore_eos": True, "min_tokens": max_tokens})
    if extra: body.update(extra)
    t0 = time.time(); ttft = None; tlast = None; text = []; reasoning = []; usage = None; chunks = 0
    with post("/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            data = line[6:]
            if data == "[DONE]": break
            d = json.loads(data)
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    now = time.time(); chunks += 1
                    if ttft is None: ttft = now - t0
                    tlast = now
                    if delta.get("content"): text.append(delta["content"])
                    if delta.get("reasoning_content") or delta.get("reasoning"):
                        reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens", chunks)
    dec = (ct - 1) / (tlast - t0 - ttft) if ttft and tlast and ct > 1 and tlast > t0 + ttft else None
    return {"ttft_s": ttft, "total_s": total, "completion_tokens": ct, "decode_tok_s": dec,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": "".join(text), "reasoning": "".join(reasoning)}

def ntok(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}) as r: return json.loads(r.read())["count"]

def code_corpus(target, seed):
    import vllm
    files = sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "**/*.py"), recursive=True))
    files = [f for f in files if os.path.getsize(f) > 3000]
    files = files[seed::7]
    parts = []; est = 0
    for f in files:
        s = open(f, errors="ignore").read()
        parts.append(f"# FILE: {os.path.relpath(f, os.path.dirname(vllm.__file__))}\n{s}\n")
        est += len(s) / 3.2
        if est > target * 1.3: break
    text = "".join(parts)
    lo, hi = 0, len(text)
    while hi - lo > 2000:                      # trim to the target token count
        mid = (lo + hi) // 2
        if ntok(text[:mid]) > target: hi = mid
        else: lo = mid
    return text[:lo]

def short(q):
    return [{"role": "user", "content": q}]

def log(k, v):
    OUT[k] = v; print(k, json.dumps({x: y for x, y in v.items() if x not in ("text", "reasoning")} if isinstance(v, dict) else v), file=sys.stderr, flush=True)

# warmup
chat(short("Say hello."), 16, force_len=False)

r = chat(short("Explain how a hash map works, in detail."), 120); log("decode_120", r)
r = chat(short("Write a long essay about the history of the printing press."), 480); log("decode_480", r)

c4 = code_corpus(4000, 1)
r = chat([{"role": "user", "content": "Here is some Python code:\n\n" + c4 + "\n\nSummarize what the first file does in two sentences."}], 64, force_len=False)
log("ttft_4k", r)

c16 = code_corpus(16000, 3)
m0 = metrics()
p16 = "Here is some Python code:\n\n" + c16 + "\n\n"
r = chat([{"role": "user", "content": p16 + "Which file defines the most classes? Answer briefly."}], 64, force_len=False)
log("ttft_16k_first", r)
m1 = metrics()
r = chat([{"role": "user", "content": p16 + "Which file defines the most classes? Answer briefly."},
          {"role": "assistant", "content": r["text"] or "I am not sure."},
          {"role": "user", "content": "Now name one function in the last file and what it returns."}], 64, force_len=False)
m2 = metrics()
r["metrics_delta_first"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
r["metrics_delta_second"] = {k: m2.get(k, 0) - m1.get(k, 0) for k in m2}
log("ttft_16k_second_prefix", r)

res = [None] * 4
qs = ["Describe the water cycle.", "Explain TCP congestion control.", "Write a story about a lighthouse keeper.",
      "Explain how compilers optimize loops."]
def worker(i): res[i] = chat(short(qs[i]), 256)
t0 = time.time(); th = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
tot = sum(x["completion_tokens"] for x in res)
log("concurrent_4", {"wall_s": wall, "total_completion_tokens": tot, "aggregate_tok_s": tot / wall,
                     "per_request_decode_tok_s": [x["decode_tok_s"] for x in res], "ttft_s": [x["ttft_s"] for x in res]})

q = ("Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that merges "
     "overlapping closed intervals, with type hints, a docstring, and three pytest tests including edge cases. "
     "Then explain its time complexity.")
r = chat(short(q), 2500, force_len=False, temperature=0.6); log("coding_question", r)
OUT["coding_question"]["question"] = q

r = None
try:
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    with post("/v1/chat/completions", {"model": MODEL, "messages": short("What is the weather in Madrid right now?"),
              "tools": tools, "max_tokens": 800, "temperature": 0}) as resp:
        d = json.loads(resp.read()); OUT["tool_call"] = d["choices"][0]["message"]
except Exception as e:
    OUT["tool_call"] = {"error": repr(e)}
print("tool_call", json.dumps(OUT["tool_call"])[:500], file=sys.stderr)
print(json.dumps(OUT, indent=1))
