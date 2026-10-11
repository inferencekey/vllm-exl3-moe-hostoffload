#!/usr/bin/env python3
# Pod side (stdlib only): measurements against the local vLLM server. Writes /root/results/bench.json.
import json, os, time, glob, urllib.request, threading, re, subprocess
BASE=os.environ.get("BASE","http://127.0.0.1:8000"); KEY=os.environ["API_KEY"]; MODEL=os.environ.get("SERVED","deepseek-v4.1-flash")
OUT=os.environ.get("OUT","/root/results/bench.json"); os.makedirs(os.path.dirname(OUT),exist_ok=True)
H={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"}
NOTHINK={"thinking":False}
def post(path, body, timeout=1800):
    r=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers=H,method="POST")
    return urllib.request.urlopen(r,timeout=timeout)
def stream_chat(messages, max_tokens, think=False, ignore_eos=False, extra=None):
    body={"model":MODEL,"messages":messages,"max_tokens":max_tokens,"temperature":0,"stream":True,
          "stream_options":{"include_usage":True}}
    if not think: body["chat_template_kwargs"]=NOTHINK
    if ignore_eos: body["ignore_eos"]=True
    if extra: body.update(extra)
    t0=time.time(); tf=None; text=""; reason=""; usage=None
    with post("/v1/chat/completions",body) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:"): continue
            p=line[5:].strip()
            if p=="[DONE]": break
            d=json.loads(p)
            if d.get("usage"): usage=d["usage"]
            for c in d.get("choices",[]):
                dl=c.get("delta",{})
                piece=(dl.get("content") or "")+(dl.get("reasoning_content") or dl.get("reasoning") or "")
                if piece and tf is None: tf=time.time()
                text+=dl.get("content") or ""; reason+=dl.get("reasoning_content") or dl.get("reasoning") or ""
    t1=time.time(); ct=(usage or {}).get("completion_tokens",0)
    dec=(ct-1)/(t1-tf) if tf and ct>1 and t1>tf else None
    return {"ttft_s":round(tf-t0,3) if tf else None,"total_s":round(t1-t0,3),"completion_tokens":ct,
            "prompt_tokens":(usage or {}).get("prompt_tokens"),"cached_tokens":((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "decode_tok_s":round(dec,2) if dec else None,"text":text,"reasoning":reason}
def metrics():
    try:
        r=urllib.request.Request(BASE+"/metrics",headers=H); m=urllib.request.urlopen(r,timeout=30).read().decode()
        out={}
        for l in m.splitlines():
            if l.startswith("#") or "_created" in l.split()[0]: continue
            for k in ("prefix_cache_hits","prefix_cache_queries","spec_decode_num_accepted_tokens","spec_decode_num_draft_tokens","spec_decode_num_drafts"):
                if l.startswith("vllm:"+k): out[k]=out.get(k,0)+float(l.split()[-1])
        return out
    except Exception as e: return {"err":repr(e)}
# code corpus from the installed vLLM sources (distinct slices for each prompt so nothing is cached by accident)
files=sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/**/*.py",recursive=True))
corpus=""
for f in files:
    try: corpus+=f"\n# ==== {f[40:]} ====\n"+open(f).read()
    except Exception: pass
def code_slice(start_char, ntok):  # ~3.2 chars per token for Python
    return corpus[start_char:start_char+int(ntok*3.2)]
R={"started":time.strftime("%F %T")}
def log(k,v):
    R[k]=v; json.dump(R,open(OUT,"w"),indent=1); print(k, json.dumps({x:y for x,y in v.items() if x not in("text","reasoning")} if isinstance(v,dict) else v)[:600], flush=True)
# warmup (like the recipe's): compile remaining kernels before measured requests
w0=time.time()
for n,mt in ((400,24),(1600,24)):
    post("/v1/completions",{"model":MODEL,"prompt":("warm up the kernels with some text. "*n).strip(),"max_tokens":mt,"temperature":0}).read()
ths=[threading.Thread(target=lambda: post("/v1/completions",{"model":MODEL,"prompt":("warm up the kernels with some text. "*300),"max_tokens":48}).read()) for _ in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]
log("warmup",{"s":round(time.time()-w0,1)})
# 1. smoke
log("smoke",stream_chat([{"role":"user","content":"What is the capital of Iceland? Answer in one sentence."}],60))
# 2. decode 120 and 480 tokens, short prompt, 3 fresh prompts for 480
q=["Write a detailed essay about the history of the Roman aqueducts.","Explain in depth how TCP congestion control works, covering slow start, AIMD and BBR.","Describe the life cycle of a star from nebula to remnant in detail."]
log("decode_120",stream_chat([{"role":"user","content":q[0]}],120,ignore_eos=True))
for i,p in enumerate(q):
    log(f"decode_480_{i}",stream_chat([{"role":"user","content":p}],480,ignore_eos=True))
# 3. TTFT ~4K and ~16K code prompts (fresh content)
Q1="\n\nQuestion: summarize in 3 bullet points what the code above does."
log("ttft_4k",stream_chat([{"role":"user","content":code_slice(0,4000)+Q1}],32))
long16=code_slice(200000,16000)
m0=metrics()
log("ttft_16k",stream_chat([{"role":"user","content":long16+Q1}],32))
m1=metrics()
# 4. same 16K prefix with a new question appended
log("ttft_16k_second_turn",stream_chat([{"role":"user","content":long16+Q1},{"role":"assistant","content":R["ttft_16k"]["text"]},
                                      {"role":"user","content":"Now list every class name defined in that code."}],32))
m2=metrics(); log("prefix_metrics",{"before":m0,"after_first":m1,"after_second":m2})
# 5. 4 concurrent requests
res=[None]*4
def worker(i): res[i]=stream_chat([{"role":"user","content":q[i%3]+f" (variant {i})"}],256,ignore_eos=True)
t0=time.time(); ths=[threading.Thread(target=worker,args=(i,)) for i in range(4)]; [t.start() for t in ths]; [t.join() for t in ths]
wall=time.time()-t0; tot=sum(r["completion_tokens"] for r in res)
log("concurrent_4",{"wall_s":round(wall,2),"total_completion_tokens":tot,"aggregate_tok_s":round(tot/wall,2),
                   "per_stream_decode":[r["decode_tok_s"] for r in res],"ttft":[r["ttft_s"] for r in res]})
# 6. tool call
body={"model":MODEL,"messages":[{"role":"user","content":"What's the weather in Reykjavik right now?"}],"max_tokens":300,"temperature":0,
      "chat_template_kwargs":NOTHINK,"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}}]}
d=json.loads(post("/v1/chat/completions",body).read()); log("tool_call",{"message":d["choices"][0]["message"],"finish":d["choices"][0]["finish_reason"]})
# 7. coding question (thinking on, the model's default)
log("coding",stream_chat([{"role":"user","content":"Write a Python function `merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]` that merges overlapping and touching closed intervals, with type hints, a docstring, edge cases handled, and 5 pytest tests. Then state its time complexity."}],4000,think=True))
log("spec_metrics_end",metrics())
smi=subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout
def rd(p):
    try: return open(p).read().strip()
    except Exception: return None
log("system",{"nvidia_smi":smi,"cgroup_memory_max":rd("/sys/fs/cgroup/memory.max") or rd("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
             "cgroup_memory_current":rd("/sys/fs/cgroup/memory.current") or rd("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
             "cgroup_memory_peak":rd("/sys/fs/cgroup/memory.peak")})
print("BENCH DONE", flush=True)
