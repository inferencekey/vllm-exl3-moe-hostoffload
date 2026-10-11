#!/usr/bin/env python3
"""Pod side: every 5 s append one CSV row with VRAM, GPU util, cgroup RAM and vLLM KV usage / queue depth.
The current scenario and N are derived from the run logs (run_all.log markers, finished levels in scenario_X.log)."""
import csv, json, os, subprocess, time, urllib.request
KEY = open("/root/.apikey").read().strip()
OUT = os.environ.get("OUT", "/root/results/mem_samples.csv")
LEVELS = [int(x) for x in os.environ.get("LEVELS", "1 2 4 8 16 32 64").split()]
WARM_LEVELS = [1, 8, 64]

def rd(p):
    try: return open(p).read()
    except Exception: return ""

def phase():
    run = rd("/root/logs/run_all.log")
    if "ALL DONE" in run: return "done", ""
    if "A DONE" in run: sc, levels = "B", LEVELS
    elif "WARMUP DONE" in run: sc, levels = "A", LEVELS
    elif "SERVER READY" in run: sc, levels = "warmup", WARM_LEVELS
    else: return "startup", ""
    log = rd(f"/root/logs/{'warmup' if sc == 'warmup' else 'scenario_' + sc}.log")
    done = sum(1 for l in log.splitlines() if l.startswith('{"N"'))
    return sc, (levels[done] if done < len(levels) else "")

def metrics():
    try:
        r = urllib.request.Request("http://127.0.0.1:8000/metrics", headers={"Authorization": "Bearer " + KEY})
        t = urllib.request.urlopen(r, timeout=4).read().decode()
    except Exception:
        return {}
    m = {}
    for l in t.splitlines():
        if l.startswith("vllm:"):
            try:
                n, v = l.rsplit(" ", 1); n = n.split("{")[0]; m[n] = m.get(n, 0) + float(v)
            except ValueError: pass
    return m

cols = ["ts", "scenario", "N", "gpu0_vram_mib", "gpu1_vram_mib", "gpu0_util", "gpu1_util",
        "ram_current_gib", "ram_anon_gib", "ram_shmem_gib", "ram_file_gib", "kv_usage_pct", "running", "waiting"]
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(cols)
    while True:
        sc, n = phase()
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip().splitlines()
        g = [[x.strip() for x in l.split(",")] for l in g] + [["", ""]] * 2
        st = dict(l.split() for l in rd("/sys/fs/cgroup/memory.stat").splitlines() if len(l.split()) == 2)
        gib = lambda b: round(int(b) / 2**30, 2) if b else ""
        m = metrics()
        kv = m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc"))
        w.writerow([time.strftime("%H:%M:%S"), sc, n, g[0][0], g[1][0], g[0][1], g[1][1],
                    gib(rd("/sys/fs/cgroup/memory.current").strip()), gib(st.get("anon")), gib(st.get("shmem")), gib(st.get("file")),
                    round(100 * kv, 2) if kv is not None else "", m.get("vllm:num_requests_running", ""), m.get("vllm:num_requests_waiting", "")])
        f.flush()
        if sc == "done": break
        time.sleep(5)
