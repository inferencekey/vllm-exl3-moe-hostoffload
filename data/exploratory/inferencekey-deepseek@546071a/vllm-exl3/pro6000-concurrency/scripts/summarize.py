#!/usr/bin/env python3
"""Local: turn results/scenario_{A,B}.json and results/mem_samples.csv into the README tables and the usable-N answers."""
import csv, json, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
BARS = {"A": {"strict": (15, 5), "loose": (10, 10)}, "B": {"strict": (15, 15), "loose": (10, 30)}}


def f(x, nd=1):
    return "" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}")


def mem_by_level():
    out = {}
    path = os.path.join(R, "mem_samples.csv")
    if not os.path.exists(path):
        return out
    for row in csv.DictReader(open(path)):
        key = (row["scenario"], row["N"])
        o = out.setdefault(key, {"vram0": 0, "vram1": 0, "ram": 0, "anon": 0, "kv": 0, "n": 0})
        num = lambda k: float(row[k]) if row[k] not in ("", None) else 0.0
        o["vram0"] = max(o["vram0"], num("gpu0_vram_mib")); o["vram1"] = max(o["vram1"], num("gpu1_vram_mib"))
        o["ram"] = max(o["ram"], num("ram_current_gib")); o["anon"] = max(o["anon"], num("ram_anon_gib"))
        o["kv"] = max(o["kv"], num("kv_usage_pct")); o["n"] += 1
    return out


mem = mem_by_level()
for sc in ("A", "B"):
    p = os.path.join(R, f"scenario_{sc}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    print(f"\n### Scenario {sc}\n")
    if sc == "B" and d.get("prefix_warm_request"):
        w = d["prefix_warm_request"]
        print(f"Prefix warm-up request (not in the table): prompt {w['prompt_tokens']} tokens, TTFT {w['ttft_s']} s, cached {w['cached_tokens']}.\n")
    print("| N | requests (ok/err) | prompt tok (mean) | cached tok (mean) | TTFT p50 s | TTFT p95 s | decode p50 tok/s | decode p5 tok/s | aggregate out tok/s | out tok/req | running max | waiting max | KV use max % | prefix hit % | DSpark accept len | peak VRAM GPU0/1 MiB | peak cgroup RAM GiB |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    best = {"strict": None, "loose": None}
    for L in d["levels"]:
        m = mem.get((sc, str(L["N"])), {})
        print(f"| {L['N']} | {L['ok']}/{L['errors']} | {f(L['mean_prompt_tokens'],0)} | {f(L['mean_cached_tokens'],0)} | {f(L['ttft_p50_s'],2)} | {f(L['ttft_p95_s'],2)} | "
              f"{f(L['decode_p50_tok_s'])} | {f(L['decode_p5_tok_s'])} | {f(L['aggregate_out_tok_s'])} | {f(L['mean_completion_tokens'],0)} | "
              f"{f(L['server_running_max'],0)} | {f(L['server_waiting_max'],0)} | {f(L['server_kv_usage_max_pct'])} | {f(L['prefix_cache_hit_rate_pct'])} | "
              f"{f(L['spec_mean_accept_len'],2)} | {f(m.get('vram0'),0)} / {f(m.get('vram1'),0)} | {f(m.get('ram'))} |")
        for bar, (p5min, ttftmax) in BARS[sc].items():
            ok = (L["errors"] == 0 and L["decode_p5_tok_s"] is not None and L["decode_p5_tok_s"] >= p5min
                  and L["ttft_p95_s"] is not None and L["ttft_p95_s"] <= ttftmax)
            if ok:
                best[bar] = L["N"]
    for bar, (p5min, ttftmax) in BARS[sc].items():
        print(f"\n{sc} {bar} (p5 >= {p5min} tok/s, TTFT p95 <= {ttftmax} s): highest passing N = {best[bar]}")
