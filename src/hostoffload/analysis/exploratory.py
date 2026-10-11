"""Readers for the exploratory (historical, single-run) data. Everything returned here is EXPLORATORY.

The original load generator stored per-request start/end wall times, TTFT and a per-request decode rate computed
from per-chunk times; we keep its definitions when reproducing its tables, and add what it did not report
(first-wave split, request-level bootstrap intervals, cost) with our own code.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import stats

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "data" / "exploratory" / "inferencekey-deepseek@546071a" / "vllm-exl3"
SWEEP = EXP / "pro6000-concurrency"
FIRST = EXP / "a100"


def sweep_file(scenario: str) -> Path:
    return SWEEP / "results" / f"scenario_{scenario}.json"


def load_sweep(scenario: str) -> dict:
    return json.loads(sweep_file(scenario).read_text())


def level_rows(scenario: str) -> list[dict]:
    """One row per concurrency level: the summary the original tool wrote, plus first-wave statistics."""
    d = load_sweep(scenario)
    rows = []
    for lv in d["levels"]:
        n = lv["N"]
        reqs = [r for r in d["requests"][str(n)] if r.get("error") is None]
        reqs.sort(key=lambda r: r["start"])
        first = reqs[:n]
        later = reqs[n:]
        ttft_later = [r["ttft_s"] for r in later if r.get("ttft_s") is not None]
        dec = [r["decode_tok_s"] for r in reqs if r.get("decode_tok_s") is not None and r["completion_tokens"] >= 32]
        row = dict(lv)
        row["ttft_p95_after_first_wave_s"] = stats.percentile(ttft_later, 95, "nearest") if ttft_later else None
        row["n_after_first_wave"] = len(later)
        row["ttft_first_wave_max_s"] = max(r["ttft_s"] for r in first) if first else None
        ci = stats.bootstrap_ci(dec, lambda v: stats.percentile(v, 50), n_boot=2000, seed=n)
        row["decode_p50_ci95"] = ci[1:] if ci else None
        row["tpot_p95_ms_from_decode_p5"] = (1000.0 / lv["decode_p5_tok_s"]) if lv.get("decode_p5_tok_s") else None
        rows.append(row)
    return rows


def per_request(scenario: str) -> dict[int, list[dict]]:
    d = load_sweep(scenario)
    return {int(k): v for k, v in d["requests"].items()}


def mem_samples() -> list[dict]:
    path = SWEEP / "results" / "mem_samples.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("ts", "scenario"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = None
    return rows


def mem_peaks_by_level() -> list[dict]:
    out: dict[tuple[str, int], dict] = {}
    for r in mem_samples():
        if r["scenario"] not in ("A", "B") or r["N"] is None:  # rows between levels have no N
            continue
        key = (r["scenario"], int(r["N"]))
        o = out.setdefault(key, {"scenario": key[0], "N": key[1], "vram_gpu0_mib": 0, "vram_gpu1_mib": 0,
                                 "ram_current_gib": 0, "ram_anon_gib": 0, "ram_shmem_gib": 0, "ram_file_gib": 0,
                                 "kv_usage_pct": 0, "n_samples": 0})
        o["vram_gpu0_mib"] = max(o["vram_gpu0_mib"], r["gpu0_vram_mib"] or 0)
        o["vram_gpu1_mib"] = max(o["vram_gpu1_mib"], r["gpu1_vram_mib"] or 0)
        for k in ("ram_current_gib", "ram_anon_gib", "ram_shmem_gib", "ram_file_gib", "kv_usage_pct"):
            o[k] = max(o[k], r[k] or 0)
        o["n_samples"] += 1
    return [out[k] for k in sorted(out)]


def first_run_bench() -> dict:
    return json.loads((FIRST / "results" / "agent30" / "bench.json").read_text())
