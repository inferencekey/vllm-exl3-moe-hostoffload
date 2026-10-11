"""Coding-agent case study: per-stream capacity at the 40 tok/s SLO, resources per level, and the seat/cost model.

Capacity C of a regime is the largest tested number of simultaneous streams N such that every tested level up to N
has a median per-request decode rate (1 / TPOT) of at least the SLO and no failed request. If the largest tested N
passes, C is a lower bound ("at least"). Two sources are kept apart: E13 (one boot, all regimes, the experiment
designed for this question) and a PRELIMINARY reading of E8 and E12, which ran on two different pods and were not
designed for it (their replies are 256 or 512 tokens, independent long prompts include their own prefill).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..cost import seats as S
from . import figures as F
from . import stats
from .metrics import level_summary

ROOT = Path(__file__).resolve().parents[3]
CTL = ROOT / "data" / "controlled"

# (key, label, source, run folder glob, scenario, dashed = independent contexts)
PRELIMINARY = [
    ("short", "short 4K, independent", "E8, pod 1", "E8-main/main/*", "A-code-4k", True),
    ("cached16k", "16K cached prefix", "E8, pod 1", "E8-main/main/*", "B-prefix-16k", False),
    ("cached128k", "128K cached prefix", "E12, pod 2", "E12-long-context/long/*", "B-prefix-128k", False),
    ("cached256k", "256K cached prefix", "E12, pod 2", "E12-long-context/long/*", "B-prefix-256k", False),
    ("fresh128k", "128K independent, fresh prefill", "E12, pod 2", "E12-long-context/long/*", "decode-128k", True),
    ("fresh256k", "256K independent, fresh prefill", "E12, pod 2", "E12-long-context/long/*", "decode-256k", True),
]
E13 = [
    ("short", "short 4K, independent", "A-short-4k", True),
    ("cached128k", "shared 128K prefix", "B-cached-128k", False),
    ("cached256k", "shared 256K prefix", "B-cached-256k", False),
    ("own128k", "own 128K context", "C-own-128k", True),
    ("own256k", "own 256K context", "C-own-256k", True),
    ("agent", "agent turns (observed ratios)", "D-agent-turns", True),
]


def _complete(run: Path) -> bool:
    try:
        return json.loads((run / "run.json").read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _levels(run: Path, scenario: str) -> list[dict]:
    """Per level of `scenario` in one run: summary, window and resources, ordered by N."""
    reqs: dict[str, list[dict]] = {}
    for line in (run / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["scenario"] == scenario and r["level"] != "warm":
            reqs.setdefault(r["level"], []).append(r)
    windows = {}
    for line in (run / "levels.jsonl").read_text().splitlines():
        lv = json.loads(line)
        if lv["scenario"] == scenario:
            windows[f"{lv['point']['mode']}:{lv['point']['value']}"] = (lv["t_start_wall"], lv["t_end_wall"])
    samples = _samples(run)
    metrics = _server_metrics(run, scenario)
    out = []
    for level, rs in reqs.items():
        n = int(level.split(":")[1])
        s = level_summary(rs)  # per-request percentiles pooled over repetitions
        # windowed aggregate rate per repetition, then averaged: pooling would span two repetitions' timelines
        by_rep: dict[int, list[dict]] = {}
        for x in rs:
            by_rep.setdefault(x["repetition"], []).append(x)
        aggs = [level_summary(v).get("aggregate_out_tok_s_window") for v in by_rep.values()]
        aggs = [a for a in aggs if a]
        s["aggregate_out_tok_s_window"] = stats.mean(aggs) if aggs else None
        s["repetitions"] = len(by_rep)
        # per-stream generation rate: (tokens - 1) / (end of stream - first token). It equals 1 / TPOT when tokens
        # stream evenly, and stays valid when the server delivers most of a reply in one late chunk, which E13 saw on
        # 6 of 366 agent-session requests (last_token_mono then misses the bulk of the reply and TPOT is meaningless)
        rates, bursts = [], 0
        for x in rs:
            ft, end, ct = x.get("first_token_mono"), x.get("end_mono"), int(x.get("completion_tokens") or 0)
            if x.get("error") is None and ft is not None and end is not None and ct >= 16 and end > ft:
                rates.append((ct - 1) / (end - ft))
                if ct / max(1, len(x.get("chunks") or [])) > 12:
                    bursts += 1
        s["stream_tok_s"] = stats.describe(rates) if rates else None
        s["bursts"] = bursts
        res = _resources(samples, *windows[level]) if level in windows else {}
        out.append({"N": n, "level": level, "summary": s, **res, **metrics.get(level, {})})
    return sorted(out, key=lambda x: x["N"])


def _samples(run: Path) -> list[dict]:
    f = run / "samples.jsonl"
    if not f.exists():
        return []
    return [r for r in (json.loads(x) for x in f.read_text().splitlines()) if r.get("type") == "sample"]


def _resources(samples: list[dict], t0: float, t1: float) -> dict:
    win = [s for s in samples if t0 <= s["t_wall"] <= t1]
    if not win:
        return {}
    gib = 2 ** 30
    gpu_mem = max(max(g["mem_used"] for g in s["gpus"]) for s in win) / gib
    rx = stats.mean([sum(g.get("pcie_rx_kBps", 0) for g in s["gpus"]) for s in win]) / 1e6
    host = max((s["cgroup"]["stat"].get("anon", 0) + s["cgroup"]["stat"].get("shmem", 0)) for s in win) / gib
    kv = max((s.get("vllm") or {}).get("gauges", {}).get("kv_usage", 0) for s in win)
    return {"gpu_mem_gib": gpu_mem, "pcie_rx_gbs": rx, "host_gib": host, "kv_usage_max": kv}


def _server_metrics(run: Path, scenario: str) -> dict:
    """Prefix-cache hit rate and preemptions per level, from the /metrics snapshots before and after it."""
    f = run / "server_metrics.jsonl"
    if not f.exists():
        return {}
    snaps: dict[str, dict] = {}
    for line in f.read_text().splitlines():
        m = json.loads(line)
        if m["scenario"] == scenario:
            snaps.setdefault(f"{m['point']['mode']}:{m['point']['value']}", {})[m["when"]] = m["counters"]
    out = {}
    for level, ba in snaps.items():
        if "before" in ba and "after" in ba:
            d = {k: ba["after"].get(k, 0) - ba["before"].get(k, 0) for k in ("prefix_hits", "prefix_queries",
                                                                              "preemptions")}
            out[level] = {"prefix_hit_rate": d["prefix_hits"] / d["prefix_queries"] if d["prefix_queries"] else None,
                          "preemptions": d["preemptions"]}
    return out


def capacity(levels: list[dict], slo: float) -> dict:
    """Largest N with every tested level up to N meeting the per-stream SLO and the error rule."""
    cap, at, first_fail = 0, None, None
    for lv in levels:
        s = lv["summary"]
        med = (s.get("stream_tok_s") or {}).get("p50")
        err = s.get("errors", 0)
        # same error rule as the harness stop rule: a level fails on at least 2 errors above 5 %, so one client-side
        # disconnect among a handful of requests (2026-10-06, since fixed) does not decide capacity
        ok = med is not None and med >= slo and not (err >= 2 and err / max(1, s.get("n", 1)) > 0.05)
        if ok and first_fail is None:
            cap, at = lv["N"], lv
        elif not ok and first_fail is None:
            first_fail = lv
    lower_bound = first_fail is None and bool(levels)
    return {"C": cap, "at": at, "first_fail": first_fail, "lower_bound": lower_bound}


MARGIN = 0.10  # overwritten from experiments/agent_workload.yaml (slo.provisioning_margin) by analyze()


def _regimes(slo: float) -> tuple[str, list[dict]]:
    """E13 regimes if a complete E13 run exists, else the preliminary reading of E8 and E12."""
    e13 = [r for r in sorted((CTL / "E13-coding-agent").glob("*/*")) if _complete(r)]
    out = []
    if e13:
        run = e13[-1]
        for key, label, scen, dashed in E13:
            lv = _levels(run, scen)
            if lv:
                out.append({"key": key, "label": label, "source": "E13", "run": run, "levels": lv,
                            "dashed": dashed, **capacity(lv, slo), "prov": capacity(lv, slo * (1 + MARGIN))})
        return "E13", out
    for key, label, src, glob, scen, dashed in PRELIMINARY:
        runs = [r for r in sorted(CTL.glob(glob)) if _complete(r)]
        if not runs:
            continue
        lv = _levels(runs[0], scen)
        if lv:
            out.append({"key": key, "label": label, "source": src, "run": runs[0], "levels": lv, "dashed": dashed,
                        **capacity(lv, slo), "prov": capacity(lv, slo * (1 + MARGIN))})
    return "preliminary", out


def _rate(lv: dict | None, p: str) -> float | None:
    return ((lv or {}).get("summary", {}).get("stream_tok_s") or {}).get(p) if lv else None


def analyze(reg, write_tex, table, notrun_block, latex_escape, fig_dir: Path) -> None:
    cfg = yaml.safe_load((ROOT / "experiments" / "agent_workload.yaml").read_text())
    pricing = yaml.safe_load((ROOT / "experiments" / "pricing.yaml").read_text())
    af, pf = ROOT / "experiments" / "agent_workload.yaml", ROOT / "experiments" / "pricing.yaml"
    slo = float(cfg["slo"]["stream_tok_s"])
    global MARGIN
    MARGIN = float(cfg["slo"].get("provisioning_margin", 0.0))
    A = cfg["assumptions"]
    mode, regimes = _regimes(slo)
    E = "controlled"

    # ---- capacity table and figure ----
    rows = []
    for r in regimes:
        srcs = [r["run"] / "requests.jsonl"]
        k = f"agent.{mode}.{r['key']}"
        reg.put(f"{k}.C", r["C"], srcs, E, "{:d}")
        reg.put(f"{k}.C_text", ("at least " if r["lower_bound"] else "") + str(r["C"]), srcs, E)
        pv = r["prov"]
        reg.put(f"{k}.Cprov", pv["C"], srcs, E, "{:d}")
        reg.put(f"{k}.Cprov_text", ("at least " if pv["lower_bound"] else "") + str(pv["C"]), srcs, E)
        at, ff = r["at"], r["first_fail"]
        for name, lv in (("at", at), ("fail", ff), ("prov", r["prov"]["at"])):
            if lv:
                s = lv["summary"]
                reg.put(f"{k}.{name}.N", lv["N"], srcs, E, "{:d}")
                for p in ("p5", "p50", "p95"):
                    if _rate(lv, p) is not None:
                        reg.put(f"{k}.{name}.rate_{p}", _rate(lv, p), srcs, E, "{:.0f}")
                if s.get("aggregate_out_tok_s_window"):
                    reg.put(f"{k}.{name}.agg", s["aggregate_out_tok_s_window"], srcs, E, "{:.0f}")
                if (s.get("ttft_s") or {}).get("p95") is not None:
                    reg.put(f"{k}.{name}.ttft_p95", s["ttft_s"]["p95"], srcs, E, "{:.2f}")
        rows.append([latex_escape(r["label"]), latex_escape(r["source"]),
                     ("$\\ge$" if r["lower_bound"] else "") + str(r["C"]),
                     ("$\\ge$" if r["prov"]["lower_bound"] else "") + str(r["prov"]["C"]),
                     *(f"{_rate(at, p):.0f}" if _rate(at, p) is not None else "-" for p in ("p5", "p50", "p95")),
                     f"{(at['summary'].get('ttft_s') or {}).get('p95', 0):.2f}" if at else "-",
                     f"{at['summary'].get('aggregate_out_tok_s_window') or 0:.0f}" if at else "-",
                     f"{ff['N']} ({_rate(ff, 'p50'):.0f})" if ff and _rate(ff, "p50") is not None else
                     (str(ff["N"]) if ff else "-")])
    cap_caption = ("Simultaneous streams at a median of at least 40 output tokens/s per stream. C is the largest N "
                   "whose level and every smaller tested level meet it (failing on 2 or more errors above 5\\%); "
                   "per-stream rates "
                   "(p5, p50, p95, tokens/s) and TTFT p95 (s) at C; the last column is the first failing N with its "
                   "median rate. C$_{+10\\%}$ is the same with a 10\\% margin (44 tokens/s), used for provisioning.")
    if mode == "preliminary":
        cap_caption = ("PRELIMINARY, from E8 and E12 (two pods, not designed for this question): " + cap_caption +
                       " Independent long rows include each request's own prefill; replies were 256 or 512 tokens.")
    write_tex("agent_capacity.tex", table(cap_caption, "tab:agent-capacity",
                                          ["regime", "source", "C", "C$_{+10\\%}$", "p5", "p50", "p95",
                                           "TTFT p95", "aggr.", "fails at"], rows, "llrrrrrrrr"))
    reg.artifact("agent_capacity.tex", E, [r["run"] / "requests.jsonl" for r in regimes], "data")
    _figure_rate(regimes, slo, mode, fig_dir / "agent_stream_rate.pdf")
    reg.artifact("figures/generated/agent_stream_rate.pdf", E, [r["run"] / "requests.jsonl" for r in regimes], "data")

    # resource extremes over all levels of a session (cited in the text)
    if regimes:
        allv = [lv for r in regimes for lv in r["levels"]]
        srcs = sorted({r["run"] / "samples.jsonl" for r in regimes})
        for k, fn, fmt in (("kv_max_pct", lambda: 100 * max(lv.get("kv_usage_max", 0) for lv in allv), "{:.0f}"),
                           ("preemptions", lambda: sum(lv.get("preemptions", 0) or 0 for lv in allv), "{:.0f}"),
                           ("gpu_mem_max", lambda: max(lv.get("gpu_mem_gib", 0) for lv in allv), "{:.1f}"),
                           ("gpu_mem_min", lambda: min(lv.get("gpu_mem_gib", 1e9) for lv in allv if "gpu_mem_gib" in lv), "{:.1f}"),
                           ("host_max", lambda: max(lv.get("host_gib", 0) for lv in allv), "{:.0f}"),
                           ("pcie_max", lambda: max(lv.get("pcie_rx_gbs", 0) for lv in allv), "{:.0f}"),
                           ("bursts", lambda: sum(lv["summary"].get("bursts", 0) for lv in allv), "{:d}"),
                           ("requests", lambda: sum(lv["summary"].get("n", 0) for lv in allv), "{:d}"),
                           ("errors", lambda: sum(lv["summary"].get("errors", 0) for lv in allv), "{:d}")):
            try:
                reg.put(f"agent.{mode}.all.{k}", fn(), srcs, E, fmt)
            except ValueError:
                pass
        own = [lv for r in regimes if r["key"].startswith("own") for lv in r["levels"]]
        if own:
            reg.put(f"agent.{mode}.own.kv_max_pct", 100 * max(lv.get("kv_usage_max", 0) for lv in own), srcs, E,
                    "{:.0f}", note="independent cached contexts (own-context regimes) only")
        for r in regimes:
            reg.put(f"agent.{mode}.{r['key']}.kv_max_pct", 100 * max(lv.get("kv_usage_max", 0) for lv in r["levels"]),
                    [r["run"] / "samples.jsonl"], E, "{:.0f}")

    # E13 per-level table with resources, or NOT RUN
    if mode == "E13":
        srows, rrows = [], []  # per-stream service and resources, two upright tables
        for r in regimes:
            for lv in r["levels"]:
                s = lv["summary"]
                tp = s.get("tpot_s") or {}
                tt = s.get("ttft_s") or {}
                key = [latex_escape(r["key"]), str(lv["N"])]
                srows.append([*key,
                              *(f"{_rate(lv, p):.0f}" if _rate(lv, p) is not None else "-" for p in ("p5", "p50", "p95")),
                              f"{1000 * (tp.get('p50') or 0):.1f}", f"{1000 * (tp.get('p95') or 0):.1f}",
                              f"{tt.get('p50') or 0:.2f}", f"{tt.get('p95') or 0:.2f}",
                              f"{100 * s.get('errors', 0) / max(1, s.get('n', 1)):.0f}",
                              f"{s.get('aggregate_out_tok_s_window') or 0:.0f}"])
                rrows.append([*key, f"{lv.get('gpu_mem_gib', 0):.1f}", f"{lv.get('host_gib', 0):.0f}",
                              f"{lv.get('pcie_rx_gbs', 0):.2f}", f"{100 * lv.get('kv_usage_max', 0):.0f}",
                              f"{100 * lv['prefix_hit_rate']:.0f}" if lv.get("prefix_hit_rate") is not None else "-",
                              f"{lv.get('preemptions', 0):.0f}", str(s.get("bursts", 0))])
        write_tex("agent_levels.tex", table(
            "Controlled: E13 per level, part 1 of 2, service per stream. Per-stream rate p5/p50/p95 (tokens/s, tokens "
            "over end of stream minus first token), TPOT p50/p95 (ms), TTFT p50/p95 (s), failed requests (\\%) and "
            "aggregate output (tokens/s). Resources at the same levels: Table~\\ref{tab:agent-resources}.",
            "tab:agent-levels",
            ["regime", "N", "rate p5", "p50", "p95", "TPOT p50", "p95", "TTFT p50", "p95", "err \\%", "aggr."],
            srows, "lrrrrrrrrrr") + table(
            "Controlled: E13 per level, part 2 of 2, resources. Peak GPU memory per GPU (GiB), peak host anon+shmem "
            "(GiB), mean PCIe RX summed over GPUs (GB/s), peak KV-cache use (\\%), prefix-cache hit rate (\\%), "
            "preemptions, and requests whose reply arrived in one late burst (their TPOT is invalid; the per-stream "
            "rate of Table~\\ref{tab:agent-levels} stays valid).", "tab:agent-resources",
            ["regime", "N", "GPU GiB", "host GiB", "PCIe GB/s", "KV \\%", "hit \\%", "preempt.", "bursts"],
            rrows, "lrrrrrrrr"))
    else:
        write_tex("agent_levels.tex", notrun_block("data/controlled/E13-coding-agent/",
                                                   "E13: interactive coding-agent workload, per level with resources"))

    # ---- seat model ----
    us = cfg["usage_sample"]
    tk = us["tokens"]
    prof = S.Profile(tk["output"], tk["cache_read"], tk["cache_write"], tk["input"], us["window_days"],
                     us["active_days"])
    reg.put("agent.sample.output_week", tk["output"], [af], "static", "{:,}")
    reg.put("agent.sample.cache_read_week", tk["cache_read"], [af], "static", "{:,}")
    reg.put("agent.sample.cache_write_week", tk["cache_write"], [af], "static", "{:,}")
    reg.put("agent.sample.input_week", tk["input"], [af], "static", "{:,}")
    reg.put("agent.sample.total_week", tk["total"], [af], "static", "{:,}")
    reg.put("agent.sample.active_days", us["active_days"], [af], "static", "{:d}")
    reg.put("agent.sample.sessions", us["sessions"], [af], "static", "{:d}")
    reg.put("agent.sample.output_share_pct", 100 * tk["output"] / tk["total"], [af], "static", "{:.1f}")
    reg.put("agent.ratio.read_per_output", prof.read_per_output, [af], "static", "{:.0f}")
    reg.put("agent.ratio.write_per_output", prof.write_per_output, [af], "static", "{:.1f}")
    shape = prof.turn_shape(A["output_tokens_per_turn"])
    reg.put("agent.turn.output", A["output_tokens_per_turn"], [af], "static", "{:d}")
    reg.put("agent.turn.context_read", shape["context_read"], [af], "static", "{:,.0f}")
    reg.put("agent.turn.new_tokens", shape["new_tokens"], [af], "static", "{:,.0f}")
    dev_month = prof.output_month(A["days_per_month"])
    reg.put("agent.dev.output_month", dev_month / 1e6, [af], "static", "{:.2f}", note="million output tokens")

    # prefill rate for the cache-write share of busy time: E12's fresh 128K prompt
    prefill_rate, pr_src = None, []
    for run in sorted((CTL / "E12-long-context").glob("*/*")):
        if _complete(run):
            lv = _levels(run, "prefill-long-131072")
            if lv and (lv[0]["summary"].get("ttft_s") or {}).get("p50"):
                prefill_rate = 131072 / lv[0]["summary"]["ttft_s"]["p50"]
                pr_src = [run / "requests.jsonl"]
    if prefill_rate is None:
        return
    reg.put("agent.prefill_tok_s", prefill_rate, pr_src, E, "{:,.0f}")
    busy = S.busy_seconds_week(prof, slo, prefill_rate)
    reg.put("agent.dev.busy_h_week", busy / 3600, [af] + pr_src, E, "{:.1f}")
    qs = {h: S.duty_cycle(prof, busy, h) for h in A["active_hours_per_active_day"]}
    for h, q in qs.items():
        reg.put(f"agent.q.h{h}", 100 * q, [af] + pr_src, E, "{:.1f}", note="percent of the active window")
    reg.put("agent.q.week", 100 * busy / (us["window_days"] * 86400), [af] + pr_src, E, "{:.1f}")
    qb = qs[A["baseline_active_hours"]]

    # ---- cost ----
    usd_h = pricing["configs"]["2x-rtx-pro-6000"]["usd_per_hour"]
    usd_eur = pricing["fx"]["usd_per_eur"]
    eur_h = usd_h / usd_eur
    pod_m = S.pod_eur_month(usd_h, usd_eur, A["month_hours"])
    seat_eur = us["subscription_eur_month"]
    reg.put("agent.pod.eur_h", eur_h, [pf], "static", "{:.2f}")
    reg.put("agent.pod.eur_month", pod_m, [pf, af], "static", "{:,.0f}")
    reg.put("agent.seat_eur", seat_eur, [af], "static", "{:d}")
    be = S.break_even_seats(pod_m, seat_eur)
    reg.put("agent.break_even_seats", be, [pf, af], "static", "{:.1f}")
    sched = S.pod_eur_month(usd_h, usd_eur, A["scheduled_hours_month"])
    reg.put("agent.pod.sched_hours", A["scheduled_hours_month"], [af], "static", "{:d}")
    reg.put("agent.pod.sched_eur_month", sched, [pf, af], "static", "{:,.0f}")
    reg.put("agent.break_even_seats_sched", S.break_even_seats(sched, seat_eur), [pf, af], "static", "{:.1f}")

    # seats by volume and by concurrency per regime (only regimes with a usable aggregate at C)
    srows, usable = [], []
    for r in regimes:
        at = r["prov"]["at"]
        agg = (at or {}).get("summary", {}).get("aggregate_out_tok_s_window") if at else None
        if not at or not agg:
            continue
        usable.append((r, agg))
        vol = [S.seats_by_volume(agg, A["month_hours"], u, dev_month) for u in A["utilization"]]
        conc = [S.max_seats_for_capacity(r["prov"]["C"], qb, e) for e in A["overflow_probability"]]
        k = f"agent.{mode}.{r['key']}"
        for u, v in zip(A["utilization"], vol):
            reg.put(f"{k}.seats_volume_u{int(100 * u)}", v, [r["run"] / "requests.jsonl", af], E, "{:.0f}")
        for e, c in zip(A["overflow_probability"], conc):
            reg.put(f"{k}.seats_conc_e{int(100 * e)}", c, [r["run"] / "requests.jsonl", af], E, "{:d}")
        for u, v in zip(A["utilization"], vol):  # how many times the volume capacity exceeds the 1% seat count
            reg.put(f"{k}.volume_over_conc_u{int(100 * u)}", v / conc[0], [r["run"] / "requests.jsonl", af], E,
                    "{:.0f}")
        reg.put(f"{k}.break_even_util", 100 * S.break_even_utilization(pod_m, seat_eur, agg, A["month_hours"],
                                                                         dev_month),
                [r["run"] / "requests.jsonl", af, pf], E, "{:.1f}")
        if r["key"] == "agent":
            _figure_two_capacities(vol, conc, A["utilization"], A["overflow_probability"], r["prov"]["C"],
                                   fig_dir / "agent_two_capacities.pdf")
            reg.artifact("figures/generated/agent_two_capacities.pdf", E, [r["run"] / "requests.jsonl", af], "data")
        srows.append([latex_escape(r["label"]), ("$\\ge$" if r["prov"]["lower_bound"] else "") + str(r["prov"]["C"]), f"{agg:.0f}",
                      *(f"{v:.0f}" for v in vol), *(str(c) for c in conc),
                      f"{100 * S.break_even_utilization(pod_m, seat_eur, agg, A['month_hours'], dev_month):.1f}"])
    pre = "PRELIMINARY (E8/E12): " if mode == "preliminary" else ""
    write_tex("agent_seats.tex", table(
        pre + "Two capacities of one pod, provisioned on C with a 10\\% margin (44 tokens/s). Volume: "
        "capacity-equivalent developer seats by observed monthly generation volume, at the aggregate output rate "
        "measured at that level and the stated utilization (these are NOT concurrent "
        "users). Concurrency: largest number of provisioned seats M with P(Binomial(M, q) > C) at most 1\\% or 5\\%, "
        "with q the duty cycle for an 8-hour active window. Last column: pod utilization at which the seats it "
        "serves would cost as much in subscriptions as the pod.", "tab:agent-seats",
        ["regime", "C", "aggr.", *(f"vol. {int(100 * u)}\\%" for u in A["utilization"]),
         *(f"conc. {int(100 * e)}\\%" for e in A["overflow_probability"]), "b.e. util. \\%"],
        srows, "lrr" + "r" * (len(A["utilization"]) + len(A["overflow_probability"]) + 1)))
    reg.artifact("agent_seats.tex", E, [af], "data")

    # duty-cycle sensitivity: max seats per pod by concurrency for each q and regime capacity
    qrows = []
    for h, q in qs.items():
        qrows.append([str(h), f"{100 * q:.1f}", *(str(S.max_seats_for_capacity(r["prov"]["C"], q, 0.01)) for r, _ in usable)])
    write_tex("agent_duty.tex", table(
        pre + "Duty-cycle sensitivity. q is one developer's generating time (output at 40 tokens/s plus prefill of "
        "new tokens, from the usage sample) over an active window of the given hours on each of the 5 active days; "
        "cells are the largest M with P(Binomial(M, q) > C) at most 1\\%.", "tab:agent-duty",
        ["active h/day", "q \\%", *(latex_escape(r["key"]) for r, _ in usable)], qrows,
        "rr" + "r" * len(usable)))
    reg.artifact("agent_duty.tex", E, [af], "data")

    # cost table at the requested seat counts: pods = max(volume, concurrency) at u = 50%, eps = 1%, q baseline
    usable_all = usable
    usable = sorted(((r, agg) for r, agg in usable if r["key"] in COST_KEYS), key=lambda x: COST_KEYS.index(x[0]["key"]))
    crow = []
    for m in A["seats"]:
        row = [str(m), f"{seat_eur * m:,.0f}"]
        for r, agg in usable:
            need = S.pods_needed(m, agg, r["prov"]["C"], qb, 0.01, A["month_hours"], 0.5, dev_month)
            pods = need["pods"]
            row.append(f"{pods} ({need['binding'][0]})" if pods else "-")
            if pods:
                reg.put(f"agent.{mode}.{r['key']}.cost.m{m}.pods", pods, [af, pf], E, "{:d}")
                reg.put(f"agent.{mode}.{r['key']}.cost.m{m}.saving", seat_eur * m - pods * pod_m, [af, pf], E, "{:,.0f}")
                reg.put(f"agent.{mode}.{r['key']}.cost.m{m}.saving_pct", 100 * (1 - pods * pod_m / (seat_eur * m)),
                        [af, pf], E, "{:.0f}")
                # extra monthly cost of self-hosting when it is above the subscriptions (printed as a positive amount)
                if pods * pod_m > seat_eur * m:
                    reg.put(f"agent.{mode}.{r['key']}.cost.m{m}.extra", pods * pod_m - seat_eur * m, [af, pf], E,
                            "{:,.0f}")
            row.append(f"{seat_eur * m - pods * pod_m:,.0f}" if pods else "-")
        crow.append(row)
    write_tex("agent_cost.tex", table(
        pre + "Subscriptions against self-hosted pods, infrastructure only (no engineering or operations). Pods per "
        "regime are the larger of the volume requirement (50\\% utilization) and the concurrency requirement "
        "(P(demand > pods x C) at most 1\\%, 8-hour window); v or c marks which binds. Savings are subscription "
        "cost minus pod cost per month, EUR; negative means self-hosting costs more.", "tab:agent-cost",
        ["seats", "subs. EUR", *[x for r, _ in usable for x in (f"pods {latex_escape(r['key'])}", "saving")]],
        crow, "rr" + "rr" * len(usable)))
    reg.artifact("agent_cost.tex", E, [af, pf], "data")

    orows = []
    for ops in A["ops_eur_month"]:
        orows.append([f"{ops:,}", f"{pod_m + ops:,.0f}", f"{S.break_even_seats(pod_m + ops, seat_eur):.1f}",
                      f"{sched + ops:,.0f}", f"{S.break_even_seats(sched + ops, seat_eur):.1f}"])
    write_tex("agent_ops.tex", table(
        "Operational overhead sensitivity for one pod: monthly cost and break-even subscription seats, 24/7 and "
        f"scheduled ({A['scheduled_hours_month']} h/month). Overhead values are assumptions, not measurements.",
        "tab:agent-ops", ["ops EUR/month", "24/7 EUR", "break-even", "scheduled EUR", "break-even"], orows, "rrrrr"))
    reg.artifact("agent_ops.tex", "static", [af, pf], "data")

    _figure_cost(usable, qb, dev_month, pod_m, seat_eur, A, mode, fig_dir / "agent_cost_seats.pdf",
                 [r["key"] for r, _ in usable_all])
    reg.artifact("figures/generated/agent_cost_seats.pdf", E, [af, pf], "data")


# regimes shown in the cost table and figure: the short baseline and the two bounds on a developer with an own
# repository (shared cached prefix: optimistic; independent with its own prefill: pessimistic) until E13 measures
# independent cached contexts (C) and agent turns (D)
COST_KEYS = ("cached128k", "short", "fresh128k", "own128k", "agent")
MARKERS = ["o", "s", "^", "D", "v", "P"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated slots 1-6, fixed order


def _figure_rate(regimes: list[dict], slo: float, mode: str, path: Path) -> None:
    fig, ax = F._new(3.4, 3.2)
    for i, r in enumerate(regimes):
        lv = [x for x in r["levels"] if _rate(x, "p50") is not None]
        if not lv:
            continue
        n = [x["N"] for x in lv]
        p50 = [_rate(x, "p50") for x in lv]
        p5 = [_rate(x, "p5") or _rate(x, "p50") for x in lv]
        p95 = [_rate(x, "p95") or _rate(x, "p50") for x in lv]
        ax.fill_between(n, p5, p95, color=COLORS[i], alpha=0.12, linewidth=0)
        ax.plot(n, p50, color=COLORS[i], linewidth=1.4, linestyle="--" if r["dashed"] else "-",
                marker=MARKERS[i], markersize=3.5, label=r["label"])
    ax.axhline(slo, color=F.INK2, linewidth=0.8, linestyle=":")
    ax.annotate(f"SLO {slo:.0f} tok/s", (1, slo), xytext=(2, -9), textcoords="offset points", fontsize=6.5,
                color=F.INK2, ha="left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ns = sorted({x["N"] for r in regimes for x in r["levels"]})
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 100, 200, 400]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("simultaneous active streams N", fontsize=8)
    ax.set_ylabel("output tok/s per stream (p50, p5-p95 band)", fontsize=7)
    ax.set_title(("PRELIMINARY: E8 and E12, two pods" if mode == "preliminary" else "E13, one boot"), fontsize=8,
                 color=F.INK)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    F.save(fig, path)


def _figure_cost(usable, q, dev_month, pod_m, seat_eur, A, mode, path: Path, order: list[str]) -> None:
    fig, ax = F._new(3.4, 2.6)
    ms = list(range(1, max(A["seats"]) + 1))
    ax.plot(ms, [seat_eur * m for m in ms], color=F.INK, linewidth=1.4, label=f"subscriptions ({seat_eur} EUR/seat)")
    drawn: list[list[float]] = []
    for i, (r, agg) in enumerate(usable):
        ys = []
        for m in ms:
            need = S.pods_needed(m, agg, r["prov"]["C"], q, 0.01, A["month_hours"], 0.5, dev_month)
            ys.append(need["pods"] * pod_m if need["pods"] else float("nan"))
        c = order.index(r["key"])  # same color as the regime in the rate figure
        same = any(ys == d for d in drawn)  # an identical step curve: draw it thinner on top, say so in the label
        drawn.append(ys)
        ax.step(ms, ys, where="post", color=COLORS[c % len(COLORS)], linewidth=0.8 if same else 1.6,
                linestyle=(0, (2, 2)) if same else ("--" if r["dashed"] else "-"),
                label=f"pods, {r['label']}" + (" (same steps)" if same else ""), zorder=3 if same else 2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("provisioned developer seats M", fontsize=8)
    ax.set_ylabel("EUR per month (infrastructure only)", fontsize=7)
    ax.set_title(("PRELIMINARY (E8/E12)" if mode == "preliminary" else "E13") + ": q at 8 h/day, overflow 1%",
                 fontsize=7, color=F.INK)
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    F.save(fig, path)


def _figure_two_capacities(vol, conc, utils, overflows, c: int, path: Path) -> None:
    """Developer seats of one pod for agent turns: by monthly volume at each utilization against by concurrency at
    each overflow probability. Two jobs, two colors, every bar labelled."""
    rows = [(f"by volume, {int(100 * u)}% util.", v, 0) for u, v in zip(utils, vol)]
    rows += [(f"by {c} streams, {int(100 * e)}% overflow", n, 1) for e, n in zip(overflows, conc)]
    fig, ax = F._new(3.4, 2.4)
    ys = list(range(len(rows)))[::-1]
    for y, (label, v, j) in zip(ys, rows):
        ax.barh(y, v, height=0.62, color=COLORS[j], edgecolor="white", linewidth=1)
        ax.annotate(f"{v:.0f}", (v, y), xytext=(3, 0), textcoords="offset points", fontsize=7, color=F.INK,
                    va="center")
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("developers per pod", fontsize=8)
    ax.set_title("E13, agent turns, one boot", fontsize=7.5, color=F.INK)
    F.save(fig, path)
