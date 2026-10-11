"""DSpark speculative decoding acceptance per level, from vLLM's own counters (E8 pod 1, E13 pod 3).

The harness (hostoffload.harness.prom.SERVER_COUNTERS) records, before and after every level
(server_metrics.jsonl) and every 0.5 s (samples.jsonl, sample.vllm.counters), the vLLM counters
- spec_accepted     = vllm:spec_decode_num_accepted_tokens_total: draft tokens accepted by the target model
- spec_drafts       = vllm:spec_decode_num_drafts_total: drafting steps, one per request per verification step
- spec_draft_tokens = vllm:spec_decode_num_draft_tokens_total: draft tokens proposed (k per draft)
A level's values are the difference of the after and before snapshots (taken at the window's two edges, after the
warm request). Mean accepted tokens per draft = accepted / drafts. Acceptance length = 1 + accepted / drafts, the
mean number of tokens a request gains per target verification step (the accepted drafts plus the one token the
target always produces). Draft acceptance rate = accepted / draft_tokens. Repetitions and levels are pooled by
summing the counter deltas.
"""
from __future__ import annotations

from pathlib import Path

from . import runs as RN
from .pcie import REGIMES, RUN_DIRS, selected_ns

KEYS = ("spec_accepted", "spec_drafts", "spec_draft_tokens", "generation_tokens")


def acceptance(accepted: float, drafts: float, draft_tokens: float | None = None) -> dict | None:
    """Accepted tokens per draft, acceptance length and draft acceptance rate; None without drafts."""
    if not drafts or drafts <= 0:
        return None
    out = {"accepted_per_draft": accepted / drafts, "accept_len": 1.0 + accepted / drafts, "drafts": drafts}
    if draft_tokens:
        out["accept_rate"] = accepted / draft_tokens
        out["k"] = draft_tokens / drafts
    return out


def counter_delta(lv: RN.Level) -> dict | None:
    if lv.before is None or lv.after is None:
        return None
    if any(lv.after.get(k) is None or lv.before.get(k) is None for k in KEYS[:3]):
        return None
    return {k: (lv.after.get(k) or 0) - (lv.before.get(k) or 0) for k in KEYS}


def per_level(run: Path, scenario: str) -> tuple[dict[int, dict], dict]:
    """N -> pooled counter deltas over repetitions, plus the pool over all levels of the scenario."""
    by_n: dict[int, dict] = {}
    for lv in RN.levels(run):
        if lv.scenario != scenario:
            continue
        d = counter_delta(lv)
        if d is None:
            continue
        acc = by_n.setdefault(lv.N, {k: 0.0 for k in KEYS})
        for k in KEYS:
            acc[k] += d[k]
    total = {k: sum(v[k] for v in by_n.values()) for k in KEYS}
    return by_n, total


def analyze(reg, write_tex, table, latex_escape, caveats: list[str]) -> None:
    rows, srcs, ks = [], [], set()
    for exp, short, scen, label in REGIMES:
        runs = RN.complete_runs(RUN_DIRS[exp])
        if not runs:
            continue
        run = runs[-1]
        src = [run / "server_metrics.jsonl", run / "levels.jsonl"]
        by_n, total = per_level(run, scen)
        if not by_n or total["spec_drafts"] <= 0:
            caveats.append(f"{exp} {scen}: speculative-decoding counters missing or zero; nothing registered")
            continue
        srcs += src
        base = f"ctl.spec.{exp}.{short}"
        for n, d in by_n.items():
            a = acceptance(d["spec_accepted"], d["spec_drafts"], d["spec_draft_tokens"])
            if a is None:
                continue
            ks.add(round(a["k"], 6))
            reg.put(f"{base}.N{n}.accept_len", a["accept_len"], src, "controlled", "{:.2f}")
            reg.put(f"{base}.N{n}.accept_rate_pct", 100 * a["accept_rate"], src, "controlled", "{:.0f}")
        a = acceptance(total["spec_accepted"], total["spec_drafts"], total["spec_draft_tokens"])
        reg.put(f"{base}.accept_len", a["accept_len"], src, "controlled", "{:.2f}",
                note="pooled over all levels of the scenario: 1 + accepted / drafts")
        reg.put(f"{base}.accepted_per_draft", a["accepted_per_draft"], src, "controlled", "{:.2f}")
        reg.put(f"{base}.accept_rate_pct", 100 * a["accept_rate"], src, "controlled", "{:.0f}")
        ns = sorted(by_n)
        for n in selected_ns(ns):
            d = by_n[n]
            x = acceptance(d["spec_accepted"], d["spec_drafts"], d["spec_draft_tokens"])
            rows.append([latex_escape(label) if n == ns[0] else "", str(n), f"{d['spec_drafts']:,.0f}",
                         f"{x['accepted_per_draft']:.2f}", f"{x['accept_len']:.2f}", f"{100 * x['accept_rate']:.0f}"])
        rows.append(["", "all", f"{total['spec_drafts']:,.0f}", f"{a['accepted_per_draft']:.2f}",
                     f"{a['accept_len']:.2f}", f"{100 * a['accept_rate']:.0f}"])
    if not rows:
        return
    if len(ks) == 1:
        reg.put("ctl.spec.k", next(iter(ks)), sorted(set(srcs)), "controlled", "{:.0f}",
                note="draft tokens proposed per draft (spec_draft_tokens / spec_drafts), equal on every level")
    else:
        caveats.append(f"draft tokens per draft differ between levels: {sorted(ks)}")
    write_tex("spec_acceptance.tex", table(
        "DSpark acceptance per level from vLLM's counters: drafts (\\texttt{vllm:spec\\_decode\\_num\\_drafts}), "
        "accepted draft tokens per draft, acceptance length $1 + \\text{accepted}/\\text{drafts}$ (tokens a request "
        "gains per verification step) and the share of proposed draft tokens accepted. Rows: $N=1$, $N=8$, the "
        "largest tested $N$ and all levels pooled; E8 on pod 1 (repetitions pooled), E13 on pod 3.",
        "tab:spec-acceptance", ["regime", "$N$", "drafts", "accepted/draft", "accept. length", "accepted \\%"],
        rows, "lrrrrr"))
    reg.artifact("spec_acceptance.tex", "controlled", sorted(set(srcs)), "data")
