"""E8 (main benchmark, pod 1): per-stream output rate per request and windowed aggregate output rate against N.

Per-stream rate of one request = (completion_tokens - 1) / (end_mono - first_token_mono), the generation rate after
the first token as the client sees it. It equals 1 / TPOT when tokens stream evenly and stays valid when a reply
arrives in a few large chunks. Warm requests and failed requests are excluded. The distribution at each N pools the
requests of all repetitions; the aggregate is the windowed output rate (hostoffload.analysis.metrics) computed per
repetition and then averaged, because pooling would join two repetitions' timelines.
"""
from __future__ import annotations

from pathlib import Path

from . import figures as F
from . import runs as RN
from . import stats
from .metrics import level_summary

ROOT = Path(__file__).resolve().parents[3]
E8 = ROOT / "data" / "controlled" / "E8-main" / "main"
SCENARIOS = [("A-code-4k", "A", "A: 4K prompt, independent"), ("B-prefix-16k", "B", "B: 16K cached prefix")]


def stream_rate(rec: dict, min_tokens: int = 16) -> float | None:
    """Per-stream generation rate of one request in tokens/s, or None for errors, warm or too-short replies."""
    if rec.get("error") is not None or rec.get("level") == "warm":
        return None
    ft, end, ct = rec.get("first_token_mono"), rec.get("end_mono"), int(rec.get("completion_tokens") or 0)
    if ft is None or end is None or ct < max(2, min_tokens) or end <= ft:
        return None
    return (ct - 1) / (end - ft)


def per_n(levels: list[RN.Level], scenario: str) -> dict[int, dict]:
    """N -> per-stream rate percentiles (pooled over repetitions) and mean windowed aggregate over repetitions."""
    by_n: dict[int, list[RN.Level]] = {}
    for lv in levels:
        if lv.scenario == scenario:
            by_n.setdefault(lv.N, []).append(lv)
    out = {}
    for n, lvs in sorted(by_n.items()):
        rates = [x for lv in lvs for x in (stream_rate(r) for r in lv.requests) if x is not None]
        aggs = [a for a in (level_summary(lv.requests).get("aggregate_out_tok_s_window") for lv in lvs) if a]
        out[n] = {"n": len(rates), "reps": len(lvs),
                  "p5": stats.percentile(rates, 5), "p50": stats.percentile(rates, 50),
                  "p95": stats.percentile(rates, 95), "agg": stats.mean(aggs), "agg_reps": len(aggs)}
    return out


def analyze(reg, fig_dir: Path) -> None:
    runs = RN.complete_runs(E8)
    if not runs:
        return
    run = runs[-1]
    src = [run / "requests.jsonl", run / "levels.jsonl"]
    lvs = RN.levels(run)
    data = {}
    for scen, short, _label in SCENARIOS:
        d = per_n(lvs, scen)
        data[scen] = d
        for n, v in d.items():
            k = f"ctl.E8.stream.{short}.N{n}"
            for q in ("p5", "p50", "p95"):
                if v[q] is not None:
                    reg.put(f"{k}.{q}", v[q], src, "controlled", "{:.1f}")
            reg.put(f"{k}.n", v["n"], src, "controlled", "{:d}")
            if v["agg"] is not None:
                reg.put(f"{k}.agg", v["agg"], src, "controlled", "{:.0f}")
                reg.put(f"{k}.agg_reps", v["agg_reps"], src, "controlled", "{:d}")
    _figure(data, fig_dir / "controlled_E8.pdf")
    reg.artifact("figures/generated/controlled_E8.pdf", "controlled", src, "data")


def _figure(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax in (ax1, ax2):
        F._style(ax)
    ticks = [1, 2, 4, 8, 16, 32, 64]
    for i, (scen, _short, label) in enumerate(SCENARIOS):
        d = data[scen]
        ns = sorted(d)
        c = F.SERIES[i]
        med = [d[n]["p50"] for n in ns]
        ax1.fill_between(ns, [d[n]["p5"] for n in ns], [d[n]["p95"] for n in ns], color=c, alpha=0.16,
                         linewidth=0)
        ax1.plot(ns, med, color=c, linewidth=1.6, marker="o", markersize=3.2)
        agg = [d[n]["agg"] for n in ns]
        ax2.plot(ns, agg, color=c, linewidth=1.6, marker="o", markersize=3.2)
        # direct labels: B above its band, pointing at its N = 4 median (where the series are furthest apart);
        # A in the empty area under its own band; aggregate panel at the right end
        n4 = 4 if 4 in d else ns[len(ns) // 2]
        if i == 0:
            ax1.annotate(label, (1.0, 11), fontsize=6.5, color=c, va="center", fontweight="bold")
        else:
            ax1.annotate(label, (n4, d[n4]["p50"]), xytext=(5.5, 150), textcoords="data", fontsize=6.5,
                         color=c, va="center", fontweight="bold",
                         arrowprops={"arrowstyle": "-", "color": c, "linewidth": 0.6, "shrinkA": 1, "shrinkB": 3})
        ax2.annotate(label.split(":")[0], (ns[-1], agg[-1]), xytext=(4, 0), textcoords="offset points",
                     fontsize=7, color=c, va="center", fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(0.8, 90)
        ax.set_xlabel("concurrent streams N (closed loop)", fontsize=7.5)
        ax.set_ylim(bottom=0)
    ax1.set_ylabel("output tok/s per stream", fontsize=7.5)
    ax1.set_title("per-stream rate: median, band p5 to p95", fontsize=7.5, color=F.INK)
    ax2.set_ylabel("aggregate output tok/s", fontsize=7.5)
    ax2.set_title("windowed aggregate (mean over repetitions)", fontsize=7.5, color=F.INK)
    F.save(fig, path)
