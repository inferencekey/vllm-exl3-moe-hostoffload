"""Reading against writing on pod 2: time to first token and time per output token of one request against the length
of its prompt, fresh or behind a cached prefix.

The figure reuses values already registered from E12 (long context) and the E3 point run on the same pod, so it adds
no measurement. Prompt lengths are the nominal lengths of the workload files.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import figures as F

ROOT = Path(__file__).resolve().parents[3]
LONG = ROOT / "experiments" / "workloads" / "long_context.yaml"
SUBSET = ROOT / "experiments" / "workloads" / "lean_subset.yaml"
E3 = "ctl.E3.gen30-session2.A-code-4k.closed:1"
E12 = "ctl.E12.long"


def _lengths() -> dict[str, int]:
    """Scenario name in the registry keys -> nominal prompt tokens."""
    out = {}
    for s in yaml.safe_load(LONG.read_text())["scenarios"]:
        if s["name"] == "prefill-long":
            out.update({f"prefill-long-{n}": n for n in s["lengths"]})
        elif "prompt_tokens" in s:
            out[s["name"]] = s["prompt_tokens"]
        else:
            out[s["name"]] = s["prompt"]["tokens"]
    for s in yaml.safe_load(SUBSET.read_text())["scenarios"]:
        if s["name"] == "A-code-4k":
            out["A-code-4k"] = s["prompt"]["tokens"]
    return out


def _points(reg, keys: list[tuple[str, str]], metric: str, lengths: dict[str, int]) -> tuple[list, list, list]:
    xs, ys, srcs = [], [], []
    for prefix, scen in keys:
        e = reg.entries.get(f"{prefix}.{metric}")
        if e is None or scen not in lengths:
            continue
        xs.append(lengths[scen])
        ys.append(float(e.value))
        srcs += e.sources
    return xs, ys, srcs


def analyze(reg, fig_dir: Path) -> None:
    lengths = _lengths()
    fresh_ttft = [(E3, "A-code-4k")] + [(f"{E12}.prefill-long-{n}.closed:1", f"prefill-long-{n}")
                                       for n in (32768, 65536, 131072, 260000)]
    fresh_tpot = [(E3, "A-code-4k"), (f"{E12}.decode-128k.closed:1", "decode-128k"),
                  (f"{E12}.decode-256k.closed:1", "decode-256k")]
    cached = [(f"{E12}.B-prefix-128k.closed:1", "B-prefix-128k"), (f"{E12}.B-prefix-256k.closed:1", "B-prefix-256k")]
    series = {
        "ttft": [("fresh prompt", *_points(reg, fresh_ttft, "ttft_p50", lengths)),
                 ("cached prefix", *_points(reg, cached, "ttft_p50", lengths))],
        "tpot": [("fresh prompt", *_points(reg, fresh_tpot, "tpot_p50_ms", lengths)),
                 ("cached prefix", *_points(reg, cached, "tpot_p50_ms", lengths))],
    }
    if any(len(s[1]) < 2 for v in series.values() for s in v):
        return
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    titles = {"ttft": "reading: seconds to first token", "tpot": "writing: ms per output token"}
    for ax, (metric, ss) in zip(axes, series.items()):
        F._style(ax)
        for i, (name, xs, ys, _) in enumerate(ss):
            ax.plot(xs, ys, color=F.SERIES[i], linewidth=1.6, linestyle="-" if i == 0 else "--", marker="os"[i],
                    markersize=4, label=name)
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=6.5,
                        color=F.INK, va="center")
        ax.set_xscale("log", base=2)
        ticks = [4096, 32768, 131072, 262144]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1024:.0f}K"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(3000, 420000)
        if metric == "ttft":
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(0, 12)
        ax.set_xlabel("prompt length (tokens)", fontsize=8)
        ax.set_title(titles[metric], fontsize=8, color=F.INK)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    path = fig_dir / "read_write_length.pdf"
    F.save(fig, path)
    srcs = sorted({ROOT / src for ss in series.values() for s in ss for src in s[3]} | {LONG, SUBSET})
    reg.artifact("figures/generated/read_write_length.pdf", "controlled", srcs, "data")
