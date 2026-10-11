"""Third-party model capability context: LiveBench scores (data/static/livebench/), as the site computes them.

These scores describe each model as served by its provider at the named reasoning effort. The DeepSeek-V4.1-Flash row
is the unquantized model at maximum effort with thinking on, not the 3-bit EXL3 stack this paper measures (thinking
off); the paper uses them only to motivate the choice of model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import figures as F

ROOT = Path(__file__).resolve().parents[3]
LB = ROOT / "data" / "static" / "livebench"
TABLE, CATS = LB / "table_2026_06_25.csv", LB / "categories_2026_06_25.json"

# (LiveBench id, label, open weights) in the order of the paper's figure and table
MODELS = [
    ("deepseek-v4.1-flash-max", "DeepSeek-V4.1-Flash (max)", True),
    ("claude-opus-5-5-max-effort", "Claude Opus 5.5 (max)", False),
    ("claude-fable-5-1-max-effort", "Claude Fable 5.1 (max)", False),
    ("claude-opus-5-max-effort", "Claude Opus 5 (max)", False),
    ("qwen3.8-max", "Qwen 3.8 Max", True),
    ("kimi-k3", "Kimi K3", True),
    ("glm-5.3", "GLM-5.3", True),
    ("claude-sonnet-5-xhigh-effort", "Claude Sonnet 5 (xhigh)", False),
]
COLUMNS = [("All", "Overall"), ("Coding", "Coding"), ("Agentic Coding", "Agentic coding"),
           ("IF", "Instruction following")]


def scores() -> dict[str, dict[str, float]]:
    """Category score = mean of its task scores; overall = mean of the category scores (the site's method)."""
    cats = json.loads(CATS.read_text())
    out = {}
    for r in csv.DictReader(TABLE.open()):
        s = {}
        for c, tasks in cats.items():
            v = [float(r[t]) for t in tasks if r.get(t) not in (None, "")]
            if len(v) == len(tasks):
                s[c] = sum(v) / len(v)
        if len(s) == len(cats):
            s["All"] = sum(s[c] for c in cats) / len(cats)
            out[r["model"]] = s
    return out


def analyze(reg, write_tex, table, latex_escape, fig_dir: Path) -> None:
    if not TABLE.exists():
        return
    sc = scores()
    src = [TABLE, CATS]
    for mid, _label, _open in MODELS:
        for c, _ in COLUMNS:
            reg.put(f"livebench.{mid}.{c.replace(' ', '_')}", sc[mid][c], src, "static", "{:.1f}")
    ranked = sorted(sc, key=lambda m: sc[m]["Agentic Coding"], reverse=True)
    reg.put("livebench.n_models", len(sc), src, "static", "{:d}")
    reg.put("livebench.agentic.rank_deepseek", ranked.index("deepseek-v4.1-flash-max") + 1, src, "static", "{:d}")
    reg.put("livebench.agentic.runner_up", sc[ranked[1]]["Agentic Coding"], src, "static", "{:.1f}")

    rows = [[latex_escape(label) + (" $^{\\circ}$" if is_open else ""),
             *(f"{sc[mid][c]:.1f}" for c, _ in COLUMNS)] for mid, label, is_open in MODELS]
    write_tex("livebench.tex", table(
        "Third-party context: LiveBench scores of the unquantized models as served by their providers at the named "
        "reasoning effort (question set 2026-06-25, data downloaded 2026-10-07, \\texttt{data/static/livebench/}). "
        "$^{\\circ}$ open weights. The DeepSeek-V4.1-Flash row is maximum effort with thinking on; the 3-bit stack "
        "measured in this paper (thinking off) was not evaluated on LiveBench.", "tab:livebench",
        ["model", *(n for _, n in COLUMNS)], rows, "lrrrr"))
    reg.artifact("livebench.tex", "static", src, "data")
    _figure(sc, fig_dir / "livebench.pdf")
    reg.artifact("figures/generated/livebench.pdf", "static", src, "data")


def _figure(sc: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    hi, gray = F.SERIES[0], "#8c8b86"
    fig, axes = plt.subplots(1, len(COLUMNS), figsize=(7.0, 2.5), sharey=True)
    ys = list(range(len(MODELS)))[::-1]
    for ax, (c, name) in zip(axes, COLUMNS):
        F._style(ax)
        best = max(sc[m][c] for m, _, _ in MODELS)
        for y, (mid, _label, is_open) in zip(ys, MODELS):
            v = sc[mid][c]
            ds = mid.startswith("deepseek")
            ax.plot([55, v], [y, y], color=F.GRID, linewidth=0.8, zorder=1)
            ax.scatter([v], [y], s=46 if ds else 26, zorder=3, linewidths=1.2,
                       facecolors=("white" if is_open else (hi if ds else gray)), edgecolors=(hi if ds else gray),
                       marker="o")
            if ds or v == best:
                ax.annotate(f"{v:.1f}", (v, y), xytext=(0, 5), textcoords="offset points", ha="center",
                            fontsize=6, color=F.INK, fontweight="bold" if ds else "normal")
        ax.set_xlim(55, 100)
        ax.set_ylim(-0.6, len(MODELS) - 0.1)  # room for the value label above the top row
        ax.set_title(name, fontsize=7.5, color=F.INK)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label + (" (open)" if is_open else "") for _, label, is_open in MODELS], fontsize=6.5)
    for t, (mid, _, _) in zip(axes[0].get_yticklabels(), MODELS):
        if mid.startswith("deepseek"):
            t.set_fontweight("bold")
            t.set_color(F.INK)
    fig.text(0.5, 0.01, "LiveBench score (third-party; unquantized models as served by their providers). "
             "Hollow markers: open weights.", ha="center", fontsize=6, color=F.INK2)
    fig.subplots_adjust(bottom=0.2)
    F.save(fig, path)
