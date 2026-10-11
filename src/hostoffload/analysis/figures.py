"""Figures for the paper (static PDF, print). Palette: the first three categorical slots of the validated reference
palette (blue, orange, aqua), recessive axes and grid, one y-axis per chart, direct labels where few series."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _new(w: float = 3.4, h: float = 2.3):
    fig, ax = plt.subplots(figsize=(w, h))
    _style(ax)
    return fig, ax


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def per_request_decode(per_req: dict[str, dict[int, list[dict]]], path: Path, title: str) -> None:
    """Per-request decode rate vs concurrency, every request as a dot plus the median line, per scenario."""
    fig, ax = _new()
    for i, (name, levels) in enumerate(per_req.items()):
        xs, ys, med_x, med_y = [], [], [], []
        for n, reqs in sorted(levels.items()):
            v = sorted(r["decode_tok_s"] for r in reqs if r.get("decode_tok_s") is not None)
            off = (i - 0.5) * 0.08
            xs += [n * (1 + off)] * len(v)
            ys += v
            if v:
                med_x.append(n * (1 + off))
                med_y.append(v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2)
        ax.scatter(xs, ys, s=8, color=SERIES[i], alpha=0.45, linewidths=0)
        ax.plot(med_x, med_y, color=SERIES[i], linewidth=1.6, marker="o", markersize=3.5, label=name)
        ax.annotate(name, (med_x[-1], med_y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=INK, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("concurrent requests N (closed loop)", fontsize=8)
    ax.set_ylabel("decode tok/s per request", fontsize=8)
    ax.set_title(title, fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    save(fig, path)


def aggregate_vs_n(rows: dict[str, list[dict]], path: Path, title: str) -> None:
    fig, ax = _new()
    for i, (name, rs) in enumerate(rows.items()):
        x = [r["N"] for r in rs]
        y = [r["aggregate_out_tok_s"] for r in rs]
        ax.plot(x, y, color=SERIES[i], linewidth=1.6, marker="o", markersize=3.5, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK,
                    va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent requests N (closed loop)", fontsize=8)
    ax.set_ylabel("aggregate output tok/s", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    save(fig, path)


def ttft_by_wave(per_req: dict[int, list[dict]], path: Path, title: str) -> None:
    """TTFT of first-wave requests (sent at level start) vs later requests, per N."""
    fig, ax = _new()
    for n, reqs in sorted(per_req.items()):
        rs = sorted((r for r in reqs if r.get("ttft_s") is not None), key=lambda r: r["start"])
        first, later = rs[:n], rs[n:]
        ax.scatter([n * 0.93] * len(first), [r["ttft_s"] for r in first], s=8, color=SERIES[1], alpha=0.6,
                   linewidths=0, label="first wave" if n == 1 else None)
        ax.scatter([n * 1.07] * len(later), [r["ttft_s"] for r in later], s=8, color=SERIES[0], alpha=0.6,
                   linewidths=0, label="after first wave" if n == 1 else None)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("concurrent requests N (closed loop)", fontsize=8)
    ax.set_ylabel("TTFT (s)", fontsize=8)
    ax.set_title(title, fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    save(fig, path)


def placement_bars(classes: list[tuple[str, float, float]], path: Path, title: str) -> None:
    """Horizontal bars of file GiB per tensor class, split VRAM / host (two series, one axis)."""
    fig, ax = _new(3.4, 2.4)
    names = [c[0] for c in classes]
    vram = [c[1] for c in classes]
    host = [c[2] for c in classes]
    y = range(len(names))
    ax.barh(list(y), vram, color=SERIES[0], height=0.6, label="VRAM", edgecolor="white", linewidth=1)
    ax.barh(list(y), host, left=vram, color=SERIES[1], height=0.6, label="pinned host", edgecolor="white",
            linewidth=1)
    for yi, (v, h) in enumerate(zip(vram, host)):
        ax.annotate(f"{v + h:.1f}", (v + h, yi), xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                    color=INK2, va="center")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("checkpoint bytes (GiB)", fontsize=8)
    ax.set_title(title, fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, path)


def host_experts_per_layer(counts: list[tuple[int, int]], path: Path, title: str) -> None:
    fig, ax = _new(3.4, 2.0)
    ax.bar([c[0] for c in counts], [c[1] for c in counts], color=SERIES[1], width=0.8)
    ax.set_xlabel("layer (40 to 42: drafter)", fontsize=8)
    ax.set_ylabel("experts in host memory", fontsize=8)
    ax.set_title(title, fontsize=8, color=INK)
    save(fig, path)


def memory_timeline(samples: list[dict], path: Path, title: str) -> None:
    fig, ax = _new(3.4, 2.2)
    xs = list(range(len(samples)))
    for i, (k, lab) in enumerate((("ram_anon_gib", "anon"), ("ram_shmem_gib", "shmem"), ("ram_file_gib", "file"))):
        ys = [s[k] for s in samples]
        ax.plot(xs, ys, color=SERIES[i], linewidth=1.4, label=lab)
        ax.annotate(lab, (xs[-1], ys[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK,
                    va="center")
    ax.set_xlabel("sample index (5 s interval)", fontsize=8)
    ax.set_ylabel("cgroup memory (GiB)", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=8, color=INK)
    save(fig, path)
