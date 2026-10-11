"""Read lm-evaluation-harness outputs and compare two systems item by item.

lm-eval writes, per task, a results_*.json ({"results": {task: {"<metric>,<filter>": value,
"<metric>_stderr,<filter>": se}}}) and, with --log_samples, samples_<task>_*.jsonl with one line per item
(doc_id plus the metric values). The comparison uses the items both systems answered (paired), and a paired
bootstrap over items for the difference in mean score.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import stats


def find_results(run_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(run_dir).rglob("results_*.json")):
        d = json.loads(f.read_text())
        for task, metrics in d.get("results", {}).items():
            out[task] = {"file": f, "metrics": metrics}
    return out


def headline(metrics: dict, metric: str) -> tuple[float | None, float | None]:
    """Value and stderr of `metric` (first filter that has it)."""
    for k, v in metrics.items():
        name, _, filt = k.partition(",")
        if name == metric and isinstance(v, (int, float)):
            se = metrics.get(f"{metric}_stderr,{filt}")
            return float(v), (float(se) if isinstance(se, (int, float)) else None)
    return None, None


def load_samples(run_dir: Path, task: str, metric: str) -> dict[int, float]:
    out = {}
    for f in sorted(Path(run_dir).rglob(f"samples_{task}_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if metric in r and isinstance(r[metric], (int, float, bool)):
                out[int(r["doc_id"])] = float(r[metric])
    return out


def paired_difference(a: dict[int, float], b: dict[int, float], n_boot: int = 5000, seed: int = 0) -> dict | None:
    """Mean(b) - mean(a) over shared items, with a paired percentile bootstrap 95% interval."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(ids), "mean_a": stats.mean([a[i] for i in ids]), "mean_b": stats.mean([b[i] for i in ids]),
            "diff": sum(d) / len(d), "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]),
            "discordant": sum(1 for x in d if x != 0)}
