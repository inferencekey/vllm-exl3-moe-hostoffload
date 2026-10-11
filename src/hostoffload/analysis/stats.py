"""Statistics: percentiles, bootstrap confidence intervals, repetition variance.

Percentiles use linear interpolation between order statistics (the 'linear' / type 7 definition, as numpy's
default) unless `method='nearest'` is asked for, which reproduces the nearest-rank percentiles of the exploratory
load generator. With n < 20 samples a p95 is close to the maximum; `describe` reports n so readers can judge.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

PCTS = (5, 50, 90, 95, 99)


def mean(v: Sequence[float]) -> float | None:
    v = list(v)
    return sum(v) / len(v) if v else None


def stdev(v: Sequence[float]) -> float | None:
    v = list(v)
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def percentile(v: Sequence[float], p: float, method: str = "linear") -> float | None:
    s = sorted(v)
    if not s:
        return None
    if method == "nearest":
        # nearest-rank as in the exploratory loadgen.py: k = round(p/100 * n + 0.5) - 1, clamped
        k = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
        return s[k]
    if len(s) == 1:
        return s[0]
    h = (len(s) - 1) * p / 100.0
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def weighted_percentile(v: Sequence[float], w: Sequence[float], p: float) -> float | None:
    """Percentile of a weighted sample (inverse CDF, no interpolation)."""
    pairs = sorted(zip(v, w))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    target, acc = p / 100.0 * total, 0.0
    for x, wi in pairs:
        acc += wi
        if acc >= target:
            return x
    return pairs[-1][0]


def describe(v: Sequence[float]) -> dict:
    v = list(v)
    d = {"n": len(v), "mean": mean(v), "sd": stdev(v), "min": min(v), "max": max(v)}
    for p in PCTS:
        d[f"p{p}"] = percentile(v, p)
    return d


def describe_weighted(v: Sequence[float], w: Sequence[float]) -> dict:
    tot = sum(w)
    d = {"n": len(v), "weight": tot, "mean": sum(x * wi for x, wi in zip(v, w)) / tot if tot else None}
    for p in PCTS:
        d[f"p{p}"] = weighted_percentile(v, w, p)
    return d


def bootstrap_ci(v: Sequence[float], stat: Callable[[Sequence[float]], float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float] | None:
    """Percentile bootstrap: (point estimate, lower, upper) of `stat` at confidence 1 - alpha. Seeded."""
    v = list(v)
    if len(v) < 2:
        return None
    rng = random.Random(seed)
    point = stat(v)
    boots = sorted(stat([v[rng.randrange(len(v))] for _ in range(len(v))]) for _ in range(n_boot))
    lo = boots[int(math.floor(alpha / 2 * n_boot))]
    hi = boots[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return point, lo, hi


def cluster_bootstrap_ci(groups: Sequence[Sequence[float]], stat: Callable[[Sequence[float]], float],
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float] | None:
    """Bootstrap that resamples whole repetitions (clusters), then pools their requests. Requests of one
    repetition share a server state, so resampling requests alone understates the uncertainty."""
    groups = [list(g) for g in groups if g]
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    pooled = [x for g in groups for x in g]
    point = stat(pooled)
    boots = []
    for _ in range(n_boot):
        sample = [x for _ in groups for x in groups[rng.randrange(len(groups))]]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int(math.floor(alpha / 2 * n_boot))]
    hi = boots[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return point, lo, hi


def repetition_summary(values: Sequence[float]) -> dict:
    """Across-repetition summary of one per-repetition statistic: mean, sd, coefficient of variation, range."""
    v = [x for x in values if x is not None]
    if not v:
        return {"n_reps": 0}
    m, s = mean(v), stdev(v)
    return {"n_reps": len(v), "mean": m, "sd": s, "cv": (s / m) if (s is not None and m) else None,
            "min": min(v), "max": max(v), "values": v}


def variance_components(groups: Sequence[Sequence[float]]) -> dict:
    """One-way random-effects decomposition: between-repetition and within-repetition variance of a per-request
    metric, and the intraclass correlation (share of variance due to the repetition)."""
    groups = [list(g) for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return {"k": k, "n": n}
    grand = sum(sum(g) for g in groups) / n
    ss_b = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_w = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    ms_b, ms_w = ss_b / (k - 1), ss_w / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    var_b = max(0.0, (ms_b - ms_w) / n0)
    icc = var_b / (var_b + ms_w) if (var_b + ms_w) > 0 else None
    return {"k": k, "n": n, "ms_between": ms_b, "ms_within": ms_w, "var_between": var_b, "var_within": ms_w,
            "icc": icc}


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def _ranks(v: Sequence[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    return pearson(_ranks(x), _ranks(y))
