"""Statistics and cost model. Cost tests use the exploratory data, and their results are exploratory."""
import math
import random

import pytest

from hostoffload.analysis import exploratory as X
from hostoffload.analysis import stats
from hostoffload.cost import model as cost


def test_percentiles_linear_and_nearest():
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert stats.percentile(v, 50) == pytest.approx(5.5)
    assert stats.percentile(v, 95) == pytest.approx(9.55)
    assert stats.percentile(v, 95, "nearest") == 10   # the exploratory tool's p95 of n=10 is the max
    assert stats.percentile(v, 5, "nearest") == 1
    assert stats.percentile([], 50) is None


def test_nearest_rank_reproduces_exploratory_summary():
    """Our nearest-rank percentile reproduces the summaries the original load generator stored."""
    d = X.load_sweep("A")
    for lv in d["levels"]:
        reqs = [r for r in d["requests"][str(lv["N"])] if r["error"] is None]
        ttft = [r["ttft_s"] for r in reqs]
        dec = [r["decode_tok_s"] for r in reqs if r["completion_tokens"] >= 32]
        assert stats.percentile(ttft, 95, "nearest") == lv["ttft_p95_s"]
        assert stats.percentile(dec, 5, "nearest") == lv["decode_p5_tok_s"]


def test_bootstrap_ci_covers_and_is_seeded():
    rng = random.Random(1)
    v = [rng.gauss(10, 2) for _ in range(200)]
    a = stats.bootstrap_ci(v, stats.mean, n_boot=500, seed=3)
    b = stats.bootstrap_ci(v, stats.mean, n_boot=500, seed=3)
    assert a == b
    point, lo, hi = a
    assert lo < point < hi and lo < 10 < hi and hi - lo < 1.0
    assert stats.bootstrap_ci([1.0], stats.mean) is None


def test_cluster_bootstrap_wider_than_request_bootstrap_when_reps_differ():
    rng = random.Random(2)
    groups = [[rng.gauss(mu, 0.5) for _ in range(50)] for mu in (9.0, 10.0, 11.0)]
    pooled = [x for g in groups for x in g]
    _, l1, h1 = stats.bootstrap_ci(pooled, stats.mean, n_boot=1000, seed=1)
    _, l2, h2 = stats.cluster_bootstrap_ci(groups, stats.mean, n_boot=1000, seed=1)
    assert (h2 - l2) > 2 * (h1 - l1)


def test_variance_components():
    vc = stats.variance_components([[1, 1, 1], [3, 3, 3]])
    assert vc["var_within"] == 0 and vc["icc"] == pytest.approx(1.0)
    vc2 = stats.variance_components([[1, 2, 3], [1, 2, 3]])
    assert vc2["var_between"] == 0 and vc2["icc"] == 0
    rs = stats.repetition_summary([10.0, 12.0, 11.0])
    assert rs["mean"] == pytest.approx(11.0) and rs["cv"] == pytest.approx(1.0 / 11.0)


def test_correlations():
    x = list(range(20))
    assert stats.pearson(x, [2 * a + 1 for a in x]) == pytest.approx(1.0)
    assert stats.spearman(x, [a ** 3 for a in x]) == pytest.approx(1.0)
    assert stats.spearman(x, [-a for a in x]) == pytest.approx(-1.0)


def test_weighted_percentile():
    assert stats.weighted_percentile([1, 2, 3], [1, 1, 98], 50) == 3
    assert stats.weighted_percentile([1, 2, 3], [98, 1, 1], 50) == 1


# ---- cost model ------------------------------------------------------------------------------------------

def test_usd_per_million_formula():
    assert cost.usd_per_million(3.6, 1.0) == pytest.approx(1000.0)  # 3.6 USD/h at 1 tok/s: 3,600 tokens per USD 3.6
    assert cost.usd_per_million(4.18, 0) is None


def test_price_and_fx_from_file():
    p = cost.load_price("2x-rtx-pro-6000")
    assert p.usd_per_hour == 4.18 and p.fx_date == "2026-10-05"
    assert p.eur_per_hour == pytest.approx(4.18 / 1.1204)


def test_split_ratio_recovers_cost():
    s = cost.split_ratio(3.6, in_tok_s=1000, out_tok_s=100, k=0.1)
    per_s = 1000 * s["usd_per_m_input"] / 1e6 + 100 * s["usd_per_m_output"] / 1e6
    assert per_s == pytest.approx(3.6 / 3600)
    assert s["usd_per_m_input"] == pytest.approx(0.1 * s["usd_per_m_output"])


def test_dedicated_split_and_per_request():
    d = cost.split_dedicated(3.6, prefill_tok_s=1000, decode_tok_s=10)
    assert d["usd_per_m_input"] == pytest.approx(1.0) and d["usd_per_m_output"] == pytest.approx(100.0)
    assert cost.per_request_cost(3.6, wall_s=100, n_requests=10) == pytest.approx(0.01)
    assert cost.per_concurrent_request_hour(4.18, 8) == pytest.approx(0.5225)


def test_exploratory_cost_at_N8_scenario_A():
    """EXPLORATORY: at N = 8 in scenario A the measured aggregate was 152.08 output tok/s (file), so the
    output-only cost is 4.18 / (152.08 * 3600) * 1e6 USD per 1M output tokens."""
    row = next(r for r in X.level_rows("A") if r["N"] == 8)
    assert row["aggregate_out_tok_s"] == 152.08
    usd = cost.usd_per_million(cost.load_price("2x-rtx-pro-6000").usd_per_hour, row["aggregate_out_tok_s"])
    assert usd == pytest.approx(4.18 / (152.08 * 3600) * 1e6)
    assert 7.6 < usd < 7.7


def test_best_point_under_sla_exploratory():
    rows = X.level_rows("A")
    pts = [{"label": r["N"], "out_tok_s": r["aggregate_out_tok_s"],
            "meets_sla": r["ttft_p95_s"] <= 5 and r["decode_p5_tok_s"] >= 15} for r in rows]
    best = cost.best_point_under_sla(pts)
    assert best["label"] == 8   # the exploratory strict bar for A (post hoc) passes up to N = 8
    assert cost.best_point_under_sla([{"out_tok_s": 1, "meets_sla": False}]) is None
    assert math.isfinite(best["out_tok_s"])
