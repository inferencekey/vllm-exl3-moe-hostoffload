"""Seat and cost model of the coding-agent case study."""
import math

import pytest
import yaml

from hostoffload.cost import plan_budget
from hostoffload.cost import seats as S


def profile():
    t = yaml.safe_load((plan_budget.ROOT / "experiments" / "agent_workload.yaml").read_text())["usage_sample"]
    tk = t["tokens"]
    return S.Profile(tk["output"], tk["cache_read"], tk["cache_write"], tk["input"], t["window_days"], t["active_days"])


def test_ratios_and_turn_shape():
    p = profile()
    assert p.read_per_output == pytest.approx(90_800_000 / 387_700)
    shape = p.turn_shape(400)
    assert shape["context_read"] == pytest.approx(400 * 90_800_000 / 387_700)
    assert shape["new_tokens"] == pytest.approx(400 * 3_400_000 / 387_700)
    assert p.output_month(30.44) == pytest.approx(387_700 * 30.44 / 7)


def test_duty_cycle_is_busy_time_over_active_window():
    p = profile()
    busy = S.busy_seconds_week(p, 40, 7000)
    assert busy == pytest.approx(387_700 / 40 + 3_400_000 / 7000)
    assert S.duty_cycle(p, busy, 8) == pytest.approx(busy / (5 * 8 * 3600))


def test_binomial_tail_and_inversion():
    assert S.binom_sf(0, 1, 0.3) == pytest.approx(0.3)
    assert S.binom_sf(1, 2, 0.5) == pytest.approx(0.25)
    m = S.max_seats_for_capacity(4, 0.07, 0.01)
    assert S.binom_sf(4, m, 0.07) <= 0.01 < S.binom_sf(4, m + 1, 0.07)
    assert S.pods_for_concurrency(m, 4, 0.07, 0.01) == 1
    assert S.pods_for_concurrency(m + 1, 4, 0.07, 0.01) == 2


def test_pods_needed_takes_the_binding_constraint():
    # plenty of volume, little concurrency: concurrency binds
    r = S.pods_needed(100, agg_tok_s=200, capacity=2, q=0.1, eps=0.01, month_hours=730, utilization=0.5,
                      dev_output_month=1.7e6)
    assert r["binding"] == "concurrency" and r["pods"] == r["by_concurrency"] > r["by_volume"]
    # low volume per pod, rarely active users: volume binds
    r = S.pods_needed(100, agg_tok_s=5, capacity=50, q=0.01, eps=0.01, month_hours=730, utilization=0.3,
                      dev_output_month=1.7e6)
    assert r["binding"] == "volume" and r["pods"] == math.ceil(100 / S.seats_by_volume(5, 730, 0.3, 1.7e6))


def test_costs():
    eur = S.pod_eur_month(4.18, 1.1204, 730)
    assert eur == pytest.approx(4.18 / 1.1204 * 730)
    assert S.break_even_seats(eur, 180) == pytest.approx(eur / 180)
    u = S.break_even_utilization(eur, 180, 100, 730, 1.7e6)
    assert u == pytest.approx(eur / 180 * 1.7e6 / (100 * 3600 * 730))
