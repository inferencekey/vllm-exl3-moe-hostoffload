"""Coding-agent seat and cost model (E13 case study). Pure functions; inputs come from experiments/agent_workload.yaml,
measured capacities from the analysis pipeline.

Two different capacities are kept apart everywhere:
- simultaneous-stream capacity C: the largest number of streams generating at the same time while each keeps the
  per-stream SLO (measured);
- volume capacity: output tokens per month the pod can produce at its SLO-compliant aggregate rate, divided by one
  developer's monthly output ("capacity-equivalent developer seats by observed generation volume").
A pod can have volume for many intermittent developers and still give 40 tok/s to only C of them at once; the number
of pods needed for M seats is the larger of the two requirements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """One developer's observed token mix and the workload shape derived from it."""
    output_week: float
    cache_read_week: float
    cache_write_week: float
    input_week: float
    window_days: float
    active_days: float

    @property
    def read_per_output(self) -> float:
        return self.cache_read_week / self.output_week

    @property
    def write_per_output(self) -> float:
        return self.cache_write_week / self.output_week

    def output_month(self, days_per_month: float) -> float:
        return self.output_week * days_per_month / self.window_days

    def turn_shape(self, output_per_turn: float) -> dict:
        """Context read and new tokens written per turn for a given reply length (the ratios fix the rest)."""
        return {"output": output_per_turn, "context_read": output_per_turn * self.read_per_output,
                "new_tokens": output_per_turn * self.write_per_output,
                "turns_week": self.output_week / output_per_turn}


def busy_seconds_week(p: Profile, stream_tok_s: float, prefill_tok_s: float) -> float:
    """Upper bound on one developer's generating time per week if served at the SLO floor: output at `stream_tok_s`
    plus the prefill of new (cache-write) tokens. Cache reads are not charged here: their cost shows up in the measured
    per-stream rate at that context length."""
    return p.output_week / stream_tok_s + p.cache_write_week / prefill_tok_s


def duty_cycle(p: Profile, busy_s_week: float, active_hours_per_active_day: float) -> float:
    """Probability that the developer is generating at a random instant of their active window."""
    return busy_s_week / (p.active_days * active_hours_per_active_day * 3600.0)


def binom_sf(k: int, m: int, q: float) -> float:
    """P(X > k) for X ~ Binomial(m, q)."""
    if k >= m:
        return 0.0
    if k < 0:
        return 1.0
    return sum(math.comb(m, i) * q ** i * (1 - q) ** (m - i) for i in range(k + 1, m + 1))


def max_seats_for_capacity(capacity: int, q: float, eps: float, limit: int = 100_000) -> int:
    """Largest M with P(Binomial(M, q) > capacity) <= eps (independent developers, same q)."""
    if capacity <= 0:
        return 0
    m = capacity
    while m < limit and binom_sf(capacity, m + 1, q) <= eps:
        m += 1
    return m


def pods_for_concurrency(seats: int, capacity: int, q: float, eps: float, max_pods: int = 1000) -> int | None:
    """Smallest number of pods k with P(Binomial(seats, q) > k * capacity) <= eps (pooled load balancing)."""
    if capacity <= 0:
        return None
    for k in range(1, max_pods + 1):
        if binom_sf(k * capacity, seats, q) <= eps:
            return k
    return None


def seats_by_volume(agg_tok_s: float, month_hours: float, utilization: float, dev_output_month: float) -> float:
    """Capacity-equivalent developer seats by observed monthly generation volume (NOT concurrent users)."""
    return agg_tok_s * 3600.0 * month_hours * utilization / dev_output_month


def pods_needed(seats: int, agg_tok_s: float, capacity: int, q: float, eps: float, month_hours: float,
                utilization: float, dev_output_month: float) -> dict:
    by_volume = math.ceil(seats / seats_by_volume(agg_tok_s, month_hours, utilization, dev_output_month)) \
        if agg_tok_s > 0 else None
    by_conc = pods_for_concurrency(seats, capacity, q, eps)
    both = [x for x in (by_volume, by_conc) if x is not None]
    return {"by_volume": by_volume, "by_concurrency": by_conc, "pods": max(both) if both else None,
            "binding": None if not both else ("concurrency" if (by_conc or 0) >= (by_volume or 0) else "volume")}


def pod_eur_month(usd_per_hour: float, usd_per_eur: float, hours: float) -> float:
    return usd_per_hour / usd_per_eur * hours


def break_even_seats(pod_month_eur: float, seat_eur: float) -> float:
    return pod_month_eur / seat_eur


def break_even_utilization(pod_month_eur: float, seat_eur: float, agg_tok_s: float, month_hours: float,
                           dev_output_month: float) -> float:
    """Utilization (share of the pod's SLO-compliant monthly output) at which the developers it serves would cost as
    much in subscriptions as the pod."""
    return break_even_seats(pod_month_eur, seat_eur) * dev_output_month / (agg_tok_s * 3600.0 * month_hours)
