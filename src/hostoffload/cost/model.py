"""Cost model: pod price to cost per token, per concurrent request and under an SLA.

All inputs come from files: prices from experiments/pricing.yaml (with the FX rate and its date), throughput and
latency from measured level summaries. Nothing here has defaults that stand in for measurements.

Attribution of the pod cost between input and output tokens is a modelling choice, so three are offered:
- output_only: the whole cost per hour is charged to output tokens (input is free). Upper bound for output.
- dedicated: input cost from a prefill-only probe (the pod doing nothing but prefill), output cost from a
  decode-only probe. Each is the cost if the pod did only that kind of work.
- ratio: a fixed price ratio k = price_in / price_out (as API price lists do); solved from a mixed-load point so
  that rate_in * price_in + rate_out * price_out = cost per second.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Price:
    config: str
    usd_per_hour: float
    usd_per_eur: float
    fx_date: str
    source: str

    @property
    def eur_per_hour(self) -> float:
        return self.usd_per_hour / self.usd_per_eur


def load_price(config: str, pricing_file: Path = ROOT / "experiments" / "pricing.yaml") -> Price:
    p = yaml.safe_load(Path(pricing_file).read_text())
    c = p["configs"][config]
    return Price(config, float(c["usd_per_hour"]), float(p["fx"]["usd_per_eur"]), str(p["fx"]["date"]), c["source"])


def usd_per_million(usd_per_hour: float, tokens_per_s: float | None) -> float | None:
    """USD per 1M tokens at a sustained rate of tokens_per_s."""
    if not tokens_per_s or tokens_per_s <= 0:
        return None
    return usd_per_hour / (tokens_per_s * 3600.0) * 1e6


def per_request_cost(usd_per_hour: float, wall_s: float, n_requests: int) -> float | None:
    """USD per completed request over a level: pod cost of the level's wall time divided by its requests."""
    if n_requests <= 0 or wall_s <= 0:
        return None
    return usd_per_hour * wall_s / 3600.0 / n_requests


def per_concurrent_request_hour(usd_per_hour: float, concurrency: float) -> float | None:
    """USD per active-request-hour at a concurrency level (cost of keeping one request in flight for an hour)."""
    if concurrency <= 0:
        return None
    return usd_per_hour / concurrency


def split_dedicated(usd_per_hour: float, prefill_tok_s: float | None, decode_tok_s: float | None) -> dict:
    return {"usd_per_m_input": usd_per_million(usd_per_hour, prefill_tok_s),
            "usd_per_m_output": usd_per_million(usd_per_hour, decode_tok_s)}


def split_ratio(usd_per_hour: float, in_tok_s: float, out_tok_s: float, k: float) -> dict:
    """Prices with price_in = k * price_out that recover the pod cost at a mixed load (in_tok_s, out_tok_s)."""
    denom = k * in_tok_s + out_tok_s
    if denom <= 0:
        return {"usd_per_m_input": None, "usd_per_m_output": None}
    price_out_per_token = (usd_per_hour / 3600.0) / denom
    return {"usd_per_m_input": k * price_out_per_token * 1e6, "usd_per_m_output": price_out_per_token * 1e6, "k": k}


def best_point_under_sla(points: list[dict], ok_key: str = "meets_sla", rate_key: str = "out_tok_s") -> dict | None:
    """The load point with the highest output rate among those that meet the SLA."""
    ok = [p for p in points if p.get(ok_key) and p.get(rate_key)]
    return max(ok, key=lambda p: p[rate_key]) if ok else None


def cost_table(price: Price, points: list[dict]) -> list[dict]:
    """points: [{label, concurrency, out_tok_s, in_tok_s, wall_s, n_requests, meets_sla?}] -> cost rows."""
    rows = []
    for p in points:
        rows.append({
            "label": p["label"], "concurrency": p.get("concurrency"),
            "usd_per_hour": price.usd_per_hour, "eur_per_hour": price.eur_per_hour,
            "usd_per_m_output_output_only": usd_per_million(price.usd_per_hour, p.get("out_tok_s")),
            "usd_per_request": per_request_cost(price.usd_per_hour, p.get("wall_s", 0), p.get("n_requests", 0)),
            "usd_per_active_request_hour": per_concurrent_request_hour(price.usd_per_hour, p.get("concurrency") or 0),
            "meets_sla": p.get("meets_sla"),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cost per token from a pod price and a measured throughput")
    ap.add_argument("--config", required=True, help="pricing.yaml config name")
    ap.add_argument("--out-tok-s", type=float, required=True)
    ap.add_argument("--in-tok-s", type=float, default=0.0)
    ap.add_argument("--k", type=float, default=None, help="input/output price ratio for the ratio split")
    args = ap.parse_args(argv)
    price = load_price(args.config)
    res = {"config": args.config, "usd_per_hour": price.usd_per_hour, "eur_per_hour": round(price.eur_per_hour, 4),
           "fx": {"usd_per_eur": price.usd_per_eur, "date": price.fx_date},
           "usd_per_m_output_output_only": usd_per_million(price.usd_per_hour, args.out_tok_s)}
    if args.k is not None:
        res["ratio_split"] = split_ratio(price.usd_per_hour, args.in_tok_s, args.out_tok_s, args.k)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
