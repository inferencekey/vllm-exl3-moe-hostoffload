"""Budget arithmetic for the experiment plans (full, minimal, lean), shared by experiments/tools/make_plan_md.py and
the paper pipeline so that PLAN.md and the paper print the same totals."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load(root: Path = ROOT) -> tuple[dict, dict]:
    cfg = yaml.safe_load((root / "experiments" / "experiments.yaml").read_text())
    pricing = yaml.safe_load((root / cfg["pricing_file"]).read_text())
    return cfg, pricing


def full_and_minimal(cfg: dict, pricing: dict) -> dict:
    prices = pricing["configs"]
    full = sum(prices[e["gpu"]]["usd_per_hour"] * e["hours"] for e in cfg["experiments"] if e["gpu"] != "none")
    mini = sum(prices[e["gpu"]]["usd_per_hour"] * e.get("minimal_hours", e["hours"])
               for e in cfg["experiments"] if e["gpu"] != "none" and e.get("minimal"))
    c = float(cfg["contingency"])
    return {"full_usd": full, "minimal_usd": mini, "contingency": c,
            "full_usd_cont": full * (1 + c), "minimal_usd_cont": mini * (1 + c),
            "full_hours": sum(e["hours"] for e in cfg["experiments"])}


def api_cost(lean: dict, pricing: dict) -> dict:
    p = pricing["api"][lean["api_reference"]["price"]["key"]]
    tin = sum(v["items"] * v["input"] for v in lean["api_reference"]["tokens_per_item"].values())
    tout = sum(v["items"] * v["output"] for v in lean["api_reference"]["tokens_per_item"].values())
    out = {"input_tokens": tin, "output_tokens": tout}
    for period in ("peak", "off_peak"):
        out[f"usd_{period}"] = tin / 1e6 * p[period]["input_cache_miss"] + tout / 1e6 * p[period]["output"]
    return out


def lean(cfg: dict, pricing: dict) -> dict:
    L = cfg["lean"]
    usd_h = pricing["configs"][L["pod"]]["usd_per_hour"]
    rows, cum = [], 0.0
    for s in L["steps"]:
        cum += s["minutes"]
        rows.append({**s, "usd": s["minutes"] / 60 * usd_h, "cum_minutes": cum, "cum_usd": cum / 60 * usd_h})
    core = sum(s["minutes"] for s in L["steps"] if s["priority"] == "core")
    allm = sum(s["minutes"] for s in L["steps"])
    c = float(L["contingency"])
    return {"usd_per_hour": usd_h, "rows": rows, "core_minutes": core, "all_minutes": allm,
            "core_usd": core / 60 * usd_h, "all_usd": allm / 60 * usd_h,
            "core_minutes_cont": core * (1 + c), "core_usd_cont": core / 60 * usd_h * (1 + c),
            "all_usd_cont": allm / 60 * usd_h * (1 + c), "contingency": c,
            "hard_cap_usd": L["hard_cap_usd"], "soft_cap_usd": L["soft_cap_usd"],
            "target_minutes": L["target_minutes"], "target_usd": L["target_minutes"] / 60 * usd_h,
            "hard_cap_minutes": round(L["hard_cap_usd"] / usd_h * 60),
            "soft_cap_minutes": round(L["soft_cap_usd"] / usd_h * 60),
            "watchdog_seconds": round(L["hard_cap_usd"] / usd_h * 3600), "api": api_cost(L, pricing)}
