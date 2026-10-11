"""Lean plan arithmetic and the plan-union tool."""
import json

import pytest

from hostoffload.cost import plan_budget
from hostoffload.weights import plan as P


def test_lean_budget_fits_two_hours():
    cfg, pricing = plan_budget.load()
    b = plan_budget.lean(cfg, pricing)
    assert b["usd_per_hour"] == 4.18
    assert b["target_minutes"] == b["hard_cap_minutes"] == 120
    assert b["core_minutes_cont"] <= b["target_minutes"]
    assert b["core_usd"] < b["all_usd"] < b["soft_cap_usd"] < b["hard_cap_usd"]
    assert b["watchdog_seconds"] == 7200
    assert b["rows"][-1]["cum_usd"] == pytest.approx(b["all_usd"])
    ids = [r["id"] for r in b["rows"]]
    assert ids.index("L1") < ids.index("L2") < ids.index("L3") < ids.index("L4") < ids.index("S1")


def test_lean_runner_matches_plan():
    cfg, _ = plan_budget.load()
    sh = (plan_budget.ROOT / "experiments" / "runs" / "lean.sh").read_text()
    for s in cfg["lean"]["steps"]:
        if s["id"] != "L5":
            assert f"step {s['id']}-" in sh and f" {s['minutes']} " in sh.split(f"step {s['id']}-")[1].split("\n")[0]
    assert "SOFT_CAP_USD:-%s}" % cfg["lean"]["soft_cap_usd"] in sh


def test_api_cost_formula():
    cfg, pricing = plan_budget.load()
    a = plan_budget.api_cost(cfg["lean"], pricing)
    p = pricing["api"]["deepseek-flash"]["peak"]
    assert a["usd_peak"] == pytest.approx(a["input_tokens"] / 1e6 * p["input_cache_miss"] + a["output_tokens"] / 1e6 * p["output"])
    assert a["usd_off_peak"] == pytest.approx(a["usd_peak"] / 2)


def test_union_plan_is_nested(tmp_path):
    a = {"m.layers.0.ffn.experts": [1, 2], "m.layers.40.ffn.experts": [5]}
    b = {"m.layers.0.ffn.experts": [2, 3], "m.layers.1.ffn.experts": [7]}
    u = P.union([a, b])
    assert u == {"m.layers.0.ffn.experts": [1, 2, 3], "m.layers.1.ffn.experts": [7], "m.layers.40.ffn.experts": [5]}


def test_lean_quality_subsets_fixed():
    d = json.load(open(plan_budget.ROOT / "quality" / "subsets_lean2h_seed20261006.json"))
    assert {k: len(v) for k, v in d.items()} == {"gsm8k_cot_llama": 150, "ifeval": 150, "humaneval_instruct": 164}
    cfg, _ = plan_budget.load()
    assert {k: v["items"] for k, v in cfg["lean"]["api_reference"]["tokens_per_item"].items()} == {
        k: len(v) for k, v in d.items()}
    assert d["gsm8k_cot_llama"] == sorted(set(d["gsm8k_cot_llama"])) and max(d["gsm8k_cot_llama"]) < 1319
