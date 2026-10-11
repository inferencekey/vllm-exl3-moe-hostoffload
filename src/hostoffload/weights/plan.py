"""Host-expert plans for the ablations (our implementation; the plan file format is the one the patched plugin reads).

  from-counts  rank experts by routing share (max over profiles) and put the coldest X% on host, like the recipe's
               make_plan.py does; counts come from a calibration workload disjoint from the benchmark prompts
  random       seeded uniform random experts with the same per-layer counts as a reference plan
  union        experts that are cold in any of several plans (a nested, larger host share built from published
               plans only, without new routing counts)
  bytes        host bytes of a plan per GPU (to size upstream vLLM's --cpu-offload-gb at equal host bytes)

Counts JSON (written on the pod by experiments/tools/stats_pt_to_json.py from EXL3_EXPERT_STATS files):
  {"<layer key>": {"decode": [counts per expert], "prefill": [...]}, ...}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from .split import load_plan

TARGET_LAYERS = 40


def layer_of(key: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", key).group(1))


def plan_from_counts(profiles: list[dict], pct: float, rows: str = "decode") -> dict[str, list[int]]:
    """Globally coldest pct% of the target experts (layers < 40) by the maximum over profiles of each expert's
    share of all routings. Drafter layers are left out (all their experts stay in VRAM) unless given separately."""
    keys = sorted({k for p in profiles for k in p if layer_of(k) < TARGET_LAYERS}, key=layer_of)
    score: dict[tuple[str, int], float] = {}
    for p in profiles:
        total = sum(sum(p[k][rows]) for k in keys if k in p) or 1.0
        for k in keys:
            for e, c in enumerate(p.get(k, {}).get(rows, [])):
                score[(k, e)] = max(score.get((k, e), 0.0), c / total)
    n = round(len(score) * pct / 100.0)
    ranked = sorted(score, key=lambda ke: (score[ke], layer_of(ke[0]), ke[1]))
    out: dict[str, list[int]] = {k: [] for k in keys}
    for k, e in ranked[:n]:
        out[k].append(e)
    return {k: sorted(v) for k, v in out.items()}


def random_like(reference: dict[str, list[int]], n_experts: dict[str, int], seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    return {k: sorted(rng.sample(range(n_experts[k]), len(v))) for k, v in reference.items()}


def union(plans: list[dict[str, list[int]]]) -> dict[str, list[int]]:
    keys = sorted({k for p in plans for k in p}, key=layer_of)
    return {k: sorted(set().union(*(set(p.get(k, [])) for p in plans))) for k in keys}


def host_bytes(plan_path: Path, tensor_table: Path, tp: int = 2) -> dict:
    from .split import Placement, account, read_tensor_table
    acc = account(read_tensor_table(tensor_table), Placement(plan=load_plan(plan_path)))
    host = acc["by_class"]["routed_expert"].get("host", 0) + acc["by_class"]["drafter_routed_expert"].get("host", 0)
    return {"host_expert_bytes": host, "per_gpu_gib": host / tp / 2**30, "host_experts": acc["host_experts_total"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("from-counts")
    a.add_argument("--pct", type=float, required=True)
    a.add_argument("--counts", type=Path, action="append", required=True)
    a.add_argument("--rows", default="decode", choices=["decode", "prefill"])
    a.add_argument("--keep-drafter-from", type=Path, help="copy drafter layers (>= 40) from this plan")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("random")
    b.add_argument("--like", type=Path, required=True)
    b.add_argument("--seed", type=int, default=20261006)
    b.add_argument("--experts-per-layer", type=int, default=384)
    b.add_argument("--drafter-experts", type=int, default=128)
    b.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("union")
    u.add_argument("--plan", type=Path, action="append", required=True)
    u.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("bytes")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--table", type=Path, default=Path("data/static/weights/diffbot-a1cf1bd2/tensors.csv.gz"))
    args = ap.parse_args(argv)
    if args.cmd == "from-counts":
        plan = plan_from_counts([json.loads(p.read_text()) for p in args.counts], args.pct, args.rows)
        if args.keep_drafter_from:
            ref = json.loads(args.keep_drafter_from.read_text())
            plan.update({k: v for k, v in ref.items() if layer_of(k) >= TARGET_LAYERS})
        args.out.write_text(json.dumps(plan))
        print(f"{args.out}: {sum(len(v) for v in plan.values())} host experts")
    elif args.cmd == "random":
        ref = json.loads(args.like.read_text())
        n = {k: (args.experts_per_layer if layer_of(k) < TARGET_LAYERS else args.drafter_experts) for k in ref}
        plan = random_like(ref, n, args.seed)
        args.out.write_text(json.dumps(plan))
        print(f"{args.out}: {sum(len(v) for v in plan.values())} host experts (seed {args.seed})")
    elif args.cmd == "union":
        plan = union([json.loads(p.read_text()) for p in args.plan])
        args.out.write_text(json.dumps(plan))
        tgt = sum(len(v) for k, v in plan.items() if layer_of(k) < TARGET_LAYERS)
        print(f"{args.out}: {sum(len(v) for v in plan.values())} host experts ({tgt} target experts)")
    else:
        print(json.dumps(host_bytes(args.plan, args.table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
