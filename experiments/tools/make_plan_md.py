#!/usr/bin/env python3
"""Generate experiments/PLAN.md (budget table) from experiments/experiments.yaml and experiments/pricing.yaml.
Totals are computed here, never typed by hand."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hostoffload.cost import plan_budget  # noqa: E402


def lean_section(cfg: dict, pricing: dict) -> str:
    b = plan_budget.lean(cfg, pricing)
    L = cfg["lean"]
    rows = "\n".join(
        f"| {r['id']} | {r['priority']} | {r['title']} | {r['minutes']} | {r['cum_minutes']:.0f} | {r['usd']:.2f} | "
        f"{r['cum_usd']:.2f} | {r['basis']} |" for r in b["rows"])
    a = b["api"]
    na = "\n".join(f"- {x}" for x in L["not_answered"])
    return f"""## RECOMMENDED: lean plan (one 2x RTX PRO 6000 RunPod session, at most two hours)

The owner rejected the full and minimal budgets below and asked for a single session of at most
{b['target_minutes']} minutes of pod time; this is the plan to approve. One pod at {b['usd_per_hour']:.2f} USD/h; boot,
bootstrap and the 309 GiB download are paid once. Steps run in priority order, each resumable, and
`experiments/runs/lean.sh` starts a step only if it ends, copy-back included, under the soft cap of
{b['soft_cap_usd']} USD ({b['soft_cap_minutes']} min). The watchdog deletes the pod at the hard cap of
**{b['hard_cap_minutes']} min, {b['hard_cap_usd']} USD** (`experiments/pod/watchdog.sh <pod> {b['watchdog_seconds']}`).
Command: `{L['command']}`.

RunPod credentials: `RUNPOD_API_KEY` is in the SOPS profile `main` of the inferencekey repo. Inject it, never print it:
`cd ~/me/inferencekey && scripts/secrets.sh run main sh -c 'cd ~/me/vllm-exl3-moe-hostoffload && exec <command>'`.

| step | priority | what | min | cumulative min | USD | cumulative USD | basis of the estimate |
|---|---|---|---:|---:|---:|---:|---|
{rows}

| | minutes | USD |
|---|---:|---:|
| core steps (L0 to L5) | {b['core_minutes']} | {b['core_usd']:.2f} |
| core + {b['contingency']:.0%} contingency | {b['core_minutes_cont']:.0f} | **{b['core_usd_cont']:.2f}** |
| core + stretch (S1) | {b['all_minutes']} | {b['all_usd']:.2f} |
| hard cap (watchdog) | {b['hard_cap_minutes']} | {b['hard_cap_usd']:.2f} |

Why two hours are enough: the core steps take {b['core_minutes']} minutes ({b['core_usd']:.2f} USD), and a
{b['contingency']:.0%} margin for slow boots and downloads gives {b['core_minutes_cont']:.0f} minutes, under the
{b['target_minutes']}-minute limit. To fit, the lean plan drops the upstream boots with the original checkpoint (and its
475 GiB download), runs E8 with two repetitions instead of three, shrinks the quality subsets, and moves the second
host share to a stretch step. If boots are slower than estimated, the stretch step (which runs last) is skipped first and
the UVA comparison (the last core step) next, because lean.sh checks the budget before each step; the watchdog ends the
session at {b['hard_cap_minutes']} minutes whatever happens.

**Quality reference without a second pod (E11-api).** The official DeepSeek API serves DeepSeek-V4.1-Flash as
`deepseek-flash` (pricing page fetched 2026-10-06, saved in `data/static/pricing/`): USD per 1M tokens, peak
{pricing['api']['deepseek-flash']['peak']['input_cache_miss']} input (cache miss) and
{pricing['api']['deepseek-flash']['peak']['output']} output, off-peak half. With the estimated token counts in
`experiments.yaml` ({a['input_tokens']:,} input and {a['output_tokens']:,} output tokens for the three subsets) the API
run costs about **{a['usd_peak']:.2f} USD at peak and {a['usd_off_peak']:.2f} USD off-peak**, outside the pod budget.
The proxy (`quality/api_proxy.py`) logs real usage, so the actual cost is recomputed from data.

**What the lean plan cannot answer:**
{na}

"""


def main() -> int:
    cfg = yaml.safe_load((ROOT / "experiments" / "experiments.yaml").read_text())
    pricing = yaml.safe_load((ROOT / cfg["pricing_file"]).read_text())["configs"]
    cont = float(cfg["contingency"])
    rows, tot, tot_min = [], 0.0, 0.0
    by_gpu, by_gpu_min = {}, {}
    for e in cfg["experiments"]:
        price = pricing[e["gpu"]]["usd_per_hour"] if e["gpu"] != "none" else 0.0
        usd = price * e["hours"]
        mh = e.get("minimal_hours", e["hours"]) if e.get("minimal") else 0.0
        musd = price * mh
        tot += usd
        tot_min += musd
        by_gpu[e["gpu"]] = by_gpu.get(e["gpu"], 0) + e["hours"]
        if mh:
            by_gpu_min[e["gpu"]] = by_gpu_min.get(e["gpu"], 0) + mh
        rows.append(f"| {e['id']} | {e['title']} | `{e['gpu']}` | {price:.2f} | {e['hours']:.2f} | {usd:.2f} | "
                    f"{'yes' if e.get('minimal') else 'no'} | {mh:.2f} | {musd:.2f} | `{e['command']}` |")
    gpu_lines = "\n".join(f"| `{g}` | {by_gpu[g]:.2f} | {by_gpu_min.get(g, 0):.2f} |" for g in sorted(by_gpu))
    lean_md = lean_section(cfg, yaml.safe_load((ROOT / cfg["pricing_file"]).read_text()))
    md = f"""# Experiment plan and budget

Generated by `make plan` from `experiments/experiments.yaml` and `experiments/pricing.yaml`. Do not edit by hand.

**Only the lean plan has been run** (one pod session on 2026-10-06, `data/controlled/LEAN/session.json`); the full and
minimal plans have not. Every GPU run needs the budget approved by the owner first. Hours are pod wall-clock estimates from the exploratory runs (boot, download, measurement,
teardown); prices are the quotes in `experiments/pricing.yaml` (2x RTX PRO 6000 at 4.18 USD/h in EUR-IS-1,
B300 at 7.89 USD/h per GPU, A100 SXM 80 GB at 1.59 USD/h). Results go to the `data` folder of each experiment,
which holds only a README saying "not run yet" until a run exists.

{lean_md}## Reference only: full and minimal plans (rejected by the owner)


| id | experiment | pod config | USD/h | hours | USD | in minimal subset | minimal hours | minimal USD | command |
|---|---|---|---:|---:|---:|---|---:|---:|---|
{chr(10).join(rows)}

## Totals

| | full plan | minimal subset |
|---|---:|---:|
| pod cost (USD) | {tot:.2f} | {tot_min:.2f} |
| contingency ({cont:.0%}) | {tot * cont:.2f} | {tot_min * cont:.2f} |
| **total with contingency (USD)** | **{tot * (1 + cont):.2f}** | **{tot_min * (1 + cont):.2f}** |

Pod hours by configuration:

| config | full plan h | minimal h |
|---|---:|---:|
{gpu_lines}

## The minimal subset and why it answers the core question

The paper's question is: what does the offload code make possible that was not possible before, and at what
performance cost? The minimal subset answers it with:

1. **E2** (feasibility): shows, with the exact error, that on 2x RTX PRO 6000 neither the original checkpoint nor the
   EXL3 pack serves with all experts in VRAM, and whether upstream vLLM's own UVA offload serves the original.
2. **E8 minimal** (main benchmark): the measured stack's TTFT, TPOT, E2E percentiles, throughput, memory split and PCIe
   traffic from N = 1 to 64 with 3 repetitions.
3. **E3 minimal** (host share 30 vs 50 percent, and upstream UVA offload at equal host bytes): the marginal cost of
   host-resident experts and what Diffbot's per-expert split adds over the upstream mechanism.
4. **E9 minimal** (2x B300 reference): the speed of the unquantized model fully in VRAM.
5. **E10/E11 minimal** (quality): GSM8K, IFEval and HumanEval for the EXL3 stack against the reference.

## Order and stop rules

- Run E1 on a CPU pod first (or locally with docker): it validates the build without GPU spend.
- Then E2 and E8 on one 2x RTX PRO 6000 pod (one weight download), E3 and E10 on the same pod if approved.
- E9 and E11 share one 2x B300 pod (one download of the original checkpoint).
- Hard stop per pod: `experiments/pod/watchdog.sh <pod> <seconds>` deletes the pod at the approved budget.
- If a run fails twice for infrastructure reasons, stop and report; do not retry blindly.

## How to run one experiment once a pod exists

```bash
# on the pod (vLLM image), from a checkout of this repository:
make pod-bootstrap VARIANT=exl3-offload    # build the variant in place (stack/build/bootstrap_pod.sh)
make download MODEL=exl3                     # pinned weights to /root/models (never into the repo)
make bench-E8                                # writes data/controlled/E8-main/<variant>/<run_id>/
# back on the workstation:
make analyze figures paper
```
"""
    (ROOT / "experiments" / "PLAN.md").write_text(md)
    b = plan_budget.lean(cfg, yaml.safe_load((ROOT / cfg["pricing_file"]).read_text()))
    print(f"lean: core {b['core_minutes']} min {b['core_usd']:.2f} USD, with contingency {b['core_minutes_cont']:.0f} min "
          f"{b['core_usd_cont']:.2f} USD, with stretch {b['all_minutes']} min, cap {b['hard_cap_minutes']} min")
    print(f"PLAN.md: full {tot:.2f} USD (+{cont:.0%} = {tot * (1 + cont):.2f}), "
          f"minimal {tot_min:.2f} USD (+{cont:.0%} = {tot_min * (1 + cont):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
