# E4: Placement policy at a fixed 30 percent share

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: how much of the result is due to profile-guided placement and how much it depends on the traffic.
- Command (on a pod prepared with `make pod-bootstrap`): `make bench-E4`
- GPU configuration: `2x-rtx-pro-6000` (prices in `experiments/pricing.yaml`), estimated 1.75 h.
- Expected layout: `data/controlled/E4-placement-policy/<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
- variant: exl3-offload, plan: gen30
- variant: exl3-offload, plan: agent30
- variant: exl3-offload, plan: blend30
- variant: exl3-offload, plan: random30, note: seeded uniform random, same per-layer counts as gen30
- variant: exl3-offload, plan: profile-30, note: our plan from E3 counts (disjoint calibration prompts)
