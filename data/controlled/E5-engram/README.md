# E5: Engram tables int4 vs FP8 in pinned host memory

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: the host-memory and speed cost of the Engram format.
- Command (on a pod prepared with `make pod-bootstrap`): `make bench-E5`
- GPU configuration: `2x-rtx-pro-6000` (prices in `experiments/pricing.yaml`), estimated 1.0 h.
- Expected layout: `data/controlled/E5-engram/<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
- variant: exl3-offload, engram: int4
- variant: exl3-offload, engram: fp8, note: needs 189 GiB pinned for Engram; feasibility check first (cgroup 351 GiB)
