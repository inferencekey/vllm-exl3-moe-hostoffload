# E7: DSpark on vs off (ablation d)

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: how much of the decode speed is speculation, and its dependence on content (code vs prose).
- Command (on a pod prepared with `make pod-bootstrap`): `make bench-E7`
- GPU configuration: `2x-rtx-pro-6000` (prices in `experiments/pricing.yaml`), estimated 2.5 h.
- Expected layout: `data/controlled/E7-dspark/<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
- variant: exl3-offload, spec: dspark
- variant: exl3-offload, spec: none
