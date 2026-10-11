# E6: Side-stream SM budget for host experts

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: sensitivity of decode and prefill to the overlap budget.
- Command (on a pod prepared with `make pod-bootstrap`): `make bench-E6`
- GPU configuration: `2x-rtx-pro-6000` (prices in `experiments/pricing.yaml`), estimated 1.75 h.
- Expected layout: `data/controlled/E6-overlap-sms/<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
- variant: exl3-offload, overlap_sms: decode {16,32,64} at prefill 64
- variant: exl3-offload, overlap_sms: prefill {32,96} at decode 32
