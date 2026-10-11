# E9: Upstream all-in-VRAM reference on 2x B300 (original checkpoint)

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: the speed of the same model without quantization or offload, on hardware where it fits.
- Command (on a pod prepared with `make pod-bootstrap`): `make bench-E9`
- GPU configuration: `2x-b300` (prices in `experiments/pricing.yaml`), estimated 2.5 h.
- Expected layout: `data/controlled/E9-reference-b300/<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
- variant: upstream, model: original, workload: experiments/workloads/main.yaml
