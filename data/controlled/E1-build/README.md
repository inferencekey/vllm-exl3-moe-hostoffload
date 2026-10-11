# E1: Reproducible build of every variant (image digests, pip freeze, build logs)

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: the stack builds from pinned sources and patches; provenance of the binaries.
- Command (on a pod prepared with `make pod-bootstrap`): `make bench-E1`
- GPU configuration: `cpu-build` (prices in `experiments/pricing.yaml`), estimated 1.5 h.
- Expected layout: `data/controlled/E1-build/<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
- see experiments/experiments.yaml
