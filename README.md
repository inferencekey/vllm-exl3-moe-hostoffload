# vllm-exl3-moe-hostoffload

A reproducible research artifact on serving DeepSeek-V4.1-Flash, an extremely large MoE, quantized to 3-bit EXL3
experts, with **vLLM** on **two 96 GB RTX PRO 6000 GPUs plus pinned host memory**. It asks one question:

> What exactly does the host-memory offload code make possible that was not possible before, and at what performance
> cost?

## Status (2026-10-06)

| Part | State |
|---|---|
| Attribution audit (`audit/`) | **done**: every model, kernel, patch and placement algorithm is third-party (Diffbot, vcruz305, turboderp, the vLLM project); upstream vLLM already has zero-copy UVA weight offload. Our contribution is integration, configuration and measurement |
| Static byte accounting (`data/static/`) | **done** from safetensors headers: without host offload the pack needs 102.1 GiB of weights per GPU against 95.6 GiB; with the published plan, 72.4 GiB |
| Exploratory data (`data/exploratory/`) | imported verbatim with hashes, re-analyzed, **labelled exploratory** |
| Controlled experiments E1 to E11 (`experiments/`) | **lean plan run** on 2026-10-06 (one 2x RTX PRO 6000 pod, 101 min, 7.06 USD): E2, E8, E3 (gen30 and gen30 union agent30; the UVA rung failed on a KV block size misconfiguration), E10, and E11 through the DeepSeek API; a second session on 2026-10-07 (43 min, 3.01 USD) ran E12 (long context at 128K and 256K) and showed that upstream UVA offload cannot serve the model without Diffbot's hooks; a third session (57 min, 3.98 USD) ran E13, the coding-agent case study (4 simultaneous streams at 40 tok/s or more in every context regime; seat and cost model in `experiments/agent_workload.yaml`). E1, E4 to E7 and E9 not run; the paper shows NOT RUN placeholders for them |
| Paper (`paper/`) | *Serving a 763B Mixture-of-Experts Model on Workstation GPUs: A Measurement Study of Expert Offload in vLLM*; `make paper` builds `paper/build/main.pdf`, and every file the paper names links to this repository |

The numbers in this table come from `paper/generated/values.tex` (keys `static.noplan.vram_per_gpu_gib`,
`expl.gpu_total_gib`, `static.gen30.vram_per_gpu_gib`).

## Quickstart

```bash
make env            # Python 3.12 venv from the hash-pinned requirements.lock (needs uv)
make test           # harness, parsers, cost model, statistics, no-fake-data guard, manifests
make paper          # fetch pinned recipe files, analyze data/, build paper/build/main.pdf (needs TeX + graphviz)
make help           # every target
```

Other workstation targets (no GPU, no spending): `make audit` (re-run the attribution audit), `make static` (re-measure
the byte accounting), `make stack-check` (dry-run every patch), `make plan` (budget table), `make release-check`.

On a GPU pod, after the budget is approved (`experiments/PLAN.md`):

```bash
make stack-fetch pod-bootstrap VARIANT=exl3-offload
make download MODEL=exl3
POD_CREATED_EPOCH=<create time> bash experiments/runs/lean.sh   # the recommended lean plan
```

## Budget (`experiments/PLAN.md`, generated)

**Recommended: the lean plan**, one 2x RTX PRO 6000 RunPod session of at most two hours
(`experiments/runs/lean.sh`), steps in priority order with a budget check before each, watchdog at 120 minutes:

| | pod minutes | USD |
|---|---:|---:|
| core steps (setup, E2 EXL3 boots, E8 with 2 repetitions, reduced E10, E3 gen30 vs UVA, copy-back) | 98 | 6.83 |
| core with 20% contingency | 118 | 8.19 |
| with the stretch step (second host share, gen30 union agent30) | 108 | 7.52 |
| hard cap (watchdog) | 120 | 8.36 |

`RUNPOD_API_KEY` comes from the SOPS profile `main` of the inferencekey repo; inject it with
`scripts/secrets.sh run main ...` from that repo, never print it (see `experiments/PLAN.md`).

Quality reference in the lean plan: the official DeepSeek API (about 0.23 USD at peak for the three subsets, from
estimated token counts and the published price). Not measured in the lean plan: the unquantized all-in-VRAM speed
and upstream vLLM with the original checkpoint (future work). The full (186.40 USD) and minimal (84.96 USD) plans,
both with contingency, are kept for reference.

## Layout

| Path | Contents |
|---|---|
| `audit/` | `ATTRIBUTION.md` (who wrote what, file:line evidence, verdict), fetch and check scripts, evidence |
| `stack/` | pins, recipe lock, fetch scripts, per-variant build (`build/bootstrap_pod.sh`, `build/Dockerfile`), `serve.sh`, variant env files |
| `src/hostoffload/` | `harness/` (load generator, client, sampler, Prometheus parser), `analysis/` (metrics, statistics, pipeline, registry, parsers), `cost/`, `weights/` (safetensors headers, placement accounting, plans) |
| `experiments/` | experiment registry, plan, pricing, SLOs, corpora, workloads, pod runners, pod tooling |
| `quality/` | lm-evaluation-harness protocol and runner |
| `data/` | `static/` (offline measurements), `exploratory/` (historical runs, hash manifest), `controlled/` (empty until runs) |
| `paper/` | LaTeX sources, verified bibliography, review protocol, generated values, tables and figures |
| `diagrams/` | graphviz sources |
| `tests/` | pytest suite (includes a mock vLLM server for harness tests) |

## Rules this artifact enforces

- No number without a file: every value in the paper goes through `\dv{}` and `paper/generated/provenance.json`; a
  missing value renders NOT RUN, and `make paper STRICT=1` fails.
- Exploratory data never feeds controlled tables.
- No third-party source code and no model weights in the tree (`THIRD_PARTY.md`, `PROVENANCE.md`).

## License

Code: Apache-2.0 (`LICENSE`). Text and data: CC-BY-4.0 (`LICENSE-data`). Built `exl3-*` images contain AGPL-3.0
components (vllm-exl3 and Diffbot's patch to it); see `THIRD_PARTY.md`. Cite with `CITATION.cff`.

**Why Apache-2.0:** no AGPL code is included here (it is fetched and applied at build time), so the repository need
not be AGPL; Apache-2.0 matches vLLM's license, carries an explicit patent grant, and can be combined with the AGPL-3.0
components in a built image.
