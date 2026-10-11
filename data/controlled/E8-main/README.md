# E8: Main benchmark of the measured stack as run (gen30 plan; scenarios A and B at N = 1 to 64, two repetitions in B and in A only at N = 1; prefill and decode probes, one repetition)

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `main/20261006T200848Z` | 1 (`LEAN/session.json`) | complete | loadgen_rc=0 | ready after 507 s | 2 |
| `probes/20261006T203021Z` | 1 (`LEAN/session.json`) | complete | loadgen_rc=0 | ready after 507 s | 1 |

- Question: TTFT, TPOT, E2E percentiles, throughput, memory split and PCIe use of the stack, with variance.
- Command: `make bench-E8`; GPU configuration: `2x-rtx-pro-6000`.
- Designed as: Main controlled benchmark of the measured stack (concurrency 1 to 64, 3 repetitions, PCIe). What ran is reduced; see the paper's Appendix (experiment plan and status).
