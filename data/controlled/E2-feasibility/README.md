# E2: Feasibility boots on 2x RTX PRO 6000 (ablation a and b without offload)

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `exl3-min-allvram/20261006T200309Z` | 1 (`LEAN/session.json`) | complete | boot_failed | server exited during boot | - |
| `exl3-offload-noplan/20261006T200651Z` | 1 (`LEAN/session.json`) | complete | boot_failed | server exited during boot | - |

- Question: what was not possible before the offload patch on this hardware, with the exact failure.
- Command: `make bench-E2`; GPU configuration: `2x-rtx-pro-6000`.
- Designed as: Feasibility boots on 2x RTX PRO 6000 (ablation a and b without offload). What ran is reduced; see the paper's Appendix (experiment plan and status).
