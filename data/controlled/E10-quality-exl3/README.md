# E10: Quality of the EXL3 stack (lm-evaluation-harness, pinned)

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `exl3-offload-gen30-lean/20261006T203546Z` | 1 (`LEAN/session.json`) | invalid | done | ready after 291 s | - |
| `exl3-offload-gen30-lean-v2/20261006T211059Z` | 1 (`LEAN/session.json`) | complete | done | ready after 296 s | - |

- Question: accuracy on GSM8K, MATH-500, HumanEval, MBPP, IFEval, RULER subsets.
- Command: `make quality-E10`; GPU configuration: `2x-rtx-pro-6000`.
- Designed as: Quality of the EXL3 stack (lm-evaluation-harness, pinned). What ran is reduced; see the paper's Appendix (experiment plan and status).
