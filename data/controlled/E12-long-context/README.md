# E12: Long context at 128K and 256K tokens as run (pod 2; prefill, decode and cached shared prefix, one repetition)

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `long/20261007T061605Z` | 2 (`LEAN/session2.json`) | complete | loadgen_rc=0 | ready after 341 s | 1 |

- Question: whether the E8 rates hold when prompts approach the 262,144-token context, and how many such requests the recipe's KV cache admits.
- Command: `POD_CREATED_EPOCH=<create time> bash experiments/runs/long.sh`; GPU configuration: `2x-rtx-pro-6000`.
- Designed as: Long context at 128K and 256K tokens (prefill, decode and cached shared prefix). What ran is reduced; see the paper's Appendix (experiment plan and status).
