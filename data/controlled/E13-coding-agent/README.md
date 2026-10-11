# E13: Interactive coding-agent workload (streams at 40 output tokens/s per stream, by context-sharing regime)

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `agent/20261007T120153Z` | 3 (`LEAN/session3.json`) | complete | loadgen_rc=0 | ready after 360 s | 1 |

- Question: how many streams generate at the same time with a median of at least 40 output tokens/s each, for short, shared-cached, independent long and agent-turn contexts.
- Command: `POD_CREATED_EPOCH=<create time> bash experiments/runs/agent.sh`; GPU configuration: `2x-rtx-pro-6000`.
- Designed as: Interactive coding-agent workload (streams at 40 output tokens/s per stream, by context-sharing regime). What ran is reduced; see the paper's Appendix (experiment plan and status).
