# E11: Quality of the reference (original checkpoint, upstream vLLM, 2x B300)

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `api/20261006T180421Z` | unknown | invalid | - | - | - |
| `api/20261006T181137Z` | unknown | complete | - | - | - |

- Question: the baseline for E10 under identical prompts, seeds and sampling.
- Command: `make quality-E11`; GPU configuration: `2x-b300`.
- Designed as: Quality of the reference (original checkpoint, upstream vLLM, 2x B300). What ran is reduced; see the paper's Appendix (experiment plan and status).

Lean plan: the reference is the official DeepSeek API (`deepseek-flash`, served by DeepSeek-V4.1-Flash), run through
`quality/api_proxy.py` with the fixed-seed subsets; results go to `api/<run_id>/` with `api_usage.jsonl` (real token
counts for the cost).
