# E3: Host share as run (rung (c) of the ablation ladder): the gen30 plan against the union of the gen30 and agent30 cold sets on pod 1, and the gen30 subset repeated on pod 2 (variant gen30-session2), one repetition per point

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
| `gen30/20261006T203206Z` | 1 (`LEAN/session.json`) | complete | loadgen_rc=0 | ready after 507 s | 1 |
| `gen30-session2/20261007T063646Z` | 2 (`LEAN/session2.json`) | complete | loadgen_rc=0 | ready after 341 s | 1 |
| `union-gen30-agent30/20261006T210155Z` | 1 (`LEAN/session.json`) | complete | loadgen_rc=1 | ready after 322 s | 1 |
| `uva-equal-gen30/20261006T204843Z` | 1 (`LEAN/session.json`) | complete | boot_failed | server exited during boot | - |
| `uva-equal-gen30-bs128/20261007T064127Z` | 2 (`LEAN/session2.json`) | failed | loadgen_rc=1 | ready after 191 s | 1 |
| `uva-equal-gen30-failed/20261006T205505Z` | 1 (`LEAN/session.json`) | complete | boot_failed | server exited during boot | - |

- Question: the marginal performance cost of moving experts to host memory; the smallest share that fits; how much Diffbot's per-expert split gains over upstream vLLM's own selective UVA offload at equal host bytes.
- Command: `make bench-E3`; GPU configuration: `2x-rtx-pro-6000`.
- Designed as: Host-share sweep (ablation c, share of experts on host). What ran is reduced; see the paper's Appendix (experiment plan and status).
