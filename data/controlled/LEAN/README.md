# LEAN: pod sessions of the lean plan

**Status: three sessions run**, each on a different 2x RTX PRO 6000 host.

| session | file | date | runner | steps | manifest | experiments |
|---|---|---|---|---|---|---|
| 1 | `session.json` | 2026-10-06 | `experiments/runs/lean.sh` | `steps/` | `manifest.sha256` | E2, E8, E3 (gen30, union, UVA at KV block 64), E10 |
| 2 | `session2.json` | 2026-10-07 | `experiments/runs/long.sh` | `steps-session2/` | `manifest-session2.sha256` | E12, E3 (gen30-session2, UVA at KV block 128) |
| 3 | `session3.json` | 2026-10-07 | `experiments/runs/agent.sh` | `steps-session3/` | `manifest-session3.sha256` | E13 |

Each `session*.json` records the pod, create and delete times, cost and the attempts (session 1: two bootstrap
attempts aborted before the valid one; session 2: one aborted bootstrap attempt; session 3: one failed stack fetch).
Invalid outputs of aborted attempts stayed on the pod and are not in `data/`. The step folders hold one marker per step
(`<id>.done` with minutes and USD spent, or `<id>.skipped` when the budget check skipped it) and the download,
bootstrap and runner logs. Each manifest lists every file of `data/controlled/` as hashed on the pod before
copy-back; the copy was verified against it. The experiment READMEs were rewritten on the workstation afterwards
(`make readmes`), so their hashes no longer match the manifests. The measurements themselves are in the experiment
folders. E11 ran from the workstation through DeepSeek's API (`E11-quality-reference/api/`), not on a pod.
