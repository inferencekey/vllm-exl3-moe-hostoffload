# Exploratory data (historical runs, not controlled)

**Status: EXPLORATORY.** Nothing in this folder may appear in a controlled table or figure. The analysis
pipeline reads it only through `hostoffload.analysis.exploratory`, and every table or figure built from it is
labelled "exploratory" in its caption and file name (`paper/generated/exploratory_*`). A test
(`tests/test_no_fake_data.py`) fails if a controlled table reads from this folder.

## Origin

- Source repository: `github.com/inferencekey/inferencekey-deepseek`, commit
  `546071a76de290b47a43149924c95866f5b8ebf0` (2026-10-06).
- Imported by `import_exploratory.py` from git objects at that commit (`git show`), according to
  `import_spec.json`. `MANIFEST.tsv` lists every file with its git blob id, the sha256 of the source blob and the
  sha256 of the imported file. `tests/test_exploratory_manifest.py` checks those hashes.
- Re-import: `python3 data/exploratory/import_exploratory.py --src <clone of inferencekey-deepseek>`.

| Folder | What it is | Mode |
|---|---|---|
| `vllm-exl3/a100/` | First run on 2x RTX PRO 6000 (named "a100" because it was the A100-class test): scripts, `results/agent30/` (bench JSON and logs), `results/gen30/console.txt` | verbatim, except `README.md` (redacted, see below) |
| `vllm-exl3/pro6000-concurrency/` | N = 1 to 64 closed-loop sweep: `results/scenario_A.json`, `scenario_B.json`, `mem_samples.csv`, `logs/serve.log`, `logs/memory_startup.txt`, the 1M-context boot | verbatim |
| `vllm-exl3/b300/` | Failed B300 track: scripts and pod-creation logs. The built wheels were never tracked and are not imported | verbatim |
| `docs/REPORT_vllm_excerpt.md` | The vLLM parts of `docs/REPORT.md`: cost method and the S10 to S12 cost rows, metric definitions, sheets S10, S11, S12, the vLLM parts of the threats to validity and of the quality note | excerpt with line-level redactions |

Redactions remove only content about a different runtime that is out of scope here. Every edited line carries a
visible `[redacted ...]` marker, and `import_spec.json` lists the line numbers.

## Known weaknesses of these runs (stated in the paper)

1. **Single runs.** Every configuration was measured once. No repetition, no variance estimate.
2. **Post-hoc usable bars.** The "usable" thresholds (decode p5 >= 15 tok/s and TTFT p95 <= 5 s / 15 s; loose
   10 tok/s and 10 s / 30 s) were chosen after seeing the data.
3. **Burst arrivals.** Each concurrency level starts with all N requests at once; the first wave dominates TTFT p95.
   Only closed-loop arrivals were used, no open-loop (Poisson) arrivals.
4. **DSpark content dependence.** Speculative decoding makes decode speed depend on the generated text (prose about
   a third of code). Prompts were synthetic (slices of the vLLM sources, fixed question templates).
5. **Quality never evaluated.** No accuracy benchmark was run by the measuring team on this stack.
6. Further: tiny per-level samples (n = 8 to 128, nearest-rank percentiles), the host-expert plan of the sweep is not
   recorded in any log (agent30 or gen30), per-chunk rather than per-token timing with speculative decoding, the
   aggregate throughput includes ramp-up and drain, and one pod, one host, one day.

## Privacy note before publication

`vllm-exl3/b300/logs/created*.txt` contain a RunPod account user id and an SSH public key, as returned by the RunPod
API. They are kept verbatim here. `make release-check` flags them; decide before any public release whether to
redact them (the redaction would then be recorded in `import_spec.json`).

## License

The source repository has no license file. These files are the measuring team's own logs and scripts, imported by
their owner. They are distributed here under CC-BY-4.0 (see `LICENSE-data`), pending the owner's confirmation at
release time (`RELEASE_CHECKLIST.md`). The logs quote third-party software output (vLLM, vllm-exl3). The imported
`scripts/serve.sh` and `scripts/bootstrap.sh` are adaptations of Diffbot's MIT-licensed recipe (`serve/serve.sh`,
`docker/build.sh`, `kernels/build.sh`); the MIT notice that covers them is reproduced in `THIRD_PARTY.md` at the repo
root. No other third-party source code is included.
