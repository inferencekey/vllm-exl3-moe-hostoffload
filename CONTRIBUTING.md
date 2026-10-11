# Contributing

Thank you for helping. This artifact has a few hard rules, enforced by tests and `make release-check`.

1. **No number without a file.** Measured values reach the paper only through the analysis pipeline
   (`src/hostoffload/analysis/pipeline.py` -> `paper/generated/values.tex`) and `\dv{key}`. Never type a measured value
   into a `.tex` file; the literal lint in `tests/test_no_fake_data.py` will fail.
2. **Exploratory stays exploratory.** Nothing under `data/exploratory/` may feed a controlled table. Name every artifact
   derived from it `exploratory_*` and label it in its caption.
3. **Controlled data comes from the runners.** Only `experiments/runs/*.sh` write `data/controlled/`. Keep every
   repetition raw; never edit result files by hand. Mock-server output never goes under `data/`.
4. **No third-party code in the tree.** Fetch and patch at build time with pinned revisions and hashes
   (`stack/pins.env`, `stack/recipe.lock.tsv`). Never commit model weights.
5. **Scope.** This artifact covers the vLLM + EXL3 line only.
6. **Style.** English everywhere, no em or en dashes, commit messages with a subject and a body.
7. **Spending.** No GPU pod or other paid resource without an approved budget (`experiments/PLAN.md`).

Workflow:

```bash
make env          # .venv from requirements.lock
make test         # must pass
make analyze paper
make release-check
```
