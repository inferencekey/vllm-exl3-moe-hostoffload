# Release checklist

Outward-facing steps need the owner's explicit approval. Nothing below the line "needs approval" has been done.

## Prepared (done)

- [x] Code license Apache-2.0 (`LICENSE`), text and data CC-BY-4.0 (`LICENSE-data`), third-party notices (`THIRD_PARTY.md`)
- [x] `CITATION.cff`, `.zenodo.json`, `PROVENANCE.md`
- [x] `make release-check` (dashes, out-of-scope terms, identifiers, license files) and `make release` (tarball in `dist/`,
      no publication)
- [x] CI workflow (`.github/workflows/ci.yml`), not run

## Before any public release (needs approval)

- [ ] Owner confirms the license of the imported exploratory data (the source repository has no license file;
      `data/exploratory/README.md` proposes CC-BY-4.0)
- [x] Identifiers in `data/exploratory/.../b300/logs/created*.txt` (a RunPod account user id and an SSH public key)
      redacted in place, recorded in `data/exploratory/import_spec.json` and `MANIFEST.tsv` (mode `redacted`)
- [ ] Decide whether to quote Diffbot's model-card quality claims with numbers (currently cited without numbers)
- [ ] Create the GitHub repository `inferencekey/vllm-exl3-moe-hostoffload` (private first) with a single initial
      commit, review it, then make it public (needs approval)
- [ ] At the tag, pin `\repoblob` in `paper/main.tex` from `tree/main` to `tree/v0.1.0`
- [ ] Tag `v0.1.0` and create a GitHub release with `dist/*.tar.gz` and its sha256 (not done)
- [ ] Connect Zenodo and archive the release; add the DOI to `CITATION.cff`, `.zenodo.json` and the paper (not done)
- [ ] Run `make paper STRICT=1` only once all controlled experiments exist; until then the paper is a protocol paper
      with NOT RUN placeholders and must be described as such

## Commands

```bash
make release-check      # must report 0 errors
make test paper         # tests and PDF
make release            # dist/vllm-exl3-moe-hostoffload-<version>.tar.gz (+ .sha256); publishes nothing
```
