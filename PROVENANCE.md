# Provenance

How every input of this artifact is pinned and verified.

## Software stack (`stack/`)

| Input | Pin | Verification |
|---|---|---|
| vLLM image | `vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89@sha256:0ad6fa4178c34f4e04402452bbc4294df6dd2fb2785a0ecd402b2861aea2a189` | the digest; `bootstrap_pod.sh` refuses a different vLLM version string |
| vllm-exl3 | `d3cfd394920360d69f820d2dc96f8292a9e10283` | `fetch_stack.sh` checks `HEAD`; `bootstrap_pod.sh` refuses a dirty or moved tree |
| ExLlamaV3 | `5be886578ec80324c2c715269387be2058724b6e` | same |
| sfxnz recipe | `460ac89c7df1eea4db8a731f1feb807f1584f359` | same |
| Diffbot recipe | Hub `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000@a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5`, folder `recipe/` | `stack/fetch_recipe.py` verifies 95 files against `stack/recipe.lock.tsv` (git blob id from the Hub and sha256); the audit found the copy used by the exploratory runs byte-identical |
| Python environment | `requirements.lock` (uv, hash-pinned, Python 3.12) | `pip install --require-hashes` semantics through uv |
| lm-evaluation-harness | 0.4.13 (tag object `eb4678c3`) | task names validated (`quality/lm_eval_0.4.13_task_listing.txt`) |

Every build writes `build_provenance.json` (versions, git revisions, dirty flags, sha256 of patched files, pip freeze),
and every run copies it next to its data.

## Models

| Checkpoint | Revision | Size | Verification |
|---|---|---|---|
| EXL3 pack | `a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5` | 309.3 GiB | per-file LFS sha256 in `data/static/hub/` and `audit/evidence/hf_diffbot_tree_a1cf1bd2.tsv`; `VERIFY=1 make download` checks them |
| DeepSeek-V4.1-Flash | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | 475.2 GiB | same, `data/static/hub/` |

Weights are never stored in or redistributed by this repository.

## Data

- `data/static/`: regenerated from public files by `make static`; headers are cached outside git, derived tables are committed.
- `data/exploratory/`: imported from `github.com/inferencekey/inferencekey-deepseek` at commit
  `546071a76de290b47a43149924c95866f5b8ebf0` by `data/exploratory/import_exploratory.py`; `MANIFEST.tsv` has the git blob
  id and sha256 of every source blob and imported file. Redactions and excerpts are listed in `import_spec.json`.
- `data/controlled/`: written only by `experiments/runs/*.sh` on GPU pods; empty except for "not run yet" READMEs.

## Paper

`paper/generated/provenance.json` maps every printed value to its source files and their sha256.
