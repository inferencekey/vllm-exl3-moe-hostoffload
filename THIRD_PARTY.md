# Third-party components and licenses

This repository contains **no third-party source code**. Everything the serving stack needs is fetched at build time
from pinned revisions (`stack/pins.env`, `stack/recipe.lock.tsv`) into `third_party/`, which is gitignored. The
per-component license evidence is in `audit/evidence/licenses.md`. This is a provenance record, not legal advice.

| Component | Pin | License | In this repo |
|---|---|---|---|
| vLLM (vllm-project) | `36768d1`, image digest `sha256:0ad6fa41...` | Apache-2.0 | not included; base image |
| vLLM PRs #58132, #58560, #59585, #59894 and part of #57028 (as carried by Diffbot's patches) | PR heads | Apache-2.0 | not included; fetched with the recipe |
| vllm-exl3 (vcruz305) | `d3cfd394` | **AGPL-3.0-only** | not included; cloned and installed at build time |
| Diffbot's `exl3-thor.patch` to vllm-exl3 | Hub `a1cf1bd2` | **AGPL-3.0-only** | not included; fetched and applied at build time |
| Diffbot recipe (all other files) | Hub `a1cf1bd2` | MIT, Copyright (c) 2026 Diffbot | not included; fetched and verified. Two imported exploratory scripts derive from it (below) |
| ExLlamaV3 (turboderp-org) | `5be88657` | MIT, Copyright (c) 2025 Turboderp | not included; built from source at build time |
| sfxnz recipe | `460ac89c` | MIT (its LICENSE reproduces DeepSeek's MIT text and names DeepSeek as copyright holder) | not included; helper files copied into the image at build time |
| LiveBench leaderboard data files (livebench.ai): `table_`, `categories_`, `cost_2026_06_25` | downloaded 2026-10-07, sha256 in `data/static/livebench/README.json` | not stated on the data files (LiveBench code is Apache-2.0) | included verbatim as a dated snapshot for provenance; check terms before a public release |
| DeepSeek-V4.1-Flash weights (deepseek-ai) | `2cba9e42` | MIT (Hub card and LICENSE) | **never redistributed**; downloaded on the pod |
| EXL3 pack (diffbot, with coolbho3k's MIT experts) | `a1cf1bd2` | MIT (Hub card) | **never redistributed**; downloaded on the pod |
| lm-evaluation-harness | 0.4.13 | MIT | not included; installed on the pod |
| Benchmark datasets (GSM8K, MATH, HumanEval, MBPP, IFEval, RULER, LongBench) | via lm-eval | licenses as stated by each dataset card; check before redistributing | not included; downloaded by lm-eval |
| Prompt corpora: vLLM sources at `36768d1`; Project Gutenberg eBooks 2600, 1342, 2701, 1661 | sha256 in `experiments/corpora.yaml` | Apache-2.0; public domain in the USA (Gutenberg license header kept in the text) | not included; downloaded to `.cache/` |

## Derived files in this repository

`data/exploratory/inferencekey-deepseek@546071a/vllm-exl3/{a100,pro6000-concurrency}/scripts/serve.sh` and
`scripts/bootstrap.sh` are the measuring team's adaptations of Diffbot's `recipe/serve/serve.sh`, `docker/build.sh` and
`kernels/build.sh`. `stack/serve.sh` and `stack/build/bootstrap_pod.sh` re-express the same build and serving steps and
use the recipe's default settings. They are covered by the recipe's MIT license notice, reproduced here:

```
MIT License

Copyright (c) 2026 Diffbot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## AGPL-3.0 and built images

The `exl3-*` images built by `stack/build/Dockerfile` or `stack/build/bootstrap_pod.sh` contain vllm-exl3 (and, for
`exl3-offload`, Diffbot's patch to it), which are AGPL-3.0-only. Distributing such an image, or serving a modified
version to users over a network, carries AGPL-3.0 obligations (section 13: offer the corresponding source). This
repository never pushes images. Our Apache-2.0 code may be combined with AGPL-3.0 code in such an image (Apache-2.0
is compatible with GPLv3-family licenses in that direction); the combined work is then distributed under the AGPL.
