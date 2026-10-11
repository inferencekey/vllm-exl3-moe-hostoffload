# Attribution and provenance audit

Who wrote each part of the serving stack measured in this repo: DeepSeek-V4.1-Flash on 2x RTX PRO 6000 (sm_120) with
vLLM, the vllm-exl3 plugin and Diffbot's recipe. The inferencekey runs did the deployment and measurement. Date:
2026-10-06. The paper's contribution statement (`paper/sections/02_people.tex`) follows from section 7.

## 1. Scope and method

In scope:

- The inferencekey run scripts at `inferencekey-deepseek@546071a`, `vllm-exl3/a100/scripts/` and
  `vllm-exl3/pro6000-concurrency/scripts/`. The in-repo copies under `data/exploratory/inferencekey-deepseek@546071a/`
  are byte-identical to the commit (`evidence/inferencekey_vs_recipe.txt:3-21`).
- Everything those scripts install or run:
  - vLLM `36768d1bfd39094681cdbc8cb37d4b31c0729c89` and the 5 vLLM patches;
  - vllm-exl3 `d3cfd394920360d69f820d2dc96f8292a9e10283` and `exl3-thor.patch`;
  - ExLlamaV3 `5be886578ec80324c2c715269387be2058724b6e` and the recipe kernels;
  - the sfxnz recipe `460ac89c7df1eea4db8a731f1feb807f1584f359`;
  - the Diffbot recipe from the Hugging Face repo `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000@a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5`;
  - the weight repos it builds on.
- Mia's AI Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks, which vllm-exl3 credits as its origin. It is checked at HEAD
  `a246263a`, because nobody pins it.

Method:

1. Fetch every source at its pin, outside the repo, with no weights.
2. Hash every recipe file against the Hugging Face tree API and against inferencekey's local copy.
3. Apply every patch to its pinned upstream.
4. Compare each patch, line by line, with the upstream PR it claims to be.
5. Grep each tree, before and after patching, for the markers of each feature.
6. Read the code at the hits.
7. Diff inferencekey's scripts against the recipe's.
8. Use the Hugging Face commit history to date when each recipe part first appeared.

Reproduce. Downloads go to `$AUDIT_SRC` (default `~/.cache/vllm-exl3-audit-src`), about 120 MB. vLLM is a blobless
sparse clone.

```bash
make audit
# which runs:
bash audit/scripts/fetch_sources.sh     # pinned clones + HF recipe/ files, prints resolved SHAs
bash audit/scripts/check_patches.sh     # writes audit/evidence/*
```

`check_patches.sh` takes these switches:

- `LOCAL_RECIPE`: inferencekey's local recipe copy;
- `IK_REPO` and `IK_COMMIT`: the inferencekey repo and commit;
- `NO_GH=1`: skip the GitHub API (needs `gh auth`);
- `NO_NET=1`: skip the Hugging Face history.

It writes these files under `audit/evidence/`:

| File | Content |
|---|---|
| `sources_resolved.txt` | resolved SHA, date and URL of every checkout, and the vLLM sparse paths |
| `hf_diffbot_tree_a1cf1bd2.tsv` | the full listing of the Diffbot HF repo at `a1cf1bd2`: 191 files, 332.1 GB, with LFS sha256 and git blob oid |
| `recipe_hashes.tsv` | sha256 of the 95 `recipe/` files, the upstream check (0 mismatches), and the local copy (0 differences, 0 missing) |
| `hf_diffbot_history.txt` | the 13 commits of the Diffbot HF repo, and which recipe parts each tree contains |
| `patch_apply.txt`, `patch_diffstats.txt` | apply results and numstat of every patch |
| `pr_vs_recipe.txt`, `gh_pr_*.json`, `gh_issue_57144.json` | PR metadata, and the line-level comparison of each PR with the recipe file |
| `feature_grep.txt` | feature markers in 9 trees: upstream and patched |
| `xmoe_vs_exllamav3.txt` | each `kernels/xmoe` file against ExLlamaV3 `exllamav3_ext`, before and after the recipe patch |
| `sitecustomize_vs_sfxnz.txt` | which runtime hooks come from sfxnz, and which were added in the recipe |
| `fat_any_item.txt` | the capture-size failure, located (section 6) |
| `license_heads.txt`, `licenses.md` | licenses (section 5) |
| `inferencekey_vs_recipe.txt` | inferencekey scripts against the recipe, and the inferencekey files with no recipe counterpart |

## 2. Patch apply results

| Patch | Target | Result | Evidence |
|---|---|---|---|
| `pr58132-decoder-swa-replay.diff` | vLLM 36768d1 | applies (alone and in build order) | `patch_apply.txt:2,7` |
| `sm120-thor-fixes.diff` | vLLM 36768d1 | applies | `patch_apply.txt:3,8` |
| `pr59585-dsml-finite-args.diff` | vLLM 36768d1 | applies | `patch_apply.txt:4,9` |
| `pr59894-decoder-replay-cut-trim.diff` | vLLM 36768d1 | fails alone, because it needs #58132's `decoder_replay_layers.py`; applies in the build order of `docker/build.sh` | `patch_apply.txt:5,10` |
| `pr58560-compressor-null-block.diff` | vLLM 36768d1 | applies | `patch_apply.txt:6,11` |
| `exl3-thor.patch` (+676 / -50 lines in `src/vllm_exl3/exl3.py`) | vllm-exl3 d3cfd394 | applies (`git apply --check` and `patch -p1`) | `patch_apply.txt:13-14`, `patch_diffstats.txt:25-26` |
| `exllamav3-5be8865-exl3_moe.patch` | ExLlamaV3 5be88657 `exllamav3/exllamav3_ext` | applies; the result is byte-identical to every file in `recipe/kernels/xmoe` | `patch_apply.txt:16`, `xmoe_vs_exllamav3.txt` |
| `quantize/quantize_mtp_exl3.diff` | sfxnz `tools/quantize_experts_exl3.py` | applies | `patch_apply.txt:23` |

inferencekey's local recipe copy (`~/me/inferencekey-deepseek/vllm-exl3/a100/pack/recipe/`) is byte-identical to the HF
revision: all 95 files have the same sha256, with no extra or missing files (`recipe_hashes.tsv`, last line).
inferencekey modified nothing in the recipe.

## 3. Components

| Component | Author / owner (as shown in the sources) | Upstream @ pin | License | What it does here | Evidence |
|---|---|---|---|---|---|
| vLLM | vLLM project (vllm-project) | github.com/vllm-project/vllm @ `36768d1` (2026-09-29), image digest `sha256:0ad6fa41...` | Apache-2.0 | serving engine; DeepSeek-V4.1 model graph (`vllm/models/deepseek_v41/`), DSpark drafter, sparse MLA, Engram with pinned-host UVA offload, includes #56227 (encoder-side SWA replay, ivanium, merged 2026-09-18) | `license_heads.txt`, `gh_pr_56227.json`, vLLM `vllm/models/deepseek_v41/nvidia/dspark.py:3`, `nvidia/engram.py:298-299` |
| vLLM PR #58132 | ivanium | merged 2026-09-29, `4b2e1cfa` | Apache-2.0 | decoder-side SWA bounded replay | `pr_vs_recipe.txt:4-17` |
| vLLM PR #58560 | bluemelov1 | merged 2026-10-05, `49e0f479` | Apache-2.0 | keeps compressor ring writes out of the null KV block | `pr_vs_recipe.txt:19-25` |
| vLLM PR #59585 | bojiang-li | open | Apache-2.0 | DSML tool arguments stay valid JSON | `pr_vs_recipe.txt:27-33` |
| vLLM PR #59894 | gcanlin | open | Apache-2.0 | decoder replay skips the cut layer's unused work | `pr_vs_recipe.txt:35-45` |
| vLLM PR #57028 | zhao6300 | open (created 2026-09-15) | Apache-2.0 | SM120 FlashInfer serving fixes; part of it is inside `sm120-thor-fixes.diff` | `pr_vs_recipe.txt:47-61`, `gh_pr_57028.json` |
| vllm-exl3 | vcruz305 (Victor Cruz) | github.com/vcruz305/vllm-exl3 @ `d3cfd394` (2026-09-09) | AGPL-3.0-only | EXL3 quantization plugin for vLLM: routed-expert EXL3 path, pointer tables, fat-expert path, MTP experts as EXL3 | `license_heads.txt`; vllm-exl3 `NOTICE`, `THIRD_PARTY_NOTICES.md:9-25`, `src/vllm_exl3/exl3.py:1614` |
| Mia's AI Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks | MiaAI-Lab, @plotarmordev, other contributors (e.g. @nood-co1) | github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (HEAD `a246263a`) | MIT until 2026-09-07, then AGPL-3.0 | origin of vllm-exl3's `exl3.py` routed-expert path and of its Fat GEMM kernels; no host-RAM expert placement found | vllm-exl3 `THIRD_PARTY_NOTICES.md:25`; `feature_grep.txt:11,21` |
| ExLlamaV3 | turboderp (turboderp-org) | github.com/turboderp-org/exllamav3 @ `5be88657` (v1.4.9) | MIT | EXL3 format, trellis/MCG/MUL1 codebooks, `exl3_moe` kernel, which xmoe is built from; installed in the image | `license_heads.txt`, `xmoe_vs_exllamav3.txt` |
| sfxnz recipe | sfxnz | github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ `460ac89c` | MIT (copyright line names DeepSeek, see `licenses.md`) | first DS-V4.1 + EXL3 + vLLM recipe (2x DGX Spark); its `docker/patch` helpers (`fix_vllm_exl3_setup.py`, `sm120_page.py`) are used by both build scripts; base of `sitecustomize.py` and of the drafter quantizer; `DSV41_ENGRAM_DISK` | `sitecustomize_vs_sfxnz.txt`, `patch_apply.txt:23`, `feature_grep.txt:100` |
| Diffbot recipe (`recipe/`) | Diffbot (all HF commits by HF user `miketung`; the code comments say "thor") | HF `diffbot/...-2x-RTX-PRO-6000` @ `a1cf1bd2` | MIT, Copyright (c) 2026 Diffbot; `patches/vllm-exl3/` AGPL-3.0-only | `exl3-thor.patch` (host-RAM experts, split launch, side-stream overlap, lean prep, decode-once dispatch, expert stats), `sm120-thor-fixes.diff`, `sitecustomize.py` THOR_* hooks, `kernels/xmoe`, `kernels/xmoe_do`, `thor_dense*.py`, plans and `make_plan.py`, quantize scripts, serve/build/bench | `recipe/THIRD_PARTY_NOTICES.md:3-12`, `hf_diffbot_history.txt` |
| EXL3 3.0 bpw routed experts | coolbho3k | HF `coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw` @ `650cae2c` | MIT | shards 1-51 of the pack, unchanged | Diffbot `README.md:171`, `license_heads.txt` |
| DeepSeek-V4.1-Flash | DeepSeek (deepseek-ai) | HF `deepseek-ai/DeepSeek-V4.1-Flash` | MIT | base model; DSpark drafter and Engram tables before re-encoding; tokenizer | `licenses.md` |
| inferencekey run scripts | inferencekey (commit author Javier Abrego) | `inferencekey-deepseek@546071a` | (repo's own) | run the stack on RunPod pods without docker, and measure it | `inferencekey_vs_recipe.txt` |

## 4. Features that matter for the paper

Classifications:

- **algorithmic contribution**: a new method or scheduling idea.
- **implementation**: code that realizes a method in kernels or the engine.
- **integration**: wiring existing parts together.
- **engineering-configuration**: settings and knobs.
- **measurement**: load generation, sampling and analysis.

Line numbers in `exl3.py (patched)` are in vllm-exl3 `d3cfd394` with `exl3-thor.patch` applied
(`$AUDIT_SRC/work/vllm-exl3-patched/src/vllm_exl3/exl3.py`). Those in `sitecustomize.py` are in the recipe's
`patches/vllm/sitecustomize.py`.

| Feature | Author | Where it lives | Evidence | Class |
|---|---|---|---|---|
| Host-RAM experts. `EXL3_HOST_EXPERTS` plan: cold experts' trellis kept in exact-size `cudaHostRegister`'d host memory behind a zero-copy UVA alias; hot experts in a compact VRAM tensor | **Diffbot** | `exl3-thor.patch`, i.e. `exl3.py (patched)`:2288-2392 (`_uva_alias` at :2365, `cudaHostRegister` at :2389) | Absent from vllm-exl3 `d3cfd394`, vLLM, sfxnz and Mia's GLM repo: `feature_grep.txt:4-12` (0 hits outside the patched plugin and the recipe). First published in the HF commit `44899f5e0d` of 2026-09-30; the 2026-09-11 trees do not have it (`hf_diffbot_history.txt:11-15`). Related prior art: ExLlamaV3 at the pin has CPU MoE offload, with experts in system RAM computed by the CPU, and `EXL3_VISION_PINNED`, an EXL3 trellis in pinned memory read through a zero-copy device alias (`xmoe_vs_exllamav3.txt:42-46`). vLLM's Engram offload uses the same `cudaHostRegister` + UVA pattern (vLLM `nvidia/engram.py:67,204`). | implementation of a known technique (GPU zero-copy reads of host-resident expert weights), specialized per expert |
| Host-expert plan selection. `plans/*.json`, `make_plan.py` ranks experts by their maximum decode share over several traffic profiles; 30% cold, plus drafter layers 40-42 | **Diffbot** | `recipe/plans/`, `make_plan.py:2-10`; counters from `EXL3_EXPERT_STATS` (`exl3.py (patched)`:1749) | `plans/README.md`; `hf_diffbot_history.txt:9` (agentic plan, 2026-10-02) | algorithmic contribution (placement heuristic) |
| Split launch with a side-stream SM budget. Hot experts on the xmoe flat kernel on the main stream; cold experts on the group kernel on a second stream, limited to `EXL3_HOST_OVERLAP_SMS` (32 decode / 64 prefill); negative expert counts mean "skip but keep offsets" | **Diffbot** | `exl3.py (patched)`:107-113, :1589 (`xmoe_set_force_group`); `kernels/xmoe/bindings.cpp:8,10`; `kernels/xmoe/quant/exl3_moe.cu:56,62` | `feature_grep.txt:23-42` (only in the patched trees and the recipe); recipe `README.md:55-60,116-121` | algorithmic contribution (scheduling), implementation |
| xmoe flat decode kernel (<= 64 tokens, one block per SM over (expert, column tile, K tile)); multi-row prefill; K=3 MUL1/MCG instances | **Diffbot**, derived from ExLlamaV3 `exl3_moe` (turboderp, MIT) | `recipe/kernels/xmoe/`; new files `quant/exl3_moe_flat_kernel.cuh`, `quant/exl3_gemm_inner_flat.cuh`, `quant/exl3_gemm_inner_mt.cuh`; 13 files modified, 10 unchanged | `xmoe_vs_exllamav3.txt:3-29`; already in the first recipe upload of 2026-09-11 (`hf_diffbot_history.txt:12-14`); recipe `README.md:111-115` | implementation (kernel), derivative work |
| Decode-once prefill MoE kernel (`EXL3_DO`, `xmoe_do`): grouped GEMM that decodes each 16x16 trellis tile once per job of up to 128 rows | **Diffbot** (uses ExLlamaV3 device functions) | `recipe/kernels/xmoe_do/do_gemm.cu:1-5` ("Copyright (c) 2026 Diffbot"); dispatch at `exl3.py (patched)`:1308 | `feature_grep.txt:43-52`; first appears in the HF commit `797f3265e9` of 2026-10-02 (`hf_diffbot_history.txt:9`) | implementation (kernel), with an algorithmic element (decode amortization) |
| Int4 Engram in pinned host memory over UVA (`DSV41_ENGRAM_INT4=1`) | Pinned-host UVA Engram offload: **vLLM upstream**. Int4 format, Triton lookup kernel and converter: **Diffbot** | vLLM `vllm/models/deepseek_v41/nvidia/engram.py:67,204,298-299` (upstream); `sm120-thor-fixes.diff:104-106,169,220`; `recipe/quantize/convert_engram_int4.py:2-4` | `feature_grep.txt:74-75` (`ENGRAM_INT4` is absent upstream and present after the recipe patch), `feature_grep.txt:84` (upstream `cpu_offload`). The Engram parts of `sm120-thor-fixes.diff` are not in #57028 (`pr_vs_recipe.txt:57-58`). `DSV41_ENGRAM_DISK` is sfxnz's disk-backed Engram (`feature_grep.txt:100`), set to 0 here. vllm-exl3 has a separate EXL3 n-gram embedding codec (TP1 only), not used here. | implementation (quantization format and kernel) on an upstream mechanism |
| Embedding table on host (`THOR_EMBED_HOST`) | **Diffbot** | `sitecustomize.py:792-834` | `sitecustomize_vs_sfxnz.txt` (recipe-only hook) | engineering-configuration / implementation |
| FP8 dense one-copy (`THOR_DENSE_ONE_COPY`) and other THOR_* VRAM trims: `THOR_DENSE_GEMV` with `kernels/thor_dense*.py`, `THOR_WOA_FP8`, `THOR_FP8_LMHEAD`, `THOR_INDEXER_BUF_MULT`, `THOR_INDEXER_EXPAND_CAP`, `THOR_ROPE_CAP`, `THOR_ROUTER_SPLITK`, `THOR_MHC_SPLITS`, `THOR_ZERO_KV_AFTER_CAPTURE` | **Diffbot** | `sitecustomize.py:282-456` (one-copy at :389), :457, :565, :714, :835, :906, :934; `kernels/thor_dense.py:1`, `thor_dense2.py` | `sitecustomize_vs_sfxnz.txt` (all are recipe-only). Possible prior art for the indexer workspace trim: "indexer prefill-workspace right-sizing" by @nood-co1 in Mia's GLM repo (`docs/DESIGN-indexer-workspace.md`, 2026-09-01), credited in vllm-exl3 `ACKNOWLEDGMENTS.md:18-19`. Whether Diffbot's hook derives from it is not known. | implementation / engineering-configuration |
| Runtime hooks inherited from sfxnz: load the plugin in every process, MXFP8 `weight_block_size`, skip the sparse-MLA warmup, SM120 page 64, packed indexer page sizes, text-only SWA rows | **sfxnz**, lightly edited by Diffbot (env gates, the `deepseek_v41` rename, `#57028` page sizes, image-token budget) | `sitecustomize.py:1-150` vs sfxnz `docker/patch/sitecustomize.py` | `sitecustomize_vs_sfxnz.txt:5-16` (every sfxnz hook kept; 831 lines added, 6 removed at :2-4), and the hunk list at its end | integration |
| DSpark drafter | Speculative method and draft model code: **vLLM** (DeepSeek-V4.1 support); drafter weights: **DeepSeek**; 3-bit EXL3 drafter experts: **Diffbot**, with a quantizer derived from **sfxnz**; loaded as EXL3 by **vllm-exl3** | vLLM `nvidia/dspark.py:3,72`; `recipe/quantize/quantize_mtp_exl3.py:2`; vllm-exl3 `exl3.py:1614` (`mtp_experts` default `"exl3"`) | `patch_apply.txt:23`; `feature_grep.txt:114`; Diffbot `README.md:172` | integration + implementation (quantization) |
| EXL3 3.0 bpw routed experts | quantized by **coolbho3k** (format and quantizer: **ExLlamaV3**, turboderp) | HF `coolbho3k/...@650cae2c`, shards 1-51 unchanged | Diffbot `README.md:171`, `recipe/quantize/README.md` step 1 | implementation (third-party artifact) |
| EXL3 support in vLLM | **vcruz305** (vllm-exl3), derived from **Mia's AI Lab** `overlay/exl3.py` and Fat GEMM kernels, on **ExLlamaV3** kernels | vllm-exl3 `d3cfd394` | vllm-exl3 `NOTICE`, `THIRD_PARTY_NOTICES.md:9-25` | implementation |
| DeepSeek-V4.1 model graph in vLLM | **vLLM project** | vLLM `vllm/models/deepseek_v41/` @ `36768d1` | `license_heads.txt` (SPDX), `gh_pr_56227.json` | implementation |
| The four upstream PRs (#58132, #58560, #59585, #59894) | **ivanium**, **bluemelov1**, **bojiang-li**, **gcanlin** (the gh metadata also lists co-author identities such as `claude`, `codex`, `cursoragent` on some commits) | `recipe/patches/vllm/pr*.diff` | Every non-test line of each recipe file is identical to the PR diff: 661/661, 6/6, 16/16, 553/553 (`pr_vs_recipe.txt:17,25,33,45`). The recipe files leave out the PRs' tests and CI files. #58132 and #58560 were merged after the pin (their merge commits are not ancestors of `36768d1`; #56227's is: `pr_vs_recipe.txt:69-75`); #59585 and #59894 are open. | implementation (third-party), integration by Diffbot |
| sm_120 fixes (`sm120-thor-fixes.diff`, 7 files) | **mixed**: 31 of 139 changed lines are identical to the open vLLM PR #57028 by **zhao6300** (MXFP8 empty batch in `mxfp8/flashinfer.py` and `mxfp8_utils.py`, the multi-stream capture guard, the compressed-cache page width). The rest is **Diffbot**: the indexer page rule, int4 Engram, the FlashInfer 0.7 masked-slot pad | `recipe/patches/vllm/sm120-thor-fixes.diff` | `pr_vs_recipe.txt:47-61`. #57028's commits are dated 2026-09-15 (`gh_pr_57028.json`), before Diffbot's first sm120 diff (2026-09-30). Attribution gap in Diffbot's docs: `THIRD_PARTY_NOTICES.md:9` calls the file "our changes to seven vLLM files", while `README.md:92-94` credits #57028 only for the indexer page geometry. | implementation (mixed authorship) |
| Custom all-reduce disabled, NCCL settings, `--block-size 64`, `FULL_DECODE_ONLY` graphs, DSpark k=5, `KV_MEM` 4.25 GiB, `MAX_BATCHED` 16384 | **Diffbot** (`--block-size 64` comes from sfxnz `460ac89c`, "Serve with --block-size 64") | `recipe/serve/serve.sh` | `inferencekey_vs_recipe.txt:24-126` (identical in inferencekey's serve.sh) | engineering-configuration |
| `MAX_NUM_SEQS=64`, `CAPTURE_SIZES` up to 256, `PLAN=plan-gen30-dh.json` (pro6000 run); `MAX_MODEL_LEN=262144`, `VISION=0`, `--enable-prompt-tokens-details` (both runs) | **inferencekey** | pro6000 launch line `pro6000-concurrency/README.md:203`; `scripts/serve.sh` defaults | `inferencekey_vs_recipe.txt:47,81,96,321-326`. The recipe already documents the knob: "`CAPTURE_SIZES` must reach `MAX_NUM_SEQS x (1 + SPEC_K)`" (recipe `README.md:140-141`). 64 x 6 = 384, so the 256 cap was a workaround (section 6). | engineering-configuration |
| Recipe build replayed without docker (`bootstrap.sh`) | **inferencekey** | `a100/scripts/bootstrap.sh` (same file in pro6000) | Same patches in the same order, the same pins, and the same sfxnz helper as `docker/build.sh` and `kernels/build.sh` (`inferencekey_vs_recipe.txt:131-167`). Differences: no docker commit; no image digest pin in the script (the pod image is set at pod creation, `a100/README.md`); `pip install ninja`; `MAX_JOBS=48`; `sitecustomize.py` copied into the image instead of mounted. | integration |
| serve.sh rewritten as a direct `vllm serve` | **inferencekey** | `a100/scripts/serve.sh` (identical to the pro6000 copy) | Header: "diffbot recipe/serve/serve.sh turned into a direct `vllm serve` (no docker). Same env knobs and args, TP2." (`inferencekey_vs_recipe.txt:338`, `a100/scripts/serve.sh:2`). Every `EXL3_*`, `THOR_*`, `DSV41_*`, NCCL and FlashInfer knob is identical to the recipe default (`inferencekey_vs_recipe.txt:24-86`; flags at :87-126). Removed: docker run, `CONTAINER_MEM`, `WARMUP`, `THINKING`, `NO_CUSTOM_AR`/`LINEAR_BACKEND` switches (hardcoded to the recipe defaults), `--ulimit memlock` (replaced by `ulimit -l unlimited`). | integration |
| Load generator (`loadgen.py`), summarizer (`summarize.py`), A100-class bench (`bench.py`), run drivers (`run_all.sh`, `restart_A.sh`) | **inferencekey** | `pro6000-concurrency/scripts/`, `a100/scripts/bench.py` | No recipe counterpart (`inferencekey_vs_recipe.txt:343-359`). The recipe has its own benchmarks (`bench/bench-sbs.py`, `bench-fanout.py`, `validate.sh`), which inferencekey did not use for the concurrency sweep. | measurement |
| Memory sampler (`mem_sampler.py`) | **inferencekey** | `pro6000-concurrency/scripts/mem_sampler.py` | `inferencekey_vs_recipe.txt:343-359` | measurement |
| Pod creation and cost guard (`rp_create_loop.py`, `rp_stock2.py`, `watchdog.sh`, `pod.sh`, `download.py`) | **inferencekey** | `a100/scripts/`, `pro6000-concurrency/scripts/` | `inferencekey_vs_recipe.txt:343-359` | engineering-configuration (orchestration) |

## 5. Licenses

The details are in `evidence/licenses.md`. In short:

- vLLM is Apache-2.0.
- vllm-exl3 at `d3cfd394` is AGPL-3.0-only. It was relicensed from Apache-2.0 on 2026-09-08, the day before the pin.
- ExLlamaV3 at `5be88657` is MIT, Copyright (c) 2025 Turboderp.
- The sfxnz recipe is MIT, but its LICENSE file is byte-identical to DeepSeek's, so its only copyright line reads
  "Copyright (c) 2023 DeepSeek". The recipe's `docker/LICENSE.sfxnz-recipe` reproduces it faithfully.
- The Diffbot recipe is MIT, Copyright (c) 2026 Diffbot, except `patches/vllm-exl3/`, which is AGPL-3.0-only.
- The Diffbot HF card says `license: mit`. The repo's top-level `LICENSE` is DeepSeek's text, covering the weights.
- coolbho3k's and deepseek-ai's repos are MIT. DeepSeek ships no model license with use restrictions.
- None of the three Hugging Face repos is gated.
- Mia's AI Lab GLM repo moved from MIT to AGPL-3.0 on 2026-09-07. vllm-exl3's derivation from it predates that switch.

## 6. Known issue: CUDA graph capture size 384

inferencekey reported this in `pro6000-concurrency/README.md:29-31`. With `CAPTURE_SIZES` including 384, graph capture
failed with `cudaErrorStreamCaptureUnsupported`:

- Above `VLLM_EXL3_FAT_THRESHOLD` (default 256) tokens, the fat-expert path calls `fat.any().item()`.
- That is a host sync, which CUDA graph capture does not allow.

Location of the line:

- **Installed stack**, vllm-exl3 `d3cfd394` + `exl3-thor.patch`: `src/vllm_exl3/exl3.py:1535`
  (`elif fat_possible and bool(fat.any().item()):`) in `apply_exl3_fused_moe`.
  - The threshold is set at `exl3.py:246`, and `fat_possible = tokens > FAT_EXPERT_THRESHOLD` at `exl3.py:1497`.
  - The call path is `apply` at `exl3.py:2753`, then `apply_exl3_experts` at `exl3.py:1837`.
  - These line numbers match the traceback in `logs/serve_attempt1_capture384.log` exactly. That also confirms the
    deployed plugin was `d3cfd394` with the recipe patch.
- **Upstream** `d3cfd394` without the patch: the same check is at `src/vllm_exl3/exl3.py:1240`, with a second one at
  `:1296`, inside `apply_exl3_fused_moe` (`:1156`). The threshold is at `:108` and `fat_possible` at `:1238`.

So the host sync comes from the upstream fat-expert path, which vllm-exl3 credits to Mia's AI Lab ("fat-expert prefill
batching", PR #77 by @plotarmordev). The recipe patch keeps it as the `elif` branch. Diffbot's own config never captures
above 48, so it never hits this. Evidence: `fat_any_item.txt:1-22`.

## 6b. Prior art inside the stack's own upstreams (added after the main audit)

Two mechanisms close to the host-expert feature already existed upstream before Diffbot published it
(`evidence/vllm_upstream_offload.txt`):

- **Upstream vLLM at the pin has UVA zero-copy weight offload with parameter selection.** `vllm/config/offload.py`
  at `36768d1` defines `UVAOffloadConfig` (`cpu_offload_gb` per GPU, and `cpu_offload_params`, a set of parameter
  name segments such as `experts` to offload selectively, "zero-copy access from CPU-pinned memory"), a
  `PrefetchOffloadConfig` (layer groups with async H2D prefetch), and the CLI flags `--offload-backend`,
  `--cpu-offload-gb`, `--cpu-offload-params` (`arg_utils.py`). Its granularity is the parameter tensor (for a fused
  MoE layer, all experts of the layer at once), filled until the budget is reached; it has no per-expert hot/cold
  split and no side stream.
- **vcruz305 documented an experimental "vLLM UVA expert placement" path for vllm-exl3 on 2026-09-11** (commits
  `f88aaac` and `1218889`, after the pin `d3cfd394` of 2026-09-09 and before Diffbot's host-expert commit of
  2026-09-30). It places the packed EXL3 expert payloads in pinned host memory with upstream vLLM's
  `--offload-backend uva --cpu-offload-params`, read by the GPU EXL3 kernels over UVA, with a fail-closed guard. It is
  not in the pinned plugin the measured stack uses.

Consequence for the classification in section 4: GPU zero-copy reads of host-resident expert weights are an upstream
mechanism (vLLM) that the plugin author was already applying to EXL3 experts. What Diffbot's patch adds on top is the
per-expert granularity driven by a routing profile (hot experts in VRAM, cold in host memory within the same layer),
the split launch with a bounded-SM side stream, and the kernel work that makes the split fast. The ablation design
(`experiments/experiments.yaml`, E3) therefore compares Diffbot's per-expert split against upstream vLLM's own
selective UVA offload of the same EXL3 experts.

## 7. Verdict: what is inferencekey's own contribution

**The working hypothesis holds, with two refinements on third-party authorship.**

inferencekey contributed orchestration, integration, configuration and measurement. They did not contribute any model,
kernel, patch or serving algorithm.

- **The recipe is unmodified.** The recipe files they ran are byte-identical to Diffbot's HF revision `a1cf1bd2`
  (`recipe_hashes.tsv`, 95/95). Their `bootstrap.sh` applies the same five vLLM patches, `exl3-thor.patch`, ExLlamaV3 and
  the xmoe builds as the recipe's `docker/build.sh` and `kernels/build.sh`, in the same order (`inferencekey_vs_recipe.txt:131-167`).
- **serve.sh is the recipe's, minus docker.** It is the recipe `serve/serve.sh` without docker, with every runtime knob
  equal to the recipe default (`inferencekey_vs_recipe.txt:24-126`). Their a100 README says so in its own words: their
  serve.sh is "diffbot recipe/serve/serve.sh turned into a direct vllm serve" (`a100/scripts/serve.sh:2`).
- **Settings.** Their own settings are `MAX_NUM_SEQS=64`, capture sizes up to 256, `MAX_MODEL_LEN=262144`, `VISION=0`,
  `PLAN=plan-gen30-dh.json` and `--enable-prompt-tokens-details`. They also found and documented the capture-384 limit.
- **Their own code.** The rest is their own: the RunPod pod creation and watchdog, the download, the closed-loop load
  generator, the memory sampler, the summarizer, and the A100-class bench.

Who authored the features the paper measures:

- **Host-RAM expert placement, plans, split launch and side-stream SM budget: Diffbot**, as a per-expert,
  profile-driven refinement of an upstream vLLM mechanism (selective UVA weight offload, section 6b). They are not in vllm-exl3 at
  `d3cfd394`, in Mia's GLM work, or in the sfxnz recipe (`feature_grep.txt:3-42`). `exl3-thor.patch` adds them, and
  they first appear on HF on 2026-09-30. The general idea (MoE experts in host RAM) has prior art in ExLlamaV3 itself,
  through CPU-computed experts and zero-copy pinned vision weights, and in the literature: the closest published
  mechanism is direct GPU access to host-resident weights with a static offline split for dense models (DAK); the
  MoE offloading systems either copy experts to the GPU or compute cold experts on the CPU (`paper/sections/13_related.tex`).
- **xmoe flat decode kernel and decode-once prefill kernel: Diffbot**, as derivative work of turboderp's ExLlamaV3
  `exl3_moe` (MIT).
- **Int4 Engram: Diffbot**, through `sm120-thor-fixes.diff` and `convert_engram_int4.py`. The pinned-host UVA Engram
  mechanism it rides on is upstream vLLM, and is already present at `36768d1`. `DSV41_ENGRAM_DISK` is sfxnz's (unused
  here, =0).
- **Embedding on host and FP8 dense one-copy, with the other THOR_* trims: Diffbot.**
- **The four PR patches are exactly the upstream PRs' non-test changes,** by ivanium (#58132, merged), bluemelov1
  (#58560, merged), bojiang-li (#59585, open) and gcanlin (#59894, open).
- **`sm120-thor-fixes.diff` has mixed authorship:** about a fifth of it is zhao6300's open PR #57028, which Diffbot's
  notices do not fully credit. #57144 is an open feature-request issue (Ampere support, thomaslwang), not a PR.
- **sitecustomize.py:** lines 1-150 are sfxnz's hooks; the THOR_* blocks from line 151 are Diffbot's.
- **The rest of the stack:**
  - **EXL3 in vLLM:** vcruz305 (vllm-exl3), derived from Mia's AI Lab and built on ExLlamaV3.
  - **DeepSeek-V4.1 graph and DSpark:** the vLLM project.
  - **EXL3 3.0 bpw experts:** coolbho3k.
  - **Base model, drafter and Engram weights:** DeepSeek.

## 8. Uncertainties and limits

- **Individuals behind Diffbot.** Diffbot's code is attributed to Diffbot as an organization: its LICENSE and
  NOTICES say "our", and every HF commit is by the HF user `miketung`. Code comments use the codename "thor".
  Individual authors are not visible.
- **Diffbot's own sources.** Diffbot may have drawn on work that is not public or not cited. Two examples:
  - The repo `0xSero/dsv41-flash-offload` (created 2026-10-02) uses the same `EXL3_HOST_EXPERTS` / `EXL3_HOST_DMA`
    names. It appeared after Diffbot's 2026-09-30 publication, so it looks downstream, but its patch was not audited.
  - The indexer workspace trim has an earlier analogue in Mia's GLM repo.
- **PR authors' affiliations are unknown.** Whether any of them is affiliated with Diffbot cannot be told from the
  sources.
- **PR heads may have moved.** The PR comparison uses each PR's current head, not its state when Diffbot copied it. The
  identity of every non-test line shows nothing changed in those files since.
- **Keyword greps.** The absence checks are keyword greps (`feature_grep.txt`), backed by reading the code at the hits.
  A feature under unrelated names could be missed. The vLLM checkout is sparse: only the patched paths, the
  DeepSeek-V4.1 model, the spec-decode code, the indexer backend and the license.
- **Not fetched.** Mia's GLM repo is checked at HEAD, not at a historical pin. Mia's DeepSeek-v4.1 repo (AGPL) was
  not fetched.
- **No runtime check.** No GPU run is part of this audit: "what was deployed" rests on the scripts, the hashes, and the
  traceback line numbers matching the patched plugin.
- **Not legal advice.** The license reading is a provenance record.
