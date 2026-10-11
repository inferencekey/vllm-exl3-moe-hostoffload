# DeepSeek-V4.1-Flash with vllm-exl3, "A100 class" test (run 2026-10-06)

Result: **deployed and usable, but NOT on an A100.** Run on **2x RTX PRO 6000 Blackwell Server Edition (96 GB each,
sm_120)**, RunPod SECURE, EUR-IS-1, 4.18 USD/h, the closest equivalent that can run this stack.
Pod `r0il578r3uzaty` (`ik-vllmexl3-a100class-2xpro6000`) lived 10:18:08 to 10:39:22 UTC: about 21 min, **about 1.5 USD**.

## Why not a single A100 80GB

Three independent blockers, any one of them is enough:

1. **No Ampere path.** vllm-exl3 is only a vLLM plugin (`--quantization exl3`); the DeepSeek-V4.1 model graph comes from
   vLLM. Upstream vLLM's V4.1 sparse-MLA backends admit only sm_9x/10x (FlashMLA) and sm_10x/12x (FlashInfer sparse);
   vllm-project/vllm#57144 (A100 support) is open. The only sm_80 V4.1 vLLM (wtdcode/vllm-backport) was never tested
   with vllm-exl3.
2. **VRAM.** The smallest non-Engram EXL3 pack (1.59 bpw) is 118.6 GiB, against ~74.5 GiB usable on an A100 80GB.
3. **Host RAM.** UVA offload of experts plus the 189 GiB FP8 Engram needs ~290 GiB pinned; a 1x A100 pod gets ~117-125 GB.

[redacted: two lines about an out-of-scope runtime]

## What was run

- Image `vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89` as the pod image (sshd bootstrapped in
  `dockerStartCmd`; entrypoint overridden to `bash -c`). vLLM 0.30.1rc1.dev312+g36768d1bf, torch 2.13.0+cu130,
  driver 595.91.07.
- diffbot's recipe replayed without docker (`scripts/bootstrap.sh`): 5 vLLM patches, vllm-exl3 `d3cfd394` +
  `exl3-thor.patch` (pure Python install, `VLLM_EXL3_NO_CUDA=1`), ExLlamaV3 `5be88657` built for `12.0a`, xmoe and
  xmoe_do kernels, sfxnz `fix_vllm_exl3_setup.py`. **Build took 217 s.**
- Weights: `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000@a1cf1bd2` (309 GiB: EXL3 3.0 bpw routed experts by
  coolbho3k, 3-bit DSpark drafter, int4 Engram), downloaded to the container disk with `snapshot_download`
  (hf_xet high performance, 32 workers). **Download took 135 s (~2.4 GB/s).**
- Server (`scripts/serve.sh`): TP2, CUDA graphs FULL_DECODE_ONLY, DSpark k=5, fp8 KV 4.25 GiB (623,891 tokens),
  `--max-model-len 262144`, text only, prefix caching on (vLLM default), tool parser and reasoning parser deepseek_v41.
  Cold experts (30%), Engram and embeddings in pinned host RAM.
- First boot to ready: about 4 min (weights 93 s, model load 126 s, graph capture 47 s). A relaunch on the same pod
  with warm page cache and kernel caches: under 5 min including the full benchmark.
- Benchmark: `scripts/bench.py`, run on the pod against 127.0.0.1 (no network in the numbers), streaming, temperature 0,
  thinking off unless noted, `ignore_eos` for the fixed-length decode tests.

## Numbers

Decode tok/s = (completion tokens - 1) / (last token time - first token time), single stream, short prompt.
Two host-expert plans were measured on the same pod: `agent30` (recipe default, tuned for agentic coding traffic) and
`gen30` (general text). DSpark speculation makes decode depend strongly on the text being produced.

| Test | agent30 (default) | gen30 |
|---|---:|---:|
| Decode, 120-token answer (essay prompt) | 29.4 | 35.6 |
| Decode, 480-token answer, prompt 1 (essay) | 31.2 | 39.4 |
| Decode, 480-token answer, prompt 2 (TCP explanation) | 58.6 | 77.8 |
| Decode, 480-token answer, prompt 3 (star life cycle) | 43.3 | 70.1 |
| Decode, coding answer with thinking (3.5-4K tokens) | 99.0 | 149.8 |
| TTFT, 3,595-token code prompt | 1.07 s | 1.00 s |
| TTFT, 14,855-token code prompt (fresh) | 3.09 s (~4,800 tok/s) | 4.73 s |
| TTFT, same 14.8K prefix + new question (2nd turn) | **0.83 s, 14,848 cached tokens** | **0.64 s, 14,848 cached** |
| 4 concurrent requests x 256 tokens, aggregate | 56.1 tok/s (14-17 per stream) | 81.0 tok/s (20-24 per stream) |
| Tool call | valid `get_weather {"city": "Reykjavik"}`, finish `tool_calls` | same |
| DSpark mean acceptance length | | 4.20 |

Prefix cache verified from both the usage field (`cached_tokens` 14,848 of 14,901) and `/metrics`
(`vllm:prefix_cache_hits` +0 on the first 16K request, +14,848 on the second).

Answer quality (agent30 run, thinking on): the `merge_intervals` question got a correct, idiomatic solution (sort then
single pass, touching intervals merged with `start <= current_end`), docstring with doctests, 5 sensible pytest tests
(empty, single, overlap plus touch, unsorted and nested, negatives) and the right complexity, O(n log n). 10.7K
characters of reasoning before the answer. The gen30 run hit the 4,000-token cap while thinking.

Resources:
- Pod RAM limit (`/sys/fs/cgroup/memory.max`): 376,999,997,440 B = **351 GiB** (memoryInGb 377); host has 1.5 TB, 256 vCPU.
  At steady state: anon 68-73 GB + shmem 155 GB (pinned experts, Engram, embeddings), plus ~300 GB page cache from
  the downloaded weights (reclaimable; `memory.current` sits at the limit because of it).
- `ulimit -l` is 8 MB and cannot be raised in a RunPod pod, but CUDA pinned and `cudaHostRegister` memory worked anyway.
- VRAM: 90,111-90,471 MiB (88.0-88.4 GiB) of 97,887 MiB (95.6 GiB) used per GPU while serving (model 76.8 GiB per GPU + 4.25 GiB KV + graphs/workspaces).

## Comparison

| Stack | Decode short | Prefill / TTFT | Prompt cache | Cost/h |
|---|---|---|---|---|
| [redacted: two table rows about an out-of-scope runtime] | | | | |
| **vllm-exl3 + diffbot recipe, 2x RTX PRO 6000** | **29-78 (text), 99-150 (code)** | **16K in 3-5 s, 2nd turn 0.6-0.8 s** | **yes** | **4.18 USD** |

[redacted: one sentence comparing with an out-of-scope runtime]
Weak points: decode on general prose with DSpark is 30-40 tok/s, and 4-stream aggregate is only 56-81 tok/s on
these prompts (diffbot reports 130-232 on their agentic traffic).

## Reproduce

All from the inferencekey repo with `scripts/secrets.sh run jenkins ...` for the RunPod key.

1. Locally, fetch the recipe (no weights) into `pack/` and the sfxnz helpers (`docker/patch` of
   sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @460ac89c), tar them as `stage/recipe.tgz` and `stage/dsv41-patch.tgz`.
2. `scripts/rp_stock2.py` (price/stock, read-only), then `scripts/rp_create_loop.py` (create with retry every 45 s
   over EUR-IS-1, EU-NL-1, CA-MTL-3; deletes a pod with < 250 GB RAM or > 4.40 USD/h). Start `scripts/watchdog.sh <id> 6000`.
3. Copy the tarballs and `bootstrap.sh serve.sh bench.py download.py` to `/root`; untar to `/opt/recipe` and
   `/opt/dsv41-patch`; `pip install --target /opt/hfpkgs "huggingface_hub[hf_xet]" hf_transfer` (do not upgrade the
   image's huggingface_hub); run `PYTHONPATH=/opt/hfpkgs python3 download.py` and `bash bootstrap.sh` in parallel.
4. `API_KEY=... bash serve.sh` (`PLAN=plan-gen30-dh.json` for general traffic), then `API_KEY=... python3 bench.py`.
5. Copy `/root/results/*` and logs back **with an unquoted remote glob or one scp per file, and check the copy before
   deleting the pod** (this run lost the gen30 JSON that way; its numbers come from the console, `results/gen30/console.txt`).
6. `rp_delete.py <id>` and `rp_list.py`.

## Files

- `scripts/`: create loop, stock check, watchdog, pod-side bootstrap, download, serve and bench scripts, `pod.sh` (SSH).
- `results/agent30/`: bench.json (with full answers), bench, serve, bootstrap and download logs.
- `results/gen30/console.txt`: gen30 numbers as printed.
- `results/create_loop.log`: the create response line (cost, RAM, vCPU).

Licensing: vllm-exl3 and its patch are AGPL-3.0; the diffbot recipe and the weights are MIT. Serving it to customers
with modifications triggers AGPL obligations.
