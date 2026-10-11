<!-- Excerpt of docs/REPORT.md at https://github.com/inferencekey/inferencekey-deepseek commit 546071a76de290b47a43149924c95866f5b8ebf0. Only the vLLM sections are in scope: header, cost method and the S10/S11/S12 cost rows, metric definitions 4.2 to 4.4, test sheets S10, S11, S12, the vLLM parts of the threats to validity and of the model-quality note. Sections S1 to S9 and S13 concern other runtimes and are not imported. Kept line ranges of the original: [[1, 7], [22, 44], [54, 56], [98, 132], [311, 486], [540, 543], [549, 553], [555, 555], [558, 567]]. Lines rewritten with a [redacted] marker: [542, 549, 552, 553, 562, 566]. -->
# DeepSeek-V4.1-Flash self-hosting tests on RunPod: full report

Period covered: 2026-10-05 11:00 UTC to 2026-10-06 11:36 UTC. Written 2026-10-06.
Every number names the file it comes from. When there is no file, the entry says "no source file" and why.
Paths are relative to this repo unless they start with `~`.

Model checkpoints and pinned revisions: [MODELS.md](../MODELS.md).
## 2. Cost

### 2.1 Method

- Pod costs: RunPod REST v1 `GET /v1/billing/pods?bucketSize=hour`, 2026-09-28 to 2026-10-07, queried 2026-10-06 11:55 UTC and
  saved as `docs/billing/2026-10-05_06-hourly.json` by `runpod/rp_billing_dump.py`. Pod ids and billed amounts in 2.2
  come from that file; pod lifetimes come from the session's create and delete events, which are only partly filed. Volumes: `GET /v1/billing/networkvolumes`.
- The last hour bucket present was 10:00 UTC on 2026-10-06, and it is complete. Pods in the 11:00 hour have an
  **estimate** = lifetime x `costPerHr` from the create response.
- RunPod bills a GPU pod only while its container is running. Pods rented but never started bill less than lifetime x rate
  (compare `timeBilledMs` per pod in the billing file with the lifetimes below). Estimates for such
  pods are an **upper bound**.
- CPU pods are not in `/billing/pods`; their cost is **estimated** as lifetime x rate.
- Lifetimes: the `CREATE 201 id=...` and `DELETE <id> -> 204` events of the session, `createdAt` in the saved create
  responses (`vllm-exl3/*/logs/`), and `vllm-exl3/pro6000-concurrency/README.md` for the sweep pod.
- Excluded (not DeepSeek): pods billed before 2026-10-05 11:00 UTC: Qwen tests 2026-09-28 to 2026-10-02, and inferencekey
  worker pods on 2026-10-04 (`rbpr59i04qjipd`, `e5ehyq721s03de`, `tcwm168dwd2l2m`, `88z7tquy6kh9sa`, `lfzk6f9iy3s0su`,
  `l5ct9lpifvnk19`).

### 2.2 Per test

| Sheet | Pods (lifetime) | Billed USD | Estimate USD | What it bought |
|---|---|---:|---:|---|
| S10 vllm-exl3 2x RTX PRO 6000, first run | `r0il578r3uzaty` (21 min) | 1.50 | | decode, TTFT, prefix cache, tool call |
| S11 vllm-exl3 1x B300 track (stopped) | CPU `tj6leo5y5j47gj` (14 min, 0.48 USD/h), B300 `a3bzy3x959hdw2` (22 min), `68b5tndcox1o1r` (20 min), `8obc30jjcya44j` (1 min) | 3.85 | 0.11 CPU + up to about 1.3 for the 11:00 hour | nothing: containers never started |
| S12 vllm-exl3 2x RTX PRO 6000 concurrency sweep | `v0tk6f34k3mkpe` (11:02:44 to 11:36:14, 33.5 min) | not yet | 2.33 | scenarios A, B, RAM/VRAM log, 1M boot |
### 4.2 vllm-exl3 runs (S10, S12)

As computed in `vllm-exl3/pro6000-concurrency/scripts/loadgen.py` and `vllm-exl3/a100/scripts/bench.py`, run on the pod
against 127.0.0.1 (no network in the numbers):
- **TTFT** = arrival of the first streamed chunk with content or reasoning, minus request start.
- **Per-user decode** = (completion_tokens - 1) / (last content chunk time - first content chunk time).
  `completion_tokens` comes from `usage`; `loadgen.py` keeps only requests with >= 32 completion tokens (all had 256).
- **Aggregate** = completion tokens of all successful requests in a level / wall time of the level.
- **N** = requests kept in flight (closed loop), not connected users. Each level starts with all N requests at once
  (first wave), which inflates TTFT p95 (5.12).
- **Percentiles**: nearest rank (`pct()` in `loadgen.py`), on the n requests of a level:
  - n=8 (N=1, 2, 4) and n=16 (N=8): p95 = max, p5 = min.
  - n=32 (N=16): p95 is the 2nd highest.
  - n=64 (N=32): p95 is the 4th highest.
  - n=128 (N=64): p95 is the 7th highest.

  Read p95 and p5 as "near the worst request", not as stable tail estimates.
- **DSpark mean acceptance length** = 1 + accepted draft tokens / drafts, from server `/metrics` deltas over the level.

### 4.3 The usable bars

From `vllm-exl3/pro6000-concurrency/README.md`. Both bars were **chosen after seeing the data**; the full tables in 5.12 let
readers apply their own.
- **Strict:** decode p5 >= 15 tok/s, plus TTFT p95 <= 5 s (scenario A) or <= 15 s (scenario B).
- **Loose:** decode p5 >= 10 tok/s, plus TTFT p95 <= 10 s (A) or <= 30 s (B).
- **Results:** A passes up to N = 8 under both bars. B passes up to N = 8 strict (N = 16 misses with p5 14.99) and N = 16 loose.

### 4.4 Units

- GiB = 2^30 B, GB = 10^9 B. RunPod `memoryInGb` and `/sys/fs/cgroup/memory.max` are converted: 377 GB = 351.1 GiB,
  251 GB = 233.8 GiB.
- Tokens are the model tokenizer's, counted by the server (`usage`, `/tokenize`).
- Cost is USD per pod-hour (`costPerHr` of the create response); a 2-GPU pod at 4.18 USD/h is 4.18 for both GPUs.
- Cost per active request-hour = pod USD/h / N. It is not cost per connected user.

### 5.10 S10: vllm-exl3 on 2x RTX PRO 6000, first run

- **Date** 2026-10-06 10:18:08 to 10:39:22 UTC. **Pod** `r0il578r3uzaty` (`ik-vllmexl3-a100class-2xpro6000`), EUR-IS-1.
- **Environment**:
  - GPUs: 2x RTX PRO 6000 Blackwell Server Edition (97,887 MiB each, sm_120), driver 595.91.07.
  - Image `vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89`, sshd installed in `dockerStartCmd`.
  - Pod: `memoryInGb` 377, cgroup 376,999,997,440 B (351.1 GiB), 64 vCPU, `ulimit -l` 8 MB, weights on the overlay
    container disk, 4.18 USD/h. Host 1.5 TB and 256 vCPU.
- **Software**: vLLM 0.30.1rc1.dev312+g36768d1bf, torch 2.13.0+cu130, vllm-exl3 `d3cfd394` + `exl3-thor.patch`,
  ExLlamaV3 `5be88657` built for `12.0a`, 5 vLLM patches (`scripts/bootstrap.sh`). Build took 217 s, download 135 s.
- **Model**: `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000@a1cf1bd2` (309.3 GiB).
- **Parameters** (`vllm-exl3/a100/scripts/serve.sh`; args echoed in `vllm-exl3/a100/results/agent30/serve.log`):
  - Flags: `--tensor-parallel-size 2 --max-model-len 262144 --kv-cache-dtype fp8 --kv-cache-memory 4563402752
    --gpu-memory-utilization 0.92 --max-num-seqs 8 --max-num-batched-tokens 16384 --block-size 64 --quantization exl3
    --disable-custom-all-reduce --linear-backend marlin --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41
    --enable-auto-tool-choice --reasoning-parser deepseek_v41 --enable-prompt-tokens-details --language-model-only
    --trust-remote-code --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}'
    --cudagraph-capture-sizes 1 2 4 6 8 12 16 24 32 48
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
    --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}'`.
  - Prefix caching: vLLM default (on).
  - Env: `EXL3_HOST_EXPERTS=/opt/recipe/plans/plan-agent30-dh.json` (then `plan-gen30-dh.json`),
    `EXL3_HOST_OVERLAP_SMS=32`, `EXL3_HOST_OVERLAP_SMS_PREFILL=64`, `EXL3_HOST_OVERLAP_GROUP=16`, `DSV41_ENGRAM_INT4=1`,
    `DSV41_ENGRAM_DISK=0`, plus the `THOR_*` and `NCCL_*` values in the script.
- **Load** (`vllm-exl3/a100/scripts/bench.py`): streaming, temperature 0, thinking off unless noted, `ignore_eos` for the
  fixed lengths. Prompts:
  - prose: Roman aqueducts essay, TCP congestion control, star life cycle;
  - code: 3,595-token and 14,855-token slices of the vLLM sources, plus a second turn on the 14.8K prefix;
  - 4 parallel prose requests x 256 tokens;
  - a tool call;
  - `merge_intervals` with thinking, 4,000 max tokens.
- **Results** (agent30: `results/agent30/bench.json`; gen30: `results/gen30/console.txt`):

| Test | agent30 | gen30 |
|---|---:|---:|
| Decode, 120-token essay | 29.42 | 35.56 |
| Decode, 480 tokens, prompts 1 / 2 / 3 | 31.22 / 58.64 / 43.33 | 39.36 / 77.82 / 70.11 |
| Decode, 32 tokens after the 3,595-token code prompt | 124.45 | 121.08 |
| Decode, coding with thinking | 99.04 (3,569 tokens) | 149.82 (4,000, capped) |
| TTFT, 3,595-token code prompt | 1.069 s | 1.002 s |
| TTFT, 14,855-token code prompt | 3.092 s | 4.733 s |
| TTFT, second turn (14,848 of 14,901 tokens cached) | 0.834 s | 0.641 s |
| 4 parallel x 256, aggregate | 56.1 (14.2-17.0 per stream) | 80.96 (20.3-24.2 per stream) |
| DSpark mean acceptance length | not recorded | 4.20 (server log line in `console.txt`) |

- **Memory**: VRAM 90,111 to 90,471 MiB (88.0 to 88.4 GiB) of 97,887 MiB (95.6 GiB) per GPU (`results/agent30/bench.log`,
  `results/gen30/console.txt`; the a100 README had the MiB figures labelled GiB, fixed). Host RAM: anon 68-73 GB, shmem 155 GB,
  page cache fills the rest.
- The gen30 JSON was lost (P15).
- **Cost** 1.50 USD.

### 5.11 S11: vllm-exl3 on 1x B300 (stopped, no result)

- **Date** 2026-10-06 10:16 to 11:11 UTC. **Create script** `vllm-exl3/b300/rp_create_b300.py`:
  - `gpuTypeIds ["NVIDIA B300 SXM6 AC"]`, `containerDiskInGb` 600, `minRAMPerGPU` 240, `allowedCudaVersions ["13.0"]`;
  - image `vllm/vllm-openai:deepseekv41-flash-0909`, with an entrypoint that installs openssh.
- **Datacenter**: `logs/created_attempt1_stuck.txt` and `created_attempt2_stuck.txt` say EU-NL-1 for both attempts.
  `vllm-exl3/b300/README.md` says EUR-IS-1 for attempt 1; the create log is the primary source.
- **Result**: the containers never started (runtime null, uptime 0). Attempts were deleted after 22 and 20 min, the
  third after ~1 min when the user stopped the track.
- **Side work**: exllamav3 1.4.9 and vllm_exl3 0.4.2 wheels built on the CPU pod (`vllm-exl3/b300/wheels/install.log`;
  wheels untracked, P18).
- **Cost** 3.85 USD billed (10:00 hour), plus 0.11 for the CPU pod and up to about 1.3 more for the minutes of attempts 2
  and 3 in the 11:00 hour (estimated).

### 5.12 S12: vllm-exl3 on 2x RTX PRO 6000, concurrency sweep

Primary write-up: `vllm-exl3/pro6000-concurrency/README.md` (commit `cd2c86c`).
- **Timeline**: pod created 11:02:44 UTC, deleted 11:36:14 (33.5 min, `rp_list.py` printed `NO PODS RUNNING (clean)`).
  Server ready 11:19:08, warmup done 11:20:24 (restart 11:20:57), A from 11:21:03 to 11:26:56, B from 11:27:03 to
  11:31:02 (`logs/run_all.log`, JSON `started`/`finished`).
- **Pod** `v0tk6f34k3mkpe`, EUR-IS-1, 2x RTX PRO 6000 Blackwell Server Edition. `memoryInGb` 377 (cgroup 351.1 GiB),
  64 vCPU, `containerDiskInGb` 450, disk throughput 24,301 MBps, image as S10 (`logs/pod_id.json`, local only:
  gitignored). Driver 595.91.07 (`logs/nvidia-smi-after-sweep.txt`).
- **Software, model**: as S10 (`scripts/bootstrap.sh`, `scripts/download.py`).
- **Parameters**, from the "non-default args" line in `logs/serve.log`:
  - Sizes: `max_model_len 262144`, `tensor_parallel_size 2`, `kv_cache_memory_bytes 4563402752`, `kv_cache_dtype fp8`,
    `max_num_seqs 64` (script default is 8), `max_num_batched_tokens 16384`, `block_size 64`.
  - Quantization and speculation: `quantization exl3`,
    `speculative_config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`.
  - CUDA graphs: `cudagraph_capture_sizes [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]`,
    `cudagraph_mode FULL_DECODE_ONLY`.
  - Other: `linear_backend marlin`, `disable_custom_all_reduce`, `language_model_only`, `enable_prompt_tokens_details`.
    Prefix caching: default (on). Env as S10.
  - Also in the args line: `tokenizer_mode deepseek_v41`, `tool_call_parser deepseek_v41`, `reasoning_parser deepseek_v41`,
    `enable_auto_tool_choice`, `trust_remote_code`, `custom_ops ["all"]`, and the `kernel_config` of S10.
  - Expert plan: no log records it. The script default is `plan-agent30-dh.json`; the sweep README says gen30.
    Treat the plan as unknown between those two.
  - `gpu_memory_utilization 0.92` was passed but has **no effect**: with an explicit KV size vLLM skips memory profiling,
    as the log says.
- **CUDA graphs**: they cover decode steps up to 256 tokens, which is 42 sequences x (1 + 5 draft tokens). Above about
  42 running sequences, decode runs without graphs (README; the code is `vllm_exl3/exl3.py:1535` on the pod, patched by `vllm-exl3/a100/pack/recipe/patches/vllm-exl3/exl3-thor.patch:146`,
  local only since `pack/` is gitignored; `VLLM_EXL3_FAT_THRESHOLD=256`).
- **KV pool** (`logs/serve.log`): "GPU KV cache size: 623,891 tokens, Maximum concurrency for 262,144 tokens per request:
  2.38x". Only two requests can hold the full 262K at once.
- **Memory at startup** (`logs/memory_startup.txt`, README):
  - Per GPU: weights 76.78 GiB (loaded in 120.6 s), KV 4.25 GiB, CUDA graphs 0.82 GiB, VRAM 91,345 MiB (89.2 GiB) of
    97,887 MiB (95.6 GiB) in use (headroom 6,542 MiB, 6.4 GiB).
  - Pinned host memory (shmem) 144.3 GiB: Engram tables (25.75 GiB per table per rank) and 65-145 of 384 experts per layer (layer 25 keeps 145).
  - cgroup: anon about 71 GiB, memory.current 350.4 GiB, peak = limit 351.1 GiB (the rest is page cache from the weights).
- **Load** (`scripts/loadgen.py`, `scripts/run_all.sh`, `scripts/restart_A.sh`):
  - Closed loop, N in {1, 2, 4, 8, 16, 32, 64}, R = max(2 x N, 8) requests per level.
  - `max_tokens` 256, `temperature` 0.6, `top_p` default, `ignore_eos` false, thinking off, streaming with usage.
  - Timeout 600 s per request; level cap 360 s; stop rule TTFT p95 > 60 s or any error (never triggered).
  - Warmup: levels 1, 8, 64 with 32 tokens on an unused corpus region (`A_OFFSET` 12,000,000).
  - **A**: each request is a different 16,000-character slice of the installed vLLM Python sources (from offset
    20,000,000), wrapped in "Here is part of a Python code base ... Review this code: explain briefly what it does and
    point out one possible bug or improvement". Mean prompt 3,878 to 4,379 tokens.
  - **B**: one 56,800-character code prefix plus one of 108 short questions and a counter (16,109 tokens). One warm
    request first puts the prefix in the cache.
  - RAM/VRAM sampler every 5 s (`scripts/mem_sampler.py`, `results/mem_samples.csv`).
- **Scenario A results** (`results/scenario_A.json`). All requests produced 256 tokens, finish `length`, 0 errors:

| N | n | TTFT p50 s | TTFT p95 s | decode p50 | decode p5 | aggregate | mean prompt | mean cached | DSpark accept | running / waiting max | KV max % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 8 | 0.959 | 1.024 (max) | 116.24 | 111.19 (min) | 82.06 | 4,048.8 | 0 | 3.28 | 1 / 0 | 3.3 |
| 2 | 8 | 0.994 | 1.797 (max) | 70.87 | 54.04 (min) | 112.16 | 3,878.1 | 0 | 3.39 | 2 / 0 | 6.6 |
| 4 | 8 | 1.483 | 3.113 (max) | 40.13 | 29.99 (min) | 125.99 | 4,379.1 | 0 | 3.29 | 4 / 0 | 9.4 |
| 8 | 16 | 2.043 | 4.990 (max) | 23.13 | 16.71 (min) | 152.08 | 3,929.7 | 0 | 3.37 | 8 / 0 | 17.0 |
| 16 | 32 | 3.391 | 9.199 | 13.81 | 9.45 | 175.96 | 3,977.7 | 0 | 3.38 | 16 / 10 | 28.1 |
| 32 | 64 | 5.053 | 18.473 | 8.23 | 5.83 | 209.34 | 4,211.0 | 0 | 3.37 | 32 / 26 | 35.6 |
| 64 | 128 | 8.340 | 33.483 | 4.59 | 3.43 | 234.11 | 4,255.1 | 0 | 3.34 | 64 / 59 | 49.7 |

  Scenario A had 0 prefix-cache hits at every level (0 hit tokens out of 31,025 to 544,657 queried), so its TTFT
  includes the full prefill.

  Requests wait from N=16 on although `max_num_seqs` is 64: with 16,384-token prefill chunks, about four new 4K prompts
  enter per step. The high TTFT p95 values are the first wave (all N start at once). Requests sent after the first wave
  had a TTFT p95 of 2.6 s at N=8, 3.4 s at N=16, 5.8 s at N=32 and 8.7 s at N=64 (README, from per-request start times
  in `scenario_A.json`).

- **Scenario B results** (`results/scenario_B.json`). Warm request: 16,109 tokens, TTFT 2.113 s, 0 cached. Prefix-cache
  hit 99.3% at every level:

| N | n | TTFT p50 s | TTFT p95 s | decode p50 | decode p5 | aggregate | mean prompt | mean cached | DSpark accept | KV max % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.407 | 0.490 (max) | 136.37 | 121.85 (min) | 112.95 | 16,109.2 | 16,000 | 3.87 | 1.3 |
| 2 | 8 | 0.428 | 0.806 (max) | 95.45 | 72.00 (min) | 149.46 | 16,107.5 | 16,000 | 3.92 | 1.5 |
| 4 | 8 | 0.486 | 0.943 (max) | 61.70 | 51.22 (min) | 208.20 | 16,107.5 | 16,000 | 3.90 | 1.9 |
| 8 | 16 | 1.010 | 1.306 (max) | 31.53 | 23.84 (min) | 223.69 | 16,105.8 | 16,000 | 3.90 | 2.7 |
| 16 | 32 | 1.439 | 1.814 | 18.84 | 14.99 | 264.83 | 16,105.0 | 16,000 | 3.95 | 4.2 |
| 32 | 64 | 2.087 | 10.318 | 11.09 | 8.97 | 270.10 | 16,106.2 | 16,000 | 3.95 | 7.1 |
| 64 | 128 | 2.231 | 4.395 | 7.11 | 5.95 | 398.21 | 16,105.7 | 16,000 | 4.03 | 13.1 |

  The 10.3 s TTFT p95 at N=32 comes only from the first wave (10.8 s max, all at t=0). Requests after it had a p95 of
  2.1 s (README). It looks like a one-time stall, not a capacity limit, but this was a single run.

- **Memory during the sweep** (`results/mem_samples.csv`, peak per level):
  - VRAM 91,345 to 91,367 MiB per GPU at every level: the KV pool is preallocated.
  - cgroup RAM 349.4 to 350.6 GiB, anon 70.1 to 71.3 GiB, shmem 144.3 GiB, file 278.3 GiB.
  - KV pool peak: 48.17% (A, N=64) and 12.41% (B, N=64) in the 5 s sampler; 49.7% and 13.1% in loadgen's 1 s samples.
- **Bars** (4.3): A = 8 strict and 8 loose (N=16 misses with p5 9.45 and TTFT p95 9.2 s). B = 8 strict (N=16 misses
  with p5 14.99) and 16 loose.
- **1M-context boot** (`logs/serve_maxlen1048576.log`, `results/maxlen1m_scenario_B_N8.json`):
  - "GPU KV cache size: 1,071,558 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.02x".
  - Scenario B at N=8 on that server: decode p50 38.51 tok/s, TTFT p95 1.194 s, VRAM 93,531 MiB per GPU. One run.
- **Runs not in the tables**: the first boot with CUDA graph sizes up to 384 failed with `cudaErrorStreamCaptureInvalidated`
  (`logs/serve_attempt1_capture384.log`); a first scenario A run with an old calibration was aborted
  (`logs/scenario_A_aborted_old_calibration.log`); warmup results are in `results/warmup_scenario_A.json`. Per-scenario
  console logs, nvidia-smi snapshots and bootstrap/download logs are in `logs/`.
- **Cost** about 2.33 USD (not billed yet at 11:55 UTC).

#### Why 116 tok/s at N=1 here and 29-39 tok/s in S10

`scripts/serve.sh` is byte-identical in both folders. S10 ran `max_num_seqs` 8 and fewer CUDA graph sizes, which do not
matter for one request.

The difference is the text being generated, because DSpark accepts more draft tokens on predictable text:
- S10's 29-39 tok/s come from prose essays.
- S10 also decoded at **124.45 tok/s** (agent30) and 121.08 (gen30) after its 3,595-token code prompt. That is the
  same range as S12's 116 at N=1, whose answers are code reviews of code.

S12 records acceptance per level (3.28 in A, 3.87 in B). S10 did not record it per prompt, so the prose-versus-code
acceptance gap is inferred from the decode numbers, not measured in one run. Expect prose traffic at about a third of
the S12 decode numbers.

## 8. Threats to validity

- **Single runs.** Each configuration was measured once, with no repeats. Small differences ([redacted: out-of-scope run], the
  N=32 vs N=64 TTFT p95 in B) are not significant. Repeats are future work.
- **Synthetic prompts.** [redacted: out-of-scope run] vllm-exl3 used vLLM source slices and fixed question
  templates. No real traffic replay.
- **Burst arrivals.** The sweep starts all N requests together; the first wave dominates TTFT p95 (5.12).
- **Speculative decoding.** DSpark makes vllm-exl3 decode content-dependent (5.12). [redacted: out-of-scope comparison]
- **Prefix cache.** A had 0 hits and B 99.3%, both by design. S10's second-turn TTFT is a cache hit. [redacted: out-of-scope comparison]
- **Post-hoc thresholds.** Both usable bars were chosen after seeing the data.
- **Billing lag.** Costs in the 11:00 UTC hour of 2026-10-06 are estimates.

## 9. Model quality

**Not evaluated**: no perplexity, no KL against the reference, no eval suite. [redacted: out-of-scope quantization] EXL3
3 bpw are aggressive quantizations.

Only anecdotal checks were done:
- [redacted: anecdotal check on an out-of-scope run]
- a correct `merge_intervals` solution with tests and a valid tool call (S10);