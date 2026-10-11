# DeepSeek-V4.1-Flash on vllm-exl3, 2x RTX PRO 6000: how many concurrent users (run 2026-10-06)

**Answer:** with 256-token answers, per-user decode speed is what runs out first. TTFT only becomes a problem when
requests arrive all at once.

| Scenario | Usable (p5 decode >= 15 tok/s, TTFT p95 <= 5 s / 15 s) | Looser bar (p5 >= 10 tok/s, TTFT p95 <= 10 s / 30 s) |
|---|---:|---:|
| A: independent users, a different ~4K-token code prompt each | **N = 8** | **N = 8** (N = 16 misses: p5 9.4 tok/s, TTFT p95 9.2 s) |
| B: assistant sessions, shared ~16K-token prefix + short question | **N = 8** (N = 16 misses by a hair: p5 14.99 tok/s) | **N = 16** |

Aggregate output keeps growing past the usable point (A: 152 tok/s at N=8, 234 at N=64; B: 224 at N=8, 398 at N=64),
and there were no errors or timeouts at any level up to N=64. So the server takes the load, but each user gets slower.

Pod `v0tk6f34k3mkpe` (`ik-vllmexl3-pro6000-conc`), RunPod SECURE, EUR-IS-1, 2x RTX PRO 6000 Blackwell Server Edition,
4.18 USD/h. It was created on the first try at 11:02:44 UTC and deleted at 11:36:14 UTC (HTTP 204, then `rp_list.py`
printed `NO PODS RUNNING (clean)`). That is **33.5 min, about 2.33 USD**.

## What was measured on

The stack is the same as `../a100/` (that run's README covers the build). This run used the faster general plan:

- Image `vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89`, the 5 vLLM patches, vllm-exl3 `d3cfd394` +
  `exl3-thor.patch`, and ExLlamaV3 `5be88657` built for `12.0a`. Bootstrap took 214 s.
- Pack `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000@a1cf1bd2` (310 GB on the container disk). Download took 99 s.
- `scripts/serve.sh` with `PLAN=plan-gen30-dh.json`, TP2, CUDA graphs `FULL_DECODE_ONLY`, DSpark k=5, fp8 KV 4.25 GiB,
  `--max-model-len 262144`, prefix caching on, text only.
- **Two changes from the recipe defaults, needed to admit 64 users:** `MAX_NUM_SEQS=64` (default 8) and
  `CAPTURE_SIZES="1 2 4 6 8 12 16 24 32 48 64 96 128 192 256"`.
  - The first boot also captured 384 and failed graph capture with `cudaErrorStreamCaptureUnsupported`. The cause is
    in `vllm_exl3/exl3.py:1535`: above `VLLM_EXL3_FAT_THRESHOLD=256` tokens, the fat-expert path calls
    `fat.any().item()`, and that host sync is not allowed during capture. The log is `logs/serve_attempt1_capture384.log`.
  - So CUDA graphs cover decode steps of up to 256 tokens, which is 42 sequences x (1 + 5 draft tokens). Decode steps
    with more than about 42 running sequences run without graphs.
  - The second boot was ready 3 min 35 s after launch (weights 90 s, model load 121 s, graph capture 57 s).
- Load generator `scripts/loadgen.py` (Python asyncio + aiohttp) ran on the pod against `127.0.0.1:8000`, using
  `/v1/chat/completions` with `stream=true`, `max_tokens 256`, `temperature 0.6`, `ignore_eos false` and thinking off
  (`chat_template_kwargs.thinking=false`).
  - Every request generated the full 256 tokens (finish reason `length`).
  - Closed loop: N requests are in flight at all times until `max(2N, 8)` requests are done.
  - TTFT is the time to the first content token. Per-stream decode is (completion_tokens - 1) / (last token time -
    first token time). p5 is the slowest 5% of users.
  - `/metrics` was sampled every 1 s (running and waiting requests, KV usage, prefix-cache hits, DSpark acceptance),
    `nvidia-smi` every 5 s.
- Prompts are real code: slices of the installed vLLM Python sources, 43.4 M characters in total.
  - **A:** each request gets a different, non-overlapping slice. Prompts were 3,495 to 5,639 tokens, about 4.0-4.4K on
    average per level. Prefix-cache hit rate was 0.0%.
  - **B:** every request starts with the same 16,1xx-token code context, then a different question about one of the
    functions or classes in it ("what does `X` do", "how would you test `X`", ...).
    - One request put the prefix in the cache before B started: 16,109 tokens, TTFT 2.11 s.
    - After that, every B request had 16,000 of its ~16,106 prompt tokens cached (99.3% prefix-cache hit rate).
- Warmup before the sweep: 1, 8 and 64 concurrent 32-token requests on a separate corpus region.
- Before the sweep, scenario A was started once with a miscalibrated prompt size (about 3.2K tokens). It was stopped
  after about 30 s and rerun on a fresh corpus region. Only the rerun is reported.

## Scenario A: independent users, ~4K-token code prompt, 256-token answer

| N | requests ok/err | prompt tok | TTFT p50 s | TTFT p95 s | decode p50 tok/s | decode p5 tok/s | aggregate out tok/s | running max | waiting max | KV use max % | DSpark accept len |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/0 | 4,049 | 0.96 | 1.02 | 116.2 | 111.2 | 82.1 | 1 | 0 | 3.3 | 3.28 |
| 2 | 8/0 | 3,878 | 0.99 | 1.80 | 70.9 | 54.0 | 112.2 | 2 | 0 | 6.6 | 3.39 |
| 4 | 8/0 | 4,379 | 1.48 | 3.11 | 40.1 | 30.0 | 126.0 | 4 | 0 | 9.4 | 3.29 |
| **8** | 16/0 | 3,930 | 2.04 | **4.99** | 23.1 | **16.7** | 152.1 | 8 | 0 | 17.0 | 3.37 |
| 16 | 32/0 | 3,978 | 3.39 | 9.20 | 13.8 | 9.4 | 176.0 | 16 | 10 | 28.1 | 3.38 |
| 32 | 64/0 | 4,211 | 5.05 | 18.47 | 8.2 | 5.8 | 209.3 | 32 | 26 | 35.6 | 3.37 |
| 64 | 128/0 | 4,255 | 8.34 | 33.48 | 4.6 | 3.4 | 234.1 | 64 | 59 | 49.7 | 3.34 |

The sweep did not stop early: TTFT p95 stayed under 60 s and there were no errors.

## Scenario B: shared ~16K-token prefix + different question, 256-token answer

| N | requests ok/err | prompt tok (cached) | TTFT p50 s | TTFT p95 s | decode p50 tok/s | decode p5 tok/s | aggregate out tok/s | running max | waiting max | KV use max % | prefix hit % | DSpark accept len |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/0 | 16,109 (16,000) | 0.41 | 0.49 | 136.4 | 121.8 | 113.0 | 1 | 0 | 1.3 | 99.3 | 3.87 |
| 2 | 8/0 | 16,108 (16,000) | 0.43 | 0.81 | 95.5 | 72.0 | 149.5 | 2 | 0 | 1.5 | 99.3 | 3.92 |
| 4 | 8/0 | 16,108 (16,000) | 0.49 | 0.94 | 61.7 | 51.2 | 208.2 | 4 | 0 | 1.9 | 99.3 | 3.90 |
| **8** | 16/0 | 16,106 (16,000) | 1.01 | 1.31 | 31.5 | **23.8** | 223.7 | 8 | 0 | 2.7 | 99.3 | 3.90 |
| **16** | 32/0 | 16,105 (16,000) | 1.44 | 1.81 | 18.8 | **14.99** | 264.8 | 16 | 0 | 4.2 | 99.3 | 3.95 |
| 32 | 64/0 | 16,106 (16,000) | 2.09 | 10.32 | 11.1 | 9.0 | 270.1 | 32 | 0 | 7.1 | 99.3 | 3.95 |
| 64 | 128/0 | 16,106 (16,000) | 2.23 | 4.39 | 7.1 | 6.0 | 398.2 | 64 | 0 | 13.1 | 99.3 | 4.03 |

## Reading the numbers

- **Decode per user is the limit.** One stream decodes at 111-136 tok/s. With N streams, each gets roughly
  aggregate / N, and aggregate grows slowly: about 1.9x from N=1 to N=8 in A and 2x in B.
  - The model decodes the experts that live in host RAM over PCIe, and DSpark's verification batch grows 6x per
    sequence. Both make a batched decode step much more expensive than a single-stream one.
  - The 15 tok/s floor is crossed between N=8 and N=16 in both scenarios.
- **Prefill also slows decode in A.** With chunked prefill (16,384-token chunks), each new 4K prompt takes GPU time away
  from the streams that are already decoding. A's per-stream decode is 25-30% below B's at the same N, and B prefills
  only about 110 new tokens per request.
- **The high TTFT p95 values are the burst at the start of each level.** All N requests arrive at once.
  - Split by arrival, A's TTFT p95 for requests sent after the first wave is 2.6 s at N=8, 3.4 s at N=16, 5.8 s at
    N=32 and 8.7 s at N=64 (`results/scenario_A.json` has every request's start time).
  - With users arriving at random, TTFT would pass the A bar up to about N=32. Decode p5 would still fail it above N=8.
- **B at N=32:** the 10.3 s TTFT p95 comes only from the first wave (10.8 s max, all at t=0). Requests after it had a
  p95 of 2.1 s, and N=64 had a TTFT p95 of 4.4 s. This looks like a one-time stall the first time the server saw 32
  new requests at once, not a capacity limit.
- **The prefix cache works under load.** B's TTFT stays at 0.4-2.2 s median from 1 to 64 users with a 16K context. A
  fresh 16K prompt takes 2.1 s alone.
- **KV cache was never the limit** at these prompt sizes: peak 49.7% (A, N=64, about 64 x 4.5K tokens live). Waiting
  requests in A come from the prefill queue (16K tokens per step), not from a lack of KV space.

## Memory

### VRAM per GPU, `max_model_len 262144` (the measured configuration)

Sources: `logs/serve.log`, `logs/memory_startup.txt`, `results/mem_samples.csv`.

| Item | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell Server Edition, 97,887 MiB (95.6 GiB) in nvidia-smi |
| Free at engine start | 93.93 GiB |
| Model weights in VRAM (vLLM "Model loading took") | 76.78 GiB per GPU |
| KV cache | 4.25 GiB per GPU (`--kv-cache-memory 4563402752`), fp8 |
| KV capacity | **623,891 tokens; "Maximum concurrency for 262,144 tokens per request: 2.38x"** |
| `gpu_memory_utilization` | 0.92 passed, but **not used**: with an explicit `kv_cache_memory_bytes`, vLLM skips memory profiling (the log says so) |
| CUDA graphs | 0.82 GiB per GPU (capture sizes up to 256; the recipe default, up to 48, took 0.30 GiB) |
| nvidia-smi used, idle after warmup | 91,345 MiB per GPU |
| nvidia-smi used, peak during the sweep | **91,367 MiB per GPU** (A and B, N=64) |
| **Headroom at 262144** | **6,520 MiB (6.4 GiB) per GPU** at peak |

VRAM barely moves with load: 91,345 to 91,367 MiB from N=1 to N=64. The KV pool and the workspaces are allocated up
front.

Per-level peaks from the 5 s sampler (`results/mem_samples.csv`):

| Scenario | N | peak VRAM GPU0 / GPU1 MiB | peak cgroup RAM GiB | peak anon GiB | shmem GiB | peak KV use % |
|---|---:|---|---:|---:|---:|---:|
| A | 1 | 91,345 / 91,345 | 350.6 | 71.3 | 144.3 | 3.7 |
| A | 2 | 91,345 / 91,345 | 349.4 | 70.1 | 144.3 | 0.9 |
| A | 4 | 91,345 / 91,345 | 349.4 | 70.1 | 144.3 | 6.8 |
| A | 8 | 91,345 / 91,345 | 349.4 | 70.1 | 144.3 | 14.2 |
| A | 16 | 91,345 / 91,345 | 349.4 | 70.1 | 144.3 | 28.1 |
| A | 32 | 91,345 / 91,345 | 349.4 | 70.1 | 144.3 | 34.8 |
| A | 64 | 91,365 / 91,365 | 349.4 | 70.1 | 144.3 | 48.2 |
| B | 1 | 91,365 / 91,365 | 350.6 | 71.2 | 144.3 | 1.3 |
| B | 2 | 91,365 / 91,365 | 349.4 | 70.1 | 144.3 | 1.5 |
| B | 4 | 91,365 / 91,365 | 349.4 | 70.1 | 144.3 | 1.7 |
| B | 8 | 91,365 / 91,365 | 349.4 | 70.1 | 144.3 | 2.6 |
| B | 16 | 91,365 / 91,365 | 349.4 | 70.1 | 144.3 | 3.8 |
| B | 32 | 91,367 / 91,367 | 349.5 | 70.2 | 144.3 | 7.0 |
| B | 64 | 91,367 / 91,367 | 349.5 | 70.3 | 144.3 | 12.4 |

Notes on this table:

- The sampler's 5 s KV figure can miss short peaks. The 1 s `/metrics` peak in the scenario tables is the better
  number (A, N=64: 49.7%).
- The A, N=1 row also covers the 30 s aborted first A attempt.
- GPU utilization averaged 97% (A) and 95% (B) in the samples.

### Pod RAM (cgroup v2, `logs/memory_startup.txt`)

- Limit `memory.max` = 376,999,997,440 B = **351.1 GiB** (the pod's `memoryInGb` 377). The host has 1.5 TB and 256 vCPU.
- While serving (`memory.stat`):
  - `memory.current` 350.4 GiB, at the limit, because the page cache from the downloaded weights fills the rest.
  - `anon` 71.1 GiB
  - `shmem` 144.3 GiB
  - `file` 278.3 GiB (cgroup v2 counts shmem inside `file`, so about 134 GiB of it is reclaimable page cache)
- Pinned host memory, from the vLLM log:
  - Engram tables: 25.75 GiB per rank per Engram layer, 2 layers x 2 ranks = 103 GiB.
  - Cold experts of the gen30 plan, kept in pinned host RAM: 65-145 of 384 per layer, plus 32/64/80 of 128 for the
    3 DSpark layers.
  - The embedding table (0.62 GiB per GPU freed from VRAM).
  - All of these show up in `shmem`.
- RAM did not move with load: anon 69-71 GiB, shmem 144.3 GiB at every N.
- **Headroom:** about 351 - 71 - 144 = **136 GiB** of the limit is not anon or shmem, but most of it holds reclaimable
  page cache. vLLM printed a cgroup preflight warning at load ("240 MiB required ... 120 MiB available") and carried on.
  Nothing failed.

### Larger context: `max_model_len 1048576` (one extra boot, same pod, no extra download)

| | 262,144 (measured) | 1,048,576 (boot check only) |
|---|---:|---:|
| Model loading (VRAM per GPU) | 76.78 GiB | 77.15 GiB (bigger rope / indexer buffers) |
| KV cache bytes | 4.25 GiB | 4.25 GiB |
| KV capacity reported | 623,891 tokens, 2.38x at max len | 1,071,558 tokens, 1.02x at max len |
| CUDA graphs | 0.82 GiB | 1.47 GiB |
| nvidia-smi after an 8 x 16K burst | 91,367 MiB peak in the sweep | 93,531 MiB |
| Headroom per GPU | 6,520 MiB | 4,356 MiB |

With the same 4.25 GiB, vLLM reports more KV tokens at the larger `max_model_len`. The V4.1 hybrid cache appears to
split its bytes between the per-request sliding-window part and the compressed long-range part according to
`max_model_len`. We did not investigate further.

The 1M server answered a B-style probe (8 concurrent requests, 16K cached prefix) at a TTFT p50 of 1.19 s and a decode
p5 of 36.8 tok/s (`results/maxlen1m_scenario_B_N8.json`). That is a single 8-request burst, so it is not comparable to
the B table row.

A smaller `max_model_len` was not booted, to keep within budget.

## Reproduce

From `~/me/inferencekey`, wrap every RunPod call in `scripts/secrets.sh run jenkins ...`.

1. Run `OUT=.../logs/pod_id python3 scripts/rp_create_loop.py`. It retries every 45 s for up to 2 h, and deletes any pod
   with less than 250 GB RAM or a price above 4.40 USD/h.
2. Start `scripts/watchdog.sh <id> 4000` (hard stop at about 4.6 USD).
3. Copy `../a100/stage/*.tgz`, `bootstrap.sh serve.sh download.py loadgen.py mem_sampler.py run_all.sh` to `/root`.
4. Untar into `/opt/recipe` and `/opt/dsv41-patch`. Both tarballs have a top-level folder (`recipe/`, `patch/`): move
   their contents up one level.
5. Run the hf `pip install --target /opt/hfpkgs` plus `download.py`, and `bootstrap.sh`, in parallel, each started
   from `/root`.
6. `PLAN=plan-gen30-dh.json MAX_NUM_SEQS=64 CAPTURE_SIZES="1 2 4 6 8 12 16 24 32 48 64 96 128 192 256" API_KEY=... bash serve.sh`
7. In parallel, run `API_KEY=... bash run_all.sh` (waits for the server, warms up, then runs A and B) and
   `python3 mem_sampler.py`.
8. Summarize locally with `python3 scripts/summarize.py results`.
9. Copy one file at a time, compare `sha256sum` on both ends, then `rp_delete.py` and `rp_list.py`.

`restart_A.sh` is the one-off script that reran A and B after the calibration fix.

Do not use `pkill -f <pattern>` over SSH when the pattern also appears in the SSH command line. It kills the remote
shell (exit 255). Kill by PID instead.

## Files

- `scripts/loadgen.py`: the load generator. `scripts/mem_sampler.py`: the 5 s CSV sampler. `scripts/summarize.py`:
  builds the tables. Also in `scripts/`: run, serve, bootstrap, download, create-loop and watchdog scripts.
- `results/scenario_A.json` and `results/scenario_B.json` hold every request (start time, TTFT, decode, tokens,
  cached tokens), the per-level summaries and the 1 s server samples.
- `results/mem_samples.csv` is the 5 s VRAM / RAM / KV / queue sampler output.
- `results/warmup_scenario_A.json` is the warmup run. `results/maxlen1m_scenario_B_N8.json` is the 1M-context probe.
- `logs/`: `serve.log` (the measured server), `serve_attempt1_capture384.log` (the failed capture),
  `serve_maxlen1048576.log`, `memory_startup.txt`, plus the bootstrap, download, scenario, warmup, nvidia-smi and
  create-loop logs.

Licensing is the same as `../a100/`: vllm-exl3 and its patch are AGPL-3.0, the recipe and the weights are MIT.
