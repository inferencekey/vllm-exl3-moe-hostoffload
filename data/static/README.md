# Static data (measured offline, no GPU)

These files are measurements of public artifacts, taken on the dev machine at no cost. They are *controlled* in the
sense that anyone can regenerate them byte for byte (except fetch timestamps) with the commands below.

| Path | What | Regenerate |
|---|---|---|
| `hub/*.json` | Hugging Face Hub metadata (resolved revision, license, gated flag, per-file sizes and LFS sha256) of the checkpoints in scope | `make static-hub` |
| `weights/diffbot-a1cf1bd2/tensors.csv.gz` | Every tensor of the served pack (name, shard, dtype, shape, bytes) from the safetensors headers (HTTP range requests, no weights downloaded) | `make static-weights` |
| `weights/diffbot-a1cf1bd2/shards.json` | Shard list with size, LFS sha256 and the sha256 of each raw header | same |
| `weights/diffbot-a1cf1bd2/placement.json` | File bytes per tensor class and location (VRAM, host, not loaded) for no plan and for the three published host-expert plans (with their sha256) | same |
| `weights/deepseek-ai-2cba9e42/` | The same for DeepSeek's original checkpoint (FP4 experts, FP8 Engram) | same |

The placement rules are an explicit model of the documented serving configuration (`src/hostoffload/weights/split.py`).
Runtime sizes differ from file sizes (repacking, FP8 copies, per-rank sharding); the controlled runs measure those.
