"""Static weight accounting: which bytes of the checkpoint a placement puts in VRAM and which in host memory.

Inputs are the safetensors headers of the pack (tensor names, dtypes, shapes, byte sizes) and a host-expert plan
(the JSON format read by the patched vllm-exl3 plugin: {"<prefix>.layers.<L>.ffn.experts": [expert ids]}).
The placement rules mirror what the serving configuration documents (recipe README, serve.sh): routed experts in
the plan go to pinned host memory, Engram embedding tables and the token embedding go to pinned host memory, the
vision tower is not loaded in text-only mode, everything else stays in VRAM. These rules are an explicit model;
`hostoffload.analysis.serve_log` checks them against what the server reports at load time.

Outputs are file bytes (as stored in the checkpoint). Runtime sizes can differ (repacking, FP8 copies, replication of
host tables per tensor-parallel rank); the paper reports both and never mixes them.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .safetensors_headers import TensorInfo, fetch_header, list_shards, tensors_from_header

GiB = 2 ** 30

# Order matters: the first matching rule wins.
CLASS_RULES: list[tuple[str, re.Pattern]] = [
    ("routed_expert", re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")),
    ("drafter_routed_expert", re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.")),
    ("engram_table", re.compile(r"^layers\.(\d+)\.engram\.embed\.")),
    ("engram_proj", re.compile(r"^layers\.(\d+)\.engram\.")),
    ("shared_expert", re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.shared_experts\.")),
    ("router", re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.gate\.")),
    ("attention", re.compile(r"^layers\.(\d+)\.attn\.")),
    ("drafter_dense", re.compile(r"^mtp\.")),
    ("embedding", re.compile(r"^embed\.")),
    ("lm_head", re.compile(r"^head\.")),
    ("vision", re.compile(r"^(vision\.|aligner\.|image_)")),
    ("norms_hc", re.compile(r"^(layers\.\d+\.(hc_|attn_norm|ffn_norm)|norm\.)")),
]

N_TARGET_LAYERS = 40  # text_config.num_hidden_layers; drafter (mtp) layer m is plan layer 40 + m


def classify(name: str) -> tuple[str, int | None, int | None]:
    """(class, layer, expert) of a tensor name; layer is the plan layer index (drafter layers offset by 40)."""
    for cls, pat in CLASS_RULES:
        m = pat.match(name)
        if not m:
            continue
        if cls == "routed_expert":
            return cls, int(m.group(1)), int(m.group(2))
        if cls == "drafter_routed_expert":
            return cls, N_TARGET_LAYERS + int(m.group(1)), int(m.group(2))
        if cls in ("engram_table", "engram_proj", "attention"):
            return cls, int(m.group(1)), None
        return cls, None, None
    return "other", None, None


def load_plan(path: Path | None) -> dict[int, set[int]]:
    """Plan layer -> host expert ids. An absent plan means every expert in VRAM."""
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    out: dict[int, set[int]] = {}
    for key, ids in raw.items():
        m = re.search(r"layers\.(\d+)\.ffn\.experts$", key)
        if not m:
            raise ValueError(f"unexpected plan key {key!r}")
        out[int(m.group(1))] = set(int(i) for i in ids)
    return out


@dataclass
class Placement:
    """Placement rules; defaults are the documented serving configuration (text only)."""
    plan: dict[int, set[int]] = field(default_factory=dict)
    engram_on_host: bool = True
    embedding_on_host: bool = True
    load_vision: bool = False

    def where(self, cls: str, layer: int | None, expert: int | None) -> str:
        if cls in ("routed_expert", "drafter_routed_expert"):
            return "host" if expert in self.plan.get(layer, set()) else "vram"
        if cls == "engram_table":
            return "host" if self.engram_on_host else "vram"
        if cls == "embedding":
            return "host" if self.embedding_on_host else "vram"
        if cls == "vision":
            return "vram" if self.load_vision else "not_loaded"
        return "vram"


def _open_text(path: Path, mode: str):
    if str(path).endswith(".gz"):
        if "w" in mode:
            # mtime=0 keeps the compressed file byte-identical across runs
            return io.TextIOWrapper(gzip.GzipFile(path, mode + "b", mtime=0), newline="")
        return io.TextIOWrapper(gzip.open(path, mode + "b"), newline="")
    return open(path, mode, newline="")


def read_tensor_table(path: Path) -> list[TensorInfo]:
    rows = []
    with _open_text(path, "r") as f:
        for r in csv.DictReader(f):
            rows.append(TensorInfo(r["name"], r["shard"], r["dtype"],
                                   tuple(int(x) for x in r["shape"].split("x")) if r["shape"] else (),
                                   int(r["nbytes"])))
    return rows


def write_tensor_table(tensors: list[TensorInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "w") as f:
        w = csv.writer(f)
        w.writerow(["name", "shard", "dtype", "shape", "nbytes"])
        for t in sorted(tensors, key=lambda t: t.name):
            w.writerow([t.name, t.shard, t.dtype, "x".join(str(s) for s in t.shape), t.nbytes])


def account(tensors: list[TensorInfo], placement: Placement) -> dict:
    """Bytes per (class, location), per layer for routed experts, and expert counts."""
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_layer: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    experts: dict[int, set[int]] = defaultdict(set)
    expert_bytes: dict[tuple[int, int], int] = defaultdict(int)
    for t in tensors:
        cls, layer, expert = classify(t.name)
        loc = placement.where(cls, layer, expert)
        by_class[cls][loc] += t.nbytes
        if expert is not None:
            per_layer[layer][loc] += t.nbytes
            experts[layer].add(expert)
            expert_bytes[(layer, expert)] += t.nbytes
    totals: dict[str, int] = defaultdict(int)
    for locs in by_class.values():
        for loc, b in locs.items():
            totals[loc] += b
    n_host = sum(len(placement.plan.get(L, set()) & experts[L]) for L in experts)
    return {
        "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "per_layer": {k: dict(v) for k, v in sorted(per_layer.items())},
        "experts_per_layer": {k: len(v) for k, v in sorted(experts.items())},
        "host_experts_per_layer": {k: len(placement.plan.get(k, set()) & v) for k, v in sorted(experts.items())},
        "host_experts_total": n_host,
        "experts_total": sum(len(v) for v in experts.values()),
        "expert_bytes_min": min(expert_bytes.values()) if expert_bytes else 0,
        "expert_bytes_max": max(expert_bytes.values()) if expert_bytes else 0,
        "totals": dict(totals),
    }


def fetch_tensor_table(repo: str, revision: str, cache_dir: Path) -> tuple[list[TensorInfo], list[dict]]:
    tensors: list[TensorInfo] = []
    shards = []
    for s in list_shards(repo, revision):
        header, digest = fetch_header(repo, revision, s["path"], cache_dir)
        tensors.extend(tensors_from_header(header, s["path"]))
        shards.append({"path": s["path"], "size": s["size"], "lfs_sha256": (s.get("lfs") or {}).get("oid"),
                       "header_sha256": digest})
    return tensors, shards


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000")
    ap.add_argument("--revision", default="a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5")
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/safetensors-headers"))
    ap.add_argument("--table", type=Path, help="use an existing tensor table instead of fetching headers")
    ap.add_argument("--plan", type=Path, action="append", default=[], help="host-expert plan JSON (repeatable)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.table:
        tensors = read_tensor_table(args.table)
    else:
        tensors, shards = fetch_tensor_table(args.repo, args.revision, args.cache_dir)
        write_tensor_table(tensors, args.out / "tensors.csv.gz")
        (args.out / "shards.json").write_text(json.dumps(
            {"repo": args.repo, "revision": args.revision, "shards": shards}, indent=1))
    results = {"repo": args.repo, "revision": args.revision, "n_tensors": len(tensors),
               "total_file_bytes": sum(t.nbytes for t in tensors), "placements": {}}
    results["placements"]["no_plan"] = account(tensors, Placement())
    for p in args.plan:
        results["placements"][p.stem] = account(tensors, Placement(plan=load_plan(p)))
        results["placements"][p.stem]["plan_file"] = p.name
        results["placements"][p.stem]["plan_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out / "placement.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    print(f"{len(tensors)} tensors, {results['total_file_bytes'] / GiB:.2f} GiB; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
