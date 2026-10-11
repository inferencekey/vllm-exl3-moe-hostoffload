"""Read safetensors headers (tensor names, dtypes, shapes, byte offsets) without downloading the weights.

A safetensors file starts with an unsigned little-endian 64-bit header length followed by a JSON header. Two HTTP
range requests per shard are enough, so the 309 GiB pack can be described from about 30 MB of downloads. Headers are
cached under a local directory (never committed); the derived per-tensor table is what the repo keeps.
"""
from __future__ import annotations

import hashlib
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF = "https://huggingface.co"
USER_AGENT = "hostoffload-weights/0.1 (+reproducibility artifact)"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


def _get_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def parse_header(raw: bytes) -> dict:
    """Parse a complete safetensors prefix (8-byte length + JSON) into the header dict."""
    (n,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + n])


def fetch_header(repo: str, revision: str, filename: str, cache_dir: Path) -> tuple[dict, str]:
    """Header dict and the sha256 of the raw header bytes, cached in cache_dir."""
    cache = cache_dir / repo.replace("/", "__") / revision / (filename + ".header")
    if not cache.exists():
        url = f"{HF}/{repo}/resolve/{revision}/{filename}"
        (n,) = struct.unpack("<Q", _get_range(url, 0, 7))
        raw = _get_range(url, 0, 7 + n)
        if len(raw) != 8 + n:
            raise IOError(f"short read for {filename}: {len(raw)} != {8 + n}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    raw = cache.read_bytes()
    return parse_header(raw), hashlib.sha256(raw).hexdigest()


def tensors_from_header(header: dict, shard: str) -> list[TensorInfo]:
    out = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out.append(TensorInfo(name, shard, meta["dtype"], tuple(meta["shape"]), b - a))
    return out


def list_shards(repo: str, revision: str) -> list[dict]:
    """Tree listing (path, size, lfs oid) of the safetensors shards at a revision."""
    url = f"{HF}/api/models/{repo}/tree/{revision}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        tree = json.load(r)
    return [x for x in tree if x["path"].endswith(".safetensors")]
