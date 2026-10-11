#!/usr/bin/env python3
"""Fetch Diffbot's serving recipe (the `recipe/` folder of the Hugging Face model repo) at a pinned revision.

The recipe is not copied into this repository: it is fetched at build time and verified file by file against
`stack/recipe.lock.tsv` (git blob id as reported by the Hub, plus sha256). No model weights are downloaded.

  python3 stack/fetch_recipe.py                 # fetch and verify into third_party/diffbot-recipe
  python3 stack/fetch_recipe.py --write-lock    # (maintainers) regenerate the lock from the Hub tree listing

License of the fetched files: MIT (Copyright (c) 2026 Diffbot), except recipe/patches/vllm-exl3/, which is
AGPL-3.0-only. See THIRD_PARTY.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000"
REVISION = "a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5"
ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "stack" / "recipe.lock.tsv"
UA = {"User-Agent": "hostoffload-fetch/0.1"}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def tree() -> list[dict]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}/recipe?recursive=1"
    return [x for x in json.loads(get(url)) if x["type"] == "file"]


def read_lock() -> list[dict]:
    rows = []
    for line in LOCK.read_text().splitlines():
        if line.startswith("#") or line.startswith("path\t"):
            continue
        path, size, blob, lfs, sha = line.split("\t")
        rows.append({"path": path, "size": int(size), "blob": blob, "lfs": lfs, "sha256": sha})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT / "third_party" / "diffbot-recipe")
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()
    if args.write_lock:
        rows = []
        for x in sorted(tree(), key=lambda x: x["path"]):
            data = get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{x['path']}")
            lfs = (x.get("lfs") or {}).get("oid", "-")
            if lfs == "-" and git_blob_sha1(data) != x["oid"]:
                raise SystemExit(f"blob mismatch for {x['path']}")
            if lfs != "-" and hashlib.sha256(data).hexdigest() != lfs:
                raise SystemExit(f"lfs sha256 mismatch for {x['path']}")
            rows.append([x["path"], str(x["size"]), x["oid"], lfs, hashlib.sha256(data).hexdigest()])
        with LOCK.open("w") as f:
            f.write(f"# repo\t{REPO}\n# revision\t{REVISION}\n")
            f.write("path\tsize\tgit_blob\tlfs_sha256\tsha256\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"wrote {LOCK} ({len(rows)} files)")
        return 0
    bad = 0
    for r in read_lock():
        dest = args.dest / r["path"].removeprefix("recipe/")
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == r["sha256"]:
            continue
        data = get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{r['path']}")
        if hashlib.sha256(data).hexdigest() != r["sha256"] or len(data) != r["size"]:
            print(f"MISMATCH {r['path']}", file=sys.stderr)
            bad += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    if bad:
        return 1
    (args.dest / ".fetched-from").write_text(f"{REPO}@{REVISION}\n")
    print(f"recipe verified in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
