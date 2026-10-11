"""Snapshot Hugging Face Hub metadata (revision, license, gated flag, file sizes) for the checkpoints in scope.

The snapshots are the data source for the static feasibility arithmetic (which checkpoint fits where). They hold
metadata only, never weights.

  python3 -m hostoffload.weights.hub_meta --out data/static/hub
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# (repo, pinned revision or None for "current main, record what it resolves to")
CHECKPOINTS = [
    ("deepseek-ai/DeepSeek-V4.1-Flash", None),
    ("nvidia/DeepSeek-V4.1-Flash-NVFP4", None),
    ("diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000", "a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5"),
    ("coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw", "650cae2c13aaaec303871a35301503570889c0be"),
]


def fetch(repo: str, revision: str | None) -> dict:
    url = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{revision}" if revision else "") + "?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hostoffload/0.1"}),
                                timeout=120) as r:
        d = json.load(r)
    files = [{"path": s["rfilename"], "size": s.get("size"),
              "lfs_sha256": (s.get("lfs") or {}).get("sha256")} for s in d.get("siblings", [])]
    st = [f for f in files if f["path"].endswith(".safetensors")]
    return {
        "repo": repo, "requested_revision": revision, "sha": d.get("sha"), "fetched_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gated": d.get("gated"), "private": d.get("private"),
        "license": (d.get("cardData") or {}).get("license"), "last_modified": d.get("lastModified"),
        "safetensors_params_total": (d.get("safetensors") or {}).get("total"),
        "safetensors_bytes_total": sum(f["size"] or 0 for f in st), "n_safetensors": len(st),
        "all_files_bytes_total": sum(f["size"] or 0 for f in files), "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for repo, rev in CHECKPOINTS:
        meta = fetch(repo, rev)
        path = args.out / f"{repo.replace('/', '__')}@{meta['sha'][:8]}.json"
        path.write_text(json.dumps(meta, indent=1))
        print(f"{repo}@{meta['sha'][:8]}: {meta['safetensors_bytes_total'] / 2**30:.1f} GiB safetensors, "
              f"license {meta['license']}, gated {meta['gated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
