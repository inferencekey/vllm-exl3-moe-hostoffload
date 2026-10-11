#!/usr/bin/env python3
"""Download a pinned checkpoint to the pod's local disk (never into the repository). MODEL=exl3|orig.
Verifies every safetensors file against the LFS sha256 recorded in data/static/hub/ when VERIFY=1 (slow)."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pins():
    out = {}
    for line in (ROOT / "stack" / "pins.env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v
    return out


def main() -> int:
    from huggingface_hub import snapshot_download
    p = pins()
    which = os.environ.get("MODEL", sys.argv[1] if len(sys.argv) > 1 else "exl3")
    repo, rev = {"exl3": (p["MODEL_EXL3_REPO"], p["MODEL_EXL3_REVISION"]),
                 "orig": (p["MODEL_ORIG_REPO"], p["MODEL_ORIG_REVISION"])}[which]
    dest = Path(os.environ.get("MODELS_DIR", "/root/models")) / f"{which}-{'3bpw-' if which == 'exl3' else ''}{rev}"
    snapshot_download(repo, revision=rev, local_dir=dest, max_workers=int(os.environ.get("HF_WORKERS", "32")))
    if os.environ.get("VERIFY") == "1":
        meta = next((ROOT / "data" / "static" / "hub").glob(f"{repo.replace('/', '__')}@{rev[:8]}.json"))
        for f in json.loads(meta.read_text())["files"]:
            if f["lfs_sha256"]:
                h = hashlib.sha256()
                with open(dest / f["path"], "rb") as fh:
                    while b := fh.read(1 << 24):
                        h.update(b)
                assert h.hexdigest() == f["lfs_sha256"], f["path"]
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
