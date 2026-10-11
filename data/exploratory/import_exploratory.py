#!/usr/bin/env python3
"""Import the historical (exploratory) vllm-exl3 runs from the inferencekey-deepseek repo.

Every file is read from git objects at the pinned source commit with `git show <commit>:<path>`, so the
working tree of the source repo cannot leak into the import. The spec (import_spec.json) says which files are
copied verbatim, which are redacted (line-level, with visible markers) and which are excerpted.

MANIFEST.tsv records, per imported file: destination path, source path, mode, git blob id at the source commit,
sha256 of the source blob, sha256 of the imported file and its size. tests/test_exploratory_manifest.py checks it.

Usage: python3 data/exploratory/import_exploratory.py --src ~/me/inferencekey-deepseek
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git(src: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(src), *args], check=True, capture_output=True).stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_files(src: Path, commit: str, prefixes: list[str]) -> list[tuple[str, str]]:
    """(path, blob id) of every file tracked at `commit` under one of `prefixes`."""
    out = git(src, "ls-tree", "-r", commit, "--", *prefixes).decode()
    rows = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _mode, kind, blob = meta.split()
        if kind == "blob":
            rows.append((path, blob))
    return rows


def blob_id(src: Path, commit: str, path: str) -> str:
    return git(src, "rev-parse", f"{commit}:{path}").decode().strip()


def apply_line_edits(text: str, keep: list[list[int]] | None, drop: list[int], replace: dict[str, str]) -> str:
    """Keep the given 1-based inclusive ranges (all lines if None), drop lines, replace lines."""
    lines = text.split("\n")
    keep_set = set(range(1, len(lines) + 1)) if keep is None else {n for a, b in keep for n in range(a, b + 1)}
    out = []
    for n, line in enumerate(lines, start=1):
        if str(n) in replace:
            out.append(replace[str(n)])
        elif n in keep_set and n not in drop:
            out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="local clone of inferencekey-deepseek")
    ap.add_argument("--spec", type=Path, default=HERE / "import_spec.json")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    commit = spec["source_commit"]
    resolved = git(args.src, "rev-parse", commit).decode().strip()
    if resolved != commit:
        print(f"source commit mismatch: {resolved} != {commit}", file=sys.stderr)
        return 1
    dest_root = HERE / spec["dest_root"]
    rows = []

    def write(dest_rel: str, src_path: str, mode: str, blob: str, src_bytes: bytes, out_bytes: bytes) -> None:
        dest = dest_root / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(out_bytes)
        rows.append([f"{spec['dest_root']}/{dest_rel}", src_path, mode, blob, sha256(src_bytes),
                     sha256(out_bytes), str(len(out_bytes))])

    excluded = set(spec.get("exclude_paths", []))
    for path, blob in tracked_files(args.src, commit, spec["verbatim_prefixes"]):
        if path in excluded:
            continue
        data = git(args.src, "show", f"{commit}:{path}")
        write(path, path, "verbatim", blob, data, data)

    for r in spec.get("redacted", []):
        data = git(args.src, "show", f"{commit}:{r['path']}")
        text = apply_line_edits(data.decode(), None, r.get("drop_lines", []), r.get("replace_lines", {}))
        write(r["path"], r["path"], "redacted", blob_id(args.src, commit, r["path"]), data, text.encode())

    for e in spec.get("excerpts", []):
        data = git(args.src, "show", f"{commit}:{e['path']}")
        text = apply_line_edits(data.decode(), e["keep_ranges"], [], e.get("replace_lines", {}))
        header = (f"<!-- Excerpt of {e['path']} at {spec['source_repo']} commit {commit}. {e['reason']} "
                  f"Kept line ranges of the original: {e['keep_ranges']}. Lines rewritten with a [redacted] "
                  f"marker: {sorted(int(k) for k in e.get('replace_lines', {}))}. -->\n")
        write(e["dest_name"], e["path"], "excerpt", blob_id(args.src, commit, e["path"]), data,
              (header + text).encode())

    manifest = HERE / "MANIFEST.tsv"
    with manifest.open("w") as f:
        f.write(f"# source_repo\t{spec['source_repo']}\n# source_commit\t{commit}\n")
        f.write("dest\tsource_path\tmode\tgit_blob\tsource_sha256\tdest_sha256\tbytes\n")
        for row in sorted(rows):
            f.write("\t".join(row) + "\n")
    print(f"imported {len(rows)} files into {dest_root}; manifest {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
