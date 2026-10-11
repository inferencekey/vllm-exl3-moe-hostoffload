"""The exploratory import is hash-pinned: every file listed in MANIFEST.tsv must exist with its recorded sha256,
verbatim files must equal their source blob, and nothing outside the manifest may sit in the import folder."""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "exploratory"


def manifest_rows():
    rows = []
    for line in (EXP / "MANIFEST.tsv").read_text().splitlines():
        if line.startswith("#") or line.startswith("dest\t"):
            continue
        rows.append(dict(zip(["dest", "source_path", "mode", "git_blob", "source_sha256", "dest_sha256", "bytes"],
                             line.split("\t"))))
    return rows


def test_manifest_hashes_match():
    rows = manifest_rows()
    assert len(rows) > 50
    for r in rows:
        data = (EXP / r["dest"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r["dest_sha256"], r["dest"]
        assert len(data) == int(r["bytes"]), r["dest"]
        if r["mode"] == "verbatim":
            assert r["dest_sha256"] == r["source_sha256"], r["dest"]


def test_no_unlisted_files():
    listed = {r["dest"] for r in manifest_rows()}
    root = EXP / "inferencekey-deepseek@546071a"
    present = {str(p.relative_to(EXP)) for p in root.rglob("*") if p.is_file()}
    assert present == listed


def test_no_out_of_scope_runtime_mentions():
    # Names of the out-of-scope runtime and its file format, assembled from fragments so that this guard does not
    # itself mention them. The same list is used by tests/test_scope_guard.py for the whole repo.
    from scope_terms import out_of_scope_pattern
    pattern = out_of_scope_pattern()
    for p in (EXP / "inferencekey-deepseek@546071a").rglob("*"):
        if p.is_file():
            assert not pattern.search(p.read_text(errors="replace")), p
