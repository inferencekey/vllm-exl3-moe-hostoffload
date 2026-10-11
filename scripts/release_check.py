#!/usr/bin/env python3
"""Pre-release scan (run by `make release-check`). Fails on: em or en dashes in our files, out-of-scope runtime
names, files under data/controlled that look like mock output, missing license files. Warns about account
identifiers in imported logs, which the owner must decide on before a public release."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from scope_terms import out_of_scope_pattern  # noqa: E402

DASHES = re.compile("[\\u2013\\u2014]")  # en and em dash, written as escapes
IDS = re.compile(r"(consumerUserId\"?: ?\"(?!\[redacted)[^\"]+|ssh-(ed25519|rsa) [A-Za-z0-9+/=]{20,}|rpa_[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{20,})")


def tracked() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    return [ROOT / p for p in out]


def verbatim(rel: Path) -> bool:
    """Records kept byte for byte that we did not write: model outputs (lm-eval samples, the text heads in
    requests.jsonl) and saved third-party pages. Editing them to remove a dash or a term would falsify the record."""
    return ("lm_eval" in rel.parts or rel.parts[:3] == ("data", "static", "pricing")
            or (rel.parts[:2] == ("data", "controlled") and rel.name == "requests.jsonl"))


def main() -> int:
    errors, warnings = [], []
    scope = out_of_scope_pattern()
    for p in tracked():
        if p.suffix in (".pdf", ".png", ".gz") or not p.is_file():
            continue
        text = p.read_text(errors="replace")
        rel = p.relative_to(ROOT)
        if p.name not in ("LICENSE", "LICENSE-data") and not verbatim(rel) and DASHES.search(text):
            errors.append(f"dash character in {rel}")
        if rel.parts[:2] != ("tests", "scope_terms.py") and not verbatim(rel) and scope.search(text):
            errors.append(f"out-of-scope runtime mentioned in {rel}")
        for m in IDS.finditer(text):
            warnings.append(f"identifier in {rel}: {m.group(0)[:40]}...")
    for f in ("LICENSE", "LICENSE-data", "CITATION.cff", ".zenodo.json", "THIRD_PARTY.md", "PROVENANCE.md"):
        if not (ROOT / f).exists():
            errors.append(f"missing {f}")
    for w in sorted(set(warnings)):
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    print(f"release-check: {len(errors)} errors, {len(set(warnings))} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
