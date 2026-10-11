#!/usr/bin/env bash
# Build a release tarball in dist/ from the committed tree plus the built paper. It does NOT push, tag, publish to
# GitHub or deposit on Zenodo: those steps are listed in RELEASE_CHECKLIST.md and need the owner's approval.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
[[ -z $(git status --porcelain) ]] || { echo "working tree not clean"; exit 1; }
VER=$(grep -E '^version:' CITATION.cff | awk '{print $2}' | tr -d '"')
REV=$(git rev-parse --short HEAD)
mkdir -p dist
NAME=vllm-exl3-moe-hostoffload-$VER
git archive --format=tar --prefix="$NAME/" HEAD > "dist/$NAME.tar"
tar --append -f "dist/$NAME.tar" --transform "s,^paper/build/main.pdf,$NAME/paper/main.pdf," paper/build/main.pdf
gzip -n -f "dist/$NAME.tar"
sha256sum "dist/$NAME.tar.gz" > "dist/$NAME.tar.gz.sha256"
echo "built dist/$NAME.tar.gz from $REV (not published)"
