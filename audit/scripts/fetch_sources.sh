#!/usr/bin/env bash
# Fetch every third-party source this audit compares against, at pinned commits, OUTSIDE the repo.
#
#   AUDIT_SRC   cache dir (default ~/.cache/vllm-exl3-audit-src)
#
# Small downloads only: vLLM is a blobless clone with a sparse checkout of the paths the recipe patches touch,
# the Hugging Face recipe is fetched file by file from the tree API at a pinned revision (no weights).
# Prints the resolved SHA of every checkout at the end.
set -euo pipefail

AUDIT_SRC=${AUDIT_SRC:-$HOME/.cache/vllm-exl3-audit-src}
mkdir -p "$AUDIT_SRC"

VLLM_URL=https://github.com/vllm-project/vllm.git
VLLM_REF=36768d1bfd39094681cdbc8cb37d4b31c0729c89
VLLM_EXL3_URL=https://github.com/vcruz305/vllm-exl3.git
VLLM_EXL3_REF=d3cfd394920360d69f820d2dc96f8292a9e10283
EXLLAMAV3_URL=https://github.com/turboderp-org/exllamav3.git
EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e
SFXNZ_URL=https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark.git
SFXNZ_REF=460ac89c7df1eea4db8a731f1feb807f1584f359
# Mia's AI Lab work credited by vllm-exl3's NOTICE. No pin is published anywhere in the recipe, so the default branch is
# fetched with full history (the audit reads its license history) and the resolved SHA is printed.
MIA_URL=https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git
MIA_REF=${MIA_REF:-}

HF_REPO=diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
HF_REV=a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5

# Paths of vLLM needed by the five recipe patches, plus the DeepSeek-V4.1 model (Engram lives there), the speculative
# decoding code (DSpark), the V4.1 indexer backend the recipe hooks, and the license.
VLLM_SPARSE=(
  /LICENSE
  /vllm/models/deepseek_v41/
  /vllm/config/cache.py
  /vllm/parser/
  /vllm/model_executor/kernels/linear/mxfp8/
  /vllm/model_executor/layers/quantization/utils/mxfp8_utils.py
  /vllm/utils/multi_stream_utils.py
  /csrc/custom_all_reduce.cuh
  /vllm/config/speculative.py
  /vllm/v1/spec_decode/
  /vllm/v1/attention/backends/mla/indexer.py
)

log() { printf '[fetch] %s\n' "$*" >&2; }

# git_pinned <dir> <url> <ref> [sparse paths...]
git_pinned() {
  local dir=$1 url=$2 ref=$3; shift 3
  if [[ ! -d $dir/.git ]]; then
    log "clone $url"
    git clone -q --filter=blob:none --no-checkout "$url" "$dir"
  fi
  if (( $# > 0 )); then
    git -C "$dir" sparse-checkout init --no-cone >/dev/null
    printf '%s\n' "$@" | git -C "$dir" sparse-checkout set --no-cone --stdin
  fi
  if [[ -n $ref ]]; then
    git -C "$dir" cat-file -e "$ref^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin "$ref"
    git -C "$dir" -c advice.detachedHead=false checkout -q --force "$ref"
  else
    git -C "$dir" fetch -q origin
    git -C "$dir" -c advice.detachedHead=false checkout -q --force origin/HEAD
  fi
}

git_pinned "$AUDIT_SRC/vllm" "$VLLM_URL" "$VLLM_REF" "${VLLM_SPARSE[@]}"
git_pinned "$AUDIT_SRC/vllm-exl3" "$VLLM_EXL3_URL" "$VLLM_EXL3_REF"
git_pinned "$AUDIT_SRC/exllamav3" "$EXLLAMAV3_URL" "$EXLLAMAV3_REF"
git_pinned "$AUDIT_SRC/sfxnz-recipe" "$SFXNZ_URL" "$SFXNZ_REF"
git_pinned "$AUDIT_SRC/mia-glm53" "$MIA_URL" "$MIA_REF"

# Hugging Face: recipe/ plus the small top-level files (model card, config, LICENSE). Anything stored in LFS with a
# size above 10 MB (the safetensors shards) is skipped, so no weights are ever downloaded.
HF_DIR=$AUDIT_SRC/hf-diffbot-$HF_REV
mkdir -p "$HF_DIR"
curl -fsSL "https://huggingface.co/api/models/$HF_REPO/tree/$HF_REV?recursive=1" -o "$HF_DIR/.tree.json"
curl -fsSL "https://huggingface.co/api/models/$HF_REPO/revision/$HF_REV" -o "$HF_DIR/.revision.json"
python3 - "$HF_DIR" "$HF_REPO" "$HF_REV" <<'PY'
import json, os, sys, urllib.request
out, repo, rev = sys.argv[1:4]
tree = json.load(open(os.path.join(out, ".tree.json")))
n = 0
for e in tree:
    if e.get("type") != "file":
        continue
    p = e["path"]
    top = "/" not in p
    if not (p.startswith("recipe/") or (top and (p.endswith((".md", ".json", ".txt")) or p.startswith("LICENSE")))):
        continue
    if p.endswith(".safetensors") or e.get("size", 0) > 10_000_000:
        continue
    dst = os.path.join(out, p)
    if os.path.exists(dst) and os.path.getsize(dst) == e.get("size"):
        continue
    os.makedirs(os.path.dirname(dst) or out, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{p}"
    with urllib.request.urlopen(url) as r, open(dst + ".part", "wb") as f:
        f.write(r.read())
    os.replace(dst + ".part", dst)
    n += 1
print(f"[fetch] hf: {n} files downloaded into {out}", file=sys.stderr)
PY

# Model cards and repo metadata of the other Hugging Face repos the pack is built from (small JSON + README only).
for spec in "coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw@650cae2c13aaaec303871a35301503570889c0be" "deepseek-ai/DeepSeek-V4.1-Flash@main"; do
  repo=${spec%@*}; rev=${spec#*@}; d=$AUDIT_SRC/hf-meta/${repo//\//__}
  mkdir -p "$d"
  curl -fsSL "https://huggingface.co/api/models/$repo/revision/$rev" -o "$d/revision.json" || log "WARN: no metadata for $repo@$rev"
  curl -fsSL "https://huggingface.co/$repo/raw/$rev/README.md" -o "$d/README.md" || log "WARN: no README for $repo@$rev"
  curl -fsSL "https://huggingface.co/$repo/raw/$rev/LICENSE" -o "$d/LICENSE" || rm -f "$d/LICENSE"
done

echo "# resolved sources ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
for d in vllm vllm-exl3 exllamav3 sfxnz-recipe mia-glm53; do
  printf '%-14s %s %s\n' "$d" "$(git -C "$AUDIT_SRC/$d" rev-parse HEAD)" "$(git -C "$AUDIT_SRC/$d" remote get-url origin)"
done
printf '%-14s %s %s\n' hf-diffbot "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["sha"])' "$HF_DIR/.revision.json")" "https://huggingface.co/$HF_REPO"
