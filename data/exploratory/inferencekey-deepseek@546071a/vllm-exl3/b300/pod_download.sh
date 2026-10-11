#!/usr/bin/env bash
# Download the 3.30bpw EXL3 pack to the container disk (run on the pod, in tmux).
set -uo pipefail
. /root/.pod_env 2>/dev/null || true
REPO=${REPO:-vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw}; REV=${REV:-e831e9e4d6bfeafa6d630848296417b1393404a3}
DEST=${DEST:-/root/m/DSV4.1-Flash-SAGE-EXL3-3.30bpw}
mkdir -p "$DEST"; export HF_HOME=/root/m/.hf; export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1
python -m pip install -q -U huggingface_hub hf_transfer hf_xet
t0=$(date +%s)
for i in 1 2 3; do hf download "$REPO" --revision "$REV" --local-dir "$DEST" --max-workers 16 && break; echo "retry $i"; done
echo "download finished in $(( $(date +%s) - t0 )) s"; du -sh "$DEST"
python - "$DEST" <<'PY'
import json, os, sys
root=sys.argv[1]; idx=json.load(open(os.path.join(root,"model.safetensors.index.json")))["weight_map"]
sh=sorted(set(idx.values())); miss=[s for s in sh if not os.path.isfile(os.path.join(root,s))]
print("shards", len(sh), "missing", miss)
PY
echo DOWNLOAD_DONE
