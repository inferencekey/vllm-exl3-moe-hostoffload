#!/usr/bin/env bash
# Build one variant of the serving stack in place, inside the pinned vLLM image (a RunPod pod started from that
# image, or `docker build` via stack/build/Dockerfile). No GPU is needed for the build itself.
#
#   VARIANT=upstream       nothing installed on top of the image (also: upstream-uva)
#   VARIANT=exl3-min       + the five vLLM patches of the recipe (sm_120 correctness and int4 Engram need them)
#                          + vllm-exl3 d3cfd394 UNPATCHED + ExLlamaV3 5be88657, sfxnz runtime hooks only
#   VARIANT=exl3-uva       same install as exl3-min (the difference is serve-time: upstream UVA offload flags)
#   VARIANT=exl3-offload   + Diffbot's exl3-thor.patch on vllm-exl3, the xmoe / xmoe_do kernels and Diffbot's
#                          sitecustomize.py (THOR_* hooks)
#
# The sequence follows Diffbot's recipe docker/build.sh and kernels/build.sh (MIT, see THIRD_PARTY.md); the steps
# are re-expressed here so that each variant stops at its own layer. Inputs come from third_party/ (stack/fetch_stack.sh)
# and are verified against their pins before use. Writes $OUT/build_provenance.json.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/stack/pins.env"
VARIANT=${VARIANT:?set VARIANT=upstream|exl3-min|exl3-uva|exl3-offload}
OUT=${OUT:-/root/build-$VARIANT}
TP=$ROOT/third_party
VP=${VP:-$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')}
export MAX_JOBS=${MAX_JOBS:-32}
mkdir -p "$OUT"
log() { echo "[bootstrap $(date -u +%T)] $*" | tee -a "$OUT/build.log"; }
t0=$(date +%s)

have=$(python3 -c 'import vllm; print(vllm.__version__)')
[[ $have == "$VLLM_VERSION" ]] || { log "vLLM $have is not the pinned $VLLM_VERSION: wrong base image"; exit 1; }
[[ -f $TP/FETCHED.txt ]] || { log "third_party missing: run stack/fetch_stack.sh first"; exit 1; }
python3 "$ROOT/stack/fetch_recipe.py" --dest "$TP/diffbot-recipe" | tee -a "$OUT/build.log"   # re-verify hashes
R=$TP/diffbot-recipe

# Idempotent, so that one pod can switch variants (the lean plan does upstream -> exl3-min -> exl3-offload -> exl3-uva):
# - an untouched copy of the image's vLLM is kept in /opt/vllm-pristine before any patch; upstream variants run from it
#   (stack/serve.sh puts it first on PYTHONPATH and disables plugins);
# - vLLM patches already applied are detected with a reverse dry run and skipped;
# - ExLlamaV3 and the xmoe kernels are built once (marker files), vllm-exl3 is reinstalled on every switch;
# - runtime hooks live in /opt/hob-site/sitecustomize.py, which only exl3 variants put on PYTHONPATH.
SITE_DIR=/opt/hob-site
if [[ ! -d /opt/vllm-pristine/vllm ]]; then
  log "pristine copy of the image's vLLM"
  mkdir -p /opt/vllm-pristine && cp -a "$VP/vllm" /opt/vllm-pristine/
fi
if [[ $VARIANT == upstream || $VARIANT == upstream-uva ]]; then
  log "upstream variant: nothing to install (serve.sh runs /opt/vllm-pristine)"
else
  if [[ ! -f /opt/.hob-deps ]]; then
    log "apt and pip build deps"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git patch libcusparse-dev-13-0 libcusolver-dev-13-0
    python3 -m pip install -q ninja
    touch /opt/.hob-deps
  fi
  # git refuses sources copied with another owner ("dubious ownership"); git exists only after the deps step above
  git config --global --add safe.directory '*'
  # The patches overlap (later ones touch files the first ones changed), so a per-patch reverse dry run cannot tell
  # whether the set was applied; a marker written after the whole set applied cleanly does.
  if [[ -f /opt/.hob-vllm-patched ]]; then log "vLLM patches already applied (marker)"
  else
    log "vLLM patches (recipe order)"
    ( cd "$VP" && for p in pr58132-decoder-swa-replay sm120-thor-fixes pr59585-dsml-finite-args \
          pr59894-decoder-replay-cut-trim pr58560-compressor-null-block; do
        patch -p1 --forward < "$R/patches/vllm/$p.diff" || exit 1; done ) && touch /opt/.hob-vllm-patched
  fi
  find "$VP/vllm" -name __pycache__ -path "*deepseek_v41*" -prune -exec rm -rf {} +
  python3 -c "import vllm.models.deepseek_v41.decoder_replay_layers, vllm.models.deepseek_v41.attention, vllm.parser.deepseek_v4"
  log "sfxnz helpers"
  rm -rf /opt/dsv41-patch && cp -r "$TP/sfxnz-recipe/docker/patch" /opt/dsv41-patch
  log "vllm-exl3 $VLLM_EXL3_COMMIT ($([[ $VARIANT == exl3-offload ]] && echo 'with exl3-thor.patch' || echo unpatched))"
  rm -rf /opt/vllm-exl3 && cp -a "$TP/vllm-exl3" /opt/vllm-exl3
  [[ $(git -C /opt/vllm-exl3 rev-parse HEAD) == "$VLLM_EXL3_COMMIT" && -z $(git -C /opt/vllm-exl3 status --porcelain) ]] || { log "vllm-exl3 not clean at pin"; exit 1; }
  if [[ $VARIANT == exl3-offload ]]; then
    ( cd /opt/vllm-exl3 && patch -p1 < "$R/patches/vllm-exl3/exl3-thor.patch" )
  fi
  python3 /opt/dsv41-patch/fix_vllm_exl3_setup.py /opt/vllm-exl3/setup.py
  VLLM_EXL3_NO_CUDA=1 python3 -m pip install -q --force-reinstall --no-build-isolation --no-deps /opt/vllm-exl3
  if [[ ! -f /opt/.hob-exllamav3-$EXLLAMAV3_COMMIT ]]; then
    log "ExLlamaV3 $EXLLAMAV3_COMMIT (sm_120a)"
    rm -rf /opt/exllamav3 && cp -a "$TP/exllamav3" /opt/exllamav3
    [[ $(git -C /opt/exllamav3 rev-parse HEAD) == "$EXLLAMAV3_COMMIT" && -z $(git -C /opt/exllamav3 status --porcelain) ]] || { log "exllamav3 not clean at pin"; exit 1; }
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0a} python3 -m pip install -q --no-build-isolation --no-deps /opt/exllamav3
    touch /opt/.hob-exllamav3-$EXLLAMAV3_COMMIT
  fi
  python3 -c "import exllamav3, exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe')"
  mkdir -p "$SITE_DIR"
  if [[ $VARIANT == exl3-offload ]]; then
    if [[ ! -f /opt/.hob-xmoe ]]; then
      log "xmoe and xmoe_do kernels"
      rm -rf /opt/recipe && cp -r "$R" /opt/recipe
      ( cd /opt/recipe/kernels && TORCH_CUDA_ARCH_LIST=12.0a python3 build_xmoe.py xmoe xmoe && TORCH_CUDA_ARCH_LIST=12.0a python3 build_xmoe_do.py )
      touch /opt/.hob-xmoe
    fi
    cp "$R/patches/vllm/sitecustomize.py" "$SITE_DIR/sitecustomize.py"
  else
    cp "$TP/sfxnz-recipe/docker/patch/sitecustomize.py" "$SITE_DIR/sitecustomize.py"
  fi
fi
echo "$VARIANT" > /opt/.hob-variant

log "provenance"
python3 - "$OUT" "$VARIANT" "$VP" <<'PY'
import hashlib, json, os, subprocess, sys, time
out, variant, vp = sys.argv[1:4]
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:
        return repr(e)
files = {}
for rel in ("vllm/models/deepseek_v41/nvidia/engram.py", "vllm/models/deepseek_v41/attention.py",
            "vllm/parser/deepseek_v4.py", "vllm/config/cache.py"):
    p = os.path.join(vp, rel)
    if os.path.exists(p):
        files[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
for p in ("/opt/vllm-exl3/src/vllm_exl3/exl3.py", "/opt/hob-site/sitecustomize.py"):
    if os.path.exists(p):
        files[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"variant": variant, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "vllm": sh("python3 -c 'import vllm; print(vllm.__version__)'"),
           "torch": sh("python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'"),
           "vllm_exl3_git": sh("git -C /opt/vllm-exl3 rev-parse HEAD 2>/dev/null"),
           "vllm_exl3_dirty": sh("git -C /opt/vllm-exl3 status --porcelain 2>/dev/null | wc -l"),
           "exllamav3_git": sh("git -C /opt/exllamav3 rev-parse HEAD 2>/dev/null"),
           "patched_file_sha256": files, "pip_freeze": sh("python3 -m pip freeze").splitlines(),
           "nvcc": sh("nvcc --version | tail -1"), "uname": sh("uname -a")},
          open(os.path.join(out, "build_provenance.json"), "w"), indent=1)
PY
log "done in $(( $(date +%s) - t0 )) s"
