#!/usr/bin/env bash
# LONG-CONTEXT SESSION (2026-10-07): E12 (128K and 256K tokens) and the E3 UVA rung with the block size fixed, on ONE
# 2x RTX PRO 6000 RunPod pod, steps in priority order, resumable, with a budget check before each step. Start the
# watchdog first (hard cap 72 min = 5.02 USD at 4.18 USD/h):
#
#   cd ~/me/inferencekey && scripts/secrets.sh run main sh -c \
#     'cd ~/me/vllm-exl3-moe-hostoffload && exec bash experiments/pod/watchdog.sh <pod_id> 4320' &
#   # on the pod:
#   POD_CREATED_EPOCH=<unix time of the create response> bash experiments/runs/long.sh
#
# Labels differ from the 2026-10-06 lean session (gen30-session2, uva-equal-gen30-bs128) so that runs from two pods
# are never pooled as repetitions of one configuration.
source "$(dirname "$0")/../lib.sh"
USD_PER_HOUR=${USD_PER_HOUR:-4.18}
SOFT_CAP_USD=${SOFT_CAP_USD:-4.81}       # 69 min: leaves room under the 72 min watchdog for copy-back and deletion
RESERVE_MIN=${RESERVE_MIN:-5}
: "${POD_CREATED_EPOCH:?set POD_CREATED_EPOCH (pod creation time, unix seconds)}"
M=$ROOT/data/controlled/LEAN/steps-session2; mkdir -p "$M"
spent_usd() { python3 -c "print(round(($(date +%s) - $POD_CREATED_EPOCH) / 3600 * $USD_PER_HOUR, 3))"; }
fits() {  # est_minutes
  python3 -c "import sys; s=($(date +%s)-$POD_CREATED_EPOCH)/60; sys.exit(0 if (s+$1+$RESERVE_MIN)/60*$USD_PER_HOUR <= $SOFT_CAP_USD else 1)"
}
step() {  # id est_minutes function
  local id=$1 est=$2 fn=$3
  if [[ -f $M/$id.done ]]; then echo "== $id already done"; return 0; fi
  if ! fits "$est"; then echo "== $id SKIPPED (budget: spent $(spent_usd) USD, needs ~$est min)" | tee "$M/$id.skipped"; return 0; fi
  echo "== $id start (spent $(spent_usd) USD, estimate $est min)"
  local t0=$SECONDS
  "$fn"; local rc=$?
  echo "rc=$rc minutes=$(( (SECONDS - t0) / 60 )) spent_usd=$(spent_usd)" > "$M/$id.done"
}
E3=data/controlled/E3-host-share E12=data/controlled/E12-long-context
R=/opt/recipe/plans

# S0 (17 min): stack fetch, build exl3-offload, EXL3 weights in parallel
s0() {
  # the vLLM image has no git; stack-fetch clones the pinned sources (2026-10-07: the third session failed here)
  command -v git >/dev/null || { DEBIAN_FRONTEND=noninteractive apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git; }
  make -C "$ROOT" stack-fetch
  ( make -C "$ROOT" download MODEL=exl3 > "$M/download-exl3.log" 2>&1 ) &
  VARIANT=exl3-offload bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-offload.log" 2>&1
  wait
}
# S1 (25 min): E12 long context, then the E3 gen30 point again on the same boot (the UVA rung's same-pod baseline)
s1() {
  do_session exl3-offload "$E12/long=experiments/workloads/long_context.yaml" \
    "$E3/gen30-session2=experiments/workloads/lean_subset.yaml" -- PLAN=$R/plan-gen30-dh.json
}
# S2 (12 min): upstream UVA offload of the EXL3 experts at the gen30 host bytes, KV block size 128
s2() {
  local gib; gib=$($PY -m hostoffload.weights.plan bytes --plan "$ROOT/third_party/diffbot-recipe/plans/plan-gen30-dh.json" \
      | $PY -c 'import json,sys; print(round(json.load(sys.stdin)["per_gpu_gib"], 1))')
  VARIANT=exl3-uva bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-uva.log" 2>&1
  BOOT_TIMEOUT=420 do_session exl3-uva "$E3/uva-equal-gen30-bs128=experiments/workloads/lean_subset.yaml" -- UVA_GIB=$gib \
    || BOOT_TIMEOUT=420 boot_only $E3 uva-equal-gen30-bs128-failed exl3-uva UVA_GIB=$gib
  for d in "$ROOT"/$E3/uva-equal-gen30-bs128-failed/*/; do
    [[ -d $d && ! -f $d/run.json ]] && echo "{\"status\": \"complete\", \"server\": {\"model\": \"feasibility\", \"variant\": \"exl3-uva\"}, \"boot\": \"$(head -1 "$d/status.txt")\"}" > "$d/run.json"
  done
}

step S0-setup 17 s0
step S1-E12-long 25 s1
step S2-E3-uva 12 s2
echo "long-context session finished; spent $(spent_usd) USD. Copy data/controlled/ back (sha256 both ends), then delete the pod."
