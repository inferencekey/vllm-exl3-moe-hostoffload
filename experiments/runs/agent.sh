#!/usr/bin/env bash
# CODING-AGENT SESSION (E13): every regime of experiments/workloads/agent.yaml on ONE boot of the gen30 stack, on one
# 2x RTX PRO 6000 RunPod pod, resumable, with a budget check before each step. Start the watchdog first (hard cap
# 80 min = 5.57 USD at 4.18 USD/h):
#
#   cd ~/me/inferencekey && scripts/secrets.sh run main sh -c \
#     'cd ~/me/vllm-exl3-moe-hostoffload && exec bash experiments/pod/watchdog.sh <pod_id> 4800' &
#   # on the pod:
#   POD_CREATED_EPOCH=<unix time of the create response> bash experiments/runs/agent.sh
source "$(dirname "$0")/../lib.sh"
USD_PER_HOUR=${USD_PER_HOUR:-4.18}
SOFT_CAP_USD=${SOFT_CAP_USD:-5.36}       # 77 min: leaves room under the 80 min watchdog for copy-back and deletion
RESERVE_MIN=${RESERVE_MIN:-5}
: "${POD_CREATED_EPOCH:?set POD_CREATED_EPOCH (pod creation time, unix seconds)}"
M=$ROOT/data/controlled/LEAN/steps-session3; mkdir -p "$M"
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
E13=data/controlled/E13-coding-agent
R=/opt/recipe/plans

# S0 (14 min): stack fetch, build exl3-offload, EXL3 weights in parallel
s0() {
  # the vLLM image has no git; stack-fetch clones the pinned sources (2026-10-07: the third session failed here)
  command -v git >/dev/null || { DEBIAN_FRONTEND=noninteractive apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git; }
  make -C "$ROOT" stack-fetch
  ( make -C "$ROOT" download MODEL=exl3 > "$M/download-exl3.log" 2>&1 ) &
  VARIANT=exl3-offload bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-offload.log" 2>&1
  wait
}
# S1 (50 min): boot 4, the six regimes about 45 (estimates per regime in the workload file's comments)
s1() { do_session exl3-offload "$E13/agent=experiments/workloads/agent.yaml" -- PLAN=$R/plan-gen30-dh.json; }

step S0-setup 14 s0
step S1-E13-agent 50 s1
echo "coding-agent session finished; spent $(spent_usd) USD. Copy data/controlled/ back (sha256 both ends), then delete the pod."
