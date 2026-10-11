#!/usr/bin/env bash
# LEAN PLAN (recommended): every step on ONE 2x RTX PRO 6000 RunPod pod session of at most two hours, in priority order,
# resumable, with a budget check before each step. Never run without an approved budget; start the watchdog first.
# RUNPOD_API_KEY lives in the SOPS profile "main" of the inferencekey repo; inject it, never print it:
#
#   cd ~/me/inferencekey && scripts/secrets.sh run main sh -c \
#     'cd ~/me/vllm-exl3-moe-hostoffload && exec bash experiments/pod/watchdog.sh <pod_id> 7200' &   # hard cap: 2 h
#   # on the pod:
#   POD_CREATED_EPOCH=<unix time of the create response> bash experiments/runs/lean.sh
#
# Each step writes a marker in data/controlled/LEAN/steps/; re-running the script skips finished steps. A step is
# started only if (pod time so far + its estimate + the copy-back reserve) stays under SOFT_CAP_USD. Estimates and
# their basis are in experiments/experiments.yaml (lean:), which also generates the table in experiments/PLAN.md.
source "$(dirname "$0")/../lib.sh"
USD_PER_HOUR=${USD_PER_HOUR:-4.18}
SOFT_CAP_USD=${SOFT_CAP_USD:-8.15}       # 117 min: leaves room under the 2 h watchdog for copy-back and deletion
RESERVE_MIN=${RESERVE_MIN:-5}
: "${POD_CREATED_EPOCH:?set POD_CREATED_EPOCH (pod creation time, unix seconds)}"
M=$ROOT/data/controlled/LEAN/steps; mkdir -p "$M"
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
E2=data/controlled/E2-feasibility E3=data/controlled/E3-host-share E8=data/controlled/E8-main
E10=data/controlled/E10-quality-exl3 P=$ROOT/data/controlled/E3-host-share/plans
R=/opt/recipe/plans

# L0 (17 min): stack fetch, build exl3-min (also saves the pristine vLLM), EXL3 weights in parallel
l0() {
  # the vLLM image has no git; stack-fetch clones the pinned sources (2026-10-07: the third session failed here)
  command -v git >/dev/null || { DEBIAN_FRONTEND=noninteractive apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git; }
  make -C "$ROOT" stack-fetch
  ( make -C "$ROOT" download MODEL=exl3 > "$M/download-exl3.log" 2>&1 ) &
  VARIANT=exl3-min bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-min.log" 2>&1
  wait
}
# L1 E2 (8 min cap): both boots are expected to fail fast; the original checkpoint is not downloaded
l1() {
  BOOT_TIMEOUT=180 boot_only $E2 exl3-min-allvram exl3-min
  VARIANT=exl3-offload bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-offload.log" 2>&1
  BOOT_TIMEOUT=180 boot_only $E2 exl3-offload-noplan exl3-offload PLAN=none
  for d in "$ROOT"/$E2/*/*/; do
    echo "{\"status\": \"complete\", \"server\": {\"model\": \"feasibility\"}, \"boot\": \"$(head -1 "$d/status.txt")\"}" > "$d/run.json"
  done
}
# L2 E8 (38 min): gen30, A and B x 2 repetitions, probes, then the E3 gen30 point on the same boot
l2() {
  do_session exl3-offload "$E8/main=experiments/workloads/lean_main.yaml" "$E8/probes=experiments/workloads/lean_probes.yaml" \
    "$E3/gen30=experiments/workloads/lean_subset.yaml" -- PLAN=$R/plan-gen30-dh.json
  $PY "$ROOT/experiments/tools/pcie_calibrate.py" --out "$ROOT/$E8/pcie-calibration.json" --seconds 3 || true
}
# L3 E10 (20 min, 15 of them capped for lm-eval): fixed-seed subsets GSM8K 150, IFEval 150, HumanEval 164. Runs before
# the UVA step so that it reuses the exl3-offload build.
l3() {
  AFTER_HOOK='quality_hook'
  quality_hook() {  # through the proxy, so that gen_prefix tasks continue the assistant message (as for E11-api)
    $PY "$ROOT/quality/api_proxy.py" --backend vllm --usage "$1/api_usage.jsonl" > "$1/api_proxy.log" 2>&1 &
    local proxy=$!; sleep 3
    CHAT_URL=http://127.0.0.1:8100/v1/chat/completions SAMPLES=$ROOT/quality/subsets_lean2h_seed20261006.json \
      TIME_CAP_S=900 bash "$ROOT/quality/run_lm_eval.sh" "$1" > "$1/lm_eval.log" 2>&1
    kill $proxy 2>/dev/null
  }
  do_run $E10 exl3-offload-gen30-lean exl3-offload none PLAN=$R/plan-gen30-dh.json THINKING=off
  AFTER_HOOK=
}
# L4 E3 (10 min): upstream UVA offload of the EXL3 experts at the gen30 host bytes
l4() {
  local gib; gib=$($PY -m hostoffload.weights.plan bytes --plan "$ROOT/third_party/diffbot-recipe/plans/plan-gen30-dh.json" \
      | $PY -c 'import json,sys; print(round(json.load(sys.stdin)["per_gpu_gib"], 1))')
  VARIANT=exl3-uva bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-uva.log" 2>&1
  BOOT_TIMEOUT=420 do_session exl3-uva "$E3/uva-equal-gen30=experiments/workloads/lean_subset.yaml" -- UVA_GIB=$gib \
    || BOOT_TIMEOUT=420 boot_only $E3 uva-equal-gen30-failed exl3-uva UVA_GIB=$gib
}
# S1 (10 min, stretch): a higher host share (gen30 union agent30, 43.5% of target experts)
s1() {
  mkdir -p "$P"
  $PY -m hostoffload.weights.plan union --plan "$ROOT/third_party/diffbot-recipe/plans/plan-gen30-dh.json" \
      --plan "$ROOT/third_party/diffbot-recipe/plans/plan-agent30-dh.json" --out "$P/union-gen30-agent30.json"
  VARIANT=exl3-offload bash "$ROOT/stack/build/bootstrap_pod.sh" > "$M/bootstrap-exl3-offload-2.log" 2>&1
  do_session exl3-offload "$E3/union-gen30-agent30=experiments/workloads/lean_subset.yaml" -- PLAN="$P/union-gen30-agent30.json"
}

step L0-setup 17 l0
step L1-E2-feasibility 8 l1
step L2-E8-main 38 l2
step L3-E10-quality 20 l3
step L4-E3-uva 10 l4
step S1-union 10 s1
echo "lean plan finished; spent $(spent_usd) USD. Copy data/controlled/ back (sha256 both ends), then delete the pod."
