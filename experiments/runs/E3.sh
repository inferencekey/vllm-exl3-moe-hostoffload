#!/usr/bin/env bash
# E3: host-share sweep. 1) routing counts on the calibration workload (eager, EXL3_EXPERT_STATS); 2) plans at
# several shares from those counts; 3) the fixed subset for every share; 4) upstream UVA offload at equal host bytes.
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E3-host-share
P=$ROOT/$E/plans; mkdir -p "$P"
do_run $E profile exl3-offload experiments/workloads/calibration.yaml EAGER=1 EXPERT_STATS_DIR=/root/expert-stats \
  PLAN=/opt/recipe/plans/plan-gen30-dh.json
$PY "$ROOT/experiments/tools/stats_pt_to_json.py" /root/expert-stats "$P/counts-calibration.json"
for pct in ${SHARES:-25 30 40 50 60}; do
  $PY -m hostoffload.weights.plan from-counts --pct $pct --counts "$P/counts-calibration.json" \
      --keep-drafter-from "$ROOT/third_party/diffbot-recipe/plans/plan-gen30-dh.json" --out "$P/profile-$pct.json"
  do_run $E profile-$pct exl3-offload experiments/workloads/subset.yaml PLAN="$P/profile-$pct.json"
done
for pct in ${UVA_SHARES:-30 50}; do
  gib=$($PY -m hostoffload.weights.plan bytes --plan "$P/profile-$pct.json" | $PY -c 'import json,sys; print(round(json.load(sys.stdin)["per_gpu_gib"],1))')
  do_run $E uva-equal-$pct exl3-uva experiments/workloads/subset.yaml UVA_GIB=$gib
done
