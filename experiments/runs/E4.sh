#!/usr/bin/env bash
# E4: placement policy at 30 percent (recipe plans, a seeded random plan, our profile plan from E3).
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E4-placement-policy
R=/opt/recipe/plans; P=$ROOT/$E/plans; mkdir -p "$P"
$PY -m hostoffload.weights.plan random --like "$ROOT/third_party/diffbot-recipe/plans/plan-gen30-dh.json" --out "$P/random30.json"
for name in gen30 agent30 blend30; do do_run $E $name exl3-offload experiments/workloads/subset.yaml PLAN=$R/plan-$name-dh.json; done
do_run $E random30 exl3-offload experiments/workloads/subset.yaml PLAN="$P/random30.json"
[[ -f $ROOT/data/controlled/E3-host-share/plans/profile-30.json ]] && \
  do_run $E profile-30 exl3-offload experiments/workloads/subset.yaml PLAN="$ROOT/data/controlled/E3-host-share/plans/profile-30.json"
