#!/usr/bin/env bash
# E8: main controlled benchmark of the measured stack, then across-boot repetitions and one nsys capture.
# MINIMAL=1 drops the long-context scenario and the nsys capture.
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E8-main
if [[ ${MINIMAL:-0} == 1 ]]; then export ONLY=A-code-4k,A-code-4k-poisson,B-prefix-16k,P-prose-1k,prefill-1024,prefill-4096,prefill-16384,prefill-32768,prefill-65536,decode; fi
do_run $E main exl3-offload experiments/workloads/main.yaml
for b in 1 2 3; do  # cold boots: across-boot variance at key points
  ONLY=A-code-4k REPS=1 do_run $E reboot-$b exl3-offload experiments/workloads/subset.yaml
done
if [[ ${MINIMAL:-0} != 1 ]] && command -v nsys >/dev/null; then
  AFTER_HOOK='nsys_capture' ; nsys_capture() { nsys profile --gpu-metrics-devices=all --duration=60 -o "$1/nsys" \
    $PY -m hostoffload.harness.loadgen --workload "$ROOT/experiments/workloads/subset.yaml" --out "$1/nsys-load" \
    --base-url "$BASE_URL" --only A-code-4k --repetitions 1 > "$1/nsys.log" 2>&1; }
  do_run $E nsys exl3-offload none
fi
$PY "$ROOT/experiments/tools/pcie_calibrate.py" --out "$ROOT/$E/pcie-calibration.json" || true
