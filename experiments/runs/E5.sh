#!/usr/bin/env bash
# E5: Engram int4 (served pack) vs FP8. FP8 needs a pack variant whose Engram shards come from the original
# checkpoint (PACK_FP8_ENGRAM, assembled with the recipe's quantize/assemble_pack.py); checked for host-memory fit first.
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E5-engram
do_run $E engram-int4 exl3-offload experiments/workloads/subset.yaml ENGRAM_INT4=1
if [[ -n ${PACK_FP8_ENGRAM:-} ]]; then
  do_run $E engram-fp8 exl3-offload experiments/workloads/subset.yaml ENGRAM_INT4=0 PACK=$PACK_FP8_ENGRAM
else
  mkdir -p "$ROOT/$E/engram-fp8"; echo "skipped: PACK_FP8_ENGRAM not set" > "$ROOT/$E/engram-fp8/SKIPPED.txt"
fi
