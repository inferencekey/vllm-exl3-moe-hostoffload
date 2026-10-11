#!/usr/bin/env bash
# E6: side-stream SM budget for host experts (decode and prefill budgets varied one at a time).
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E6-overlap-sms
for s in 16 32 64; do do_run $E dec$s-pre64 exl3-offload experiments/workloads/subset.yaml OVERLAP_SMS=$s OVERLAP_SMS_PREFILL=64; done
for s in 32 96; do do_run $E dec32-pre$s exl3-offload experiments/workloads/subset.yaml OVERLAP_SMS=32 OVERLAP_SMS_PREFILL=$s; done
