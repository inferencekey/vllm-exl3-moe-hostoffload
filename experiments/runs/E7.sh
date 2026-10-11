#!/usr/bin/env bash
# E7: DSpark speculative decoding on vs off, main workload without the long-context scenario.
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E7-dspark
export ONLY=A-code-4k,B-prefix-16k,P-prose-1k,decode
do_run $E dspark exl3-offload experiments/workloads/main.yaml SPEC=dspark
do_run $E nospec exl3-offload experiments/workloads/main.yaml SPEC=none
