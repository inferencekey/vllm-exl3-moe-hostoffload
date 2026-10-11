#!/usr/bin/env bash
# E9: upstream vLLM, original checkpoint, all in VRAM on 2x B300 (no offload, no quantization beyond DeepSeek's own).
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E9-reference-b300
if [[ ${MINIMAL:-0} == 1 ]]; then export ONLY=A-code-4k,P-prose-1k; WL=experiments/workloads/subset.yaml; else WL=experiments/workloads/main.yaml; fi
do_run $E upstream-orig upstream $WL TORCH_CUDA_ARCH_LIST=10.0a FI_ARCH=10.0a
