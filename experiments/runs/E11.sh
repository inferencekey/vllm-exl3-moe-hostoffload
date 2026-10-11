#!/usr/bin/env bash
# E11: quality of the reference (upstream vLLM, original checkpoint, 2x B300), identical prompts, seeds and sampling.
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E11-quality-reference
AFTER_HOOK='quality_hook'; quality_hook() { TASKS=${TASKS:-} bash "$ROOT/quality/run_lm_eval.sh" "$1" > "$1/lm_eval.log" 2>&1; }
do_run $E upstream-orig upstream none THINKING=off TORCH_CUDA_ARCH_LIST=10.0a FI_ARCH=10.0a
