#!/usr/bin/env bash
# E10: quality of the measured stack with lm-evaluation-harness (pinned); see quality/README.md.
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E10-quality-exl3
AFTER_HOOK='quality_hook'; quality_hook() { TASKS=${TASKS:-} bash "$ROOT/quality/run_lm_eval.sh" "$1" > "$1/lm_eval.log" 2>&1; }
do_run $E exl3-offload-gen30 exl3-offload none THINKING=off
