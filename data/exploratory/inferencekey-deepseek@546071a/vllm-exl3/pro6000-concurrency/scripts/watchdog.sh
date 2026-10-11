#!/usr/bin/env bash
# Hard stop: delete the pod after LIMIT_S seconds whatever its state (money guard).
PID="$1"; LIMIT_S="${2:-6000}"
sleep "$LIMIT_S"
cd /home/ubuntu/me/inferencekey && scripts/secrets.sh run jenkins python3 /home/ubuntu/me/inferencekey-deepseek/runpod/rp_delete.py "$PID"
echo "watchdog deleted $PID at $(date)"
