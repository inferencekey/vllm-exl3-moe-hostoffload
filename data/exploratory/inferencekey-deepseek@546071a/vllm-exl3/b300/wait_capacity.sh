#!/usr/bin/env bash
# Retry the B300 create every ~45 s until one succeeds or MAX_SECONDS elapse. No pod runs while waiting.
D=$(cd "$(dirname "$0")" && pwd); LOG=${LOG:-$D/logs/wait_capacity.log}; MAX_SECONDS=${MAX_SECONDS:-14400}
mkdir -p "$(dirname "$LOG")"; t0=$(date +%s)
cd /home/ubuntu/me/inferencekey
while :; do
  for dc in EUR-IS-1 EU-NL-1 any; do
    out=$(scripts/secrets.sh run jenkins python3 "$D/rp_create_b300.py" "$dc" 2>&1 | grep -v "setlocale\|LC_ALL")
    echo "$(date -u +%FT%TZ) $dc $(echo "$out" | head -1)" >> "$LOG"
    if echo "$out" | grep -q "^CREATE 201"; then echo "$out" > "$D/logs/created.txt"; echo SUCCESS >> "$LOG"; exit 0; fi
    sleep 15
  done
  [ $(( $(date +%s) - t0 )) -gt $MAX_SECONDS ] && { echo "GAVE UP" >> "$LOG"; exit 2; }
  sleep 15
done
