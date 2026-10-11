#!/usr/bin/env bash
# Hard budget stop: delete the pod after <seconds> no matter what. Start it right after creating a pod.
#   RUNPOD_API_KEY=... bash experiments/pod/watchdog.sh <pod_id> <seconds> &
set -u
POD=${1:?pod id}; SECS=${2:?seconds}
sleep "$SECS"
curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" -H "User-Agent: hostoffload/0.1" \
  "https://rest.runpod.io/v1/pods/$POD" -w '\nDELETE %{http_code}\n'
