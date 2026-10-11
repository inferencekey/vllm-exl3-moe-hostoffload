#!/usr/bin/env bash
# Pod side: wait for the server, warm it up, then run scenario A and B. Logs to /root/logs, JSON to /root/results.
set -u
export API_KEY
cd /root
until curl -sf -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
echo "SERVER READY $(date +%T)"
# warmup: compile remaining kernels at batch sizes up to 64, on a corpus region the measured runs never use
OUT_DIR=/root/results/warmup A_OFFSET=${WARM_OFFSET:-12000000} MAX_TOKENS=32 MIN_REQ=1 REQ_MULT=1 STOP_TTFT_P95_S=100000 \
  python3 loadgen.py A "1 8 64" > /root/logs/warmup.log 2>&1
echo "WARMUP DONE $(date +%T)"
A_OFFSET=${A_OFFSET:-20000000} python3 loadgen.py A "${LEVELS:-1 2 4 8 16 32 64}" > /root/logs/scenario_A.log 2>&1
echo "A DONE $(date +%T)"
python3 loadgen.py B "${LEVELS:-1 2 4 8 16 32 64}" > /root/logs/scenario_B.log 2>&1
echo "B DONE $(date +%T)"
nvidia-smi > /root/logs/nvidia-smi-end.txt
echo ALL DONE
