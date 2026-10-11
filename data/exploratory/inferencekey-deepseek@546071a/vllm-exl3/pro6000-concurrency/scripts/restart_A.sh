#!/usr/bin/env bash
# Pod side: used once in this run, to restart scenario A and B with the recalibrated loadgen (server already warm).
export API_KEY=$(cat /root/.apikey)
cd /root
mv /root/logs/scenario_A.log /root/logs/scenario_A_aborted_old_calibration.log 2>/dev/null
mv /root/results/scenario_A.json /root/results/scenario_A_aborted_old_calibration.json 2>/dev/null
echo "WARMUP DONE (restart $(date +%T))" >> /root/logs/run_all.log
A_OFFSET=20000000 python3 loadgen.py A "1 2 4 8 16 32 64" > /root/logs/scenario_A.log 2>&1
echo "A DONE $(date +%T)" >> /root/logs/run_all.log
python3 loadgen.py B "1 2 4 8 16 32 64" > /root/logs/scenario_B.log 2>&1
echo "B DONE $(date +%T)" >> /root/logs/run_all.log
nvidia-smi > /root/logs/nvidia-smi-end.txt
echo ALL DONE >> /root/logs/run_all.log
