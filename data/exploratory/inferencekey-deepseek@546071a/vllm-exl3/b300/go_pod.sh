#!/usr/bin/env bash
# From the dev VM: wait for the pod container (10 min cutoff -> delete), then push files and start pod_all.sh.
D=$(cd "$(dirname "$0")" && pwd); PID=$1; cd /home/ubuntu/me/inferencekey
for i in $(seq 1 40); do
  INFO=$(scripts/secrets.sh run jenkins python3 /home/ubuntu/me/inferencekey-deepseek/runpod/rp_get_full.py "$PID" 2>/dev/null)
  IP=$(echo "$INFO" | awk -F': ' '/^publicIp/{print $2}'); PORT=$(echo "$INFO" | grep -oP 'portMappings: \{"22": \K[0-9]+')
  if [ -n "$IP" ] && [ -n "$PORT" ] && timeout 15 ssh -i ~/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10 -p $PORT root@$IP true 2>/dev/null; then
    echo "$IP $PORT" > "$D/logs/ssh.txt"; echo "SSH UP $IP $PORT at $(date -u +%T)"
    SCP="scp -q -i $HOME/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P $PORT"
    ssh -i ~/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $PORT root@$IP 'mkdir -p /root/wheels'
    $SCP "$D"/wheels/*.whl root@$IP:/root/wheels/ && $SCP "$D"/pod_*.sh "$D"/bench.py root@$IP:/root/
    ssh -i ~/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $PORT root@$IP 'tmux new -d -s all "bash /root/pod_all.sh"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes; nproc; df -h /; ulimit -l'
    exit 0
  fi
  sleep 15
done
echo "NO CONTAINER after 10 min, deleting $PID"; scripts/secrets.sh run jenkins python3 /home/ubuntu/me/inferencekey-deepseek/runpod/rp_delete.py "$PID"; exit 1
