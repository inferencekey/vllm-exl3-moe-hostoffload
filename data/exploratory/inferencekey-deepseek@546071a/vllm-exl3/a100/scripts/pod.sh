#!/usr/bin/env bash
# Local helper: run a command on the pod.
exec ssh -i /home/ubuntu/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p ${POD_PORT:-10915} root@${POD_IP:-157.157.221.177} "$@" 2> >(grep -v setlocale >&2)
