# Pod tooling (not run)

- `create_pod.py`: dry run by default; creating a pod needs `--approved-budget-usd` and `HOB_SPEND_APPROVED=yes`.
- `watchdog.sh <pod> <seconds>`: deletes the pod when the approved budget is used up. Always start it.
- On the pod: clone this repository, `make stack-fetch pod-bootstrap VARIANT=<variant> download MODEL=<exl3|orig>`,
  then `make bench-<exp>`. Copy `data/controlled/<exp>/` back (check sha256 on both ends) before deleting the pod.

Lessons from the exploratory runs (`data/exploratory/.../README.md`): use an image that is already cached or that has
sshd; check the pod's cgroup RAM (`/sys/fs/cgroup/memory.max`, the 2x RTX PRO 6000 pod had 351 GiB), copy results one
file at a time, and never `pkill -f` a pattern that also matches the SSH command line.
