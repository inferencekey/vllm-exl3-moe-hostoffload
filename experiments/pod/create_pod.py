#!/usr/bin/env python3
"""Create a RunPod GPU pod for one experiment. DRY RUN BY DEFAULT: it prints the request and exits.

Creating a pod spends money. It runs only with --approved-budget-usd <amount> AND the environment variable
HOB_SPEND_APPROVED=yes, both set by the owner after approving the budget in experiments/PLAN.md. The lean-plan pod of 2026-10-06 was
created with the same request body, sent through the RunPod REST API.

  python3 experiments/pod/create_pod.py --config 2x-rtx-pro-6000            # dry run
  RUNPOD_API_KEY=... HOB_SPEND_APPROVED=yes python3 experiments/pod/create_pod.py --config 2x-rtx-pro-6000 \
      --approved-budget-usd 25 --name hob-e8
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
API = "https://rest.runpod.io/v1/pods"
GPU_TYPE = {"2x-rtx-pro-6000": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 2),
            "4x-rtx-pro-6000": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 4),
            "1x-b300": ("NVIDIA B300 SXM6 AC", 1), "2x-b300": ("NVIDIA B300 SXM6 AC", 2),
            "1x-a100-sxm-80gb": ("NVIDIA A100-SXM4-80GB", 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(GPU_TYPE))
    ap.add_argument("--name", default="hob")
    ap.add_argument("--disk-gb", type=int, default=500)  # lean plan: 309 GiB EXL3 pack only (no original checkpoint)
    ap.add_argument("--datacenter", default="EUR-IS-1")
    ap.add_argument("--approved-budget-usd", type=float, default=None)
    a = ap.parse_args()
    pins = dict(l.split("=", 1) for l in (ROOT / "stack/pins.env").read_text().splitlines() if "=" in l and l[0] != "#")
    price = yaml.safe_load((ROOT / "experiments/pricing.yaml").read_text())["configs"][a.config]["usd_per_hour"]
    gpu, n = GPU_TYPE[a.config]
    body = {"name": a.name, "cloudType": "SECURE", "computeType": "GPU", "gpuTypeIds": [gpu], "gpuCount": n,
            "dataCenterIds": [a.datacenter], "imageName": pins["VLLM_IMAGE"].split("@")[0],
            "containerDiskInGb": a.disk_gb, "ports": ["22/tcp", "8000/http"],
            "env": {"PUBLIC_KEY": os.environ.get("HOB_SSH_PUBKEY", "<ssh public key>")},
            "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": ["apt-get update -qq && apt-get install -y -qq openssh-server && mkdir -p /root/.ssh "
                               "/run/sshd && echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && /usr/sbin/sshd "
                               "&& sleep infinity"]}
    print(json.dumps({"would_post": API, "body": body, "usd_per_hour": price}, indent=1))
    if a.approved_budget_usd is None or os.environ.get("HOB_SPEND_APPROVED") != "yes":
        print("DRY RUN: no pod created (needs --approved-budget-usd and HOB_SPEND_APPROVED=yes)")
        return 0
    hours = a.approved_budget_usd / price
    print(f"approved {a.approved_budget_usd} USD = {hours:.2f} h at {price} USD/h; start watchdog.sh with "
          f"{int(hours * 3600)} s right after creation")
    req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json",
        "User-Agent": "hostoffload/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        print(r.read().decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
