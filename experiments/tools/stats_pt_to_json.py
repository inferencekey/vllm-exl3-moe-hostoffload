#!/usr/bin/env python3
"""Convert the routing counters written by the patched plugin (EXL3_EXPERT_STATS=<dir>, torch .pt files of
{layer_key: tensor[2, n_experts]} with decode-sized and prefill rows) into the JSON read by hostoffload.weights.plan.
Runs on the pod (needs torch). Sums every .pt file in the directory (one per rank / dump)."""
import glob
import json
import os
import sys

import torch


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(src, "**", "*.pt"), recursive=True))
    if not files:
        print(f"no .pt files under {src}", file=sys.stderr)
        return 1
    acc: dict[str, dict[str, list[float]]] = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for k, v in d.items():
            v = v.double()
            dec, pre = (v[0], v[1]) if v.dim() == 2 else (v, torch.zeros_like(v))
            a = acc.setdefault(k, {"decode": [0.0] * dec.numel(), "prefill": [0.0] * pre.numel()})
            a["decode"] = [x + y for x, y in zip(a["decode"], dec.tolist())]
            a["prefill"] = [x + y for x, y in zip(a["prefill"], pre.tolist())]
    json.dump({"source_files": files, **acc}, open(out, "w"))
    print(f"{out}: {len(acc)} layers from {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
