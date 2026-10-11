#!/usr/bin/env bash
# E2: feasibility boots on 2x RTX PRO 6000. Each run records whether the server booted, the failure signature, and
# for those that boot, a three-request smoke test (text heads saved for degenerate-output checks).
source "$(dirname "$0")/../lib.sh"
E=data/controlled/E2-feasibility
boot_only $E upstream-orig upstream
boot_only $E upstream-uva-orig upstream-uva UVA_GIB=${UVA_GIB:-75}
boot_only $E exl3-min-allvram exl3-min
boot_only $E exl3-offload-noplan exl3-offload PLAN=none
for d in "$ROOT"/$E/*/*/; do
  echo "{\"status\": \"complete\", \"server\": {\"model\": \"feasibility\"}, \"boot\": \"$(head -1 "$d/status.txt")\"}" > "$d/run.json"
done
