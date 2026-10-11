#!/usr/bin/env bash
# Fetch every third-party input of the stack into third_party/ (gitignored) at its pin and verify it.
# Nothing fetched here is committed to this repository.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/stack/pins.env"
TP=$ROOT/third_party
mkdir -p "$TP"
clone_at() {  # url dir commit
  local url=$1 dir=$2 commit=$3
  if [[ ! -d $dir/.git ]]; then git clone -q --filter=blob:none "$url" "$dir"; fi
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$commit"
  [[ $(git -C "$dir" rev-parse HEAD) == "$commit" ]] || { echo "pin mismatch in $dir" >&2; exit 1; }
  echo "$(basename "$dir") $(git -C "$dir" rev-parse HEAD)"
}
python3 "$ROOT/stack/fetch_recipe.py" --dest "$TP/diffbot-recipe"
clone_at "$SFXNZ_REPO" "$TP/sfxnz-recipe" "$SFXNZ_COMMIT"
clone_at "$VLLM_EXL3_REPO" "$TP/vllm-exl3" "$VLLM_EXL3_COMMIT"
clone_at "$EXLLAMAV3_REPO" "$TP/exllamav3" "$EXLLAMAV3_COMMIT"
# record what was fetched
{
  echo "fetched_utc=$(date -u +%FT%TZ)"
  for d in sfxnz-recipe vllm-exl3 exllamav3; do echo "$d=$(git -C "$TP/$d" rev-parse HEAD)"; done
  echo "diffbot-recipe=$(cat "$TP/diffbot-recipe/.fetched-from")"
  ( cd "$TP/diffbot-recipe" && find patches kernels plans serve docker -type f | sort | xargs sha256sum )
} > "$TP/FETCHED.txt"
echo "third_party ready: $TP/FETCHED.txt"
