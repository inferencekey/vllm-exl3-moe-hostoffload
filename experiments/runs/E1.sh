#!/usr/bin/env bash
# E1: build every variant with docker (on a CPU machine with docker) and record image digests and provenance.
source "$(dirname "$0")/../lib.sh"
OUT=$ROOT/data/controlled/E1-build/$(run_id); mkdir -p "$OUT"
bash "$ROOT/stack/fetch_stack.sh" > "$OUT/fetch.log" 2>&1 && cp "$ROOT/third_party/FETCHED.txt" "$OUT/"
for v in upstream exl3-min exl3-offload; do
  /usr/bin/time -v docker build -f "$ROOT/stack/build/Dockerfile" --build-arg VARIANT=$v -t hostoffload:$v "$ROOT" \
    > "$OUT/build-$v.log" 2>&1; echo "$v rc=$?" >> "$OUT/status.txt"
  docker image inspect hostoffload:$v --format '{{.Id}} {{.Size}}' >> "$OUT/images.txt" 2>&1
  docker run --rm --entrypoint cat hostoffload:$v /opt/hostoffload/build/build_provenance.json > "$OUT/provenance-$v.json" 2>/dev/null
done
echo '{"status": "complete", "server": {"model": "none"}, "kind": "build"}' > "$OUT/run.json"
