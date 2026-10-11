#!/usr/bin/env bash
# Shared functions for the experiment runners (experiments/runs/E*.sh). Runs ON THE POD, from a checkout of this
# repository, after `make pod-bootstrap VARIANT=...` and `make download MODEL=...`.
#
# One "run" = boot a server for a variant with given overrides, start the samplers, drive one workload, stop
# everything and collect logs into data/controlled/<exp>/<label>/<run_id>/.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python3}
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
: "${API_KEY:=hob-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
export API_KEY
PORT=${PORT:-8000}
BASE_URL=http://127.0.0.1:$PORT

run_id() { date -u +%Y%m%dT%H%M%SZ; }

fingerprint() {  # dir
  local d=$1
  {
    echo "{"
    echo "\"utc\": \"$(date -u +%FT%TZ)\","
    echo "\"repo_git\": \"$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)\","
    echo "\"repo_dirty\": $(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l),"
    echo "\"hostname\": \"$(hostname)\","
    echo "\"cpu\": \"$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')\","
    echo "\"nproc\": $(nproc),"
    echo "\"cgroup_memory_max\": \"$(cat /sys/fs/cgroup/memory.max 2>/dev/null)\","
    echo "\"driver\": \"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)\","
    echo "\"gpus\": \"$(nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader 2>/dev/null | tr '\n' ';')\","
    echo "\"pod_id\": \"${RUNPOD_POD_ID:-}\""
    echo "}"
  } > "$d/fingerprint.json"
  nvidia-smi -q > "$d/nvidia-smi-q.txt" 2>&1 || true
  nvidia-smi topo -m > "$d/nvidia-smi-topo.txt" 2>&1 || true
  cp /root/build-*/build_provenance.json "$d/" 2>/dev/null || true
}

wait_ready() {  # timeout_s
  local t0=$SECONDS
  until curl -sf -o /dev/null -H "Authorization: Bearer $API_KEY" "$BASE_URL/v1/models"; do
    sleep 5
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited during boot"; return 1; fi
    (( SECONDS - t0 > ${1:-3600} )) && { echo "boot timeout"; return 1; }
  done
  echo "ready after $(( SECONDS - t0 )) s"
}

start_server() {  # dir variant [VAR=value ...]
  local d=$1 variant=$2; shift 2
  mkdir -p "$d"
  ( export "$@" 2>/dev/null; VARIANT=$variant RUN_DIR=$d PORT=$PORT exec bash "$ROOT/stack/serve.sh" ) &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$d/server.pid"
  T_BOOT0=$SECONDS
}

stop_server() {
  [[ -n ${SERVER_PID:-} ]] || return 0
  pkill -TERM -P "$SERVER_PID" 2>/dev/null; kill -TERM "$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
  kill -KILL "$SERVER_PID" 2>/dev/null
  # never `pkill -f vllm` (it would also match unrelated shells); wait until the GPUs are free instead
  for _ in $(seq 60); do
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && break; sleep 2
  done
  SERVER_PID=
}

start_samplers() {  # dir
  local d=$1
  $PY -m hostoffload.harness.sampler --out "$d/samples.jsonl" --interval "${SAMPLE_INTERVAL:-0.5}" \
      --metrics-url "$BASE_URL/metrics" --smaps-at-start > "$d/sampler.log" 2>&1 &
  SAMPLER_PID=$!
  nvidia-smi dmon -s putm -d 1 -o T > "$d/dmon.txt" 2>&1 &
  DMON_PID=$!
}

stop_samplers() {
  kill -TERM "${SAMPLER_PID:-}" "${DMON_PID:-}" 2>/dev/null; wait "${SAMPLER_PID:-}" 2>/dev/null; SAMPLER_PID= DMON_PID=
}

# do_run <exp_data_dir> <label> <variant> <workload.yaml|none> [VAR=value ...]
# Boots, waits, samples, drives the workload, stops; writes a status file. Failures are recorded, not hidden.
do_run() {
  local exp=$1 label=$2 variant=$3 wl=$4; shift 4
  local d=$ROOT/$exp/$label/$(run_id)
  mkdir -p "$d"
  echo "== $label ($variant) -> $d"
  printf '%s\n' "$@" > "$d/overrides.txt"
  fingerprint "$d"
  start_server "$d" "$variant" "$@"
  if ! wait_ready "${BOOT_TIMEOUT:-3600}" > "$d/boot.txt" 2>&1; then
    echo "boot_failed" > "$d/status.txt"
    cat "$d/boot.txt"; stop_server; return 1
  fi
  echo "boot_s=$(( SECONDS - T_BOOT0 ))" >> "$d/boot.txt"
  start_samplers "$d"
  if [[ $wl != none ]]; then
    $PY -m hostoffload.harness.loadgen --workload "$ROOT/$wl" --out "$d" --base-url "$BASE_URL" \
        ${REPS:+--repetitions $REPS} ${ONLY:+--only $ONLY} > "$d/loadgen.log" 2>&1
    echo "loadgen_rc=$?" >> "$d/status.txt"
  fi
  [[ -n ${AFTER_HOOK:-} ]] && eval "$AFTER_HOOK" "$d"
  stop_samplers
  stop_server
  echo "done" >> "$d/status.txt"
}

# do_session <variant> "<exp_dir>/<label>=<workload.yaml>" [...] -- [VAR=value ...]
# One boot, several workloads in sequence, each written to its own <exp_dir>/<label>/<run_id>/ (saves a boot, about
# 4 minutes of pod time, per extra workload). Used by the lean plan.
do_session() {
  local variant=$1; shift
  local pairs=()
  while [[ $# -gt 0 && $1 != -- ]]; do pairs+=("$1"); shift; done
  [[ ${1:-} == -- ]] && shift
  local first=${pairs[0]%%=*}
  local sd=$ROOT/$first/$(run_id)
  mkdir -p "$sd"
  printf '%s\n' "$@" > "$sd/overrides.txt"
  fingerprint "$sd"
  start_server "$sd" "$variant" "$@"
  if ! wait_ready "${BOOT_TIMEOUT:-3600}" > "$sd/boot.txt" 2>&1; then
    echo "boot_failed" > "$sd/status.txt"; stop_server; return 1
  fi
  echo "boot_s=$(( SECONDS - T_BOOT0 ))" >> "$sd/boot.txt"
  local pr d
  for pr in "${pairs[@]}"; do
    d=$ROOT/${pr%%=*}/$(run_id); [[ ${pr%%=*} == "$first" ]] && d=$sd
    mkdir -p "$d"; [[ $d != "$sd" ]] && { cp "$sd"/{fingerprint.json,overrides.txt,boot.txt} "$d/"; echo "server_log=$sd/serve.log" > "$d/session.txt"; }
    start_samplers "$d"
    $PY -m hostoffload.harness.loadgen --workload "$ROOT/${pr#*=}" --out "$d" --base-url "$BASE_URL" \
        ${REPS:+--repetitions $REPS} > "$d/loadgen.log" 2>&1
    echo "loadgen_rc=$?" >> "$d/status.txt"
    stop_samplers
  done
  stop_server
  echo "done" >> "$sd/status.txt"
}

# boot_only <exp_data_dir> <label> <variant> [VAR=value ...]: feasibility check, keeps the failure signature
boot_only() {
  local exp=$1 label=$2 variant=$3; shift 3
  local d=$ROOT/$exp/$label/$(run_id)
  mkdir -p "$d"
  printf '%s\n' "$@" > "$d/overrides.txt"
  fingerprint "$d"
  start_server "$d" "$variant" "$@"
  if wait_ready "${BOOT_TIMEOUT:-2400}" > "$d/boot.txt" 2>&1; then
    echo "booted" > "$d/status.txt"
    $PY -m hostoffload.harness.loadgen --workload "$ROOT/experiments/workloads/smoke.yaml" --out "$d" \
        --base-url "$BASE_URL" > "$d/loadgen.log" 2>&1
  else
    echo "boot_failed" > "$d/status.txt"
  fi
  stop_server
  $PY -m hostoffload.analysis.failures "$d/serve.log" > "$d/failure_signature.json" 2>/dev/null || true
}
