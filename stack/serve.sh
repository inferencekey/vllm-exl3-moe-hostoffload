#!/usr/bin/env bash
# Start `vllm serve` for one variant (stack/variants/<VARIANT>.env) with every setting logged.
#
#   VARIANT=exl3-offload API_KEY=... bash stack/serve.sh        # foreground; writes $RUN_DIR/serve.log
#
# Common settings (all variants, so that only the variant's own layer differs):
#   TP=2, --max-model-len 262144, fp8 KV cache of KV_MEM bytes (4.25 GiB, the recipe value), --max-num-seqs 64,
#   --max-num-batched-tokens 16384, --block-size 64 (128 for exl3-uva), CUDA graphs FULL_DECODE_ONLY with capture sizes up to 256,
#   prefix caching on (vLLM default), text only. Capture sizes stop at 256 because vllm-exl3's fat-expert path calls
#   fat.any().item() above VLLM_EXL3_FAT_THRESHOLD=256 tokens, which graph capture forbids
#   (audit/evidence/fat_any_item.txt); decode steps of more than about 42 sequences x (1 + 5 draft tokens) run eager.
# The exl3-offload values of EXL3_* and THOR_* are the defaults of Diffbot's recipe serve/serve.sh (MIT).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/stack/pins.env"
VARIANT=${VARIANT:-${HOB_VARIANT:?set VARIANT}}
source "$ROOT/stack/variants/$VARIANT.env"
: "${API_KEY:?set API_KEY}"
MODELS_DIR=${MODELS_DIR:-/root/models}
case $MODEL in
  exl3) PACK=${PACK:-$MODELS_DIR/exl3-3bpw-$MODEL_EXL3_REVISION} ;;
  orig) PACK=${PACK:-$MODELS_DIR/orig-$MODEL_ORIG_REVISION} ;;
esac
RUN_DIR=${RUN_DIR:-/root/run}
mkdir -p "$RUN_DIR"
PORT=${PORT:-8000}
SERVED=${SERVED:-deepseek-v4.1-flash}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_BATCHED=${MAX_BATCHED:-16384}
KV_MEM=${KV_MEM:-4563402752}
CAPTURE_SIZES=${CAPTURE_SIZES:-"1 2 4 6 8 12 16 24 32 48 64 96 128 192 256"}
SPEC=${SPEC:-dspark}
SPEC_K=${SPEC_K:-5}
EAGER=${EAGER:-0}
# 64 needs Diffbot's hooks (DIFFBOT_HOOKS=1); without them vLLM's DeepSeek-V4.1 sparse attention only accepts 128
BLOCK_SIZE=${BLOCK_SIZE:-64}

ulimit -l unlimited 2>/dev/null || echo "WARN: memlock limit $(ulimit -l) (pinned memory still works on RunPod)"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0a} FLASHINFER_CUDA_ARCH_LIST=${FI_ARCH:-12.0f}
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_SERVER_DEV_MODE=1   # dev mode: /reset_prefix_cache
export TILELANG_CACHE_DIR=/root/.cache/tilelang TRITON_CACHE_DIR=/root/.cache/triton
# variant isolation on a pod that has switched variants (stack/build/bootstrap_pod.sh)
if [[ -n ${PLUGINS:-} ]]; then
  export VLLM_PLUGINS=$PLUGINS
  [[ -f /opt/.hob-variant && $(cat /opt/.hob-variant) != "$VARIANT" && $VARIANT != exl3-uva ]] && \
    { echo "installed variant $(cat /opt/.hob-variant) != $VARIANT: run make pod-bootstrap VARIANT=$VARIANT" >&2; exit 2; }
  [[ $VARIANT == exl3-uva && -f /opt/.hob-variant && $(cat /opt/.hob-variant) != exl3-min && $(cat /opt/.hob-variant) != exl3-uva ]] && \
    { echo "exl3-uva needs the unpatched plugin: run make pod-bootstrap VARIANT=exl3-uva" >&2; exit 2; }
  [[ -d /opt/hob-site ]] && export PYTHONPATH=/opt/hob-site${PYTHONPATH:+:$PYTHONPATH}
else
  export VLLM_PLUGINS=none   # upstream: load no out-of-tree plugin
  [[ -d /opt/vllm-pristine ]] && export PYTHONPATH=/opt/vllm-pristine${PYTHONPATH:+:$PYTHONPATH}
fi
export DSV41_ENGRAM_DISK=0 DSV41_ENGRAM_INT4=${ENGRAM_INT4:-0}
args=(--tensor-parallel-size "$TP" --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY"
      --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype fp8 --kv-cache-memory "$KV_MEM"
      --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" --block-size "$BLOCK_SIZE"
      --disable-custom-all-reduce --language-model-only --enable-prompt-tokens-details
      --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
      --reasoning-parser deepseek_v41 --served-model-name "$SERVED" --trust-remote-code)
[[ -n ${QUANT:-} ]] && args+=(--quantization "$QUANT" --linear-backend marlin)
if [[ $EAGER == 1 ]]; then args+=(--enforce-eager)
else args+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes $CAPTURE_SIZES); fi
# THINKING=off makes thinking-off the server default (quality runs: lm-eval cannot pass chat_template_kwargs)
[[ ${THINKING:-default} == off ]] && args+=(--default-chat-template-kwargs '{"thinking":false}')
[[ $SPEC == dspark ]] && args+=(--speculative-config '{"method":"dspark","num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"}')

if [[ ${DIFFBOT_HOOKS:-0} == 1 ]]; then
  # recipe serve/serve.sh defaults (Diffbot, MIT)
  export XMOE_EXT=xmoe EXL3_DO=${EXL3_DO:-1} EXL3_FAT_SLICE=0 EXL3_LEAN_PREP=1 EXL3_HOST_OVERLAP_GROUP=16
  export EXL3_HOST_OVERLAP_SMS=$OVERLAP_SMS EXL3_HOST_OVERLAP_SMS_PREFILL=$OVERLAP_SMS_PREFILL
  [[ ${PLAN:-none} != none ]] && export EXL3_HOST_EXPERTS=$PLAN
  [[ -n ${EXPERT_STATS_DIR:-} ]] && export EXL3_EXPERT_STATS=$EXPERT_STATS_DIR
  export THOR_MHC_SPLITS=80 THOR_DENSE_GEMV=1 THOR_ROUTER_SPLITK=1 THOR_DENSE_ONE_COPY=1 THOR_DENSE_GEMV_MAX=64
  export THOR_DENSE_MBLOCK=64 THOR_DENSE_PY=/opt/recipe/kernels/thor_dense2.py
  export THOR_DENSE_TABLE=/opt/recipe/kernels/thor_dense_table_v2.json THOR_WOA_FP8=1 THOR_EMBED_HOST=1
  export THOR_INDEXER_BUF_MULT=8 THOR_INDEXER_EXPAND_CAP=1 THOR_ROPE_CAP=1 THOR_FP8_LMHEAD=1 DENSE_HYBRID_M=128
  export VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0
fi
# shellcheck disable=SC2206
extra=(${EXTRA_ARGS:-} ${SERVE_EXTRA:-})
{
  echo "variant=$VARIANT pack=$PACK"
  env | grep -E '^(EXL3_|THOR_|XMOE_|DSV41_|VLLM_|NCCL_|PYTORCH_|FLASHINFER_|TORCH_CUDA)' | sort
  printf 'arg: %s\n' "${args[@]}" "${extra[@]}"
} > "$RUN_DIR/serve_config.txt"
exec vllm serve "$PACK" "${args[@]}" "${extra[@]}" > >(tee -a "$RUN_DIR/serve.log") 2>&1
