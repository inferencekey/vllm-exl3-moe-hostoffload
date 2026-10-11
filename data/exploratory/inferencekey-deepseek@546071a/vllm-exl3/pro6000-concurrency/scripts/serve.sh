#!/usr/bin/env bash
# Pod side: diffbot recipe/serve/serve.sh turned into a direct `vllm serve` (no docker). Same env knobs and args, TP2.
set -u
: "${API_KEY:?set API_KEY}"
PACK=${PACK:-/root/models/DeepSeek-V4.1-Flash-EXL3-3bpw}
PORT=${PORT:-8000}; SERVED=${SERVED:-deepseek-v4.1-flash}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}; MAX_BATCHED=${MAX_BATCHED:-16384}
KV_MEM=${KV_MEM:-4563402752}; UTIL=${UTIL:-0.92}; EAGER=${EAGER:-0}; SPEC=${SPEC:-dspark}; SPEC_K=${SPEC_K:-5}
VISION=${VISION:-0}; PLAN=${PLAN:-plan-agent30-dh.json}; FI_ARCH=${FI_ARCH:-12.0f}
ulimit -l unlimited 2>/dev/null || echo "WARN: cannot raise memlock ($(ulimit -l))"
export EXL3_DO=1 XMOE_EXT=xmoe
export EXL3_HOST_EXPERTS=/opt/recipe/plans/$PLAN EXL3_HOST_OVERLAP_SMS=32 EXL3_HOST_OVERLAP_SMS_PREFILL=64 EXL3_HOST_OVERLAP_GROUP=16
export EXL3_FAT_SLICE=0 EXL3_LEAN_PREP=1 THOR_MHC_SPLITS=80 THOR_DENSE_GEMV=1 THOR_ROUTER_SPLITK=1 THOR_DENSE_ONE_COPY=1
export THOR_DENSE_GEMV_MAX=64 THOR_DENSE_MBLOCK=64 THOR_DENSE_PY=/opt/recipe/kernels/thor_dense2.py
export THOR_DENSE_TABLE=/opt/recipe/kernels/thor_dense_table_v2.json THOR_WOA_FP8=1 THOR_EMBED_HOST=1 THOR_INDEXER_BUF_MULT=8
export THOR_INDEXER_EXPAND_CAP=1 THOR_ROPE_CAP=1 THOR_FP8_LMHEAD=1 DENSE_HYBRID_M=128
export TORCH_CUDA_ARCH_LIST=12.0a FLASHINFER_CUDA_ARCH_LIST=$FI_ARCH FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PLUGINS=vllm_exl3 DSV41_ENGRAM_DISK=0 DSV41_ENGRAM_INT4=1 VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0
export NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=WARN TILELANG_CACHE_DIR=/root/.cache/tilelang TRITON_CACHE_DIR=/root/.cache/triton
extra=()
CAPTURE_SIZES=${CAPTURE_SIZES:-"1 2 4 6 8 12 16 24 32 48"}
[[ $EAGER == 1 ]] && extra+=(--enforce-eager) || extra+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes $CAPTURE_SIZES)
[[ $SPEC == dspark ]] && extra+=(--speculative-config '{"method":"dspark","num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"}')
[[ $VISION == 1 ]] || extra+=(--language-model-only)
exec vllm serve "$PACK" \
  --tensor-parallel-size 2 --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype fp8 --kv-cache-memory "$KV_MEM" \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}' \
  --block-size 64 --quantization exl3 --disable-custom-all-reduce --linear-backend marlin \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --enable-prompt-tokens-details \
  --served-model-name "$SERVED" --trust-remote-code "${extra[@]}" ${EXTRA_ARGS:-}
