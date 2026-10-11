#!/usr/bin/env bash
# Launch vLLM + vllm-exl3 on one B300, TP1, Engram in pinned host RAM. MODE=eager|fast
. /root/.pod_env 2>/dev/null || true
MODEL=${MODEL:-/root/m/DSV4.1-Flash-SAGE-EXL3-3.30bpw}; MODE=${MODE:-eager}; MAXLEN=${MAXLEN:-65536}
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_V41_NATIVE_MOE=0 \
       VLLM_EXL3_TRELLIS_ARENA=1 VLLM_EXL3_ARENA_PRESCAN=1 VLLM_EXL3_MEM_WATERFALL=0
ARGS=(--quantization exl3 --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --reasoning-parser deepseek_v41 --tensor-parallel-size 1 --served-model-name deepseek-v41-exl3
  --host 0.0.0.0 --port 8000 --language-model-only --max-model-len "$MAXLEN" --max-num-seqs 4
  --max-num-batched-tokens ${MNBT:-4096} --gpu-memory-utilization ${GMU:-0.92} --block-size 128 --enable-prefix-caching
  --enable-prompt-tokens-details --engram-config "${ENGRAM_CFG:-{\"cpu_offload\":true}}"
  --hf-overrides '{"quantization_config":{"quant_method":"exl3","non_routed_quantization":{"quant_method":"deepseek_v4_fp8","activation_scheme":"dynamic","weight_block_size":[32,32]},"mtp_experts":"source","mtp_experts_start_layer":40}}')
if [ "$MODE" = eager ]; then ARGS+=(--enforce-eager); else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12]}'
         --speculative-config '{"method":"dspark","num_speculative_tokens":2,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'); fi
echo "vllm serve $MODEL ${ARGS[*]}"
exec vllm serve "$MODEL" "${ARGS[@]}"
