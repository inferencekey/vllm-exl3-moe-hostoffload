#!/usr/bin/env bash
# Build ExLlamaV3 + vllm-exl3 inside the vllm/vllm-openai V4.1 image (run on the pod).
set -euo pipefail
. /root/.pod_env 2>/dev/null || true
VLLM_EXL3_REF=${VLLM_EXL3_REF:-814d4fe38082cddd838b45418c7d13a95395a36a}
EXLLAMAV3_REF=${EXLLAMAV3_REF:-be57335b087e4f001c5caae061544df3c06ba01e}
command -v python >/dev/null || ln -sf "$(command -v python3)" /usr/local/bin/python
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
export PATH=$(dirname "$NVCC"):$PATH
if [ -n "${ARCH:-}" ]; then export TORCH_CUDA_ARCH_LIST="$ARCH"; elif "$NVCC" --list-gpu-arch 2>/dev/null | grep -q compute_103; then export TORCH_CUDA_ARCH_LIST="10.3"; else export TORCH_CUDA_ARCH_LIST="10.0+PTX"; fi
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
echo "nvcc: $($NVCC --version | tail -1)  arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS"
python -c 'import vllm, torch, sys; print("vllm", vllm.__version__, "torch", torch.__version__, torch.version.cuda, sys.version)'
python -m pip install -q --upgrade "setuptools>=77,<81" wheel ninja
[ -d /opt/vllm-exl3 ] || git clone -q https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3
git -C /opt/vllm-exl3 checkout -q --detach "$VLLM_EXL3_REF"
[ -d /opt/exllamav3 ] || git clone -q https://github.com/turboderp-org/exllamav3.git /opt/exllamav3
git -C /opt/exllamav3 checkout -q --detach "$EXLLAMAV3_REF"
export EXL3_EXT_INCLUDE=/opt/exllamav3/exllamav3/exllamav3_ext
CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia /usr/local/cuda/include -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"
export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"
echo "cusparse: $CUSPARSE_DIR"
t0=$(date +%s)
W=/root/wheels; mkdir -p $W
ls $W/exllamav3-*.whl >/dev/null 2>&1 || python -m pip wheel --no-build-isolation --no-deps -w $W /opt/exllamav3 2>&1 | tail -3
python -m pip install --no-deps $W/exllamav3-*.whl 2>&1 | tail -1
echo "exllamav3 built in $(( $(date +%s) - t0 )) s"; t1=$(date +%s)
unset VLLM_EXL3_NO_CUDA
ls $W/vllm_exl3-*.whl >/dev/null 2>&1 || python -m pip wheel --no-build-isolation --no-deps -w $W /opt/vllm-exl3 2>&1 | tail -3
python -m pip install --no-deps $W/vllm_exl3-*.whl 2>&1 | tail -1
echo "vllm-exl3 built in $(( $(date +%s) - t1 )) s"
python - <<'PY'
import json, vllm, vllm_exl3, vllm_exl3_c
vllm_exl3.register()
print("vllm", vllm.__version__, "ABI", getattr(vllm_exl3_c, "P2B_MOE_ABI_VERSION", None))
print(json.dumps(vllm_exl3.runtime_diagnostics().get("mixed_k"), default=str))
p = vllm_exl3.plan_deepseek_v41(tensor_parallel_size=1, expert_parallel=False)
print("TP1 plan:", p.to_dict())
PY
python /opt/vllm-exl3/tools/check_vllm_compat.py "$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')" || echo "compat check returned nonzero"
echo INSTALL_DONE
