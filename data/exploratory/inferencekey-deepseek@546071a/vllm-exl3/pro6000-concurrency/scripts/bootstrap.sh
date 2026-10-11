#!/usr/bin/env bash
# Pod side: replay diffbot recipe/docker/build.sh + recipe/kernels/build.sh without docker, inside the
# vllm/vllm-openai:nightly-36768d1 pod. Expects /opt/recipe (the pack's recipe/ folder) and /opt/dsv41-patch (sfxnz docker/patch).
set -euxo pipefail
t0=$(date +%s)
VP=/usr/local/lib/python3.12/dist-packages
export MAX_JOBS=${MAX_JOBS:-48}
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git patch libcusparse-dev-13-0 libcusolver-dev-13-0
python3 -m pip install -q ninja
cd $VP
for p in pr58132-decoder-swa-replay sm120-thor-fixes pr59585-dsml-finite-args pr59894-decoder-replay-cut-trim pr58560-compressor-null-block; do
  patch -p1 --forward < /opt/recipe/patches/vllm/$p.diff
done
find $VP/vllm -name __pycache__ -path "*deepseek_v41*" -prune -exec rm -rf {} +
python3 -c "import vllm.models.deepseek_v41.decoder_replay_layers, vllm.models.deepseek_v41.attention, vllm.models.deepseek_v41.nvidia.flashinfer_sparse, vllm.parser.deepseek_v4; print('vllm patches ok')"
git clone -q https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 && git -C /opt/vllm-exl3 checkout -q d3cfd394920360d69f820d2dc96f8292a9e10283
cd /opt/vllm-exl3 && patch -p1 < /opt/recipe/patches/vllm-exl3/exl3-thor.patch
python3 /opt/dsv41-patch/fix_vllm_exl3_setup.py /opt/vllm-exl3/setup.py
VLLM_EXL3_NO_CUDA=1 python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3
python3 -c "import vllm_exl3; print('vllm_exl3 ok')"
git clone -q https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 && git -C /opt/exllamav3 checkout -q 5be886578ec80324c2c715269387be2058724b6e
TORCH_CUDA_ARCH_LIST=12.0a python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3
python3 -c "import torch, exllamav3, exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'); print('exllamav3 ok')"
cd /opt/recipe/kernels
TORCH_CUDA_ARCH_LIST=12.0a python3 build_xmoe.py xmoe xmoe
TORCH_CUDA_ARCH_LIST=12.0a python3 build_xmoe_do.py
cp /opt/recipe/patches/vllm/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
echo "BOOTSTRAP DONE in $(( $(date +%s)-t0 ))s"
