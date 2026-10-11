# Pod side: download the diffbot pack to the container disk (never a network volume).
import os, time
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1"); os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download
t=time.time()
p=snapshot_download("diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000", revision="a1cf1bd227a754aa7e2ca5c21652bfaf256ee8a5",
                    local_dir="/root/models/DeepSeek-V4.1-Flash-EXL3-3bpw", max_workers=32,
                    ignore_patterns=["*.pdf", "assets/*", "inference/examples/images/*"])
print(f"DOWNLOAD DONE {p} in {time.time()-t:.0f}s", flush=True)
