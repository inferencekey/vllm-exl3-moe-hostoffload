"""Feasibility facts that do not need a GPU, reproduced from data files, plus the failure classifier on a real log.

The GPU-side reproduction of the same infeasibility is experiment E2 (experiments/runs/E2.sh), which boots each
configuration and records the server's own error (hostoffload.analysis.failures)."""
import json
from pathlib import Path

from hostoffload.analysis import exploratory as X
from hostoffload.analysis import failures, serve_log
from hostoffload.weights.split import Placement, account, classify, load_plan, read_tensor_table

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data" / "static" / "weights"
GiB = 2 ** 30


def gpu_total_gib() -> float:
    """Usable memory per GPU as nvidia-smi reported it on the exploratory pod (97,887 MiB)."""
    s = next(s for s in X.load_sweep("A")["samples"]["1"] if s.get("gpus"))
    return s["gpus"][0][2] / 1024


def test_exl3_pack_does_not_fit_without_host_experts():
    """With no host plan, the pack's VRAM-resident file bytes alone exceed the memory of two 96 GB cards, before
    KV cache, CUDA graphs, activations and the runtime overhead are counted."""
    pl = json.loads((W / "diffbot-a1cf1bd2" / "placement.json").read_text())
    vram_per_gpu = pl["placements"]["no_plan"]["totals"]["vram"] / GiB / 2
    assert vram_per_gpu > gpu_total_gib()
    gen = pl["placements"]["plan-gen30-dh"]["totals"]["vram"] / GiB / 2
    assert gen < gpu_total_gib()


def test_original_checkpoint_does_not_fit_on_two_96gb_cards():
    po = json.loads((W / "deepseek-ai-2cba9e42" / "placement.json").read_text())
    must_be_gpu = sum(sum(v.values()) for k, v in po["placements"]["no_plan"]["by_class"].items()
                      if k not in ("engram_table", "embedding", "vision"))
    assert must_be_gpu / GiB / 2 > 1.4 * gpu_total_gib()


def test_static_split_matches_what_the_server_logged():
    """File-level accounting agrees with the exploratory server log where both measure the same thing."""
    sl = serve_log.parse_file(X.SWEEP / "logs" / "serve.log")
    pl = json.loads((W / "diffbot-a1cf1bd2" / "placement.json").read_text())
    engram_file_gib = sum(pl["placements"]["no_plan"]["by_class"]["engram_table"].values()) / GiB
    assert abs(sl["engram_pinned_gib_total"] - engram_file_gib) < 0.01  # 103.0 GiB both ways
    gen = pl["placements"]["plan-gen30-dh"]
    for layer, (kept, total) in sl["host_experts_by_rank"][0].items():
        assert gen["host_experts_per_layer"][str(layer)] == kept
        assert gen["experts_per_layer"][str(layer)] == total


def test_tensor_classification_covers_every_tensor():
    tensors = read_tensor_table(W / "diffbot-a1cf1bd2" / "tensors.csv.gz")
    assert len(tensors) == 190557
    assert all(classify(t.name)[0] != "other" for t in tensors)
    acc = account(tensors, Placement())
    assert acc["experts_total"] == 40 * 384 + 384  # 15,360 target experts + 3 x 128 drafter experts


def test_failure_classifier_on_the_real_capture_failure():
    log = X.SWEEP / "logs" / "serve_attempt1_capture384.log"
    r = failures.classify(log.read_text(errors="replace"))
    assert r["kind"] == "cuda_graph_capture"
    ok = failures.classify((X.SWEEP / "logs" / "serve.log").read_text(errors="replace"))
    assert ok["kind"] == "none"


def test_failure_classifier_patterns():
    assert failures.classify("x\ntorch.OutOfMemoryError: CUDA out of memory. Tried")["kind"] == "gpu_oom"
    assert failures.classify("FlashMLA is not supported on compute capability 8.0")["kind"] == "unsupported_arch"
    assert failures.classify("Killed process 123 (python3)")["kind"] == "host_oom"
