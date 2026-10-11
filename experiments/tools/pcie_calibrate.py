#!/usr/bin/env python3
"""Calibrate NVML's PCIe RX counter against known host-to-device traffic on this pod (needs torch, a GPU, NVML).

Two traffic patterns, each for --seconds per GPU, while NVML is sampled every 50 ms:
  copy      cudaMemcpy H2D from a pinned buffer (DMA engine)
  zerocopy  a GPU kernel (torch sum) reading a pinned, cudaHostRegister'd buffer through its device alias, the way
            host-resident experts and Engram rows are read
The script reports bytes moved / time against the mean NVML RX rate, so that the PCIe numbers of the benchmark can be
converted with a measured factor rather than trusted blindly. Writes JSON.
"""
import argparse
import ctypes
import json
import threading
import time

import pynvml
import torch


def sampler(h, stop, out):
    while not stop.is_set():
        out.append((time.monotonic(), pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)))
        time.sleep(0.05)


def uva_view(t: torch.Tensor) -> torch.Tensor:
    """Device tensor aliasing pinned host memory (zero copy)."""
    cudart = ctypes.CDLL("libcudart.so")
    ptr = ctypes.c_void_p()
    rc = cudart.cudaHostGetDevicePointer(ctypes.byref(ptr), ctypes.c_void_p(t.data_ptr()), 0)
    if rc != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer rc={rc}")

    class Iface:
        __cuda_array_interface__ = {"shape": (t.numel(),), "typestr": "|u1", "data": (ptr.value, False), "version": 3}
    return torch.as_tensor(Iface(), device="cuda")


def run(dev: int, gib: float, seconds: float) -> dict:
    torch.cuda.set_device(dev)
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    n = int(gib * 2**30)
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    res = {}
    for mode in ("copy", "zerocopy"):
        view = uva_view(host) if mode == "zerocopy" else None
        samples, stop = [], threading.Event()
        th = threading.Thread(target=sampler, args=(h, stop, samples))
        torch.cuda.synchronize()
        th.start()
        t0, moved = time.monotonic(), 0
        while time.monotonic() - t0 < seconds:
            if mode == "copy":
                dst.copy_(host, non_blocking=True)
            else:
                view.view(torch.int32).sum()
            torch.cuda.synchronize()
            moved += n
        dt = time.monotonic() - t0
        stop.set()
        th.join()
        rx = [v for t, v in samples if t0 + 0.5 <= t <= t0 + dt - 0.2]
        res[mode] = {"bytes": moved, "seconds": dt, "true_GBps": moved / dt / 1e9,
                     "nvml_rx_GBps_mean": (sum(rx) / len(rx) * 1024 / 1e9) if rx else None, "n_samples": len(rx)}
        if rx:
            res[mode]["nvml_over_true"] = res[mode]["nvml_rx_GBps_mean"] / res[mode]["true_GBps"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gib", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    pynvml.nvmlInit()
    out = {"torch": torch.__version__, "gpus": {}}
    for d in range(torch.cuda.device_count()):
        out["gpus"][str(d)] = {"name": torch.cuda.get_device_name(d), **run(d, a.gib, a.seconds)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
