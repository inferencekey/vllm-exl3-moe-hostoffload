"""System sampler: GPU memory, PCIe throughput, host memory by kind, per-process memory, vLLM gauges.

Writes one JSON object per line (JSONL) every `--interval` seconds, with both wall and monotonic timestamps so that
samples can be aligned with the request log. Fixes against the exploratory sampler (5 s nvidia-smi + cgroup only):
- PCIe RX/TX per GPU from NVML (nvmlDeviceGetPcieThroughput, KB/s over a ~20 ms window), the quantity that
  zero-copy reads of host-resident experts and Engram rows show up in;
- per-process host memory from /proc/<pid>/status (RssAnon, RssFile, RssShmem, VmLck, VmPin) and the cgroup's
  memory.stat, so that pinned host buffers can be told apart from page cache;
- a one-off dump of the largest mappings (/proc/<pid>/smaps) of every vLLM process at start, to attribute
  shared-memory and anonymous regions to their allocations;
- sub-second interval (default 0.5 s).

NVML is optional: without it the sampler falls back to `nvidia-smi --query-gpu`, which has no PCIe throughput.
Cross-check PCIe with `nvidia-smi dmon -s t` (started by experiments/run_experiment.sh) and with
`scripts/pcie_calibrate.py`, which drives a known H2D volume and compares it with what NVML reports.

  python3 -m hostoffload.harness.sampler --out samples.jsonl --metrics-url http://127.0.0.1:8000/metrics
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import prom

CGROUP = Path("/sys/fs/cgroup")
PROC_STATUS_KEYS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin", "VmSwap")


def read_kv_file(path: Path) -> dict[str, int]:
    """Parse 'key value' lines (cgroup memory.stat) into ints."""
    out = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                out[parts[0]] = int(parts[1])
    except OSError:
        pass
    return out


def parse_proc_status(text: str) -> dict[str, int]:
    """Selected /proc/<pid>/status fields, in bytes."""
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in PROC_STATUS_KEYS:
            num = v.strip().split()
            if num and num[0].isdigit():
                out[k] = int(num[0]) * 1024
    return out


def parse_meminfo(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        num = v.strip().split()
        if num and num[0].isdigit():
            out[k] = int(num[0]) * (1024 if len(num) > 1 and num[1] == "kB" else 1)
    return out


def parse_smaps_largest(text: str, top: int = 40) -> list[dict]:
    """Largest mappings of a process from /proc/<pid>/smaps: range, size, Rss, Anonymous, Locked, path."""
    maps, cur = [], None
    head = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\d+\s*(.*)$")
    for line in text.splitlines():
        m = head.match(line)
        if m:
            cur = {"start": m.group(1), "end": m.group(2), "perms": m.group(3), "path": m.group(4).strip(),
                   "size": int(m.group(2), 16) - int(m.group(1), 16)}
            maps.append(cur)
            continue
        if cur is not None:
            k, _, v = line.partition(":")
            if k in ("Rss", "Anonymous", "Locked", "Shared_Clean", "Shared_Dirty", "Private_Dirty"):
                cur[k] = int(v.strip().split()[0]) * 1024
    maps.sort(key=lambda x: x.get("Rss", 0), reverse=True)
    return maps[:top]


def find_pids(pattern: str) -> list[int]:
    pids = []
    rx = re.compile(pattern)
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if rx.search(cmd) and int(d.name) != os.getpid():
            pids.append(int(d.name))
    return sorted(pids)


class Gpu:
    def __init__(self) -> None:
        self.nvml = None
        try:
            import pynvml  # nvidia-ml-py
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self.nvml = None

    def static_info(self) -> list[dict]:
        if not self.nvml:
            return []
        n, out = self.nvml, []
        for i, h in enumerate(self.handles):
            d = {"index": i, "name": n.nvmlDeviceGetName(h)}
            for key, fn in (("pcie_gen_max", n.nvmlDeviceGetMaxPcieLinkGeneration),
                            ("pcie_width_max", n.nvmlDeviceGetMaxPcieLinkWidth),
                            ("pcie_gen_cur", n.nvmlDeviceGetCurrPcieLinkGeneration),
                            ("pcie_width_cur", n.nvmlDeviceGetCurrPcieLinkWidth)):
                try:
                    d[key] = fn(h)
                except Exception:
                    d[key] = None
            try:
                d["memory_total"] = n.nvmlDeviceGetMemoryInfo(h).total
            except Exception:
                pass
            out.append(d)
        return out

    def sample(self) -> list[dict]:
        if not self.nvml:
            return self._smi()
        n, out = self.nvml, []
        for i, h in enumerate(self.handles):
            d: dict = {"i": i}
            try:
                m = n.nvmlDeviceGetMemoryInfo(h)
                d["mem_used"], d["mem_total"] = m.used, m.total
                u = n.nvmlDeviceGetUtilizationRates(h)
                d["util_gpu"], d["util_mem"] = u.gpu, u.memory
                d["pcie_rx_kBps"] = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_RX_BYTES)
                d["pcie_tx_kBps"] = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_TX_BYTES)
                d["power_mW"] = n.nvmlDeviceGetPowerUsage(h)
                d["sm_clock_MHz"] = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
                d["procs"] = [{"pid": p.pid, "used": p.usedGpuMemory}
                              for p in n.nvmlDeviceGetComputeRunningProcesses(h)]
            except Exception as e:
                d["error"] = repr(e)[:200]
            out.append(d)
        return out

    @staticmethod
    def _smi() -> list[dict]:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        except Exception:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            try:
                out.append({"i": int(f[0]), "mem_used": int(float(f[1])) * 2**20,
                            "mem_total": int(float(f[2])) * 2**20, "util_gpu": float(f[3]),
                            "power_mW": float(f[4]) * 1000})
            except (ValueError, IndexError):
                pass
        return out


def sample_once(gpu: Gpu, pids: list[int], metrics_url: str | None, headers: dict) -> dict:
    s: dict = {"t_wall": time.time(), "t_mono": time.monotonic(), "gpus": gpu.sample()}
    s["cgroup"] = {"memory.current": _read_int(CGROUP / "memory.current"),
                   "memory.max": _read_int(CGROUP / "memory.max"),
                   "stat": {k: v for k, v in read_kv_file(CGROUP / "memory.stat").items()
                            if k in ("anon", "file", "shmem", "file_mapped", "unevictable", "kernel", "sock",
                                     "pagetables", "active_file", "inactive_file")}}
    try:
        mi = parse_meminfo(Path("/proc/meminfo").read_text())
        s["meminfo"] = {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "Cached", "Shmem", "Mlocked",
                                                 "Unevictable", "AnonPages", "HugePages_Total")}
    except OSError:
        pass
    procs = {}
    for pid in pids:
        try:
            procs[str(pid)] = parse_proc_status(Path(f"/proc/{pid}/status").read_text())
        except OSError:
            continue
    s["procs"] = procs
    if metrics_url:
        try:
            req = urllib.request.Request(metrics_url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                snap = prom.snapshot(r.read().decode())
            s["vllm"] = {"gauges": snap["gauges"], "counters": snap["counters"]}
        except Exception as e:
            s["vllm"] = {"error": repr(e)[:200]}
    return s


def _read_int(p: Path) -> int | None:
    try:
        v = p.read_text().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="memory / PCIe / vLLM sampler")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--metrics-url", default=None)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"))
    ap.add_argument("--proc-pattern", default=r"vllm|VLLM|EngineCore|Worker")
    ap.add_argument("--smaps-at-start", action="store_true")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until SIGTERM/SIGINT")
    args = ap.parse_args(argv)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    gpu = Gpu()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(args.out, "a") as f:
        pids = find_pids(args.proc_pattern)
        f.write(json.dumps({"type": "header", "t_wall": time.time(), "nvml": gpu.nvml is not None,
                            "gpus": gpu.static_info(), "pids": pids, "interval": args.interval}) + "\n")
        if args.smaps_at_start:
            for pid in pids:
                try:
                    f.write(json.dumps({"type": "smaps", "pid": pid, "maps": parse_smaps_largest(
                        Path(f"/proc/{pid}/smaps").read_text())}) + "\n")
                except OSError:
                    pass
        n = 0
        while not stop["flag"] and (args.duration <= 0 or time.monotonic() - t0 < args.duration):
            if n % 20 == 0:
                pids = find_pids(args.proc_pattern)
            f.write(json.dumps({"type": "sample", **sample_once(gpu, pids, args.metrics_url, headers)}) + "\n")
            f.flush()
            n += 1
            time.sleep(max(0.0, args.interval - ((time.monotonic() - t0) % args.interval)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
