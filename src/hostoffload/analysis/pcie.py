"""PCIe receive traffic per generated token, per level of E8 (pod 1) and E13 (pod 3). Descriptive only.

Counter: `gpus[].pcie_rx_kBps` in samples.jsonl is NVML's nvmlDeviceGetPcieThroughput(NVML_PCIE_UTIL_RX_BYTES)
(hostoffload.harness.sampler), a rate in KB/s averaged by the driver over a short (~20 ms) window, sampled every
0.5 s. It is read here with 1 KB = 1000 bytes, the convention of hostoffload.analysis.agent, so the GB/s figures
match the case-study tables; experiments/tools/pcie_calibrate.py reads it as KiB (1024 bytes), 2.4 % higher.

Per level: mean over the samples inside the level window of the RX rate summed over both GPUs, times the window
length, divided by the completion tokens of the successful requests that ended inside the window. Repetitions are
pooled as sum of bytes over sum of tokens. Bytes include everything the GPUs receive (prefill of new prompt tokens,
host-resident expert and Engram reads by decode and by the drafter, input transfers), so the ratio is a description
of the traffic that accompanies each output token, not the cost of producing one.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import runs as RN
from . import stats

ROOT = Path(__file__).resolve().parents[3]
CTL = ROOT / "data" / "controlled"
CALIB = CTL / "E8-main" / "pcie-calibration.json"
KB = 1000.0
# PCIe 5.0 x16: 32 GT/s x 16 lanes with 128b/130b coding, per direction
PCIE5_X16_GBPS = 32e9 * 16 * 128 / 130 / 8 / 1e9

# (experiment, short, scenario in the run, label)
REGIMES = [
    ("E8", "A", "A-code-4k", "E8 A: 4K independent"),
    ("E8", "B", "B-prefix-16k", "E8 B: 16K cached prefix"),
    ("E13", "A", "A-short-4k", "E13 short: 4K independent"),
    ("E13", "B128k", "B-cached-128k", "E13 shared 128K prefix"),
    ("E13", "B256k", "B-cached-256k", "E13 shared 256K prefix"),
    ("E13", "C128k", "C-own-128k", "E13 own 128K context"),
    ("E13", "C256k", "C-own-256k", "E13 own 256K context"),
    ("E13", "D", "D-agent-turns", "E13 agent turns"),
]
RUN_DIRS = {"E8": CTL / "E8-main" / "main", "E13": CTL / "E13-coding-agent" / "agent"}


def rx_bytes_per_s(sample: dict) -> float:
    """Receive rate summed over the GPUs of one sample, bytes/s."""
    return sum(g.get("pcie_rx_kBps", 0) for g in sample["gpus"]) * KB


def mb_per_token(mean_rx_bytes_s: float, seconds: float, tokens: int) -> float | None:
    """Megabytes received per output token: mean rate x window length / tokens (None without tokens)."""
    if tokens <= 0 or seconds <= 0:
        return None
    return mean_rx_bytes_s * seconds / tokens / 1e6


def level_traffic(lv: RN.Level, samples: list[dict]) -> dict | None:
    win = [s for s in samples if lv.t0 <= s["t_wall"] <= lv.t1]
    if not win:
        return None
    rate = stats.mean([rx_bytes_per_s(s) for s in win])
    toks = sum(int(r.get("completion_tokens") or 0) for r in lv.requests
               if r.get("error") is None and (e := RN.end_wall(r)) is not None and lv.t0 <= e <= lv.t1)
    return {"rate": rate, "seconds": lv.duration, "bytes": rate * lv.duration, "tokens": toks,
            "n_samples": len(win)}


def per_level(run: Path, scenario: str) -> dict[int, dict]:
    """N -> pooled traffic over repetitions: GB/s (mean over repetitions) and MB per output token."""
    smp = RN.samples(run)
    by_n: dict[int, list[dict]] = {}
    for lv in RN.levels(run):
        if lv.scenario == scenario:
            t = level_traffic(lv, smp)
            if t:
                by_n.setdefault(lv.N, []).append(t)
    out = {}
    for n, ts in sorted(by_n.items()):
        b, tok, sec = sum(t["bytes"] for t in ts), sum(t["tokens"] for t in ts), sum(t["seconds"] for t in ts)
        out[n] = {"gbs": b / sec / 1e9, "mb_per_tok": mb_per_token(b / sec, sec, tok), "tokens": tok,
                  "reps": len(ts), "n_samples": sum(t["n_samples"] for t in ts)}
    return out


def selected_ns(ns: list[int]) -> list[int]:
    """Rows shown in the compact tables: N = 1, N = 8 and the largest tested N (when tested)."""
    return sorted({n for n in ns if n in (1, 8)} | {max(ns)})


def calibration(reg) -> dict:
    c = json.loads(CALIB.read_text())
    out = {}
    for gpu, d in c["gpus"].items():
        for mode in ("copy", "zerocopy"):
            m = d[mode]
            out[(gpu, mode)] = m
            reg.put(f"ctl.pcie.calib.gpu{gpu}.{mode}.true_gbs", m["true_GBps"], [CALIB], "controlled", "{:.1f}")
            reg.put(f"ctl.pcie.calib.gpu{gpu}.{mode}.nvml_gbs", m["nvml_rx_GBps_mean"], [CALIB], "controlled",
                    "{:.1f}")
            reg.put(f"ctl.pcie.calib.gpu{gpu}.{mode}.ratio", m["nvml_over_true"], [CALIB], "controlled", "{:.2f}")
    reg.put("ctl.pcie.calib.link_gbs", PCIE5_X16_GBPS, [CALIB], "controlled", "{:.0f}",
            note="PCIe 5.0 x16 per-direction line rate after 128b/130b coding (32 GT/s x 16 lanes), not measured")
    return out


def analyze(reg, write_tex, table, latex_escape) -> None:
    rows, srcs = [], []
    for exp, short, scen, label in REGIMES:
        runs = RN.complete_runs(RUN_DIRS[exp])
        if not runs:
            continue
        run = runs[-1]
        src = [run / "samples.jsonl", run / "levels.jsonl", run / "requests.jsonl"]
        srcs += src
        d = per_level(run, scen)
        for n, v in d.items():
            k = f"ctl.pcie.{exp}.{short}.N{n}"
            reg.put(f"{k}.gbs", v["gbs"], src, "controlled", "{:.1f}")
            if v["mb_per_tok"] is not None:
                reg.put(f"{k}.mb_per_tok", v["mb_per_tok"], src, "controlled", "{:.0f}")
        for n in selected_ns(list(d)):
            v = d[n]
            rows.append([latex_escape(label) if n == min(d) else "", str(n), f"{v['gbs']:.1f}",
                         f"{v['mb_per_tok']:.0f}" if v["mb_per_tok"] is not None else ""])
    if not rows:
        return
    cal = calibration(reg)
    g0c, g0z = cal[("0", "copy")]["nvml_over_true"], cal[("0", "zerocopy")]["nvml_over_true"]
    g1z = cal[("1", "zerocopy")]["true_GBps"]
    write_tex("pcie_per_token.tex", table(
        "PCIe receive traffic summed over both GPUs (NVML, raw counter) per level, and megabytes received per output "
        "token completed in the level window. E8 ran on pod 1 (two repetitions for B and for A at $N=1$, pooled), "
        "E13 on pod 3 (one repetition). Rows: $N=1$, $N=8$ and the largest tested $N$; every level is in "
        "\\texttt{values.tex} (\\texttt{ctl.pcie.*}). Descriptive: the bytes include prefill and drafter traffic.",
        "tab:pcie-per-token", ["regime", "$N$", "PCIe RX GB/s", "MB per output token"], rows, "lrrr",
        note=("Calibration on pod 1 (\\texttt{data/controlled/E8-main/pcie-calibration.json}): NVML over true bytes "
              f"{g0c:.2f} for DMA copies and {g0z:.2f} for zero-copy kernel reads on GPU 0. The GPU 1 zero-copy test "
              f"is invalid: its nominal {g1z:.0f} GB/s exceeds the {PCIE5_X16_GBPS:.0f} GB/s PCIe 5.0 x16 line rate "
              "and NVML saw almost no traffic, so the kernel did not read over PCIe. Values are not corrected.")))
    reg.artifact("pcie_per_token.tex", "controlled", sorted(set(srcs)) + [CALIB], "data")
