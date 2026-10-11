"""Cross-pod comparison on an identical workload (E3 gen30 subset on pod 1 and pod 2) and the hardware fingerprint
of the three pod sessions.

The E3 subset (scenarios A-code-4k and decode at N = 1, 8, 32, one repetition) ran with the same plan, stack and
workload file on pod 1 (data/controlled/E3-host-share/gen30, on the E8 server boot) and on pod 2
(gen30-session2, on the E12 server boot). The ratio is pod 2 over pod 1. One repetition per pod: the ratio mixes the
host difference with run-to-run and boot-to-boot variance, which these data cannot separate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import runs as RN
from .metrics import level_summary
from .registry import rel

ROOT = Path(__file__).resolve().parents[3]
CTL = ROOT / "data" / "controlled"
POD1, POD2 = CTL / "E3-host-share" / "gen30", CTL / "E3-host-share" / "gen30-session2"
CROSS = [("A-code-4k", "A", "A: 4K prompt"), ("decode", "decode", "decode")]
CROSS_NS = (1, 8, 32)
SLOW_TTFT_S = 10.0  # a request this slow to start dominates a short level's windowed aggregate
# (session key, label, run folder, LEAN session file)
SESSIONS = [("s1", "1 (pod 1)", CTL / "E8-main" / "main", CTL / "LEAN" / "session.json"),
            ("s2", "2 (pod 2)", CTL / "E12-long-context" / "long", CTL / "LEAN" / "session2.json"),
            ("s3", "3 (pod 3)", CTL / "E13-coding-agent" / "agent", CTL / "LEAN" / "session3.json")]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def level_rates(run: Path, scenario: str) -> dict[int, dict]:
    """N -> windowed aggregate output rate and median TPOT (s) of one run's levels of `scenario`."""
    out = {}
    for lv in RN.levels(run):
        if lv.scenario == scenario:
            s = level_summary(lv.requests)
            out[lv.N] = {"agg": s.get("aggregate_out_tok_s_window"), "tpot_p50": (s.get("tpot_s") or {}).get("p50"),
                         "n": s["n"], "errors": s.get("errors", 0), "ttft_max": (s.get("ttft_s") or {}).get("max")}
    return out


def ratio(a: float | None, b: float | None) -> float | None:
    """b / a, or None when either is missing or a is zero."""
    if a is None or b is None or a == 0:
        return None
    return b / a


def parse_topo(text: str) -> dict:
    """GPU rows of `nvidia-smi topo -m`: link between GPU0 and GPU1, CPU and NUMA affinity per GPU."""
    lines = [_ANSI.sub("", x) for x in text.splitlines()]
    head = next(x for x in lines if "CPU Affinity" in x)
    cols = [c.strip() for c in head.split("\t")]
    gpus = {}
    for x in lines:
        f = x.split("\t")
        if re.fullmatch(r"GPU\d+", f[0].strip()):
            gpus[f[0].strip()] = {c: (f[i].strip() if i < len(f) else "") for i, c in enumerate(cols) if c}
    link = gpus.get("GPU0", {}).get("GPU1")
    return {"link": link, "numa": [g.get("NUMA Affinity") for g in gpus.values()],
            "cpu_affinity": [g.get("CPU Affinity") for g in gpus.values()], "n_gpus": len(gpus)}


def parse_smi_q(text: str) -> dict:
    """Driver, and per GPU bus id and maximum PCIe generation and link width from `nvidia-smi -q`."""
    drv = re.search(r"Driver Version\s*:\s*(\S+)", text)
    gpus, cur, section = [], None, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Product Name"):
            cur = {}
            gpus.append(cur)
        elif cur is not None and s.startswith("Bus Id"):
            cur.setdefault("bus", s.split(":", 1)[1].strip())
        elif s in ("PCIe Generation", "Link Width"):
            section = s
        elif cur is not None and section and s.startswith("Max") and ":" in s:
            cur["gen_max" if section == "PCIe Generation" else "width_max"] = s.split(":", 1)[1].strip()
            section = None
    return {"driver": drv.group(1) if drv else None, "gpus": gpus}


def analyze(reg, write_tex, table, latex_escape) -> None:
    r1, r2 = RN.complete_runs(POD1), RN.complete_runs(POD2)
    if r1 and r2:
        _cross(reg, write_tex, table, latex_escape, r1[-1], r2[-1])
    _hardware(reg, write_tex, table, latex_escape)


def _cross(reg, write_tex, table, latex_escape, p1: Path, p2: Path) -> None:
    src = [p1 / "requests.jsonl", p2 / "requests.jsonl"]
    rows, slow = [], []
    for scen, short, label in CROSS:
        a, b = level_rates(p1, scen), level_rates(p2, scen)
        for n in CROSS_NS:
            if n not in a or n not in b:
                continue
            x, y = a[n], b[n]
            k = f"ctl.pod.cross.{short}.N{n}"
            ra, rt = ratio(x["agg"], y["agg"]), ratio(x["tpot_p50"], y["tpot_p50"])
            for name, v, fmt in (("pod1.agg", x["agg"], "{:.0f}"), ("pod2.agg", y["agg"], "{:.0f}"),
                                 ("agg_ratio", ra, "{:.2f}"), ("tpot_ratio", rt, "{:.2f}")):
                if v is not None:
                    reg.put(f"{k}.{name}", v, src, "controlled", fmt,
                            note="ratio = pod 2 / pod 1" if "ratio" in name else "")
            for pod, v in (("pod1", x), ("pod2", y)):
                if v["tpot_p50"] is not None:
                    reg.put(f"{k}.{pod}.tpot_p50_ms", 1000 * v["tpot_p50"], src, "controlled", "{:.1f}")
            for pod, v in (("pod1", x), ("pod2", y)):
                if v["ttft_max"] is not None and v["ttft_max"] > SLOW_TTFT_S:
                    reg.put(f"{k}.{pod}.ttft_max", v["ttft_max"], src, "controlled", "{:.1f}")
                    slow.append(f"{label} $N={n}$ on {pod.replace('pod', 'pod ')}, {v['ttft_max']:.1f} s")
            rows.append([latex_escape(label) if n == CROSS_NS[0] else "", str(n),
                         f"{x['agg']:.0f}", f"{y['agg']:.0f}", f"{ra:.2f}",
                         f"{1000 * x['tpot_p50']:.1f}", f"{1000 * y['tpot_p50']:.1f}", f"{rt:.2f}"])
    write_tex("cross_pod.tex", table(
        "The same E3 subset workload (\\texttt{gen30} plan, identical stack and workload file, one repetition) on "
        "pod 1 and pod 2: windowed aggregate output rate (tok/s) and median TPOT (ms) per load point, and the "
        "ratio pod 2 / pod 1. With one repetition per pod the ratio mixes the host difference with run-to-run and "
        "boot-to-boot variance.", "tab:cross-pod",
        ["scenario", "$N$", "aggr. pod 1", "pod 2", "ratio", "TPOT p50 pod 1", "pod 2", "ratio"], rows, "lrrrrrrr",
        note=(f"Sources: \\datafile{{{rel(p1)}}}, \\datafile{{{rel(p2)}}}. "
              + (f"Levels whose slowest request had a TTFT above {SLOW_TTFT_S:.0f} s: " + "; ".join(slow) + "."
                 if slow else ""))))
    reg.artifact("cross_pod.tex", "controlled", src, "data")


def _hardware(reg, write_tex, table, latex_escape) -> None:
    rows, srcs = [], []
    for key, label, d, sess in SESSIONS:
        runs = RN.complete_runs(d)
        if not runs:
            continue
        run = runs[-1]
        fp_f, topo_f, q_f = run / "fingerprint.json", run / "nvidia-smi-topo.txt", run / "nvidia-smi-q.txt"
        if not (fp_f.exists() and topo_f.exists() and q_f.exists()):
            continue
        fp = json.loads(fp_f.read_text())
        topo = parse_topo(topo_f.read_text())
        q = parse_smi_q(q_f.read_text())
        hdr = RN.sampler_header(run) or {}
        cur = sorted({(g.get("pcie_gen_cur"), g.get("pcie_width_cur")) for g in hdr.get("gpus", [])})
        k = f"ctl.pod.{key}"
        cpu = fp["cpu"].replace(" Processor", "")
        gen = sorted({g.get("gen_max") for g in q["gpus"]})
        width = sorted({"x" + (g.get("width_max") or "").rstrip("x") for g in q["gpus"]})
        pcie = f"Gen{'/'.join(gen)} {'/'.join(width)}"
        numa = ", ".join(topo["numa"])
        reg.put(f"{k}.cpu", cpu, [fp_f], "controlled")
        reg.put(f"{k}.threads", int(fp["nproc"]), [fp_f], "controlled", "{:d}")
        reg.put(f"{k}.driver", q["driver"] or fp["driver"], [q_f], "controlled")
        reg.put(f"{k}.gpu_link", topo["link"], [topo_f], "controlled",
                note="nvidia-smi topo -m between GPU0 and GPU1: NODE = same NUMA node, SYS = across the SMP link")
        reg.put(f"{k}.gpu_numa", numa, [topo_f], "controlled")
        reg.put(f"{k}.pcie_max", pcie, [q_f], "controlled")
        reg.put(f"{k}.run", rel(run), [fp_f], "controlled")
        if cur:
            reg.put(f"{k}.pcie_cur_at_start", ", ".join(f"Gen{g} x{w}" for g, w in cur), [run / "samples.jsonl"],
                    "controlled", note="current link from the sampler header (NVML) when sampling started")
        if sess.exists():
            reg.put(f"{k}.date", json.loads(sess.read_text())["created_utc"][:10], [sess], "controlled")
        srcs += [fp_f, topo_f, q_f]
        rows.append([latex_escape(label), latex_escape(run.parent.parent.name + "/" + run.parent.name),
                     latex_escape(cpu), str(fp["nproc"]), latex_escape(numa), latex_escape(topo["link"] or ""),
                     latex_escape(q["driver"] or ""), latex_escape(pcie),
                     latex_escape(", ".join(g.get("bus", "")[-7:] for g in q["gpus"]))])
    if len(rows) >= 2:
        same = {r[2] for r in rows}
        reg.put("ctl.pod.n_cpu_models", len(same), srcs, "controlled", "{:d}")
    if not rows:
        return
    write_tex("pod_hardware.tex", table(
        "Hardware fingerprint of the three pod sessions (all 2x RTX PRO 6000 Blackwell Server Edition, same "
        "datacenter), from one run folder per session: CPU model and threads visible to the container "
        "(\\texttt{fingerprint.json}), NUMA node of each GPU and the path between GPU0 and GPU1 (\\texttt{nvidia-smi topo -m}: "
        "NODE within one NUMA node, SYS across the inter-socket link), driver, maximum PCIe link and GPU bus ids "
        "(\\texttt{nvidia-smi -q}).", "tab:pod-hardware",
        ["session", "run", "CPU", "threads", "GPU NUMA", "GPU link", "driver", "PCIe max", "bus ids"], rows,
        "lp{2.4cm}p{2.2cm}rp{1.1cm}llp{1.2cm}p{1.6cm}", placement="!hb"))
    reg.artifact("pod_hardware.tex", "controlled", srcs, "data")
