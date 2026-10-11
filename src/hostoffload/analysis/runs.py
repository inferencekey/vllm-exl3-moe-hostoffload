"""Readers for one controlled run folder (requests.jsonl, levels.jsonl, server_metrics.jsonl, samples.jsonl).

A level is one (scenario, load point, repetition) of a run. Its window is [t_start_wall, t_end_wall] from
levels.jsonl; the server's /metrics snapshots `before` and `after` are taken at exactly those two instants, and the
warm request of a cached-prefix level is sent before the window opens, so it is outside both the window and the
counter deltas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def level_name(point: dict) -> str:
    return f"{point['mode']}:{point['value']}"


@dataclass
class Level:
    scenario: str
    level: str
    N: int
    rep: int
    t0: float
    t1: float
    requests: list[dict] = field(default_factory=list)
    before: dict | None = None
    after: dict | None = None

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def end_wall(r: dict) -> float | None:
    """Wall-clock end of a request (the log keeps monotonic times plus the wall time of the send)."""
    if r.get("end_mono") is None or r.get("sent_wall") is None:
        return None
    return r["sent_wall"] + (r["end_mono"] - r["sent_mono"])


def levels(run: Path) -> list[Level]:
    """Every level of a run with its non-warm requests and its counter snapshots, in run order."""
    out: dict[tuple, Level] = {}
    for lv in read_jsonl(run / "levels.jsonl"):
        name = level_name(lv["point"])
        out[(lv["scenario"], name, lv["rep"])] = Level(lv["scenario"], name, int(lv["point"]["value"]), lv["rep"],
                                                       lv["t_start_wall"], lv["t_end_wall"])
    for r in read_jsonl(run / "requests.jsonl"):
        lv = out.get((r["scenario"], r["level"], r["repetition"]))
        if r["level"] != "warm" and lv is not None:
            lv.requests.append(r)
    sm = run / "server_metrics.jsonl"
    if sm.exists():
        for m in read_jsonl(sm):
            lv = out.get((m["scenario"], level_name(m["point"]), m.get("rep", 0)))
            if lv is not None:
                setattr(lv, m["when"], m["counters"])
    return list(out.values())


def samples(run: Path) -> list[dict]:
    f = run / "samples.jsonl"
    return [s for s in read_jsonl(f) if s.get("type") == "sample"] if f.exists() else []


def sampler_header(run: Path) -> dict | None:
    f = run / "samples.jsonl"
    if not f.exists():
        return None
    with f.open() as fh:
        first = json.loads(fh.readline())
    return first if first.get("type") == "header" else None


def complete_runs(pattern_dir: Path) -> list[Path]:
    out = []
    for rj in sorted(Path(pattern_dir).glob("*/run.json")):
        try:
            if json.loads(rj.read_text()).get("status") == "complete":
                out.append(rj.parent)
        except json.JSONDecodeError:
            continue
    return out
