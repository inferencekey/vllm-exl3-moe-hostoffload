#!/usr/bin/env python3
"""Create data/controlled/<exp>/README.md for every experiment: "not run yet" for an empty folder, and for a folder
with data the real status of each run folder (run id, pod session, status, repetitions) read from its files.

The READMEs of folders with data are rewritten on the workstation after copy-back, so their hashes differ from the
pod manifests in data/controlled/LEAN/ (which list the READMEs as they were on the pod)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def fmt_run(r: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in r.items())


SESSIONS = ("session.json", "session2.json", "session3.json")


def pod_session(run_id: str) -> str:
    """Pod session whose create-to-delete window contains the run id's UTC timestamp."""
    try:
        t = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return "unknown"
    for i, name in enumerate(SESSIONS, 1):
        f = ROOT / "data" / "controlled" / "LEAN" / name
        if f.exists():
            sj = json.loads(f.read_text())
            t0, t1 = (datetime.strptime(sj[k], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                      for k in ("created_utc", "deleted_utc"))
            if t0 <= t <= t1:
                return f"{i} (`LEAN/{name}`)"
    return "unknown"


def run_rows(d: Path) -> list[str]:
    rows = []
    for rj in sorted(d.rglob("run.json")):
        run = json.loads(rj.read_text())
        st = (rj.parent / "status.txt").read_text().split()[0] if (rj.parent / "status.txt").exists() else "-"
        boot = (rj.parent / "boot.txt").read_text().splitlines()[0] if (rj.parent / "boot.txt").exists() else "-"
        reps = run.get("repetitions")
        rows.append(f"| `{rj.parent.relative_to(d)}` | {pod_session(rj.parent.name)} | {run.get('status', '-')} | "
                    f"{st} | {boot} | {reps if reps is not None else '-'} |")
    return rows


def ran_readme(e: dict, d: Path) -> str:
    rows = "\n".join(run_rows(d))
    note = f"\n{e['readme_note'].strip()}\n" if e.get("readme_note") else ""
    return f"""# {e['id']}: {e.get('table_title', e['title'])}

**Status: ran** (reduced against the design in `experiments/experiments.yaml`). The paper reports the run folders
below; `make analyze` reads the `complete` ones that hold results. Runs marked `invalid` or `failed` are kept as
evidence of what went wrong and are described in the paper, not pooled with valid runs.

| run folder | pod session | run.json status | status.txt | boot | repetitions |
|---|---|---|---|---|---|
{rows}

- Question: {' '.join(str(e['answers']).split())}.
- Command: `{e['command']}`; GPU configuration: `{e['gpu']}`.
- Designed as: {e['title']}. What ran is reduced; see the paper's Appendix (experiment plan and status).
{note}"""


def main() -> int:
    cfg = yaml.safe_load((ROOT / "experiments" / "experiments.yaml").read_text())
    for e in cfg["experiments"]:
        if e["id"] == "E0":
            continue
        d = ROOT / e["data"]
        d.mkdir(parents=True, exist_ok=True)
        if any(p.is_file() and p.name != "README.md" for p in d.rglob("*")):
            (d / "README.md").write_text(ran_readme(e, d))
            continue
        runs = "\n".join(f"- {fmt_run(r)}" for r in e.get("runs", [])) or "- see experiments/experiments.yaml"
        note = f"\n{e['readme_note'].strip()}\n" if e.get("readme_note") else ""
        (d / "README.md").write_text(f"""# {e['id']}: {e['title']}

**Status: not run yet.** No data exists in this folder. The paper shows a NOT RUN placeholder for this experiment
until `make analyze` finds a complete run here.

- Question: {' '.join(str(e['answers']).split())}.
- Command (on a pod prepared with `make pod-bootstrap`): `{e['command']}`
- GPU configuration: `{e['gpu']}` (prices in `experiments/pricing.yaml`), estimated {e['hours']} h.
- Expected layout: `{e['data']}<variant>/<run_id>/` with `run.json`, `requests.jsonl`, `levels.jsonl`,
  `server_metrics.jsonl`, `samples.jsonl`, `serve.log`, `fingerprint.json` (see `experiments/README.md`).

Runs:
{runs}
{note}""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
