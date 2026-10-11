"""The no-fake-data guard.

1. The registry refuses values without existing source files, controlled values from exploratory files, and None.
2. Running the pipeline with no controlled data yields NOT RUN placeholders for every controlled experiment, and
   strict mode fails.
3. The paper renders a visible NOT RUN marker for a missing value, and a strict build stops with an error.
4. No decimal number appears in the paper sources outside \\dv{...} unless whitelisted with its source.
5. data/controlled holds only READMEs until real runs exist, and never output of the test mock server.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from hostoffload.analysis import pipeline
from hostoffload.analysis.registry import ROOT, MissingData, Registry

PAPER = ROOT / "paper"


def test_registry_rejects_bad_values(tmp_path):
    reg = Registry()
    with pytest.raises(MissingData):
        reg.put("x", 1, [tmp_path / "missing.json"], "static")
    with pytest.raises(ValueError):
        reg.put("x", None, [ROOT / "README.md"], "static")
    with pytest.raises(ValueError):
        reg.put("x", 1, [], "static")
    exp = next((ROOT / "data" / "exploratory").rglob("scenario_A.json"))
    with pytest.raises(ValueError):
        reg.put("x", 1, [exp], "controlled")
    with pytest.raises(ValueError):
        reg.put("y", 1, [ROOT / "experiments" / "pricing.yaml"], "exploratory")
    reg.put("z", 2, [exp], "exploratory")
    with pytest.raises(ValueError):
        reg.put("z", 3, [exp], "exploratory")


def test_pipeline_without_controlled_data_writes_placeholders():
    assert pipeline.main([]) == 0
    prov = json.loads((PAPER / "generated" / "provenance.json").read_text())
    cfg = yaml.safe_load((ROOT / "experiments" / "experiments.yaml").read_text())
    for exp in cfg["experiments"]:
        if exp["id"] == "E0":
            continue
        name = f"controlled_{exp['id']}.tex"
        art = prov["artifacts"][name]
        has_data = pipeline.controlled_runs(ROOT / exp["data"]) if (ROOT / exp["data"]).exists() else []
        if not has_data:
            assert art["status"] == "not_run"
            assert "\\notrunblock{" in (PAPER / "generated" / name).read_text()
    # every value cites existing files and every controlled value cites only controlled or static data
    for key, v in prov["values"].items():
        assert v["sources"], key
        for s in v["sources"]:
            assert (ROOT / s).exists(), (key, s)
        if v["kind"] == "controlled":
            assert all(not s.startswith("data/exploratory/") for s in v["sources"]), key
        if key.startswith("expl."):
            assert v["kind"] == "exploratory", key
    # every exploratory artifact is named as such
    for name, art in prov["artifacts"].items():
        if art["kind"] == "exploratory":
            assert "exploratory_" in name, name


def test_pipeline_strict_fails_without_controlled_data():
    cfg = yaml.safe_load((ROOT / "experiments" / "experiments.yaml").read_text())
    if all(pipeline.controlled_runs(ROOT / e["data"]) for e in cfg["experiments"] if e["id"] != "E0"
           and (ROOT / e["data"]).exists()):
        pytest.skip("all controlled data present")
    assert pipeline.main(["--strict"]) == 2


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no TeX toolchain")
@pytest.mark.parametrize("strict,expect_ok", [(False, True), (True, False)])
def test_latex_missing_value_marker(tmp_path, strict, expect_ok):
    doc = tmp_path / "t.tex"
    doc.write_text("\\documentclass{article}\\usepackage{xcolor}\n"
                   f"\\input{{{PAPER / 'datamacros.tex'}}}\n"
                   + ("\\strictdatatrue\n" if strict else "") +
                   "\\dvset{present}{42}\n\\begin{document}\\dv{present} \\dv{absent.key}\n"
                   "\\notrunblock{data/controlled/E8-main/}{E8}\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name], cwd=tmp_path,
                       capture_output=True, text=True)
    assert (r.returncode == 0) == expect_ok, r.stdout[-2000:]
    if expect_ok:
        log = (tmp_path / "t.log").read_text()
        assert "DATAMACROS NOT RUN value: absent.key" in log
        assert "DATAMACROS NOT RUN block: data/controlled/E8-main/" in log
        assert "DATAMACROS NOT RUN value: present" not in log


DECIMAL = re.compile(r"(?<![\w.\\{-])\d+(?:\.\d+)+(?![\w])")  # 0.4.13 is one token


def _strip(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\(dv|datafile|cite[pt]?|ref|label|eqref|input|includegraphics|url|href|texttt|path|nolinkurl)"
                 r"(\[[^\]]*\])?\{[^}]*\}", "", tex)
    return tex


def test_no_literal_measurements_in_paper_sources():
    allowed = yaml.safe_load((PAPER / "literal_numbers.yaml").read_text())["allowed"]
    allowed_set = {str(a["literal"]) for a in allowed}
    for a in allowed:
        assert a.get("source"), a
    bad = []
    sources = sorted((PAPER / "sections").glob("*.tex"))
    for f in sources + [PAPER / "main.tex"]:
        for n, line in enumerate(_strip(f.read_text()).splitlines(), 1):
            for m in DECIMAL.finditer(line):
                if m.group(0) not in allowed_set:
                    bad.append(f"{f.name}:{n}: {m.group(0)}")
    assert not bad, "decimal literals outside \\dv{} (add a \\dv value or whitelist with a source):\n" + \
        "\n".join(bad)


def test_controlled_folders_hold_no_mock_or_unexplained_data():
    root = ROOT / "data" / "controlled"
    for p in root.rglob("*"):
        if p.is_file() and p.name == "run.json":
            run = json.loads(p.read_text())
            assert run.get("server", {}).get("model") != "mock-model", p
    cfg = yaml.safe_load((ROOT / "experiments" / "experiments.yaml").read_text())
    for exp in cfg["experiments"]:
        d = ROOT / exp["data"]
        if exp["id"] == "E0":
            continue
        assert (d / "README.md").exists(), d
        files = [p for p in d.rglob("*") if p.is_file() and p.name != "README.md"]
        if not files:
            assert "not run yet" in (d / "README.md").read_text().lower(), d
