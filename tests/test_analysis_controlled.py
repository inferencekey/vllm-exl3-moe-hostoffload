"""The controlled-data analyzers, exercised on mock-server output written to a temporary directory (never data/)."""
import json

import yaml

from hostoffload.analysis import pipeline, quality
from hostoffload.analysis.registry import Registry
from test_harness import mock_server  # noqa: F401  (fixture)


def test_concurrency_table_from_mock_runs(tmp_path, mock_server, monkeypatch):  # noqa: F811
    _mock, url = mock_server
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(chr(97 + (i * 11) % 26) for i in range(100_000)))
    from hostoffload.harness.workload import CorpusSpec, build_corpus_text
    _, obs = build_corpus_text(CorpusSpec("t", (corpus.as_uri(),), "", kind="texts"), tmp_path / "c")
    (tmp_path / "corpora.yaml").write_text(yaml.safe_dump(
        {"code": {"name": "t", "kind": "texts", "urls": [corpus.as_uri()], "text_sha256": obs["text_sha256"]}}))
    monkeypatch.setenv("HOB_CORPORA_FILE", str(tmp_path / "corpora.yaml"))
    monkeypatch.setenv("HOB_CACHE", str(tmp_path / "c"))
    wl = {"seed": 3, "repetitions": 3, "scenarios": [
        {"name": "A", "kind": "chat", "prompt": {"tokens": 100, "chars_per_token": 1.5},
         "output": {"max_tokens": 40}, "arrivals": {"mode": "closed", "levels": [1, 2], "min_requests": 4}}]}
    (tmp_path / "wl.yaml").write_text(yaml.safe_dump(wl))
    from hostoffload.harness import loadgen
    run = tmp_path / "E8-main" / "variantX" / "run1"
    assert loadgen.main(["--workload", str(tmp_path / "wl.yaml"), "--out", str(run), "--base-url", url,
                         "--model", "mock-model", "--ready-timeout", "10"]) == 0
    # the mock's runs are never accepted as controlled data
    assert pipeline.controlled_runs(tmp_path / "E8-main") == []
    rj = json.loads((run / "run.json").read_text())
    rj["server"]["model"] = "pretend-real"
    (run / "run.json").write_text(json.dumps(rj))
    runs = pipeline.controlled_runs(tmp_path / "E8-main")
    assert runs == [run]
    reg = Registry()
    slos = {"loose": {"ttft_p95_s": 10.0, "tpot_p95_s": 1.0, "max_error_rate": 0.0}}
    tex = pipeline.controlled_concurrency_table({"id": "E8", "title": "t"}, runs, reg, slos)
    assert "closed:1" in tex and "closed:2" in tex and "sidewaystable" not in tex  # upright, in two parts
    assert "\\label{tab:ctl-E8}" in tex and "\\label{tab:ctl-E8-thr}" in tex and "\\textit{variantX}" in tex
    keys = [k for k in reg.entries if k.startswith("ctl.E8.variantX.A.closed:1.")]
    assert any(k.endswith("ttft_p95") for k in keys) and any(k.endswith("agg_cv") for k in keys)
    assert all(e.kind == "controlled" for e in reg.entries.values())


def test_quality_paired_difference(tmp_path):
    a = {i: float(i % 2) for i in range(200)}
    b = {i: 1.0 for i in range(200)}
    d = quality.paired_difference(a, b, n_boot=500)
    assert d["n"] == 200 and abs(d["diff"] - 0.5) < 1e-9
    lo, hi = d["ci95"]
    assert lo < 0.5 < hi and d["discordant"] == 100
    assert quality.paired_difference({1: 1.0}, {1: 0.0}) is None


def test_quality_reads_lm_eval_layout(tmp_path):
    t = tmp_path / "lm_eval" / "ifeval" / "model"
    t.mkdir(parents=True)
    (t / "results_2026-10-07.json").write_text(json.dumps(
        {"results": {"ifeval": {"prompt_level_strict_acc,none": 0.8, "prompt_level_strict_acc_stderr,none": 0.02}}}))
    (t / "samples_ifeval_2026-10-07.jsonl").write_text(
        "\n".join(json.dumps({"doc_id": i, "prompt_level_strict_acc": i % 5 != 0}) for i in range(10)))
    r = quality.find_results(tmp_path)
    assert quality.headline(r["ifeval"]["metrics"], "prompt_level_strict_acc") == (0.8, 0.02)
    s = quality.load_samples(tmp_path, "ifeval", "prompt_level_strict_acc")
    assert len(s) == 10 and s[0] == 0.0 and s[1] == 1.0
