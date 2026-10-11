"""Harness tests: SSE parsing, chunk accounting, arrivals, slicing, Prometheus parsing, sampler parsers, and an
end-to-end loadgen run against the mock server (no GPU)."""
import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

from hostoffload.harness import prom, sampler
from hostoffload.harness.client import RequestRecord, apply_chunk, chat_body, parse_sse_line
from hostoffload.harness.workload import ClosedLoop, Poisson, SliceAllocator, corpus_from_tarball, lognormal_lengths
from hostoffload.analysis.metrics import level_summary, request_metrics


def test_parse_sse_line():
    assert parse_sse_line("event: x") is None
    assert parse_sse_line("data: [DONE]") == "DONE"
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}


def test_apply_chunk_tracks_tokens_per_chunk():
    r = RequestRecord("id", "s", "closed:1", 0, 0, None, sent_mono=0.0)
    apply_chunk(r, {"choices": [{"delta": {"content": "ab"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
                1.0)
    apply_chunk(r, {"choices": [{"delta": {"content": "cd"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 7,
                              "prompt_tokens_details": {"cached_tokens": 4}}}, 1.5)
    assert r.first_token_mono == 1.0 and r.last_token_mono == 1.5
    assert r.chunks == [(1.0, 3), (1.5, 7)]
    assert r.cached_tokens == 4 and r.finish_reason == "length"
    m = request_metrics({**r.__dict__, "end_mono": 1.6, "error": None}, min_tokens_for_tpot=2)
    assert m.ttft_s == pytest.approx(1.0)
    assert m.tpot_s == pytest.approx(0.5 / 6)
    assert m.itl_s == (pytest.approx(0.5 / 4),) and m.itl_w == (4,)


def test_chat_body_requests_continuous_usage():
    b = chat_body("m", [{"role": "user", "content": "x"}], 8, 0.0, 1, True, False)
    assert b["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert b["chat_template_kwargs"] == {"thinking": False} and b["seed"] == 1


def test_arrivals_are_seeded_and_staggered():
    a = ClosedLoop(users=8, requests=16, stagger_s=10.0, seed=3)
    assert a.start_offsets() == ClosedLoop(users=8, requests=16, stagger_s=10.0, seed=3).start_offsets()
    assert len(set(a.start_offsets())) == 8 and max(a.start_offsets()) < 10.0
    assert ClosedLoop(users=4, requests=8).start_offsets() == [0.0] * 4
    p = Poisson(rate_rps=2.0, requests=4000, seed=1).arrival_times()
    assert p == sorted(p)
    assert 4000 / p[-1] == pytest.approx(2.0, rel=0.1)


def test_slice_allocator_unique_and_respects_reserved():
    al = SliceAllocator(10_000, 100, seed=1, reserved=(1000, 2000))
    seen = set()
    while len(al):
        a, b = al.next()
        assert b <= 1000 or a >= 2000
        assert (a, b) not in seen
        seen.add((a, b))
    with pytest.raises(RuntimeError):
        al.next()


def test_lognormal_lengths_clipped_and_seeded():
    v = lognormal_lengths(1000, 4096, 0.8, 512, 16384, seed=5)
    assert v == lognormal_lengths(1000, 4096, 0.8, 512, 16384, seed=5)
    assert min(v) >= 512 and max(v) <= 16384


def test_corpus_from_tarball_sorted(tmp_path):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, txt in (("r/b.py", "B"), ("r/a.py", "A"), ("r/c.txt", "C")):
            data = txt.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    text = corpus_from_tarball(buf.getvalue(), ".py")
    assert text.index("A") < text.index("B") and "C" not in text.replace("# ==== file", "")


def test_prom_histogram_delta_and_quantile():
    t0 = 'vllm:x_bucket{le="0.1"} 0\nvllm:x_bucket{le="1.0"} 0\nvllm:x_bucket{le="+Inf"} 0\nvllm:x_sum 0\nvllm:x_count 0\n'
    t1 = 'vllm:x_bucket{le="0.1"} 5\nvllm:x_bucket{le="1.0"} 10\nvllm:x_bucket{le="+Inf"} 10\nvllm:x_sum 4\nvllm:x_count 10\n'
    h = prom.histogram_delta(prom.histogram(prom.parse(t0), "vllm:x"), prom.histogram(prom.parse(t1), "vllm:x"))
    assert h["count"] == 10
    assert prom.histogram_quantile(h, 0.5) == pytest.approx(0.1)
    assert prom.histogram_quantile(h, 0.75) == pytest.approx(0.55)
    json.dumps(h)  # JSON-serializable


def test_prom_counters_sum_labels():
    s = prom.parse('vllm:prefix_cache_hits_total{a="1"} 3\nvllm:prefix_cache_hits_total{a="2"} 4\n# c\n')
    assert prom.scalar(s, "vllm:prefix_cache_hits_total") == 7


def test_sampler_parsers():
    st = sampler.parse_proc_status("VmRSS:\t  100 kB\nRssAnon:\t 60 kB\nVmLck:\t 0 kB\nName:\tpython\n")
    assert st == {"VmRSS": 102400, "RssAnon": 61440, "VmLck": 0}
    smaps = ("7f00-7f10 rw-s 00000000 00:01 5 /dev/shm/x\nSize: 64 kB\nRss: 64 kB\nAnonymous: 0 kB\n"
             "7e00-7e01 rw-p 00000000 00:00 0 \nRss: 4 kB\nAnonymous: 4 kB\n")
    maps = sampler.parse_smaps_largest(smaps)
    assert maps[0]["path"] == "/dev/shm/x" and maps[0]["Rss"] == 65536
    assert sampler.parse_meminfo("MemTotal: 10 kB\nHugePages_Total: 0\n") == {"MemTotal": 10240, "HugePages_Total": 0}


def test_level_summary_window_excludes_drain():
    recs = []
    for i in range(10):
        recs.append({"id": str(i), "error": None, "sent_mono": float(i), "first_token_mono": i + 0.5,
                     "last_token_mono": i + 1.5, "end_mono": i + 1.5, "completion_tokens": 11,
                     "prompt_tokens": 100, "cached_tokens": 0, "chunks": []})
    s = level_summary(recs)
    assert s["ok"] == 10 and s["ttft_s"]["p50"] == pytest.approx(0.5)
    assert s["tpot_s"] is None  # 11 tokens is below the TPOT minimum of 16
    assert s["window_s"] == pytest.approx(9.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def mock_server():
    from aiohttp import web
    from mock_server import Mock
    mock = Mock()
    port = _free_port()
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(mock.app())

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        loop.run_forever()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(0.3)
    yield mock, f"http://127.0.0.1:{port}"
    loop.call_soon_threadsafe(loop.stop)


def test_loadgen_end_to_end(tmp_path, mock_server, monkeypatch):
    mock, url = mock_server
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(chr(97 + (i * 7) % 26) for i in range(200_000)))
    import hashlib
    from hostoffload.harness.workload import CorpusSpec, build_corpus_text
    spec = CorpusSpec("t", (corpus.as_uri(),), "", kind="texts")
    _, obs = build_corpus_text(spec, tmp_path / "cache")
    (tmp_path / "corpora.yaml").write_text(yaml.safe_dump(
        {"code": {"name": "t", "kind": "texts", "urls": [corpus.as_uri()], "text_sha256": obs["text_sha256"]}}))
    monkeypatch.setenv("HOB_CORPORA_FILE", str(tmp_path / "corpora.yaml"))
    monkeypatch.setenv("HOB_CACHE", str(tmp_path / "cache"))
    wl = {"seed": 7, "repetitions": 2, "scenarios": [
        {"name": "A", "kind": "chat", "prompt": {"tokens": 300, "corpus": "code", "template": "review",
                                                  "chars_per_token": 1.5},
         "output": {"max_tokens": 40}, "arrivals": {"mode": "closed", "levels": [1, 4], "requests_per_user": 2,
                                                    "min_requests": 4, "stagger_s": 0.05}},
        {"name": "P", "kind": "chat", "prompt": {"tokens": 200, "corpus": "code", "chars_per_token": 1.5},
         "output": {"max_tokens": 20}, "arrivals": {"mode": "poisson", "rates": [50], "requests": 6}},
        {"name": "B", "kind": "chat", "prompt": {"tokens": 600, "shared_prefix_tokens": 500, "corpus": "code",
                                                  "template": "qa_prefix", "chars_per_token": 1.5},
         "output": {"max_tokens": 20}, "arrivals": {"mode": "closed", "levels": [2], "min_requests": 4}},
        {"name": "pre", "kind": "prefill", "lengths": [100, 400], "requests": 2,
         "prompt": {"corpus": "code", "chars_per_token": 1.5}},
    ]}
    (tmp_path / "wl.yaml").write_text(yaml.safe_dump(wl))
    from hostoffload.harness import loadgen
    rc = loadgen.main(["--workload", str(tmp_path / "wl.yaml"), "--out", str(tmp_path / "out"), "--base-url", url,
                       "--model", "mock-model", "--ready-timeout", "10"])
    assert rc == 0
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run["status"] == "complete"
    reqs = [json.loads(l) for l in (tmp_path / "out" / "requests.jsonl").read_text().splitlines()]
    measured = [r for r in reqs if r["level"] != "warm"]
    # 2 reps x (A: 4 + 8, P: 6, B: 2 users x 4 requests, prefill: 2 + 2)
    assert len(measured) == 2 * (4 + 8 + 6 + 8 + 2 + 2)
    assert len([r for r in reqs if r["level"] == "warm"]) == 2
    assert all(r["error"] is None for r in reqs)
    a = [r for r in measured if r["scenario"] == "A"]
    assert all(r["prompt_tokens"] >= 300 for r in a)
    pre = [r for r in measured if r["scenario"].startswith("pre-")]
    assert all(r["completion_tokens"] == 1 for r in pre)
    levels = [json.loads(l) for l in (tmp_path / "out" / "levels.jsonl").read_text().splitlines()]
    assert len(levels) == 2 * (2 + 1 + 1 + 2)
    # different repetitions use different seeds and therefore different prompts
    seeds = {(l["scenario"], json.dumps(l["point"])): set() for l in levels}
    for l in levels:
        seeds[(l["scenario"], json.dumps(l["point"]))].add(l["seed"])
    assert all(len(v) == 2 for v in seeds.values())
    assert len({json.dumps(b["messages"]) for b in mock.bodies}) > 30


def test_loadgen_agent_sessions_and_stream_slo(tmp_path, mock_server, monkeypatch):
    """E13 agent sessions: one warm turn per user, then each user's own turns in order, each prompt extending the
    previous one; the per-stream SLO stops a sweep after two failing levels and keeps every level run."""
    mock, url = mock_server
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(chr(97 + (i * 7) % 26) for i in range(200_000)))
    from hostoffload.harness.workload import CorpusSpec, build_corpus_text
    _, obs = build_corpus_text(CorpusSpec("t", (corpus.as_uri(),), "", kind="texts"), tmp_path / "cache")
    (tmp_path / "corpora.yaml").write_text(yaml.safe_dump(
        {"code": {"name": "t", "kind": "texts", "urls": [corpus.as_uri()], "text_sha256": obs["text_sha256"]}}))
    monkeypatch.setenv("HOB_CORPORA_FILE", str(tmp_path / "corpora.yaml"))
    monkeypatch.setenv("HOB_CACHE", str(tmp_path / "cache"))
    # the mock decodes at about 1000 tok/s, so an SLO of 1e9 tok/s fails at every level
    wl = {"seed": 3, "repetitions": 1, "slo_stream_tok_s": 1e9, "stop_after_slo_failures": 2, "scenarios": [
        {"name": "D", "kind": "agent", "agent": {"base_tokens": 300, "append_tokens": 50, "turns": 3},
         "prompt": {"corpus": "code", "chars_per_token": 1.5}, "output_tokens": 30,
         "arrivals": {"mode": "closed", "levels": [1, 2, 3, 4]}}]}
    (tmp_path / "wl.yaml").write_text(yaml.safe_dump(wl))
    from hostoffload.harness import loadgen
    assert loadgen.main(["--workload", str(tmp_path / "wl.yaml"), "--out", str(tmp_path / "out"), "--base-url", url,
                         "--model", "mock-model", "--ready-timeout", "10"]) == 0
    reqs = [json.loads(l) for l in (tmp_path / "out" / "requests.jsonl").read_text().splitlines()]
    levels = [json.loads(l) for l in (tmp_path / "out" / "levels.jsonl").read_text().splitlines()]
    assert [l["point"]["value"] for l in levels] == [1, 2]          # stopped after two failing levels
    assert len([r for r in reqs if r["level"] == "warm"]) == 1 + 2  # one session opening per user
    measured = [r for r in reqs if r["level"] != "warm"]
    assert len(measured) == 3 * (1 + 2) and all(r["error"] is None for r in reqs)
    for lv in ("closed:1", "closed:2"):
        for u in {r["user"] for r in measured if r["level"] == lv}:
            mine = sorted((r for r in measured if r["level"] == lv and r["user"] == u), key=lambda r: r["sent_mono"])
            sizes = [r["prompt_tokens"] for r in mine]
            assert sizes == sorted(sizes) and sizes[-1] - sizes[0] >= 2 * 50   # each turn appends one chunk
