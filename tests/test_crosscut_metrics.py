"""Pure functions of the cross-cutting readings: per-stream rate, PCIe bytes per token, DSpark acceptance, pod parsing."""
import pytest

from hostoffload.analysis import e8_streams, pcie, pods, spec
from hostoffload.analysis import runs as RN


def req(**kw):
    base = {"level": "closed:1", "error": None, "first_token_mono": 10.0, "end_mono": 12.0, "completion_tokens": 201}
    base.update(kw)
    return base


def test_stream_rate_is_tokens_after_first_over_stream_time():
    assert e8_streams.stream_rate(req()) == pytest.approx(200 / 2.0)


@pytest.mark.parametrize("kw", [{"error": "timeout"}, {"level": "warm"}, {"first_token_mono": None},
                                {"end_mono": None}, {"end_mono": 10.0}, {"completion_tokens": 8},
                                {"completion_tokens": None}])
def test_stream_rate_excludes_errors_warm_and_degenerate(kw):
    assert e8_streams.stream_rate(req(**kw)) is None


def test_per_n_pools_rates_and_averages_aggregate_per_repetition():
    def lv(rep, t0):
        rs = [{"id": f"{rep}{i}", "level": "closed:2", "error": None, "sent_mono": t0, "first_token_mono": t0 + 1,
               "last_token_mono": t0 + 3, "end_mono": t0 + 3, "completion_tokens": 101, "chunks": []}
              for i in range(2)]
        return RN.Level("A", "closed:2", 2, rep, 0.0, 1.0, rs)
    d = e8_streams.per_n([lv(0, 0.0), lv(1, 100.0)], "A")
    assert d[2]["n"] == 4 and d[2]["reps"] == 2
    assert d[2]["p50"] == pytest.approx(50.0)
    assert d[2]["agg"] == pytest.approx(2 * 101 / 3.0)  # window ends at the last end when sends are simultaneous


def test_pcie_rate_sums_gpus_in_kilobytes():
    s = {"gpus": [{"pcie_rx_kBps": 1_000_000}, {"pcie_rx_kBps": 500_000}, {"i": 2}]}
    assert pcie.rx_bytes_per_s(s) == pytest.approx(1.5e9)


def test_mb_per_token():
    assert pcie.mb_per_token(2e9, 10.0, 100) == pytest.approx(200.0)
    assert pcie.mb_per_token(2e9, 10.0, 0) is None
    assert pcie.mb_per_token(2e9, 0.0, 10) is None


def test_level_traffic_uses_samples_and_tokens_inside_the_window():
    lv = RN.Level("A", "closed:1", 1, 0, 100.0, 110.0, [
        {"sent_wall": 100.0, "sent_mono": 0.0, "end_mono": 5.0, "completion_tokens": 50, "error": None},
        {"sent_wall": 100.0, "sent_mono": 0.0, "end_mono": 50.0, "completion_tokens": 999, "error": None},
        {"sent_wall": 100.0, "sent_mono": 0.0, "end_mono": 6.0, "completion_tokens": 999, "error": "x"}])
    smp = [{"t_wall": t, "gpus": [{"pcie_rx_kBps": k}]} for t, k in ((99.0, 9e9), (101.0, 1e6), (109.0, 3e6))]
    t = pcie.level_traffic(lv, smp)
    assert t["rate"] == pytest.approx(2e9) and t["tokens"] == 50 and t["n_samples"] == 2
    assert t["bytes"] == pytest.approx(2e10)


def test_acceptance_length_definition():
    a = spec.acceptance(accepted=1453, drafts=600, draft_tokens=3000)
    assert a["accepted_per_draft"] == pytest.approx(1453 / 600)
    assert a["accept_len"] == pytest.approx(1 + 1453 / 600)
    assert a["accept_rate"] == pytest.approx(1453 / 3000)
    assert a["k"] == pytest.approx(5)
    assert spec.acceptance(0, 0, 0) is None
    assert "accept_rate" not in spec.acceptance(10, 5)


def test_counter_delta_requires_both_snapshots_and_counters():
    c0 = {"spec_accepted": 10, "spec_drafts": 5, "spec_draft_tokens": 25, "generation_tokens": 20}
    c1 = {"spec_accepted": 40, "spec_drafts": 15, "spec_draft_tokens": 75, "generation_tokens": 60}
    lv = RN.Level("A", "closed:1", 1, 0, 0, 1, [], c0, c1)
    assert spec.counter_delta(lv) == {"spec_accepted": 30, "spec_drafts": 10, "spec_draft_tokens": 50,
                                      "generation_tokens": 40}
    assert spec.counter_delta(RN.Level("A", "closed:1", 1, 0, 0, 1, [], None, c1)) is None
    assert spec.counter_delta(RN.Level("A", "closed:1", 1, 0, 0, 1, [], {**c0, "spec_drafts": None}, c1)) is None


def test_selected_ns():
    assert pcie.selected_ns([1, 2, 4, 8, 16]) == [1, 8, 16]
    assert pcie.selected_ns([1, 2, 3]) == [1, 3]


def test_ratio():
    assert pods.ratio(2.0, 3.0) == pytest.approx(1.5)
    assert pods.ratio(0, 3.0) is None and pods.ratio(None, 1.0) is None


TOPO = ("\t\x1b[4mGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
        "GPU0\t X \tSYS\tNODE\t0-63,128-191\t0\t\tN/A\n"
        "GPU1\tSYS\t X \tSYS\t64-127,192-255\t1\t\tN/A\n"
        "NIC0\tNODE\tSYS\t X \t\t\t\t\n")


def test_parse_topo():
    t = pods.parse_topo(TOPO)
    assert t["link"] == "SYS" and t["numa"] == ["0", "1"] and t["n_gpus"] == 2
    assert t["cpu_affinity"] == ["0-63,128-191", "64-127,192-255"]


SMI_Q = """Driver Version                                         : 595.91.07
GPU 00000000:81:00.0
    Product Name                                       : NVIDIA RTX PRO 6000
        Bus Id                                         : 00000000:81:00.0
            PCIe Generation
                Max                                    : 5
                Current                                : 1
            Link Width
                Max                                    : 16x
                Current                                : 16x
        Max Clocks
            Max                                        : 3000 MHz
"""


def test_parse_smi_q():
    q = pods.parse_smi_q(SMI_Q)
    assert q["driver"] == "595.91.07"
    assert q["gpus"] == [{"bus": "00000000:81:00.0", "gen_max": "5", "width_max": "16x"}]
