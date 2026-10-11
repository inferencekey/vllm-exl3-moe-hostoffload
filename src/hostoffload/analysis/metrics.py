"""Per-request and per-level serving metrics, computed from raw request records.

Definitions (used everywhere in the paper):
- TTFT: first streamed token arrival minus send time (includes queueing and prefill).
- E2E: end of stream minus send time.
- TPOT: (last token time - first token time) / (completion_tokens - 1), the mean time per output token after the
  first. Per-request decode speed is 1 / TPOT.
- Per-token latency (ITL): for each chunk after the first, (chunk time - previous chunk time) / tokens in the chunk,
  weighted by tokens. With speculative decoding several tokens can arrive in one chunk.
- Aggregate output throughput: completion tokens of successful requests that finished inside the measurement window,
  divided by the window length. The window excludes `warmup_s` after the first send and ends at the last send
  (so that the drain of the last requests does not dilute the rate), unless the level is too short.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import stats


@dataclass(frozen=True)
class RequestMetrics:
    id: str
    ok: bool
    ttft_s: float | None
    e2e_s: float | None
    tpot_s: float | None
    decode_tok_s: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int
    sent_mono: float
    end_mono: float | None
    itl_s: tuple[float, ...]
    itl_w: tuple[int, ...]


def request_metrics(rec: dict, min_tokens_for_tpot: int = 16) -> RequestMetrics:
    ok = rec.get("error") is None
    sent = rec["sent_mono"]
    ft, lt, end = rec.get("first_token_mono"), rec.get("last_token_mono"), rec.get("end_mono")
    ct = int(rec.get("completion_tokens") or 0)
    ttft = (ft - sent) if (ok and ft is not None) else None
    e2e = (end - sent) if (ok and end is not None) else None
    tpot = None
    if ok and ft is not None and lt is not None and ct >= min_tokens_for_tpot and lt > ft:
        tpot = (lt - ft) / (ct - 1)
    itl, w = [], []
    chunks = rec.get("chunks") or []
    for (t0, c0), (t1, c1) in zip(chunks, chunks[1:]):
        if c0 < 0 or c1 < 0:
            continue
        n = c1 - c0
        if n > 0 and t1 >= t0:
            itl.append((t1 - t0) / n)
            w.append(n)
    return RequestMetrics(rec["id"], ok, ttft, e2e, tpot, (1.0 / tpot) if tpot else None, rec.get("prompt_tokens"),
                          rec.get("cached_tokens"), ct, sent, end, tuple(itl), tuple(w))


def level_summary(records: Iterable[dict], warmup_s: float = 0.0) -> dict:
    recs = list(records)
    ms = [request_metrics(r) for r in recs]
    ok = [m for m in ms if m.ok]
    sends = [m.sent_mono for m in ms]
    out: dict = {"n": len(ms), "ok": len(ok), "errors": len(ms) - len(ok)}
    if not ok:
        return out
    t_first, t_last_send = min(sends), max(sends)
    t_end = max(m.end_mono for m in ok if m.end_mono is not None)
    w0 = t_first + warmup_s
    w1 = t_last_send if t_last_send - w0 > 1.0 else t_end
    in_win = [m for m in ok if m.end_mono is not None and w0 <= m.end_mono <= w1]
    out["window_s"] = w1 - w0
    out["wall_s"] = t_end - t_first
    out["aggregate_out_tok_s_window"] = (sum(m.completion_tokens for m in in_win) / (w1 - w0)) if w1 > w0 else None
    out["aggregate_out_tok_s_wall"] = sum(m.completion_tokens for m in ok) / (t_end - t_first) if t_end > t_first else None
    out["aggregate_in_tok_s_wall"] = (sum((m.prompt_tokens or 0) - (m.cached_tokens or 0) for m in ok)
                                      / (t_end - t_first)) if t_end > t_first else None
    for name, vals in (("ttft_s", [m.ttft_s for m in ok]), ("e2e_s", [m.e2e_s for m in ok]),
                       ("tpot_s", [m.tpot_s for m in ok]), ("decode_tok_s", [m.decode_tok_s for m in ok])):
        v = [x for x in vals if x is not None]
        out[name] = stats.describe(v) if v else None
    itl = [x for m in ok for x in m.itl_s]
    itl_w = [x for m in ok for x in m.itl_w]
    out["itl_s"] = stats.describe_weighted(itl, itl_w) if itl else None
    out["mean_prompt_tokens"] = stats.mean([m.prompt_tokens for m in ok if m.prompt_tokens is not None])
    out["mean_cached_tokens"] = stats.mean([m.cached_tokens or 0 for m in ok])
    out["mean_completion_tokens"] = stats.mean([m.completion_tokens for m in ok])
    return out


def meets_sla(summary: dict, sla: dict) -> bool:
    """SLA check on a level summary. sla keys: ttft_p95_s, tpot_p95_s (or decode_p5_tok_s), max_error_rate."""
    if summary.get("ok", 0) == 0:
        return False
    err = summary["errors"] / summary["n"]
    if err > sla.get("max_error_rate", 0.0):
        return False
    if "ttft_p95_s" in sla and (summary.get("ttft_s") or {}).get("p95", float("inf")) > sla["ttft_p95_s"]:
        return False
    if "tpot_p95_s" in sla and (summary.get("tpot_s") or {}).get("p95", float("inf")) > sla["tpot_p95_s"]:
        return False
    if "decode_p5_tok_s" in sla and (summary.get("decode_tok_s") or {}).get("p5", 0.0) < sla["decode_p5_tok_s"]:
        return False
    return True
