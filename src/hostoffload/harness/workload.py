"""Workloads: pinned prompt corpora, prompt construction at a target token length, and arrival schedules.

Fixes against the exploratory load generator:
- the corpus is pinned (a source tarball or text file with a recorded sha256) instead of "whatever vLLM version is
  installed on the pod";
- prompt length is set in tokens with the server's /tokenize endpoint, not with a chars-per-token guess;
- every random choice comes from a seeded generator, and the seed is saved with the results;
- arrivals can be closed loop with a staggered start (no synchronized first wave) or open loop (Poisson).
"""
from __future__ import annotations

import hashlib
import io
import math
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# --------------------------------------------------------------------------------------------------------------
# Corpora


@dataclass(frozen=True)
class CorpusSpec:
    """A pinned text corpus.

    kind 'tarball': one source archive; files ending in `suffix` are concatenated in sorted path order.
    kind 'texts': several UTF-8 text files (`urls`), concatenated in the listed order.
    `text_sha256` pins the resulting corpus text (required). `archive_sha256` pins the downloaded archive when the
    host serves stable bytes; it is advisory because some hosts recompress archives over time.
    """
    name: str
    urls: tuple[str, ...]
    text_sha256: str
    kind: str = "tarball"
    suffix: str = ".py"
    archive_sha256: str = ""
    license: str = ""


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hostoffload/0.1"}),
                                    timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def build_corpus_text(spec: CorpusSpec, cache_dir: Path) -> tuple[str, dict]:
    """Download (once) and assemble the corpus text; returns the text and the observed hashes."""
    observed = {"archives": []}
    if spec.kind == "tarball":
        path = _download(spec.urls[0], cache_dir / f"{spec.name}.tar.gz")
        data = path.read_bytes()
        observed["archives"].append(hashlib.sha256(data).hexdigest())
        text = corpus_from_tarball(data, spec.suffix)
    elif spec.kind == "texts":
        parts = []
        for i, url in enumerate(spec.urls):
            data = _download(url, cache_dir / f"{spec.name}.{i}.txt").read_bytes()
            observed["archives"].append(hashlib.sha256(data).hexdigest())
            parts.append(data.decode("utf-8", errors="replace").replace("\r\n", "\n"))
        text = "\n\n".join(parts)
    else:
        raise ValueError(f"unknown corpus kind {spec.kind}")
    observed["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return text, observed


def load_corpus(spec: CorpusSpec, cache_dir: Path) -> str:
    """The pinned corpus text; raises if its sha256 differs from the pin."""
    text, observed = build_corpus_text(spec, cache_dir)
    if observed["text_sha256"] != spec.text_sha256:
        raise ValueError(f"corpus {spec.name}: text sha256 {observed['text_sha256']} != pinned {spec.text_sha256}")
    return text


def corpus_from_tarball(data: bytes, suffix: str) -> str:
    parts = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)),
                         key=lambda m: m.name)
        for m in members:
            f = tar.extractfile(m)
            if f is None:
                continue
            parts.append(f"\n# ==== file: {m.name.split('/', 1)[-1]} ====\n" + f.read().decode("utf-8", "replace"))
    return "".join(parts)


# --------------------------------------------------------------------------------------------------------------
# Prompts at a target token length

Tokenize = Callable[[str], list[int]]
Detokenize = Callable[[list[int]], str]


class SliceAllocator:
    """Hands out non-overlapping corpus regions in a seeded random order, so that no two prompts of a run share
    text (prefix-cache hits come only from deliberately shared prefixes)."""

    def __init__(self, corpus_len: int, region_chars: int, seed: int, reserved: tuple[int, int] | None = None):
        n = corpus_len // region_chars
        idx = list(range(n))
        if reserved:
            lo, hi = reserved
            idx = [i for i in idx if (i + 1) * region_chars <= lo or i * region_chars >= hi]
        random.Random(seed).shuffle(idx)
        self._idx = idx
        self.region_chars = region_chars

    def __len__(self) -> int:
        return len(self._idx)

    def next(self) -> tuple[int, int]:
        if not self._idx:
            raise RuntimeError("corpus exhausted: use a larger corpus or fewer requests")
        i = self._idx.pop()
        return i * self.region_chars, (i + 1) * self.region_chars


def exact_length_text(text: str, target_tokens: int, tokenize: Tokenize, detokenize: Detokenize) -> tuple[str, int]:
    """Cut `text` to `target_tokens` tokens of the served tokenizer. Returns the text and its token count after a
    round trip (which can differ from the target by a token or two at the cut)."""
    ids = tokenize(text)
    if len(ids) < target_tokens:
        raise ValueError(f"region too short: {len(ids)} tokens < {target_tokens}")
    cut = detokenize(ids[:target_tokens])
    return cut, len(tokenize(cut))


# --------------------------------------------------------------------------------------------------------------
# Arrival schedules


@dataclass(frozen=True)
class ClosedLoop:
    """N users, each with one request in flight; user i starts after stagger_s * U_i (U_i uniform in [0, 1),
    seeded), then sends its next request think_s after the previous one ends (exponential if think_exp)."""
    users: int
    requests: int
    stagger_s: float = 0.0
    think_s: float = 0.0
    think_exp: bool = False
    seed: int = 0

    def start_offsets(self) -> list[float]:
        rng = random.Random(self.seed)
        return sorted(self.stagger_s * rng.random() for _ in range(self.users)) if self.stagger_s else [0.0] * self.users

    def think_times(self) -> Iterator[float]:
        rng = random.Random(self.seed + 1_000_003)
        while True:
            yield rng.expovariate(1.0 / self.think_s) if (self.think_exp and self.think_s > 0) else self.think_s


@dataclass(frozen=True)
class Poisson:
    """Open loop: requests arrive as a Poisson process of `rate_rps`, independent of completions."""
    rate_rps: float
    requests: int
    seed: int = 0

    def arrival_times(self) -> list[float]:
        if self.rate_rps <= 0:
            raise ValueError("rate must be positive")
        rng = random.Random(self.seed)
        t, out = 0.0, []
        for _ in range(self.requests):
            t += rng.expovariate(self.rate_rps)
            out.append(t)
        return out


def lognormal_lengths(n: int, median: int, sigma: float, lo: int, hi: int, seed: int) -> list[int]:
    """Seeded lognormal token lengths clipped to [lo, hi] (for mixed-length workloads)."""
    rng = random.Random(seed)
    mu = math.log(median)
    return [int(min(hi, max(lo, round(rng.lognormvariate(mu, sigma))))) for _ in range(n)]
