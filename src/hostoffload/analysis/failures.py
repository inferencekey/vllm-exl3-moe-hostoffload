"""Classify why a vLLM boot or run failed, from its log. Used by the feasibility experiment (E2) so that an
"infeasible" result is the exact error the server printed, not a guess.

  python3 -m hostoffload.analysis.failures <serve.log>     # prints JSON {kind, line, lineno, ...}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# (kind, pattern); the first match in file order wins, so put specific patterns first
SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("cuda_graph_capture", re.compile(r"cudaErrorStreamCapture\w+|operation not permitted when stream is capturing")),
    ("gpu_oom", re.compile(r"(torch\.)?OutOfMemoryError|CUDA out of memory|cudaErrorMemoryAllocation|"
                           r"out of memory on device|Not enough memory|No available memory for the cache blocks")),
    ("host_oom", re.compile(r"Killed process|oom-kill|MemoryError|cannot allocate memory|"
                            r"cudaHostRegister failed|pin_memory.*(failed|error)", re.IGNORECASE)),
    ("unsupported_arch", re.compile(r"(not supported|unsupported|only supported).{0,80}"
                                    r"(compute capability|sm_?\d+|SM ?\d+|architecture|device capability)",
                                    re.IGNORECASE)),
    ("nan_output", re.compile(r"\bNaN\b|nan detected|inf or nan", re.IGNORECASE)),
    ("config_error", re.compile(r"(ValueError|NotImplementedError|AssertionError|TypeError): ")),
    ("engine_died", re.compile(r"EngineCore (failed|encountered a fatal error)|Engine core initialization failed")),
]


def classify(text: str) -> dict:
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for kind, pat in SIGNATURES:
            if pat.search(line):
                return {"kind": kind, "lineno": i, "line": line.strip()[:500],
                        "context": [l[:300] for l in lines[max(0, i - 4):i + 2]]}
    return {"kind": "none", "lineno": None, "line": None}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    print(json.dumps(classify(Path(argv[0]).read_text(errors="replace")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
