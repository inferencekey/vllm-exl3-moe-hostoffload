"""Out-of-scope terms for the scope guards. This artifact covers only the vLLM + EXL3 line of work."""
import re

_FRAGMENTS = [("lla", "ma", r"\.", "cpp"), ("lla", "ma", "-", "server"), ("lla", "ma", "cpp"), ("gg", "uf"),
              ("q2", "_k"), ("gg", "ml"), ("lla", "ma", "file")]


def out_of_scope_pattern() -> re.Pattern:
    return re.compile("|".join("".join(f) for f in _FRAGMENTS), re.IGNORECASE)
