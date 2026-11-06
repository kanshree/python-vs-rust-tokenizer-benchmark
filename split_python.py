"""Where Python's time goes on the books: the regex pass alone (the `regex` module is written in C), the regex pass
plus the vocabulary lookups (a dict, also C), and the whole encode. Five runs each, medians.
Writes results/python-split.json."""
import json
import statistics
import time
from pathlib import Path

from tok_python import PAT, RANKS, encode

HERE = Path(__file__).parent
docs: list[str] = json.loads((HERE / "corpus" / "docs-books.json").read_text())


def regex_only() -> None:
    for d in docs:
        for m in PAT.finditer(d):
            m.group().encode("utf-8")


def regex_and_lookup() -> None:
    for d in docs:
        for m in PAT.finditer(d):
            RANKS.get(m.group().encode("utf-8"))


def whole() -> None:
    for d in docs:
        encode(d)


out = {}
for f in (regex_only, regex_and_lookup, whole):
    times = []
    for _ in range(5):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    out[f.__name__] = {"times": times, "median": statistics.median(times)}
    print(f"{f.__name__:18} median {statistics.median(times):.3f}s", flush=True)
(HERE / "results" / "python-split.json").write_text(json.dumps(out, indent=1))
