"""Time only Python's merge loop: the pieces that are not a single vocabulary entry, cut out before the clock starts.
Also reports how many pieces needed no merging at all.
    python3 merge_only_python.py <dataset> [copies|cuts]"""
import json
import sys
import time
from pathlib import Path

from tok_python import PAT, RANKS, bpe, bpe_cuts

data = sys.argv[1] if len(sys.argv) > 1 else "books"
style = sys.argv[2] if len(sys.argv) > 2 else "copies"
merge = {"copies": bpe, "cuts": bpe_cuts}[style]
docs: list[str] = json.loads((Path(__file__).parent / "corpus" / f"docs-{data}.json").read_text())
total, pieces = 0, []
for d in docs:
    for m in PAT.finditer(d):
        total += 1
        p = m.group().encode("utf-8")
        if p not in RANKS:
            pieces.append(p)
t = time.perf_counter()
n = sum(len(merge(p)) for p in pieces)
secs = time.perf_counter() - t
print(json.dumps({"impl": f"python-{style}", "data": data, "mode": "merge-only", "pieces": len(pieces), "all_pieces": total,
                  "seconds": secs, "tokens": n}))
