"""GPT-4o's tokenizer (o200k_base) written in plain Python.

Same algorithm as OpenAI's tiktoken core and as tok_rust/src/main.rs, line for line:
1. a regex cuts the text into pieces (words, numbers, punctuation, spaces);
2. each piece starts as single bytes, and the pair with the lowest rank is merged until no pair is in the vocabulary.
The vocabulary and the regex are OpenAI's, read from tiktoken so the result can be checked token for token.
"""
import base64
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import regex

HERE = Path(__file__).parent


def load() -> tuple[dict[bytes, int], str]:
    ranks: dict[bytes, int] = {}
    for line in (HERE / "o200k.ranks").read_text().splitlines():
        tok, rank = line.split()
        ranks[base64.b64decode(tok)] = int(rank)
    return ranks, (HERE / "o200k.pattern").read_text()


RANKS, PATTERN = load()
PAT = regex.compile(PATTERN)


def bpe(piece: bytes) -> list[int]:
    parts = [piece[i:i + 1] for i in range(len(piece))]
    while len(parts) > 1:
        best_rank, best_i = None, -1
        for i in range(len(parts) - 1):
            r = RANKS.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_rank is None:
            break
        parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return [RANKS[p] for p in parts]


def bpe_cuts(piece: bytes) -> list[int]:
    """The Rust `slices` version translated back: keep cut positions instead of joined parts.
    In Python this does not avoid copies, because piece[a:b] makes a new bytes object."""
    cuts = list(range(len(piece) + 1))
    while len(cuts) > 2:
        best_rank, best_i = None, -1
        for i in range(len(cuts) - 2):
            r = RANKS.get(piece[cuts[i]:cuts[i + 2]])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_rank is None:
            break
        del cuts[best_i + 1]
    return [RANKS[piece[a:b]] for a, b in zip(cuts, cuts[1:])]


def encode(text: str) -> list[int]:
    out: list[int] = []
    for m in PAT.finditer(text):
        piece = m.group().encode("utf-8")
        r = RANKS.get(piece)
        if r is not None:
            out.append(r)
        else:
            out.extend(bpe(piece))
    return out


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data = sys.argv[2] if len(sys.argv) > 2 else "books"
    docs: list[str] = json.loads((HERE / "corpus" / f"docs-{data}.json").read_text())
    t = time.perf_counter()
    if workers == 1:
        tokens = [encode(d) for d in docs]
    else:
        with Pool(workers) as pool:
            tokens = pool.map(encode, docs, chunksize=64)
    secs = time.perf_counter() - t
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"python-{data}-{workers}.json").write_text(json.dumps(tokens))
    print(json.dumps({"impl": "python", "data": data, "workers": workers, "seconds": secs,
                      "tokens": sum(map(len, tokens))}))


if __name__ == "__main__":
    main()
