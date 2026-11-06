"""The same job done with OpenAI's tiktoken: Python on top, a Rust core underneath. This is also the reference
output every other implementation must match token for token."""
import json
import sys
import time
from pathlib import Path

import tiktoken

HERE = Path(__file__).parent


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data = sys.argv[2] if len(sys.argv) > 2 else "books"
    docs: list[str] = json.loads((HERE / "corpus" / f"docs-{data}.json").read_text())
    enc = tiktoken.get_encoding("o200k_base")
    enc.encode_ordinary("warm up")
    t = time.perf_counter()
    if workers == 1:
        tokens = [enc.encode_ordinary(d) for d in docs]
    else:
        tokens = enc.encode_ordinary_batch(docs, num_threads=workers)
    secs = time.perf_counter() - t
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"tiktoken-{data}-{workers}.json").write_text(json.dumps(tokens))
    print(json.dumps({"impl": "tiktoken", "data": data, "workers": workers, "seconds": secs,
                      "tokens": sum(map(len, tokens))}))


if __name__ == "__main__":
    main()
