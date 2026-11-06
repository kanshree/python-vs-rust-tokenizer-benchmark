"""Export OpenAI's o200k_base (GPT-4o) regex and vocabulary from tiktoken to o200k.pattern and o200k.ranks, so the
plain Python and Rust tokenizers use exactly the same vocabulary.   python3 export_vocab.py"""
import base64
from pathlib import Path

import tiktoken

here = Path(__file__).parent
enc = tiktoken.get_encoding("o200k_base")
with open(here / "o200k.ranks", "w") as f:
    for tok, rank in sorted(enc._mergeable_ranks.items(), key=lambda x: x[1]):
        f.write(f"{base64.b64encode(tok).decode()} {rank}\n")
(here / "o200k.pattern").write_text(enc._pat_str)
print(f"{len(enc._mergeable_ranks)} ranks")
