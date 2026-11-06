"""Random text built from the characters that decide where o200k's pattern cuts (every kind of whitespace, line
breaks, digits, punctuation, apostrophes, capitals, accents, Devanagari, emoji), tokenized by every Rust variant
and compared with tiktoken, document by document.
    python3 fuzz.py [docs] [seed]"""
import json
import random
import subprocess
import sys
from pathlib import Path

import tiktoken

HERE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
ATOMS = [" ", "  ", "\t", "\n", "\r\n", "\r", " ", " ", "　", "​", "\x0b", "\x0c", " ", "\u0085",
         "a", "The", "hello", "WORLD", "McDonald", "naïve", "é", "'s", "'S", "'ll", "'", "’s", "0", "12", "12345", "٣",
         ".", ",", "!?", "...", "--", "/", "(", ")", "$", "#", "@", "\\", "_", "नमस्ते", "भारत", "्", "ि", "😀", "👍🏽", "中文",
         "Ω", "ß", "İ", "x̃"]
docs = ["".join(rng.choice(ATOMS) for _ in range(rng.randint(1, 40))) for _ in range(N)]
docs += ["", " ", "  ", " \n", "\n ", "a ", "a  ", "a  b", "a \t b", "  x", "x　　y", "  \r\n  x", "\t\tx"]
(HERE / "corpus" / "docs-fuzz.json").write_text(json.dumps(docs, ensure_ascii=False))
enc = tiktoken.get_encoding("o200k_base")
ref = [enc.encode_ordinary(d) for d in docs]
binary = HERE / "tok_rust" / "target" / "release" / "tok_rust"
bad = 0
for style in ["copies", "slices", "slices+fx", "cached+fx", "slices+fastre", "cached+fx+fastre"]:
    subprocess.run([str(binary), str(HERE), "1", "fuzz", style], check=True, capture_output=True)
    got = json.loads((HERE / "out" / f"rust-{style}-fuzz-1.json").read_text())
    diff = [i for i, (a, b) in enumerate(zip(got, ref)) if a != b]
    bad += len(diff)
    print(f"{style:18s} {len(docs) - len(diff)}/{len(docs)} match", *(f"\n   {docs[i]!r}" for i in diff[:5]))
sys.exit(1 if bad else 0)
