# Python vs Rust on OpenAI's tokenizer, measured

The benchmark behind the Shree AI video *Python vs Rust: I Timed OpenAI's Tokenizer on 7.8 Million Words*.
Every number in the video comes from `results/20261105-235126.json`.

## Run it

```sh
pip install tiktoken regex
python3 export_vocab.py          # OpenAI's o200k_base regex + vocabulary, from tiktoken
python3 fetch_books.py           # 41 public-domain books from Project Gutenberg
python3 make_corpus.py && python3 make_hindi.py && python3 make_code.py
(cd tok_rust && cargo build --release)
python3 bench.py 5               # every config 5 times, each output checked against tiktoken
```

## What is compared

GPT-4o's tokenizer (`o200k_base`: OpenAI's regex and 199,998-entry vocabulary, exported from tiktoken to
`o200k.pattern` and `o200k.ranks` by `export_vocab.py`), written the same way in each language:

| Name | File | What it is |
|---|---|---|
| python | `tok_python.py` | plain Python: the `regex` module (C) cuts pieces; the merge loop is Python |
| rust-copies | `tok_rust` `copies` | the Python program line for line in Rust: every part is its own `Vec<u8>`, every pair is glued into a new vector to look it up |
| rust-slices | `tok_rust` `slices` | the same algorithm with slices: one byte string per piece and a list of cut positions; nothing is copied |
| tiktoken | `tok_tiktoken.py` | OpenAI's tiktoken 0.13.0 called from Python (Rust core). Also the reference output |

All use the same merge algorithm (scan every adjacent pair, join the lowest-ranked one, repeat), the one in tiktoken's
`_byte_pair_merge`. Standard `HashMap` in Rust, no custom allocator.

## Datasets (`corpus/`, not committed: rebuilt by the scripts)

| Name | Built by | Size |
|---|---|---|
| books | `fetch_books.py` + `make_corpus.py`: 41 Project Gutenberg books, licence header/footer removed, one paragraph per document | 174,349 docs, 43.9 MB, 7.79 M words, 10,837,266 tokens |
| hindi | `make_hindi.py`: 30 Hindi Wikipedia articles (CC BY-SA), one paragraph per document | 1,205 docs, 2.1 MB, 151 k words, 264,535 tokens |
| code | `make_code.py`: every `.py` in the Python 3.13 standard library except tests and idlelib | 573 files, 10.5 MB, 2,677,689 tokens |

## Method

`bench.py 5`: every configuration 5 times, median reported. Load time is not counted (the clock starts after the
vocabulary and the documents are in memory). Each output is compared with tiktoken's, token for token, and the run
aborts on any difference. Peak memory is the summed RSS of the process tree, sampled every 20 ms in a sixth run, so
the sampler does not take a core from the timed runs. Memory is only meaningful for runs that last well over 20 ms
(books); the short Hindi runs are under-sampled.

`merge_only_python.py` and `tok_rust ... merge-only` time the merge loop alone, on the pieces that are not one
vocabulary entry, cut out before the clock starts. `split_python.py` times Python's regex pass, regex plus lookups,
and the whole encode.

Parallel: Python uses `multiprocessing.Pool` (18 processes, spawn start method, each loads the vocabulary); Rust uses
rayon with one regex clone per thread. Rust scaled best at 6 threads (the M5 Max's 6 fastest cores); 18 threads was
slower, and both are in the results.

## Results (`results/20261105-235126.json`, Apple M5 Max, Python 3.13.13, rustc 1.97)

| books, median of 5 | time | peak memory |
|---|---|---|
| python, 1 core | 5.80 s | 458 MB |
| rust-copies, 1 core | 1.47 s | 228 MB |
| rust-slices, 1 core | 1.01 s | 212 MB |
| tiktoken, 1 thread | 1.17 s | 805 MB |
| python, 18 processes | 0.70 s | 1,990 MB |
| rust-slices, 6 threads | 0.31 s | 236 MB |
| rust-slices, 18 threads | 0.50 s | 229 MB |
| tiktoken, 18 threads (`encode_ordinary_batch`) | 1.72 s | 971 MB |

Merge loop only (books, 782,000 pieces of 9,817,827): python 2.33 s, python with cut positions 2.15 s, rust-copies
0.74 s, rust-slices 0.30 s.

Python split (books, `results/python-split.json`): regex pass 2.64 s, + lookups 3.14 s, whole 5.73 s.

Python ÷ rust-slices, 1 core: books 5.7×, hindi 6.6×, code 5.4×. On hindi, tiktoken is 15× faster than Python and
2.2× faster than rust-slices.

Not tried: Python's free-threaded build, PyPy, a Rust version with tiktoken's further optimisations (FxHash, no
per-piece allocation). Corrections and faster honest versions (in either language) are welcome as issues or pull requests.

## Round 3: making the Rust faster (`bench_opt.py`, `results/opt-20261106-173037.json`)

Each step on top of the last, one core, median of 5, every output identical to tiktoken's:

| Step | books | hindi | code |
|---|---|---|---|
| `slices` (round 2) | 0.991 s | 0.102 s | 0.284 s |
| + FxHash (`slices+fx`) | 0.895 s | 0.080 s | 0.259 s |
| + tiktoken's cached-rank merge loop (`cached+fx`) | 0.814 s | 0.039 s | 0.240 s |
| + `regex` crate, o200k's look-ahead applied by hand (`cached+fx+fastre`) | 0.609 s | 0.049 s | 0.140 s |
| + target-cpu=native | 0.615 s | 0.049 s | 0.141 s |
| + mimalloc (`--features mimalloc`) | 0.597 s | 0.049 s | 0.139 s |
| + panic=abort | 0.604 s | 0.049 s | 0.139 s |
| + PGO (needs `rustup component add llvm-tools`) | 0.560 s | 0.050 s | 0.127 s |
| all build options | 0.541 s | 0.047 s | 0.123 s |

On Hindi the `regex` crate splits 2× slower than `fancy-regex` (0.020 vs 0.010 s), so `cached+fx` is the fastest there.
On 6 threads the fully optimised build took 0.335 s, no faster than `slices` (0.310 s).

`python3 fuzz.py [docs] [seed]` checks every variant against tiktoken on random text made of every kind of
whitespace, line breaks, apostrophes, digits, Devanagari and emoji (40,026 strings with seeds 1 and 2: all match).
