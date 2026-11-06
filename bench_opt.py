"""Round 3: how much faster can the Rust tokenizer get? Each optimisation is added one at a time and timed on one
core (5 runs, median), and every output is checked token for token against tiktoken, as in bench.py.
    python3 bench_opt.py [runs]
Needs `python3 bench.py` to have run once (for tiktoken's reference outputs in out/), and for PGO the llvm-tools
component (`rustup component add llvm-tools`). Writes results/opt-<time>.json."""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
CRATE = HERE / "tok_rust"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DATA = ["books", "hindi", "code"]
BEST = "cached+fx+fastre"

# (name, style, build). Code changes first, each on top of the last; then build options on top of the best code.
STEPS = [
    ("slices (the video's version)", "slices", "base"),
    ("+ FxHash", "slices+fx", "base"),
    ("+ cached pair ranks (tiktoken's loop)", "cached+fx", "base"),
    ("+ regex crate, look-ahead by hand", BEST, "base"),
    ("+ target-cpu=native", BEST, "native"),
    ("+ mimalloc", BEST, "mimalloc"),
    ("+ panic=abort", BEST, "abort"),
    ("+ PGO", BEST, "pgo"),
    ("all build options", BEST, "all"),
]
NATIVE = "-C target-cpu=native"


def profdata() -> Path | None:
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], capture_output=True, text=True).stdout.strip()
    found = list(Path(sysroot).glob("lib/rustlib/*/bin/llvm-profdata"))
    return found[0] if found else None


def build(kind: str) -> Path:
    """Build the crate with one set of build options into its own target dir; returns the binary."""
    target = CRATE / "target" / f"opt-{kind}"
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    flags, features = [], []
    if kind in ("native", "all"):
        flags.append(NATIVE)
    if kind in ("mimalloc", "all"):
        features = ["--features", "mimalloc"]
    if kind in ("abort", "all"):
        env["CARGO_PROFILE_RELEASE_PANIC"] = "abort"
    binary = target / "release" / "tok_rust"
    if kind in ("pgo", "all"):
        tool = profdata()
        if tool is None:
            raise SystemExit("PGO needs llvm-profdata: rustup component add llvm-tools")
        prof = target / "pgo-data"
        shutil.rmtree(prof, ignore_errors=True)
        # 1. an instrumented build, run once on every dataset (the training run is not timed)
        env["RUSTFLAGS"] = " ".join(flags + [f"-Cprofile-generate={prof}"])
        subprocess.run(["cargo", "build", "--release", "-q", *features], cwd=CRATE, env=env, check=True)
        for d in DATA:
            subprocess.run([str(binary), str(HERE), "1", d, BEST], check=True, capture_output=True)
        subprocess.run([str(tool), "merge", "-o", str(prof / "merged.profdata"), str(prof)], check=True)
        # 2. the real build, optimised with what the run measured
        env["RUSTFLAGS"] = " ".join(flags + [f"-Cprofile-use={prof / 'merged.profdata'}"])
    else:
        env["RUSTFLAGS"] = " ".join(flags)
    subprocess.run(["cargo", "build", "--release", "-q", *features], cwd=CRATE, env=env, check=True)
    return binary


def run(binary: Path, data: str, style: str, workers: int = 1) -> float:
    out = subprocess.run([str(binary), str(HERE), str(workers), data, style], capture_output=True, text=True, check=True).stdout
    secs = json.loads(out)["seconds"]
    got = json.loads((HERE / "out" / f"rust-{style}-{data}-{workers}.json").read_text())
    ref = json.loads((HERE / "out" / f"tiktoken-{data}-1.json").read_text())
    if got != ref:
        raise SystemExit(f"{style} on {data}: output differs from tiktoken")
    return secs


def main() -> None:
    builds: dict[str, Path] = {}
    results = {"runs": RUNS, "steps": []}
    for name, style, kind in STEPS:
        if kind not in builds:
            print(f"building {kind}…", flush=True)
            builds[kind] = build(kind)
        row = {"name": name, "style": style, "build": kind, "data": {}}
        for d in DATA:
            times = [run(builds[kind], d, style) for _ in range(RUNS)]
            row["data"][d] = {"median": statistics.median(times), "times": times}
        results["steps"].append(row)
        print(f"{name:42s} " + "  ".join(f"{d} {row['data'][d]['median']:.3f}s" for d in DATA), flush=True)
    # the best build on the performance cores, as in the video's 6-thread run
    perf = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout or 6)
    times = [run(builds["all"], "books", BEST, perf) for _ in range(RUNS)]
    results["parallel"] = {"workers": perf, "build": "all", "style": BEST, "median": statistics.median(times), "times": times}
    print(f"all options, {perf} threads, books: {statistics.median(times):.3f}s")
    out = HERE / "results" / f"opt-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
