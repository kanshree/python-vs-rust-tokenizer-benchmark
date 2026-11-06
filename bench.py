"""Run every implementation on every dataset, several times; check each output against tiktoken token for token;
record the peak memory of the whole process tree (in a separate run); and time the merge loop on its own.
    python3 bench.py [runs]
Writes results/<timestamp>.json with every timing, the medians and the machine."""
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CORES = int(subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout)
PERF = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout)
DATASETS = ["books", "hindi", "code"]
RUST = str(HERE / "tok_rust/target/release/tok_rust")
PY = sys.executable

# name -> command (without dataset); the parallel runs use every core, and Rust is also tried on the fastest cores only
CONFIGS = {
    "python-1": lambda d: [PY, str(HERE / "tok_python.py"), "1", d],
    "rust-copies-1": lambda d: [RUST, str(HERE), "1", d, "copies"],
    "rust-slices-1": lambda d: [RUST, str(HERE), "1", d, "slices"],
    "tiktoken-1": lambda d: [PY, str(HERE / "tok_tiktoken.py"), "1", d],
    f"python-{CORES}": lambda d: [PY, str(HERE / "tok_python.py"), str(CORES), d],
    f"rust-slices-{PERF}": lambda d: [RUST, str(HERE), str(PERF), d, "slices"],
    f"rust-slices-{CORES}": lambda d: [RUST, str(HERE), str(CORES), d, "slices"],
    f"tiktoken-{CORES}": lambda d: [PY, str(HERE / "tok_tiktoken.py"), str(CORES), d],
}
MERGE_ONLY = {
    "python-copies": lambda d: [PY, str(HERE / "merge_only_python.py"), d, "copies"],
    "python-cuts": lambda d: [PY, str(HERE / "merge_only_python.py"), d, "cuts"],
    "rust-copies": lambda d: [RUST, str(HERE), "1", d, "copies", "merge-only"],
    "rust-slices": lambda d: [RUST, str(HERE), "1", d, "slices", "merge-only"],
}


def run(argv: list[str]) -> dict:
    return json.loads(subprocess.run([PY, str(HERE / "peak_memory.py"), *argv], capture_output=True, text=True,
                                     check=True).stdout)


def output_file(name: str, data: str) -> Path:
    impl, workers = name.rsplit("-", 1)
    return HERE / "out" / f"{impl}-{data}-{workers}.json"


results = {"machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True).stdout.strip(),
           "cores": CORES, "performance_cores": PERF, "python": platform.python_version(),
           "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
           "tiktoken": subprocess.run([PY, "-c", "import tiktoken; print(tiktoken.__version__)"], capture_output=True,
                                      text=True).stdout.strip(),
           "runs": RUNS, "date": time.strftime("%Y-%m-%d %H:%M"), "data": {}}
for data in DATASETS:
    docs = json.loads((HERE / "corpus" / f"docs-{data}.json").read_text())
    entry = {"documents": len(docs), "bytes": sum(len(d.encode()) for d in docs),
             "words": sum(len(d.split()) for d in docs), "configs": {}, "merge_only": {}}
    for name, argv in CONFIGS.items():
        # timed runs alone; memory from one more run under the sampler, which takes a core of its own
        outs = [json.loads(subprocess.run(argv(data), capture_output=True, text=True, check=True).stdout)
                for _ in range(RUNS)]
        times = [o["seconds"] for o in outs]
        entry["tokens"] = outs[-1]["tokens"]
        entry["configs"][name] = {"times": times, "median": statistics.median(times),
                                  "peak_mb": run(argv(data))["peak_mb"]}
        print(f"{data:6} {name:16} median {statistics.median(times):7.3f}s  range {min(times):.3f}-{max(times):.3f}"
              f"  peak {entry['configs'][name]['peak_mb']} MB", flush=True)
    ref = json.loads(output_file("tiktoken-1", data).read_text())
    for name in CONFIGS:
        same = json.loads(output_file(name, data).read_text()) == ref
        entry["configs"][name]["matches_tiktoken"] = same
        assert same, f"{name} on {data} does not match tiktoken"
    for name, argv in MERGE_ONLY.items():
        outs = [json.loads(subprocess.run(argv(data), capture_output=True, text=True, check=True).stdout)
                for _ in range(RUNS)]
        entry["merge_only"][name] = {"times": [o["seconds"] for o in outs],
                                     "median": statistics.median(o["seconds"] for o in outs),
                                     "pieces": outs[-1]["pieces"], "tokens": outs[-1]["tokens"],
                                     "all_pieces": outs[-1].get("all_pieces")}
        print(f"{data:6} merge-only {name:14} median {entry['merge_only'][name]['median']:7.3f}s", flush=True)
    print(f"{data}: every implementation matches tiktoken, {entry['tokens']:,} tokens", flush=True)
    results["data"][data] = entry
(HERE / "results").mkdir(exist_ok=True)
path = HERE / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps(results, indent=1))
print("saved", path)
