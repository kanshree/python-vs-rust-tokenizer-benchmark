"""Run a command and report the peak of the summed resident memory of it and all its child processes,
sampled every 20 ms.   python3 peak_memory.py <command...>"""
import json
import subprocess
import sys
import time


def tree_rss_kb(root: int) -> int:
    rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout.split("\n")
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for r in rows:
        if r.strip():
            pid, ppid, kb = map(int, r.split())
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = kb
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += kids.get(p, [])
    return total


proc = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, text=True)
peak = 0
while proc.poll() is None:
    peak = max(peak, tree_rss_kb(proc.pid))
    time.sleep(0.02)
out = json.loads(proc.stdout.read())
out["peak_mb"] = round(peak / 1024)
print(json.dumps(out))
