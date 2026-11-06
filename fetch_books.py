"""Download the 41 Project Gutenberg books the benchmark uses into corpus/ (public domain in the US).
    python3 fetch_books.py"""
import subprocess
import time
from pathlib import Path

IDS = [100, 105, 1080, 11, 1184, 120, 1232, 1260, 1342, 135, 1400, 145, 158, 161, 1661, 1727, 174, 219, 25344, 2554,
       2591, 2600, 2701, 28054, 345, 35, 36, 43, 4300, 5200, 580, 6130, 730, 74, 76, 766, 768, 84, 8800, 98, 996]
out = Path(__file__).with_name("corpus")
out.mkdir(exist_ok=True)
for i in IDS:
    f = out / f"pg{i}.txt"
    if not f.exists():
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(f), f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"],
                       check=True)
        time.sleep(1)
print(f"{len(list(out.glob('pg*.txt')))} books in {out}")
