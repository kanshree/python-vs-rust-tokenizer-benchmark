"""Second dataset: 30 Hindi Wikipedia articles (plain-text extracts, CC BY-SA), split into paragraphs.
Each article is cached in corpus/hindi/, so a rate-limited run can be repeated. Writes corpus/docs-hindi.json."""
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

TITLES = ["भारत", "हिन्दी", "महात्मा गांधी", "भारत का इतिहास", "गंगा नदी", "रवीन्द्रनाथ ठाकुर", "भारतीय संविधान", "मुम्बई",
          "दिल्ली", "क्रिकेट", "भगत सिंह", "ताजमहल", "हिमालय", "प्रेमचंद", "भारतीय रेल", "सिन्धु घाटी सभ्यता", "मौर्य राजवंश",
          "मुग़ल साम्राज्य", "स्वामी विवेकानन्द", "सुभाष चन्द्र बोस", "जवाहरलाल नेहरू", "इसरो", "आयुर्वेद", "योग", "महाभारत",
          "रामायण", "कबीर", "तुलसीदास", "बॉलीवुड", "भारतीय राष्ट्रीय कांग्रेस"]
CACHE = Path(__file__).with_name("corpus") / "hindi"
CACHE.mkdir(exist_ok=True)
docs: list[str] = []
for t in TITLES:
    cached = CACHE / f"{t}.txt"
    if cached.exists():
        docs += [x.strip() for x in cached.read_text().split("\n\n") if x.strip()]
        continue
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "titles": t,
                                "format": "json", "redirects": 1})
    got = subprocess.run(["curl", "-sf", "--retry", "3", "-A", "tokenizer-benchmark/0.1",
                          f"https://hi.wikipedia.org/w/api.php?{q}"], capture_output=True)
    if got.returncode:
        print(f"skipped {t} (curl {got.returncode})")
        continue
    raw = got.stdout
    text = "\n\n".join(p.get("extract", "") for p in json.loads(raw)["query"]["pages"].values())
    cached.write_text(text)
    docs += [x.strip() for x in text.split("\n\n") if x.strip()]
    time.sleep(3)
Path(__file__).with_name("corpus").joinpath("docs-hindi.json").write_text(json.dumps(docs, ensure_ascii=False))
print(f"{len(docs)} documents, {sum(len(d.encode()) for d in docs):,} bytes, {sum(len(d.split()) for d in docs):,} words")
