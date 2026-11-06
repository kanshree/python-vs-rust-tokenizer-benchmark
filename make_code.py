"""Third dataset: source code, the kind of text coding models are trained on. Every .py file in this machine's
Python 3.13 standard library (tests excluded), one file per document. Writes corpus/docs-code.json."""
import json
import sysconfig
from pathlib import Path

lib = Path(sysconfig.get_paths()["stdlib"])
docs = [p.read_text(encoding="utf-8", errors="strict") for p in sorted(lib.rglob("*.py"))
        if "test" not in p.parts and "site-packages" not in p.parts and "idlelib" not in p.parts]
Path(__file__).with_name("corpus").joinpath("docs-code.json").write_text(json.dumps(docs, ensure_ascii=False))
print(f"{len(docs)} files, {sum(len(d.encode()) for d in docs):,} bytes, {sum(len(d.splitlines()) for d in docs):,} lines")
