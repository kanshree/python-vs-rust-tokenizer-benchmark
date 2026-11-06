"""Build the benchmark dataset: 41 Project Gutenberg books (every corpus/pg*.txt), licence header/footer removed,
split into paragraphs. Each paragraph is one document, the way a training set is a list of documents.
Writes corpus/docs-books.json (a JSON list of strings)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent / "corpus"
BOOKS = sorted(HERE.glob("pg*.txt"))


def body(text: str) -> str:
    start = re.search(r"\*\*\* START OF TH[EI]S? PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* END OF TH[EI]S? PROJECT GUTENBERG EBOOK", text)
    assert start and end, "Gutenberg markers not found"
    return text[start.end():end.start()]


docs: list[str] = []
for path in BOOKS:
    text = body(path.read_text(encoding="utf-8")).replace("\r\n", "\n")
    docs += [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
(HERE / "docs-books.json").write_text(json.dumps(docs, ensure_ascii=False))
total = sum(len(d.encode()) for d in docs)
print(f"{len(docs)} documents, {total:,} bytes, {sum(len(d.split()) for d in docs):,} words")
