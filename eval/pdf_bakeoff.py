"""PDF extractor bake-off on real vault PDFs.

Ground truth is the PDF's own text layer (``pymupdf`` raw page text). For each
extractor we measure:

* token recall: fraction of ground-truth word tokens present in the extracted text
  (math-heavy tokens such as ``ΓL=(ZL−50)/(ZL+50)`` are included on purpose)
* digit-token recall: same, restricted to tokens containing digits (numeric results)
* sections / bookmarks recovered, and wall-clock seconds

    uv run python eval/pdf_bakeoff.py <vault> [--n 3] [--extractors pymupdf,docling]
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import pymupdf

from acadmem.ingest.pdf import PDFDocumentExtractor, get_pdf_extractor

_TOKEN = re.compile(r"[^\W_]+(?:[.,][^\W_]+)*", re.U)
_NORMALISE = str.maketrans({"−": "-", "–": "-", "’": "'", "“": '"', "”": '"'})


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text.translate(_NORMALISE))]


def ground_truth(path: Path) -> list[str]:
    with pymupdf.open(str(path)) as doc:
        return tokens(" ".join(page.get_text() for page in doc))


def recall(truth: list[str], got_text: str, only_digits: bool = False) -> float:
    got = set(tokens(got_text))
    pool = [t for t in truth if (any(c.isdigit() for c in t) if only_digits else True)]
    if not pool:
        return 1.0
    return sum(1 for t in pool if t in got) / len(pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vault", type=Path)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extractors", default="pymupdf,docling")
    args = ap.parse_args()

    pdfs = sorted(args.vault.rglob("*.pdf"))
    random.Random(args.seed).shuffle(pdfs)
    sample = pdfs[: args.n]
    names = [n.strip() for n in args.extractors.split(",")]
    extractors = {n: PDFDocumentExtractor(get_pdf_extractor(n)) for n in names}

    totals = {n: {"recall": [], "digits": [], "secs": 0.0, "sections": 0, "pages": 0} for n in names}
    for pdf in sample:
        truth = ground_truth(pdf)
        with pymupdf.open(str(pdf)) as d:
            bookmarks = len(d.get_toc())
        print(f"\n{pdf.name}: {len(truth)} truth tokens, {bookmarks} bookmarks")
        for n, ex in extractors.items():
            t = time.time()
            parsed = ex.parse(pdf)
            secs = time.time() - t
            text = "\n".join(s.text for s in parsed.sections)
            r, rd = recall(truth, text), recall(truth, text, only_digits=True)
            totals[n]["recall"].append(r)
            totals[n]["digits"].append(rd)
            totals[n]["secs"] += secs
            totals[n]["sections"] += len(parsed.sections)
            totals[n]["pages"] += parsed.page_count or 0
            print(f"  {n:8s} recall={r:.3f} digit-recall={rd:.3f} sections={len(parsed.sections):3d} {secs:6.1f}s")

    print("\n== summary (mean over sample) ==")
    for n, t in totals.items():
        k = max(1, len(t["recall"]))
        print(
            f"{n:8s} token-recall={sum(t['recall']) / k:.3f} digit-recall={sum(t['digits']) / k:.3f} "
            f"sec/page={t['secs'] / max(1, t['pages']):.2f} sections/doc={t['sections'] / k:.1f}"
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
