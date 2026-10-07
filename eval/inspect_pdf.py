"""Print the section tree one PDF extractor produces for a PDF.

    uv run python eval/inspect_pdf.py <pdf> [pymupdf|docling] [--text]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from acadmem.ingest.pdf import PDFDocumentExtractor, get_pdf_extractor


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_text = "--text" in sys.argv
    path = Path(args[0])
    name = args[1] if len(args) > 1 else "pymupdf"
    t = time.time()
    parsed = PDFDocumentExtractor(get_pdf_extractor(name)).parse(path)
    print(f"[{name}] {time.time() - t:.1f}s title={parsed.title!r} pages={parsed.page_count} sections={len(parsed.sections)}")
    for s in parsed.sections:
        chars = sum(len(b.text) for b in s.blocks)
        print(f"{'  ' * s.level}[L{s.level}] {s.title!r} pages={s.page_start}-{s.page_end} blocks={len(s.blocks)} chars={chars}")
        if show_text:
            for b in s.blocks:
                print(f"{'  ' * s.level}   - p{b.page} {b.kind}: {b.text[:140]!r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
