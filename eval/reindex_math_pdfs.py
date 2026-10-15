"""Re-index the browser-printed PDFs in the vault, whose equations are recovered from vector outlines (see
ingest/pdf/mathglyphs), or just the files whose path contains one of the given names.

Incremental indexing skips unchanged files, so after the extractor changed this re-extracts just those PDFs.
Usage: ACADMEM_VAULT=... python eval/reindex_math_pdfs.py [name-substring ...]
"""
import sys
import time

import pymupdf

from acadmem.cli import build_parser, _config
from acadmem.ingest.discovery import discover
from acadmem.ingest.pdf import mathglyphs
from acadmem.indexer import Indexer, IndexReport


def browser_printed(path) -> bool:
    with pymupdf.open(str(path)) as doc:
        return mathglyphs.drawn_by_browser(doc.metadata)


def main(names: list[str]) -> None:
    cfg = _config(build_parser().parse_args(["index"]))
    indexer = Indexer(cfg)
    started = time.time()
    todo = [df for df in discover(cfg) if df.path.suffix.lower() == ".pdf" and (not names or any(n in df.source_path for n in names))]
    print(len(todo), "pdfs to check", flush=True)
    report = IndexReport(mode="math-reindex")
    done = 0
    with indexer._lock:
        indexer._prepare()
        for df in todo:
            if not names and not browser_printed(df.path):
                continue
            t = time.time()
            ok = indexer._safe_index_file(df, report)
            done += 1
            print(f"{'ok ' if ok else 'ERR'} {df.source_path} {time.time() - t:.1f}s", flush=True)
        indexer.apply_pdf_tags()
        indexer.index.refresh_fts()
    print(f"re-indexed {done} pdfs, {report.chunks_written} chunks, {len(report.errors)} errors in {time.time() - started:.0f}s", flush=True)
    for e in report.errors:
        print(e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
