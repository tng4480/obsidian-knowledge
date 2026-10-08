"""Read-only real-vault check of quote-based PDF tagging.

Copies a few real PDFs into a scratch vault (the real vault is never written), indexes them with the real extractor,
then simulates a user selecting text in a PDF viewer: random word windows are taken from the PDF's *raw* page text
(PyMuPDF ``get_text()``: PDF line breaks, hyphenation, ligatures, math glyphs: not what our extractor produced) and
used as quotes. For each one we ask: was it matched, to how many sections, and does the matched section really
contain the words?

    uv run python eval/smoke_pdf_tags.py [--pdfs 6] [--per-pdf 25] [--words 12] [--seed 1]

Needs ACADMEM_VAULT. Uses a hashing embedder (matching does not involve embeddings), so it takes about a minute.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import fitz  # noqa: E402
from conftest import FakeEmbedder  # noqa: E402

from acadmem.config import Config  # noqa: E402
from acadmem.index import sql_quote  # noqa: E402
from acadmem.indexer import Indexer  # noqa: E402
from acadmem.ingest.discovery import discover  # noqa: E402
from acadmem.pdf_tags import MAX_QUOTE_MATCHES, find_passages, load_chunks, normalize_words  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", type=int, default=6)
    ap.add_argument("--per-pdf", type=int, default=25)
    ap.add_argument("--words", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    vault = os.environ.get("ACADMEM_VAULT")
    if not vault or not Path(vault).is_dir():
        print("Set ACADMEM_VAULT to the real vault.")
        return 2
    rng = random.Random(args.seed)
    real = Config(vault_path=Path(vault), index_dir=Path(tempfile.mkdtemp()))
    pdfs = [f for f in discover(real) if f.source_path.lower().endswith(".pdf")]
    rng.shuffle(pdfs)
    chosen = sorted(pdfs[: args.pdfs], key=lambda f: f.source_path)

    work = Path(tempfile.mkdtemp(prefix="acadmem-pdftags-"))
    try:
        scratch = work / "vault"
        for f in chosen:
            dest = scratch / f.source_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f.path, dest)
        cfg = Config(vault_path=scratch, index_dir=work / "index")
        indexer = Indexer(cfg, embedder=FakeEmbedder())
        report = indexer.run(full=True)
        print(report.summary())

        totals = dict(windows=0, matched=0, none=0, vague=0, wrong=0, single=0, multi=0, header=0)
        for f in chosen:
            doc = indexer.index.get_document(source_path=f.source_path)
            if not doc or not doc.get("chunk_count"):
                print(f"  skip {f.source_path}: no indexed text")
                continue
            rows = load_chunks(indexer.index, doc)
            by_id = {r["chunk_id"]: r for r in rows}
            pages = fitz.open(f.path)
            stats = dict(windows=0, matched=0, none=0, vague=0, wrong=0)
            for _ in range(args.per_pdf):
                number = rng.randrange(len(pages))
                words = pages[number].get_text().split()
                if len(words) < args.words + 2:
                    continue
                start = rng.randrange(0, len(words) - args.words)
                quote = " ".join(words[start : start + args.words])
                if len(normalize_words(quote)) < 4:
                    continue
                stats["windows"] += 1
                match = find_passages(rows, number + 1, number + 1, quote)  # as a user would: page + quote
                if match.ok:
                    stats["matched"] += 1
                    totals["single" if len(match.passages) == 1 else "multi"] += 1
                    need = set(normalize_words(quote))
                    have = set()
                    for p in match.passages:
                        have |= set(normalize_words(by_id[p.chunk_id]["text"] + " " + " ".join(by_id[p.chunk_id].get("heading_path") or [])))
                    if len(need & have) / len(need) < 0.7:
                        stats["wrong"] += 1
                elif "distinctive" in (match.reason or ""):
                    stats["vague"] += 1
                else:
                    stats["none"] += 1
                    if "WORKED REVISION GUIDE" in quote or "WORKED STUDY GUIDE" in quote:
                        totals["header"] += 1  # running page header: stripped as boilerplate on purpose
                    if os.environ.get("SHOW_MISSES") and "WORKED " not in quote:
                        print(f"      miss {Path(f.source_path).name} p.{number + 1}: {quote!r}")
                        longest = max(words[start : start + args.words], key=len)
                        for r in rows:
                            if r["page_start"] <= number + 1 <= (r["page_end"] or r["page_start"]) and longest in (r["text"] or ""):
                                i = r["text"].index(longest)
                                print(f"          ours: {r['text'][max(0, i - 70): i + 110]!r}")
                                break
            for k, v in stats.items():
                totals[k] += v
            print(f"  {f.source_path}: {stats['matched']}/{stats['windows']} matched, {stats['none']} not found, {stats['vague']} too vague, {stats['wrong']} wrong")

        n = max(1, totals["windows"])
        print("\nTOTAL")
        print(f"  windows tried      {totals['windows']}")
        print(f"  matched            {totals['matched']} ({100 * totals['matched'] / n:.0f}%)  [one section: {totals['single']}, several: {totals['multi']}]")
        print(f"  not found          {totals['none']} ({100 * totals['none'] / n:.0f}%), of which {totals['header']} are the running page header (deliberately not indexed)")
        body = n - totals["header"]
        print(f"  body text matched  {totals['matched']} of {body} ({100 * totals['matched'] / max(1, body):.0f}%)")
        print(f"  refused as vague   {totals['vague']}")
        print(f"  matched but wrong  {totals['wrong']}   (matched section lacks >=30% of the words)")
        print(f"  (MAX_QUOTE_MATCHES={MAX_QUOTE_MATCHES}; quotes are {args.words} words taken at random from raw page text)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
