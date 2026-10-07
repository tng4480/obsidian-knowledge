"""Show how a document was chunked in the index (to debug retrieval misses).

    uv run acadmem ... ; uv run python eval/inspect_chunks.py --index-dir <dir> "3F1/Notes/Z-transform checklist.md"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from acadmem.config import load_config
from acadmem.index import CHUNK_COLUMNS, VectorIndex, sql_quote


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.environ.get("ACADMEM_VAULT"))
    ap.add_argument("--index-dir")
    ap.add_argument("--chars", type=int, default=160)
    ap.add_argument("source_path")
    args = ap.parse_args()
    cfg = load_config(vault_path=args.vault, index_dir=args.index_dir)
    rows = VectorIndex(cfg).scan(f"source_path = {sql_quote(args.source_path)}", columns=CHUNK_COLUMNS)
    for r in sorted(rows, key=lambda r: r["chunk_index"]):
        pages = f" p.{r['page_start']}-{r['page_end']}" if r["page_start"] else ""
        print(f"[{r['chunk_index']}] {r['token_count']} tok{pages} | {r['heading']}")
        print("     " + " ".join(r["text"].split())[: args.chars])


if __name__ == "__main__":
    main()
