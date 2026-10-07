"""Show the top results (cosine similarity and the mode's native rank score) for a query in each mode, and where a given source ranks.

    uv run python eval/debug_query.py --index-dir <dir> --vault <vault> "query" --find "3F1/Notes/Z-transform checklist.md"
"""

from __future__ import annotations

import argparse
import os

from acadmem.config import load_config
from acadmem.search import Searcher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.environ.get("ACADMEM_VAULT"))
    ap.add_argument("--index-dir")
    ap.add_argument("--find", help="source path to locate in a deep result list")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--modes", default="vector,fts,hybrid")
    ap.add_argument("query")
    args = ap.parse_args()
    searcher = Searcher(load_config(vault_path=args.vault, index_dir=args.index_dir))
    for mode in args.modes.split(","):
        print(f"\n== {mode}")
        for h in searcher.search(args.query, top_k=args.k, mode=mode):
            print(f"  {h.rank:>2} sim {h.score:6.3f}  rank_score {h.rank_score:8.4f}  {h.location[:130]}")
        if args.find:
            deep = searcher.search(args.query, top_k=200, mode=mode)
            ranks = [h.rank for h in deep if h.source_path == args.find]
            best = next((h for h in deep if h.source_path == args.find), None)
            print(f"  -> '{args.find}' ranks {ranks[:5] or 'not in top 200'}" + (f" (sim {best.score:.3f}, rank_score {best.rank_score:.4f})" if best else ""))


if __name__ == "__main__":
    main()
