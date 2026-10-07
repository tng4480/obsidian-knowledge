"""Find which vault PDFs / pages contain a phrase (from the text layer). Used to write and re-check benchmark labels.

    uv run python eval/find_pages.py --text-dir <dir from dump_text.py> "phrase one" "phrase two"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--within", default="", help="only files whose relative path contains this")
    ap.add_argument("phrases", nargs="+")
    args = ap.parse_args()
    root = Path(args.text_dir)
    files = [p for p in sorted(root.rglob("*.txt")) if args.within.lower() in str(p.relative_to(root)).lower()]
    for phrase in args.phrases:
        needle = norm(phrase)
        print(f"\n## {phrase!r}")
        for path in files:
            pages = re.split(r"\n=== PAGE (\d+) ===\n", path.read_text(encoding="utf-8"))
            hits = [int(pages[i]) for i in range(1, len(pages), 2) if needle in norm(pages[i + 1])]
            if hits:
                name = str(path.relative_to(root)).replace("\\", "/").replace(".txt", ".pdf")
                print(f"  {name}: pages {hits}")


if __name__ == "__main__":
    main()
