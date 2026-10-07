"""Dump per-page plain text of every vault PDF to a folder, so benchmark labels can be written (and
re-checked) from ground truth rather than from the retrieval system's own output.

    uv run python eval/dump_text.py --vault <vault> --out <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pymupdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.environ.get("ACADMEM_VAULT"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vault, out = Path(args.vault), Path(args.out)
    for pdf in sorted(vault.rglob("*.pdf")):
        rel = pdf.relative_to(vault)
        if any(part.startswith(".") for part in rel.parts):
            continue
        target = out / rel.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(pdf) as doc, open(target, "w", encoding="utf-8") as fh:
            for i, page in enumerate(doc, 1):
                fh.write(f"\n=== PAGE {i} ===\n{page.get_text()}")
    print("done")


if __name__ == "__main__":
    main()
