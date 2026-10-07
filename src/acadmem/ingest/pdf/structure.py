"""PDF structure: per-page blocks + bookmarks -> the shared section tree.

Independent of which ``PDFExtractor`` produced the pages. Two ideas:

* **Boilerplate removal** - running headers/footers/page numbers repeat on most
  pages and would pollute every embedding. Text that sits in header/footer
  positions (first/last blocks of a page) and repeats on many pages is dropped.
  Content in the middle of a page is never dropped by this heuristic.
* **Outline-first hierarchy** - PDF bookmarks, when present, are the authoritative
  section hierarchy (font-size heuristics routinely mis-rank headings). In-page
  headings that merely restate a bookmark are dropped; other in-page headings
  become sub-sections beneath the active bookmark.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from ...models import Block, ParsedContent
from ..links import clean_inline
from ..sections import SectionBuilder
from .base import OutlineEntry, PDFBlock, PDFExtraction, PDFPage

_EDGE_BLOCKS = 2  # how many blocks at the top/bottom of a page count as header/footer zone
_RESTATE_MAX_LEN = 100  # max length of a non-heading block considered a restated bookmark


def _boiler_key(text: str) -> str:
    """Key under which repeated header/footer text is compared across pages.

    Digits are normalised only for short, page-number-like blocks ("12", "Page 3 of 10",
    "p. 4") so those collapse to one key. Longer text is compared verbatim, otherwise real
    content that differs only by a number ("Step 1", "Step 2") would be mistaken for boilerplate.
    """
    t = re.sub(r"[*_`#]", "", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    words = re.findall(r"[^\W\d_]+", t)
    if len(words) <= 1:
        t = re.sub(r"\d+", "#", t)
    return t


def _alnum(text: str) -> str:
    return re.sub(r"[\W_]+", "", clean_inline(text).lower())


def strip_boilerplate(
    pages: list[PDFPage], min_pages: int = 3, ratio: float = 0.5, max_len: int = 200
) -> list[PDFPage]:
    """Drop short blocks in header/footer position that repeat across pages."""
    n = len(pages)
    if n < min_pages:
        return pages

    def edge_blocks(page: PDFPage) -> list[tuple[int, PDFBlock]]:
        blocks = list(enumerate(page.blocks))
        return blocks[:_EDGE_BLOCKS] + blocks[-_EDGE_BLOCKS:]

    counts: Counter[str] = Counter()
    for page in pages:
        keys = {
            _boiler_key(b.text)
            for _, b in edge_blocks(page)
            if b.text.strip() and len(b.text) <= max_len
        }
        counts.update(k for k in keys if k)
    threshold = max(2, math.ceil(ratio * n))
    drop = {k for k, c in counts.items() if c >= threshold}
    if not drop:
        return pages

    out: list[PDFPage] = []
    for page in pages:
        edge_idx = {i for i, _ in edge_blocks(page)}
        kept = [
            b
            for i, b in enumerate(page.blocks)
            if not (i in edge_idx and len(b.text) <= max_len and _boiler_key(b.text) in drop)
        ]
        out.append(PDFPage(page.source, page.page_number, kept))
    return out


@dataclass
class _Event:
    kind: str  # "outline" | "heading" | "block"
    page: int
    entry: OutlineEntry | None = None
    block: PDFBlock | None = None


def build_pdf_content(extraction: PDFExtraction, fallback_title: str = "") -> ParsedContent:
    """Build a :class:`ParsedContent` (sections with page-tagged blocks) from a PDF extraction."""
    pages = strip_boilerplate(extraction.pages)
    page_numbers = {p.page_number for p in pages}
    outline = [e for e in extraction.outline if e.page in page_numbers and e.title.strip()]
    use_outline = bool(outline)

    by_page: dict[int, list[OutlineEntry]] = defaultdict(list)
    for e in outline:
        by_page[e.page].append(e)

    # ---- pass 1: linearise into events, dropping headings that restate a bookmark
    events: list[_Event] = []
    current_norm = ""
    for page in pages:
        entries = by_page.get(page.page_number, [])
        covers = [_alnum(e.title) for e in entries]
        for e in entries:
            events.append(_Event("outline", page.page_number, entry=e))
        if covers:
            current_norm = covers[-1]
        for block in page.blocks:
            if block.heading_level is None:
                if use_outline and len(block.text) <= _RESTATE_MAX_LEN:
                    # a short bold/kicker line that just repeats the bookmark title
                    b = _alnum(block.text)
                    if len(b) >= 4 and (any(b in c for c in covers) or (current_norm and b in current_norm)):
                        continue
                events.append(_Event("block", page.page_number, block=block))
                continue
            if use_outline:
                h = _alnum(block.text)
                if h and (any(h in c for c in covers) or (current_norm and h in current_norm)):
                    continue  # restates the bookmark
            events.append(_Event("heading", page.page_number, block=block))

    heading_levels = [ev.block.heading_level for ev in events if ev.kind == "heading" and ev.block]
    min_h = min((h for h in heading_levels if h is not None), default=1)

    # ---- pass 2: build the tree
    title = (extraction.title or "").strip() or fallback_title
    builder = SectionBuilder(root_title=title)
    outline_level = 0
    for ev in events:
        if ev.kind == "outline" and ev.entry is not None:
            builder.open(clean_inline(ev.entry.title) or "(untitled)", ev.entry.level)
            outline_level = ev.entry.level
        elif ev.kind == "heading" and ev.block is not None:
            level_hint = ev.block.heading_level or 1
            if use_outline:
                level = outline_level + 1 + max(0, level_hint - min_h)
            else:
                level = level_hint
            builder.open(clean_inline(ev.block.text) or "(untitled)", level)
        elif ev.kind == "block" and ev.block is not None:
            builder.add(Block(ev.block.text, ev.block.kind, ev.page))

    return ParsedContent(
        title=title,
        sections=builder.build(),
        page_count=extraction.page_count,
    )
