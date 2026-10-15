"""pymupdf4llm-backed ``PDFExtractor`` (fast, text-layer PDFs, per-page output)."""

from __future__ import annotations

import re
from pathlib import Path

from ..markdown import HEADING, iter_markdown_blocks
from .base import OutlineEntry, PDFBlock, PDFExtraction, PDFExtractor, PDFPage

_PICTURE_TEXT = re.compile(
    r"<!--\s*Start of picture text\s*-->\s*(.*?)\s*<!--\s*End of picture text\s*-->", re.S | re.I
)
_BR = re.compile(r"\s*<br\s*/?>\s*", re.I)


def clean_page_markdown(text: str) -> str:
    """Normalise pymupdf4llm page Markdown.

    Text recovered from diagrams is wrapped in HTML comments with ``<br>`` separators;
    turn it into one labelled line so it is searchable but clearly figure text.
    """

    def _figure(m: re.Match[str]) -> str:
        labels = "; ".join(p for p in _BR.split(m.group(1)) if p.strip())
        return f"*Figure text:* {labels}" if labels else ""

    return _PICTURE_TEXT.sub(_figure, text)


def read_outline(path: Path) -> tuple[list[OutlineEntry], str, int]:
    """Read PDF bookmarks, metadata title and page count with PyMuPDF."""
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        outline = [
            OutlineEntry(level=int(level), title=str(title).strip(), page=int(page))
            for level, title, page in doc.get_toc(simple=True)
            if int(page) >= 1
        ]
        title = (doc.metadata or {}).get("title") or ""
        return outline, str(title).strip(), doc.page_count


class PyMuPDFExtractor(PDFExtractor):
    name = "pymupdf"

    def __init__(self, use_ocr: bool = False, strip_headers_footers: bool = True, recover_math: bool = True) -> None:
        # OCR is off by default: the study material has a text layer, and OCR of
        # diagrams adds noise. Enable for scanned PDFs.
        self.use_ocr = use_ocr
        self.strip_headers_footers = strip_headers_footers
        # Some PDFs (Chrome/Skia prints of KaTeX maths) draw equations as vector outlines with no text layer, which
        # every text extractor drops. Recover them from the outlines (fast, no OCR model) before extraction.
        self.recover_math = recover_math

    def extract(self, path: Path) -> PDFExtraction:
        import pymupdf
        import pymupdf4llm

        path = Path(path)
        doc = None
        formulas: dict[int, list] = {}
        source: object = str(path)
        if self.recover_math:
            from .mathglyphs import drawn_by_browser, recover_math

            doc = pymupdf.open(str(path))
            formulas = recover_math(doc) if drawn_by_browser(doc.metadata) else {}
            if formulas:
                source = doc
        chunks = pymupdf4llm.to_markdown(
            source,
            page_chunks=True,
            use_ocr=self.use_ocr,
            header=not self.strip_headers_footers,
            footer=not self.strip_headers_footers,
        )
        pages: list[PDFPage] = []
        for chunk in chunks:
            number = int(chunk["metadata"].get("page_number") or (len(pages) + 1))
            blocks: list[PDFBlock] = []
            body = chunk.get("text", "")
            if number - 1 in formulas:
                # the layout model drops blocks it calls "formula"; put the recovered text back
                from .mathglyphs import fill_formula_boxes

                body = fill_formula_boxes(body, chunk.get("page_boxes"), formulas[number - 1])
            for kind, text, level in iter_markdown_blocks(clean_page_markdown(body)):
                if kind == HEADING:
                    blocks.append(PDFBlock(text, heading_level=level))
                else:
                    blocks.append(PDFBlock(text, kind=kind))
            pages.append(PDFPage(source=path.name, page_number=number, blocks=blocks))
        if doc is not None:
            doc.close()
        outline, title, _ = read_outline(path)
        return PDFExtraction(source=path.name, pages=pages, outline=outline, title=title, extractor=self.name)
