"""PDF ingestion: pluggable ``PDFExtractor`` + shared structure builder."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ...models import ParsedContent
from .base import OutlineEntry, PDFBlock, PDFExtraction, PDFExtractor, PDFPage
from .structure import build_pdf_content, strip_boilerplate

__all__ = [
    "OutlineEntry",
    "PDFBlock",
    "PDFDocumentExtractor",
    "PDFExtraction",
    "PDFExtractor",
    "PDFPage",
    "build_pdf_content",
    "get_pdf_extractor",
    "strip_boilerplate",
]


def get_pdf_extractor(name: str = "pymupdf", **kwargs) -> PDFExtractor:
    """Factory for the configured PDF backend (lazy imports keep startup fast)."""
    key = name.lower()
    if key == "pymupdf":
        from .pymupdf_adapter import PyMuPDFExtractor

        return PyMuPDFExtractor(**kwargs)
    if key == "docling":
        from .docling_adapter import DoclingExtractor

        return DoclingExtractor(**kwargs)
    raise ValueError(f"Unknown PDF extractor: {name!r} (expected 'pymupdf' or 'docling')")


class PDFDocumentExtractor:
    """Adapts a :class:`PDFExtractor` to the generic ``Extractor`` interface."""

    extensions: ClassVar[tuple[str, ...]] = (".pdf",)
    file_type: ClassVar[str] = "pdf"

    def __init__(self, pdf_extractor: PDFExtractor) -> None:
        self.pdf_extractor = pdf_extractor

    def parse(self, path: Path) -> ParsedContent:
        extraction = self.pdf_extractor.extract(Path(path))
        return build_pdf_content(extraction, fallback_title=Path(path).stem)
