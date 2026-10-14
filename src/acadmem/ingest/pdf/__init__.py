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
    "OcrRoutingExtractor",
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


MIN_PAGE_TEXT_CHARS = 50  # a page with less text than this but with a picture on it is treated as a scan


def has_scanned_pages(path: Path) -> bool:
    """True when at least one page has (almost) no text but does carry an image: a scanned page."""
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return any(len(page.get_text().strip()) < MIN_PAGE_TEXT_CHARS and page.get_images() for page in doc)


class OcrRoutingExtractor(PDFExtractor):
    """Uses the normal extractor, except for PDFs under chosen vault folders that contain scanned pages.

    Those go through Docling with OCR and formula recognition instead (slow, so only where it is asked for). A PDF
    in those folders that already has a text layer on every page keeps the fast path.
    """

    name = "routed"

    def __init__(self, default: PDFExtractor, ocr: PDFExtractor, vault_path: Path, folders: tuple[str, ...]) -> None:
        self.default = default
        self.ocr = ocr
        self.vault_path = Path(vault_path)
        self.prefixes = tuple(f.strip().strip("/\\").replace("\\", "/").lower() + "/" for f in folders if f.strip())

    def wants_ocr(self, path: Path) -> bool:
        try:
            rel = Path(path).resolve().relative_to(self.vault_path.resolve()).as_posix().lower()
        except ValueError:
            return False
        if not self.prefixes or not rel.startswith(self.prefixes):
            return False
        return has_scanned_pages(Path(path))

    def extract(self, path: Path) -> PDFExtraction:
        return (self.ocr if self.wants_ocr(path) else self.default).extract(path)


class PDFDocumentExtractor:
    """Adapts a :class:`PDFExtractor` to the generic ``Extractor`` interface."""

    extensions: ClassVar[tuple[str, ...]] = (".pdf",)
    file_type: ClassVar[str] = "pdf"

    def __init__(self, pdf_extractor: PDFExtractor) -> None:
        self.pdf_extractor = pdf_extractor

    def parse(self, path: Path) -> ParsedContent:
        extraction = self.pdf_extractor.extract(Path(path))
        return build_pdf_content(extraction, fallback_title=Path(path).stem)
