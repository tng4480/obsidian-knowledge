"""PDF extraction interface.

``PDFExtractor`` is the seam that keeps the retrieval architecture independent of
the PDF library. Obsidian's Text Extractor plugin was evaluated and rejected as the
default: it only runs inside Obsidian's renderer process, exposes
``extractText(file) -> Promise<string>`` (no pages), and its own README warns that
PDF extraction often fails. Any implementation of this interface can be swapped in
(Docling, pymupdf4llm, MinerU, Marker, ...) without touching chunking or indexing.

Required conceptual output per page::

    source file, page number, extracted text
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ...models import PARAGRAPH


@dataclass(frozen=True)
class PDFBlock:
    text: str
    kind: str = PARAGRAPH
    heading_level: int | None = None  # set when the extractor believes this is a heading


@dataclass
class PDFPage:
    source: str  # file name / path of the PDF
    page_number: int  # 1-based
    blocks: list[PDFBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(frozen=True)
class OutlineEntry:
    """A PDF bookmark: ``[level, title, page]`` (page is 1-based)."""

    level: int
    title: str
    page: int


@dataclass
class PDFExtraction:
    source: str
    pages: list[PDFPage]
    outline: list[OutlineEntry] = field(default_factory=list)
    title: str = ""
    extractor: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PDFExtractor(ABC):
    """Extracts per-page text and structure hints from a PDF."""

    name: str = "abstract"

    @abstractmethod
    def extract(self, path: Path) -> PDFExtraction:
        """Return per-page blocks (with heading hints) and the PDF outline, if any."""
