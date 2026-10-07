"""Default extractor registry (add a new format by registering another extractor)."""

from __future__ import annotations

from ..config import Config
from .base import ExtractorRegistry
from .markdown import MarkdownExtractor
from .pdf import PDFDocumentExtractor, get_pdf_extractor


def build_registry(config: Config) -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register(MarkdownExtractor())
    registry.register(PDFDocumentExtractor(get_pdf_extractor(config.pdf_extractor)))
    return registry
