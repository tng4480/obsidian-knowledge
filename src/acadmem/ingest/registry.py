"""Default extractor registry (add a new format by registering another extractor)."""

from __future__ import annotations

from ..config import Config
from .base import ExtractorRegistry
from .markdown import MarkdownExtractor
from .pdf import OcrRoutingExtractor, PDFDocumentExtractor, get_pdf_extractor


def build_registry(config: Config) -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register(MarkdownExtractor())
    pdf = get_pdf_extractor(config.pdf_extractor)
    if config.ocr_paths:
        pdf = OcrRoutingExtractor(
            pdf, get_pdf_extractor("docling", do_ocr=True, do_formulas=True), config.vault_path, config.ocr_paths
        )
    registry.register(PDFDocumentExtractor(pdf))
    return registry
