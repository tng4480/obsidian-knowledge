"""Docling-backed ``PDFExtractor`` (layout model; keeps pages, headings, tables).

Docling's ``DoclingDocument`` keeps per-item provenance (``prov.page_no``), which is
what makes exact page citations possible. We read the document directly rather than
round-tripping through Markdown (which loses page information). Docling does not
read PDF bookmarks, so the outline is taken from PyMuPDF.
"""

from __future__ import annotations

from pathlib import Path

from ...models import CODE, LIST, MATH, PARAGRAPH, TABLE
from .base import PDFBlock, PDFExtraction, PDFExtractor, PDFPage
from .pymupdf_adapter import read_outline


class DoclingExtractor(PDFExtractor):
    name = "docling"

    def __init__(self, do_ocr: bool = False, do_table_structure: bool = True) -> None:
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions(do_ocr=self.do_ocr, do_table_structure=self.do_table_structure)
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def extract(self, path: Path) -> PDFExtraction:
        from docling_core.types.doc import DocItemLabel

        path = Path(path)
        result = self._get_converter().convert(str(path))
        doc = result.document

        skip_labels = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
        by_page: dict[int, list[PDFBlock]] = {n: [] for n in sorted(doc.pages)}

        for item, _depth in doc.iterate_items():
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            label = getattr(item, "label", None)
            if label in skip_labels:
                continue
            page_no = int(prov[0].page_no)
            blocks = by_page.setdefault(page_no, [])
            cls = type(item).__name__
            if cls == "TableItem":
                text = item.export_to_markdown(doc=doc).strip()
                if text:
                    blocks.append(PDFBlock(text, kind=TABLE))
                continue
            if cls == "PictureItem":
                continue
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            if cls == "TitleItem":
                blocks.append(PDFBlock(text, heading_level=1))
            elif cls == "SectionHeaderItem":
                blocks.append(PDFBlock(text, heading_level=max(1, int(getattr(item, "level", 1))) + 1))
            elif cls == "ListItem":
                marker = (getattr(item, "marker", "") or "-").strip() or "-"
                blocks.append(PDFBlock(f"{marker} {text}", kind=LIST))
            elif cls == "CodeItem":
                blocks.append(PDFBlock(text, kind=CODE))
            elif label == DocItemLabel.FORMULA:
                blocks.append(PDFBlock(text, kind=MATH))
            else:
                blocks.append(PDFBlock(text, kind=PARAGRAPH))

        pages = [PDFPage(source=path.name, page_number=n, blocks=b) for n, b in sorted(by_page.items())]
        outline, title, _ = read_outline(path)
        return PDFExtraction(source=path.name, pages=pages, outline=outline, title=title, extractor=self.name)
