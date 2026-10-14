"""OCR (Docling + formula recognition) is used only for scanned PDFs under the folders named in `ocr_paths`."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from acadmem.config import Config
from acadmem.ingest.pdf import OcrRoutingExtractor, has_scanned_pages
from acadmem.ingest.pdf.base import PDFExtraction, PDFExtractor
from acadmem.ingest.registry import build_registry


class Marker(PDFExtractor):
    def __init__(self, name: str) -> None:
        self.name = name

    def extract(self, path: Path) -> PDFExtraction:
        return PDFExtraction(source=Path(path).name, pages=[], extractor=self.name)


def scanned_pdf(path: Path, text_pages: int = 0) -> Path:
    """A page that is only a picture (no text layer), optionally preceded by ordinary text pages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for _ in range(text_pages):
        doc.new_page().insert_text((72, 72), "This page has an ordinary text layer with plenty of words on it.")
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(str(path))
    return path


def text_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Everything on this page is real text, so no OCR is needed here at all.")
    doc.save(str(path))
    return path


def router(vault: Path, folders=("Databook",)) -> OcrRoutingExtractor:
    return OcrRoutingExtractor(Marker("fast"), Marker("ocr"), vault, folders)


def used(r: OcrRoutingExtractor, path: Path) -> str:
    return r.extract(path).extractor


def test_a_scanned_pdf_in_the_folder_is_ocrd(tmp_path):
    assert used(router(tmp_path), scanned_pdf(tmp_path / "Databook" / "Handouts" / "Mechanics.pdf")) == "ocr"


def test_a_mixed_pdf_with_a_single_scanned_page_is_ocrd(tmp_path):
    assert used(router(tmp_path), scanned_pdf(tmp_path / "Databook" / "Mixed.pdf", text_pages=3)) == "ocr"


def test_a_pdf_with_text_on_every_page_keeps_the_fast_path(tmp_path):
    assert used(router(tmp_path), text_pdf(tmp_path / "Databook" / "Maths.pdf")) == "fast"


def test_a_scanned_pdf_elsewhere_keeps_the_fast_path(tmp_path):
    assert used(router(tmp_path), scanned_pdf(tmp_path / "3B1" / "Handouts" / "scan.pdf")) == "fast"


def test_folder_matching_is_whole_folder_names_and_case_insensitive(tmp_path):
    r = router(tmp_path)
    assert used(r, scanned_pdf(tmp_path / "databook" / "a.pdf")) == "ocr"
    assert used(r, scanned_pdf(tmp_path / "Databook2" / "a.pdf")) == "fast"
    assert used(r, scanned_pdf(tmp_path / "MyDatabook" / "a.pdf")) == "fast"


def test_a_nested_folder_can_be_named_and_slashes_are_tolerated(tmp_path):
    r = router(tmp_path, ("3B1\\Reference/",))
    assert used(r, scanned_pdf(tmp_path / "3B1" / "Reference" / "a.pdf")) == "ocr"
    assert used(r, scanned_pdf(tmp_path / "3B1" / "Handouts" / "a.pdf")) == "fast"


def test_a_file_outside_the_vault_is_never_ocrd(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    assert used(router(vault), scanned_pdf(tmp_path / "Databook" / "a.pdf")) == "fast"


def test_no_folders_means_never(tmp_path):
    assert used(router(tmp_path, ()), scanned_pdf(tmp_path / "Databook" / "a.pdf")) == "fast"


def test_scan_detection(tmp_path):
    assert has_scanned_pages(scanned_pdf(tmp_path / "s.pdf"))
    assert not has_scanned_pages(text_pdf(tmp_path / "t.pdf"))


def test_the_registry_only_routes_when_folders_are_configured(tmp_path):
    on = build_registry(Config(vault_path=tmp_path, index_dir=tmp_path / "i", ocr_paths=("Databook",)))
    off = build_registry(Config(vault_path=tmp_path, index_dir=tmp_path / "i", ocr_paths=()))
    assert isinstance(on.for_path(tmp_path / "a.pdf").pdf_extractor, OcrRoutingExtractor)
    assert not isinstance(off.for_path(tmp_path / "a.pdf").pdf_extractor, OcrRoutingExtractor)


def test_the_folder_list_can_be_set_from_a_comma_separated_string(tmp_path):
    cfg = Config(vault_path=tmp_path).with_overrides(ocr_paths="Databook, Reference")
    assert cfg.ocr_paths == ("Databook", "Reference")


@pytest.mark.parametrize("name", ["Databook"])
def test_the_default_is_the_databook_folder(name):
    assert name in Config().ocr_paths
