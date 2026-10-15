"""Equations drawn as vector outlines (Chrome/Skia prints of KaTeX maths) are recovered from the outlines."""

from __future__ import annotations

import pymupdf
import pytest

from acadmem.ingest.pdf import mathglyphs as mg
from acadmem.ingest.pdf.pymupdf_adapter import PyMuPDFExtractor

# Distinct stand-in "glyphs": polygons with different vertex counts (the lexicon is keyed by outline).
SHAPES = {
    "a": [(0, 0), (1, 0), (0.5, 1)],
    "b": [(0, 0), (1, 0), (1, 1), (0, 1)],
    "c": [(0, 0.3), (0.5, 0), (1, 0.3), (0.8, 1), (0.2, 1)],
    "2": [(0, 0), (0.6, 0), (1, 0.5), (0.6, 1), (0, 1), (0.1, 0.5)],
    "+": [(0.4, 0), (0.6, 0), (0.6, 0.4), (1, 0.4), (1, 0.6), (0.6, 0.6), (0.6, 1), (0.4, 1), (0.4, 0.6), (0, 0.6), (0, 0.4), (0.4, 0.4)],
}
SIZE = 10.0


def draw(page, name: str, x: float, y: float, size: float = SIZE) -> None:
    """Draw glyph ``name`` with its top-left at (x, y)."""
    shape = page.new_shape()
    shape.draw_polyline([pymupdf.Point(x + px * size * 0.8, y + py * size) for px, py in SHAPES[name]])
    shape.finish(fill=(0, 0, 0), color=None, closePath=True)
    shape.commit()


def bar(page, x: float, y: float, w: float, thickness: float = 0.78) -> None:
    page.draw_rect(pymupdf.Rect(x, y, x + w, y + thickness), color=None, fill=(0, 0, 0))


@pytest.fixture
def lexicon(monkeypatch):
    """A lexicon built from the stand-in glyphs, labelled exactly as the real one is."""
    scratch = pymupdf.open().new_page()
    lex: dict = {}
    for name in SHAPES:
        draw(scratch, name, 100, 100)
        x = scratch.get_drawings()[-1]
        key, vec = mg.outline_key(x["items"], x["rect"])
        lex.setdefault(key, []).append((vec, name, x["rect"].height))
    monkeypatch.setattr(mg, "load_lexicon", lambda: lex)
    monkeypatch.setattr(mg, "MIN_GLYPHS", 2)
    return lex


def make_page(draw_fn):
    doc = pymupdf.open()
    page = doc.new_page()
    draw_fn(page)
    return doc, page


def texts(page) -> list[str]:
    return [f.text for f in mg.recover_page(page)]


def test_a_superscript_and_a_subscript(lexicon):
    def body(page):
        draw(page, "a", 100, 100)
        draw(page, "2", 109, 98, 7)  # raised, small
        draw(page, "+", 118, 104)
        draw(page, "b", 128, 100)
        draw(page, "c", 137, 106, 7)  # lowered, small

    _, page = make_page(body)
    assert texts(page) == ["a^2 + b_c"]


def test_a_fraction_is_found_from_its_bar(lexicon):
    def body(page):
        draw(page, "a", 105, 80, 8)
        bar(page, 100, 95, 16)
        draw(page, "b", 105, 99, 8)

    _, page = make_page(body)
    assert texts(page) == ["\\frac{a}{b}"]


def test_the_recovered_text_replaces_the_outlines_in_the_document(lexicon):
    def body(page):
        page.insert_text((60, 90), "Here the sum is", fontsize=11)
        draw(page, "a", 150, 82)
        draw(page, "+", 160, 86)
        draw(page, "b", 170, 82)

    doc, page = make_page(body)
    found = mg.recover_math(doc)
    assert list(found) == [0] and found[0][0].inline
    text = doc[0].get_text()
    assert "$a + b$" in text
    assert not doc[0].get_drawings()  # the glyph paths are gone, the text carries the equation


def test_a_display_equation_is_marked_as_math_block(lexicon):
    def body(page):
        draw(page, "a", 250, 300)
        draw(page, "+", 262, 304)
        draw(page, "b", 272, 300)

    doc, _ = make_page(body)
    assert mg.recover_math(doc)[0][0].markdown == "$$ a + b $$"


def test_ordinary_pdfs_are_left_alone(lexicon, tmp_path, pdf_factory):
    path = pdf_factory(tmp_path / "plain.pdf", [{"body": ["Nothing but ordinary text on this page, no drawn glyphs."]}])
    with pymupdf.open(str(path)) as doc:
        assert mg.recover_math(doc) == {}
    extraction = PyMuPDFExtractor().extract(path)
    assert "ordinary text" in extraction.pages[0].text


def test_only_browser_printed_pdfs_are_searched_for_drawn_maths(lexicon, tmp_path):
    assert mg.drawn_by_browser({"producer": "Skia/PDF m155", "creator": "HeadlessChrome/155"})
    assert not mg.drawn_by_browser({"producer": "pdfTeX-1.40", "creator": "LaTeX"})
    assert not mg.drawn_by_browser(None)

    def body(page):
        page.insert_text((60, 120), "Worked example, then the result:", fontsize=11)
        draw(page, "a", 250, 150)
        draw(page, "+", 262, 154)
        draw(page, "b", 272, 150)

    doc, _ = make_page(body)  # made by pymupdf, not a browser: looks like a table/diagram, so it is not touched
    path = tmp_path / "other.pdf"
    doc.save(str(path))
    assert not any("a + b" in b.text for p in PyMuPDFExtractor().extract(path).pages for b in p.blocks)


def test_the_extractor_returns_equations_through_the_whole_pipeline(lexicon, tmp_path):
    def body(page):
        page.insert_text((60, 120), "Worked example, then the result:", fontsize=11)
        draw(page, "a", 250, 150)
        draw(page, "+", 262, 154)
        draw(page, "b", 272, 150)

    doc, _ = make_page(body)
    doc.set_metadata({"producer": "Skia/PDF m155", "creator": "Mozilla/5.0 HeadlessChrome/155"})
    path = tmp_path / "math.pdf"
    doc.save(str(path))
    blocks = [b.text for p in PyMuPDFExtractor().extract(path).pages for b in p.blocks]
    assert any("a + b" in b for b in blocks)


def test_formulas_the_layout_model_left_out_are_put_back_in_reading_order():
    formulas = [
        mg.Formula("x^2", (100, 50, 140, 62), False),  # inside a blanked "formula" block
        mg.Formula("z", (100, 120, 110, 130), False),  # in no block at all
        mg.Formula("y", (300, 5, 310, 12), True),  # already present in the text
    ]
    boxes = [
        {"class": "text", "bbox": (0, 0, 500, 20), "pos": (0, 10)},
        {"class": "formula", "bbox": (90, 45, 150, 66), "pos": (10, 12)},
        {"class": "text", "bbox": (0, 80, 500, 100), "pos": (12, 16)},
    ]
    out = mg.fill_formula_boxes("01234$y$9  tail", boxes, formulas)
    assert out.index("$$ x^2 $$") < out.index("$$ z $$") and out.count("$y$") == 1


def test_the_shipped_lexicon_loads():
    lex = mg.load_lexicon()
    assert sum(len(v) for v in lex.values()) > 200
    assert any(label == "\\theta" for entries in lex.values() for _, label, _ in entries)
