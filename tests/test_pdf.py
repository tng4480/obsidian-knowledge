from pathlib import Path

import pytest

from acadmem.ingest.pdf import (
    PDFBlock,
    PDFDocumentExtractor,
    PDFExtraction,
    PDFExtractor,
    PDFPage,
    build_pdf_content,
    get_pdf_extractor,
    strip_boilerplate,
)
from acadmem.ingest.pdf.base import OutlineEntry
from acadmem.ingest.pdf.pymupdf_adapter import PyMuPDFExtractor, clean_page_markdown

PAGES = [
    {
        "outline": "Conditional probability",
        "heading": "Conditional probability",
        "body": ["Bayes theorem relates the posterior to the likelihood.", "P(A|B) = P(B|A) P(A) / P(B)."],
    },
    {"body": ["The evidence term normalises the posterior distribution."]},
    {
        "outline": "Independence",
        "heading": "Independence",
        "body": ["Two events are independent when P(A and B) = P(A) P(B)."],
    },
    {"body": ["Pairwise independence is weaker than mutual independence."]},
]


@pytest.fixture
def sample_pdf(tmp_path: Path, pdf_factory) -> Path:
    return pdf_factory(tmp_path / "Week 5 - Probability.pdf", PAGES)


def test_pymupdf_extractor_returns_source_page_text(sample_pdf: Path):
    ex = PyMuPDFExtractor().extract(sample_pdf)
    assert ex.source == "Week 5 - Probability.pdf"
    assert [p.page_number for p in ex.pages] == [1, 2, 3, 4]
    assert all(p.source == "Week 5 - Probability.pdf" for p in ex.pages)
    assert "posterior to the likelihood" in ex.pages[0].text
    assert "normalises the posterior" in ex.pages[1].text
    assert [(e.level, e.title, e.page) for e in ex.outline] == [
        (1, "Conditional probability", 1),
        (1, "Independence", 3),
    ]
    assert ex.title == "Test Guide"


def test_page_provenance_survives_into_sections(sample_pdf: Path):
    parsed = PDFDocumentExtractor(PyMuPDFExtractor()).parse(sample_pdf)
    assert parsed.page_count == 4
    titles = {s.title: s for s in parsed.sections}
    cond, indep = titles["Conditional probability"], titles["Independence"]
    # Section spans pages 1-2 / 3-4 and each block remembers its own page.
    assert (cond.page_start, cond.page_end) == (1, 2)
    assert (indep.page_start, indep.page_end) == (3, 4)
    p2 = [b for b in cond.blocks if "normalises" in b.text]
    assert p2 and p2[0].page == 2
    assert cond.heading_path == ("Conditional probability",)


def test_headers_footers_and_page_numbers_are_stripped(sample_pdf: Path):
    parsed = PDFDocumentExtractor(PyMuPDFExtractor()).parse(sample_pdf)
    body = "\n".join(s.text for s in parsed.sections)
    assert "COURSE 3B1" not in body
    assert "WORKED GUIDE" not in body
    # heading restating the bookmark is not duplicated as body text
    assert body.count("Conditional probability") == 0


def test_strip_boilerplate_only_touches_edge_blocks():
    def page(n: int) -> PDFPage:
        # 6 blocks: indices 0-1 are the header zone, 4-5 the footer zone, 2-3 are interior.
        return PDFPage(
            "x.pdf",
            n,
            [
                PDFBlock("RUNNING TITLE"),
                PDFBlock(f"intro paragraph number {n}"),
                PDFBlock("Answer."),  # interior + repeated on every page: must survive
                PDFBlock(f"derivation step {n}"),
                PDFBlock(f"closing remark {n}"),
                PDFBlock(f"Page {n}"),
            ],
        )

    cleaned = strip_boilerplate([page(i) for i in range(1, 7)])
    texts = [b.text for b in cleaned[0].blocks]
    assert "RUNNING TITLE" not in texts  # header zone, repeats -> dropped
    assert "Page 1" not in texts  # footer zone, digits normalised -> dropped
    assert "Answer." in texts  # interior block survives even though it repeats
    assert "intro paragraph number 1" in texts and "closing remark 1" in texts  # unique edge text kept


def test_short_documents_are_not_stripped():
    pages = [PDFPage("x.pdf", i, [PDFBlock("same"), PDFBlock(f"body {i}")]) for i in (1, 2)]
    assert strip_boilerplate(pages) == pages


class FakeExtractor(PDFExtractor):
    """Proves the interface is swappable: any implementation feeds the same pipeline."""

    name = "fake"

    def extract(self, path: Path) -> PDFExtraction:
        return PDFExtraction(
            source=path.name,
            pages=[
                PDFPage(path.name, 7, [PDFBlock("Chapter A", heading_level=1), PDFBlock("alpha text")]),
                PDFPage(path.name, 8, [PDFBlock("Section A.1", heading_level=2), PDFBlock("beta text")]),
            ],
        )


def test_pdf_extractor_interface_is_swappable(tmp_path: Path):
    parsed = PDFDocumentExtractor(FakeExtractor()).parse(tmp_path / "doc.pdf")
    a, a1 = parsed.sections
    assert a.heading_path == ("Chapter A",) and a.blocks[0].page == 7
    assert a1.heading_path == ("Chapter A", "Section A.1") and a1.blocks[0].page == 8
    assert a1.parent_id == a.section_id


def test_in_page_headings_nest_under_bookmark():
    extraction = PDFExtraction(
        source="d.pdf",
        pages=[
            PDFPage(
                "d.pdf",
                1,
                [
                    PDFBlock("QUESTION 1", heading_level=2),
                    PDFBlock("Solve the thing", heading_level=1),
                    PDFBlock("intro"),
                    PDFBlock("Sanity check", heading_level=3),
                    PDFBlock("check body"),
                ],
            )
        ],
        outline=[OutlineEntry(1, "QUESTION 1 - Solve the thing", 1)],
    )
    parsed = build_pdf_content(extraction)
    assert [s.title for s in parsed.sections] == ["QUESTION 1 - Solve the thing", "Sanity check"]
    sub = parsed.sections[1]
    assert sub.level == 2 and sub.heading_path == ("QUESTION 1 - Solve the thing", "Sanity check")


def test_no_outline_falls_back_to_extractor_heading_levels():
    extraction = PDFExtraction(
        source="d.pdf",
        pages=[PDFPage("d.pdf", 1, [PDFBlock("Top", heading_level=1), PDFBlock("x"), PDFBlock("Sub", heading_level=2), PDFBlock("y")])],
    )
    parsed = build_pdf_content(extraction)
    assert [s.heading_path for s in parsed.sections] == [("Top",), ("Top", "Sub")]


def test_picture_text_comments_become_labelled_figure_text():
    md = "intro\n\n<!-- Start of picture text -->\nR1<br>C1<br><!-- End of picture text -->\n\nafter"
    out = clean_page_markdown(md)
    assert "<!--" not in out and "<br>" not in out
    assert "*Figure text:* R1; C1" in out


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_pdf_extractor("nonsense")
    assert get_pdf_extractor("pymupdf").name == "pymupdf"


@pytest.mark.slow
def test_docling_adapter_matches_interface(sample_pdf: Path):
    ex = get_pdf_extractor("docling").extract(sample_pdf)
    assert [p.page_number for p in ex.pages] == [1, 2, 3, 4]
    assert any("posterior" in p.text for p in ex.pages)
