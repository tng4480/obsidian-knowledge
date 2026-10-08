"""Tagging passages of PDFs from notes: parsing the tag lines, resolving them onto indexed chunks, staying in sync."""

from __future__ import annotations

import pytest

from acadmem.index import sql_quote
from acadmem.ingest.markdown import parse_markdown
from acadmem.models import PdfRef
from acadmem.pdf_tags import MAX_QUOTE_MATCHES, evaluate_pdf_tags, find_passages, normalize_words, resolve_pdf
from acadmem.search import Searcher

PDF = "3B1/Handouts/Week 5 - Probability.pdf"
NOTE = "3B1/Notes/study.md"


def refs(text: str) -> list[PdfRef]:
    return parse_markdown(text).pdf_refs


# --------------------------------------------------------------------------- parsing
def test_a_tag_next_to_a_page_link_tags_that_page():
    (ref,) = refs("[[Week 5 - Probability.pdf#page=3]] the Bayes example I keep missing #revisit\n")
    assert ref == PdfRef("Week 5 - Probability.pdf", 3, 3, None, False, ("revisit",))


def test_a_link_without_a_page_tags_the_whole_pdf():
    (ref,) = refs("[[Week 5 - Probability.pdf]] #exam-prep\n")
    assert (ref.page_start, ref.page_end, ref.quote) == (None, None, None)


def test_a_page_range_is_understood():
    (ref,) = refs("[[w.pdf#page=2-4]] #revisit\n")
    assert (ref.page_start, ref.page_end) == (2, 4)


def test_words_in_double_quotes_pick_the_section():
    (ref,) = refs('[[w.pdf#page=3]] "events are independent when" #revisit\n')
    assert ref.quote == "events are independent when" and not ref.quote_is_weak


def test_several_quotes_make_several_references():
    got = refs('[[w.pdf]] "first passage here" and "second passage here" #revisit\n')
    assert [r.quote for r in got] == ["first passage here", "second passage here"]


def test_curly_quotes_work_too():
    (ref,) = refs("[[w.pdf]] \u201cevents are independent\u201d #revisit\n")
    assert ref.quote == "events are independent"


def test_pdf_plus_copy_as_quote_uses_the_quote_above_the_link():
    text = (
        "> the occurrence of one does not change\n"
        "> the probability of the other\n"
        "\n"
        "[[Week 5 - Probability.pdf#page=3&selection=4,0,5,20&color=red|Week 5 - Probability, page 3]] #revisit\n"
    )
    (ref,) = refs(text)
    assert ref.quote == "the occurrence of one does not change the probability of the other"
    assert (ref.page_start, ref.target) == (3, "Week 5 - Probability.pdf")


def test_pdf_plus_callout_uses_its_body_as_the_quote():
    text = "> [!PDF|yellow] [[w.pdf#page=3&selection=1,2,3,4&color=yellow|w, page 3]] #revisit\n> pairwise independence is weaker\n"
    (ref,) = refs(text)
    assert ref.quote == "pairwise independence is weaker" and ref.tags == ("revisit",)


def test_a_tag_on_its_own_line_under_the_link_works():
    (ref,) = refs("[[w.pdf#page=2]]\n#revisit\n")
    assert ref.tags == ("revisit",) and ref.page_start == 2


def test_selected_words_as_display_text_are_a_weak_quote():
    (ref,) = refs("[[w.pdf#page=3&selection=1,0,1,9|events are independent]] #revisit\n")
    assert ref.quote == "events are independent" and ref.quote_is_weak


def test_the_default_display_text_is_not_mistaken_for_a_quote():
    (ref,) = refs("[[w.pdf#page=3&selection=1,0,1,9|w, page 3]] #revisit\n")
    assert ref.quote is None


def test_an_ordinary_alias_without_a_selection_is_not_a_quote():
    (ref,) = refs("[[w.pdf#page=3|my handout]] #revisit\n")
    assert ref.quote is None


def test_markdown_style_links_to_pdfs_work():
    (ref,) = refs("[handout](Week%205%20-%20Probability.pdf#page=3) #revisit\n")
    assert ref.target == "Week 5 - Probability.pdf" and ref.page_start == 3


@pytest.mark.parametrize(
    "text",
    [
        "[[w.pdf#page=3]] no tag on this line\n",  # nothing to apply
        "#revisit but no link to a PDF here\n",
        "[[Other note#page=3]] #revisit\n",  # not a PDF
        "```\n[[w.pdf#page=3]] #revisit\n```\n",  # inside code
    ],
)
def test_lines_that_are_not_pdf_tags_produce_nothing(text):
    assert refs(text) == []


def test_frontmatter_tags_do_not_leak_onto_pdfs_a_note_links_to():
    assert refs("---\ntags: [maths]\n---\nSee [[w.pdf#page=3]] for details.\n") == []


def test_the_note_itself_still_carries_its_tag():
    doc = parse_markdown("[[w.pdf#page=3]] hard #revisit\n")
    assert "revisit" in doc.tags


# --------------------------------------------------------------------------- word normalisation
def test_words_ignore_case_punctuation_hyphenation_and_markup():
    assert normalize_words("The **Posterior**, is\nproportion-\nal to P(A|B)!") == normalize_words(
        "the posterior is proportional to p a b"
    )


# --------------------------------------------------------------------------- resolving names
DOCS = [
    {"source_path": "3B1/Handouts/Week 5.pdf", "file_type": "pdf"},
    {"source_path": "3F1/Handouts/Week 5.pdf", "file_type": "pdf"},
    {"source_path": "3B1/Handouts/Other.pdf", "file_type": "pdf"},
    {"source_path": "3B1/Notes/x.md", "file_type": "md"},
]


def test_resolving_by_unique_name_by_path_and_case():
    assert resolve_pdf("other.PDF", DOCS)[0]["source_path"] == "3B1/Handouts/Other.pdf"
    assert resolve_pdf("3F1/Handouts/Week 5.pdf", DOCS)[0]["source_path"] == "3F1/Handouts/Week 5.pdf"
    assert resolve_pdf("Handouts/Other.pdf", DOCS)[0]["source_path"] == "3B1/Handouts/Other.pdf"


def test_resolving_an_ambiguous_or_unknown_name_explains_why():
    doc, why = resolve_pdf("Week 5.pdf", DOCS)
    assert doc is None and "ambiguous" in why and "3B1/Handouts/Week 5.pdf" in why
    doc, why = resolve_pdf("nope.pdf", DOCS)
    assert doc is None and "no indexed PDF" in why


# --------------------------------------------------------------------------- end to end
@pytest.fixture
def built(indexer):
    indexer.run(full=True)
    return indexer


def write_note(vault, text: str, rel: str = NOTE) -> None:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tagged(index, tag: str, file_type: str = "pdf") -> list[dict]:
    """Chunks carrying ``tag``: PDF sections by default (the note's own tagged line is a tagged chunk too)."""
    where = f"file_type = '{file_type}' AND array_has_any(tags, ['{tag}'])"
    rows = index.scan(where, columns=["chunk_id", "source_path", "page_start", "page_end", "text", "tags"])
    return sorted(rows, key=lambda r: (r["source_path"], r["page_start"] or 0))


def pdf_chunks(index) -> list[dict]:
    rows = index.scan(f"source_path = {sql_quote(PDF)}", columns=["chunk_id", "chunk_index", "page_start", "page_end", "text", "tags"])
    return sorted(rows, key=lambda r: r["chunk_index"])


def test_a_page_tag_lands_on_the_sections_of_that_page_only(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] total probability again #revisit\n")
    built.run()
    hits = tagged(built.index, "revisit")
    assert hits and all(h["source_path"] == PDF for h in hits)
    assert all(h["page_start"] <= 2 <= h["page_end"] for h in hits)
    assert "law of total probability" in " ".join(h["text"] for h in hits)
    untagged_pages = {r["page_start"] for r in pdf_chunks(built.index) if not r["tags"]}
    assert 1 in untagged_pages or 3 in untagged_pages


def test_a_quote_tags_only_the_section_containing_it(built, vault):
    write_note(vault, '[[Week 5 - Probability.pdf]] "mutual independence of a whole family" #hard\n')
    built.run()
    hits = tagged(built.index, "hard")
    assert len(hits) == 1 and "Pairwise independence" in hits[0]["text"]


def test_the_pdf_plus_copy_as_quote_format_works_end_to_end(built, vault):
    write_note(
        vault,
        "> the occurrence of one does not change the probability of the other\n\n"
        "[[Week 5 - Probability.pdf#page=3&selection=4,0,5,20|Week 5 - Probability, page 3]] #revisit\n",
    )
    built.run()
    hits = tagged(built.index, "revisit")
    assert len(hits) == 1 and "occurrence of one" in hits[0]["text"]


def test_a_whole_pdf_tag_covers_every_section(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf]] #exam-prep\n")
    built.run()
    assert len(tagged(built.index, "exam-prep")) == len(pdf_chunks(built.index))


def test_tags_are_searchable_with_the_normal_tag_filter(built, vault, config, fake_embedder):
    write_note(vault, '[[Week 5 - Probability.pdf]] "mutual independence of a whole family" #hard\n')
    built.run()
    hits = Searcher(config, embedder=fake_embedder).search("independence", top_k=5, filters={"tags": ["hard"]})
    assert all("hard" in h.tags for h in hits)
    assert PDF in {h.source_path for h in hits}  # (the note's own tagged line is a hit as well)


def test_the_pdf_document_lists_the_tags_its_sections_carry(built, vault):
    write_note(vault, '[[Week 5 - Probability.pdf]] "mutual independence of a whole family" #hard\n')
    built.run()
    assert built.index.get_document(source_path=PDF)["tags"] == ["hard"]


def test_the_note_keeps_its_own_tags_and_text_unchanged(built, vault):
    write_note(vault, "# Study\n\n[[Week 5 - Probability.pdf#page=2]] hard #revisit\n")
    built.run()
    notes = tagged(built.index, "revisit", file_type="md")
    assert [r["source_path"] for r in notes] == [NOTE]


def test_removing_the_line_removes_the_tag(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #revisit\n")
    built.run()
    assert tagged(built.index, "revisit")
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] no longer tagged\n")
    built.run()
    assert [r for r in tagged(built.index, "revisit") if r["source_path"] == PDF] == []
    assert built.index.get_document(source_path=PDF)["tags"] == []


def test_deleting_the_note_removes_the_tag(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #revisit\n")
    built.run()
    (vault / NOTE).unlink()
    built.run()
    assert tagged(built.index, "revisit") == []


def test_changing_the_tag_replaces_it(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #revisit\n")
    built.run()
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #mastered\n")
    built.run()
    assert tagged(built.index, "revisit") == [] and tagged(built.index, "mastered")


def test_two_notes_can_tag_the_same_section_and_each_removal_is_independent(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #revisit\n", "3B1/Notes/a.md")
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #hard\n", "3B1/Notes/b.md")
    built.run()
    page2 = [r for r in pdf_chunks(built.index) if r["page_start"] <= 2 <= r["page_end"]]
    assert page2 and all(sorted(r["tags"]) == ["hard", "revisit"] for r in page2)
    (vault / "3B1/Notes/a.md").unlink()
    built.run()
    page2 = [r for r in pdf_chunks(built.index) if r["page_start"] <= 2 <= r["page_end"]]
    assert all(r["tags"] == ["hard"] for r in page2)


def test_tags_survive_reindexing_the_pdf(built, vault, pdf_factory):
    write_note(vault, '[[Week 5 - Probability.pdf]] "mutual independence of a whole family" #hard\n')
    built.run()
    from conftest import PDF_PAGES

    pdf_factory(vault / PDF, PDF_PAGES + [{"body": ["An extra page added later."]}])
    report = built.run()
    assert PDF in report.modified
    hits = tagged(built.index, "hard")
    assert len(hits) == 1 and "Pairwise independence" in hits[0]["text"]


def test_tags_survive_a_full_rebuild(built, vault):
    write_note(vault, '[[Week 5 - Probability.pdf]] "mutual independence of a whole family" #hard\n')
    built.run()
    built.run(full=True)
    assert len(tagged(built.index, "hard")) == 1


def test_a_tag_line_written_before_the_pdf_is_indexed_applies_once_it_is(built, vault, pdf_factory):
    write_note(vault, "[[Later.pdf#page=1]] #revisit\n")
    built.run()
    assert [e.status for e in built.last_pdf_tag_report.entries] == ["unresolved"]
    from conftest import PDF_PAGES

    pdf_factory(vault / "3B1" / "Handouts" / "Later.pdf", PDF_PAGES[:1])
    built.run()
    assert [r["source_path"] for r in tagged(built.index, "revisit") if r["source_path"].endswith("Later.pdf")]


def test_single_file_sync_applies_tags_too(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2]] #revisit\n")
    assert built.sync_path(NOTE) == "indexed"
    assert [r for r in tagged(built.index, "revisit") if r["source_path"] == PDF]
    (vault / NOTE).unlink()
    assert built.sync_path(NOTE) == "deleted"
    assert tagged(built.index, "revisit") == []


# --------------------------------------------------------------------------- reporting
def statuses(index) -> dict[str, tuple[str, str]]:
    report, _, _ = evaluate_pdf_tags(index)
    return {(e.quote or e.page or e.target): (e.status, e.detail) for e in report.entries}


def test_lines_that_cannot_be_applied_are_reported_not_silently_dropped(built, vault):
    write_note(
        vault,
        "[[Missing.pdf#page=1]] #revisit\n\n"
        '[[Week 5 - Probability.pdf]] "words that are nowhere in this handout" #revisit\n\n'
        "[[Week 5 - Probability.pdf#page=40]] #revisit\n",
    )
    built.run()
    got = statuses(built.index)
    assert got["1"][0] == "unresolved" and "no indexed PDF" in got["1"][1]
    assert got["words that are nowhere in this handout"][0] == "unmatched"
    assert got["40"][0] == "unmatched" and "no indexed text on page 40" in got["40"][1]
    assert tagged(built.index, "revisit") == [r for r in tagged(built.index, "revisit") if r["source_path"] == NOTE]


def test_a_quote_on_the_wrong_page_is_reported_not_applied(built, vault):
    write_note(vault, '[[Week 5 - Probability.pdf#page=1]] "mutual independence of a whole family" #hard\n')
    built.run()
    assert [r for r in tagged(built.index, "hard") if r["source_path"] == PDF] == []
    ((status, detail),) = statuses(built.index).values()
    assert status == "unmatched" and "page 1" in detail


def test_a_quote_that_matches_too_many_sections_is_rejected_as_too_vague(built, vault, pdf_factory):
    filler = "Probability is discussed here together with many other ideas that fill out this page of the handout nicely."
    pages = [{"heading": f"Section {i}", "body": [filler] * 6} for i in range(MAX_QUOTE_MATCHES + 3)]
    pdf_factory(vault / "3B1" / "Handouts" / "Many.pdf", pages, header=None, footer=None)
    write_note(vault, '[[Many.pdf]] "probability" #revisit\n')
    built.run()
    ((status, detail),) = statuses(built.index).values()
    assert status == "unmatched" and "more distinctive" in detail


def test_a_weak_quote_falls_back_to_the_page_when_not_found(built, vault):
    write_note(vault, "[[Week 5 - Probability.pdf#page=2&selection=1,0,1,9|garbled text from copy]] #revisit\n")
    built.run()
    ((status, _),) = statuses(built.index).values()
    assert status == "page_fallback"
    assert [r for r in tagged(built.index, "revisit") if r["source_path"] == PDF]


def test_find_passages_accepts_a_quote_that_runs_across_two_sections():
    rows = [
        {"chunk_id": "a", "chunk_index": 0, "heading_path": ["H"], "text": "alpha beta gamma delta epsilon", "page_start": 1, "page_end": 1},
        {"chunk_id": "b", "chunk_index": 1, "heading_path": ["H"], "text": "zeta eta theta iota kappa", "page_start": 1, "page_end": 1},
        {"chunk_id": "c", "chunk_index": 2, "heading_path": ["H"], "text": "unrelated words about chickens", "page_start": 1, "page_end": 1},
    ]
    match = find_passages(rows, None, None, "gamma delta epsilon zeta eta theta")
    assert [p.chunk_id for p in match.passages] == ["a", "b"]


def _one(text: str, heading: str = "H") -> list[dict]:
    return [{"chunk_id": "a", "chunk_index": 0, "heading_path": [heading], "text": text, "page_start": 1, "page_end": 1}]


@pytest.mark.parametrize(
    "extracted, selected",
    [
        # subscripts: the extractor spaces them out, the viewer does not
        ("Connect C ₁ from collector to emitter, and C ₂ from emitter to ground.", "C₁ from collector to emitter, and C₂ from emitter"),
        # formatting tags the extractor writes for superscripts
        ("so that E[X(t)<sup>2</sup> ] = r(0). This is ensemble-average power.", "E[X(t)2] = r(0). This is ensemble-average power."),
        # a table row the viewer reads straight across
        ("|Select abc|Data input|f1input|\n|---|---|---|\n|000|I0|0|", "Select abc Data input f1 input 000"),
        # ligatures and a hyphenated line break
        ("the e\ufb03cient estimator is asymptotically unbiased", "the efficient estimator is asymp-\ntotically unbiased"),
    ],
)
def test_quotes_survive_how_formulas_tables_and_ligatures_are_extracted(extracted, selected):
    assert [p.chunk_id for p in find_passages(_one(extracted), None, None, selected).passages] == ["a"]


def test_short_quotes_are_not_matched_by_ignoring_spaces():
    # "a b" squashed would be "ab", which would match inside "cab": too short to be compared that loosely
    assert find_passages(_one("a cab drove by"), None, None, "a b").reason


def test_find_passages_tolerates_a_slightly_garbled_quote():
    text = "the posterior distribution is proportional to the likelihood times the prior and is then normalised by the evidence term"
    rows = [
        {"chunk_id": "a", "chunk_index": 0, "heading_path": ["H"], "text": text, "page_start": 1, "page_end": 1},
        {"chunk_id": "b", "chunk_index": 1, "heading_path": ["H"], "text": "completely different material about queues", "page_start": 1, "page_end": 1},
    ]
    garbled = "the posterior distribution is proportional to the likelihood times the prior and is then normalized by the evidence term"
    assert [p.chunk_id for p in find_passages(rows, None, None, garbled).passages] == ["a"]
    assert find_passages(rows, None, None, "nothing like it at all in any of the text").reason


def test_the_status_report_summarises_problems(built, vault):
    write_note(vault, "[[Missing.pdf]] #revisit\n")
    built.run()
    summary = built.last_pdf_tag_report.to_dict()
    assert summary["tag_lines"] == 1 and summary["applied"] == 0 and summary["problem_count"] == 1
