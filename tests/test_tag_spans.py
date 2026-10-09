"""Span-level tags: a quote tags only the paragraph(s) it hit inside a section; the section stays the unit of retrieval."""

from __future__ import annotations

import pytest
from test_mcp_server import Client

from acadmem.index import VectorIndex, chunks_schema
from acadmem.mcp_server import AcademicMemoryService, build_server
from acadmem.pdf_tags import find_passages, passage_text, split_paragraphs, tagged_excerpt

P0 = "The first paragraph talks about conditional probability and how it is defined."
P0B = "It continues on a second line of the same paragraph."
P1 = "The second paragraph is about independence of events and what that means."
P2 = "The third paragraph covers the law of total probability in a partition."
PDF = "3B1/Handouts/Spans.pdf"
NOTE = "3B1/Notes/spans.md"


@pytest.fixture
def world(indexer, config, vault, fake_embedder, pdf_factory):
    body = [P0, P0B, "", "", P1, "", "", P2]
    pdf_factory(vault / PDF, [{"heading": "Topic", "body": body}], header=None, footer=None)
    indexer.run(full=True)
    client = Client(build_server(AcademicMemoryService(config, embedder=fake_embedder)))
    return indexer, client, vault


def note(vault, text: str, rel: str = NOTE) -> None:
    (vault / rel).write_text(text, encoding="utf-8")


def tagged(client, tag: str) -> list[dict]:
    return client.call("search_academic_memory", tags=tag, file_type="pdf")["results"]


# --------------------------------------------------------------------------- matching, in isolation
def rows(text: str, heading: str = "H") -> list[dict]:
    return [{"chunk_id": "a", "chunk_index": 0, "heading_path": [heading], "text": text, "page_start": 1, "page_end": 1}]


TEXT = "Alpha beta gamma delta one.\n\nEpsilon zeta eta theta two.\n\nIota kappa lambda mu three.\n\nNu xi omicron pi four."


def span(quote: str, text: str = TEXT, heading: str = "H"):
    (p,) = find_passages(rows(text, heading), None, None, quote).passages
    return p.paragraphs


def test_a_quote_inside_one_paragraph_reports_just_that_paragraph():
    assert span("zeta eta theta two") == (1,)


def test_a_quote_over_two_paragraphs_reports_both_and_not_the_rest():
    assert span("theta two. Iota kappa") == (1, 2)


def test_a_quote_in_the_first_and_in_the_last_paragraph():
    assert span("Alpha beta gamma delta") == (0,)
    assert span("omicron pi four") == (3,)


def test_the_shortest_run_of_paragraphs_wins():
    assert span("kappa lambda mu three") == (2,)


def test_a_quote_that_occurs_twice_reports_both_paragraphs():
    text = "The posterior is large.\n\nSomething else entirely.\n\nThe posterior is large.\n\nMore text."
    assert span("the posterior is large", text) == (0, 2)


def test_a_quote_that_only_matches_the_heading_covers_the_whole_section():
    assert span("Conditional probability heading", TEXT, "Conditional probability heading") is None


def test_a_single_paragraph_section_is_simply_whole():
    assert span("zeta eta theta", "Epsilon zeta eta theta two.") is None


def test_a_page_or_a_whole_pdf_tag_covers_the_whole_section():
    (p,) = find_passages(rows(TEXT), 1, 1, None).passages
    assert p.paragraphs is None


def test_formula_spacing_does_not_stop_a_paragraph_being_found():
    text = "Intro paragraph here.\n\nConnect C ₁ from collector to emitter, and C ₂ to ground.\n\nClosing paragraph."
    assert span("C₁ from collector to emitter, and C₂ to ground", text) == (1,)


def test_a_slightly_garbled_quote_still_finds_its_paragraph():
    text = (
        "Intro paragraph about something else entirely.\n\n"
        "the posterior distribution is proportional to the likelihood times the prior and is then normalised by the evidence term\n\n"
        "Closing paragraph about queues."
    )
    garbled = "the posterior distribution is proportional to the likelihood times the prior and is then normalized by the evidence term"
    assert span(garbled, text) == (1,)


def test_a_quote_running_over_two_sections_tags_each_whole():
    both = [
        {"chunk_id": "a", "chunk_index": 0, "heading_path": ["H"], "text": "One.\n\nalpha beta gamma delta", "page_start": 1, "page_end": 1},
        {"chunk_id": "b", "chunk_index": 1, "heading_path": ["H"], "text": "epsilon zeta eta theta\n\nTwo.", "page_start": 1, "page_end": 1},
    ]
    match = find_passages(both, None, None, "gamma delta epsilon zeta")
    assert [(p.chunk_id, p.paragraphs) for p in match.passages] == [("a", None), ("b", None)]


def test_paragraph_helpers_agree_on_what_a_paragraph_is():
    assert split_paragraphs(TEXT)[1].startswith("Epsilon")
    assert passage_text(TEXT, (1, 3)) == "Epsilon zeta eta theta two.\n\nNu xi omicron pi four."
    assert passage_text(TEXT, None) == TEXT


def test_the_excerpt_rules():
    spans = {"hard": (1,), "easy": (2,)}
    assert tagged_excerpt(TEXT, ["hard"], spans) == "Epsilon zeta eta theta two."
    assert tagged_excerpt(TEXT, ["hard", "easy"], spans).startswith("Epsilon")
    assert tagged_excerpt(TEXT, ["hard", "easy"], spans, wanted=["easy"]) == "Iota kappa lambda mu three."
    assert tagged_excerpt(TEXT, ["hard", "whole"], spans) is None  # one tag covers everything
    assert tagged_excerpt(TEXT, ["hard", "whole"], spans, wanted=["hard"]) == "Epsilon zeta eta theta two."
    assert tagged_excerpt(TEXT, [], spans) is None


# --------------------------------------------------------------------------- end to end
def test_the_section_has_three_paragraphs(world):
    indexer, client, vault = world
    chunks = [r for r in indexer.index.scan(f"source_path = '{PDF}'", columns=["text"])]
    assert len(chunks) == 1 and len(split_paragraphs(chunks[0]["text"])) == 3


def test_a_quote_tags_one_paragraph_and_the_result_marks_it(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events and what that means" #hard\n')
    indexer.run()
    (hit,) = tagged(client, "hard")
    assert hit["tagged_text"] == P1
    assert P0 in hit["text"] and P1 in hit["text"] and P2 in hit["text"]  # the whole section is still the context


def test_a_page_tag_marks_nothing_narrower(world):
    indexer, client, vault = world
    note(vault, "[[Spans.pdf#page=1]] #revisit\n")
    indexer.run()
    (hit,) = tagged(client, "revisit")
    assert "tagged_text" not in hit


def test_a_whole_pdf_tag_marks_nothing_narrower(world):
    indexer, client, vault = world
    note(vault, "[[Spans.pdf]] #exam\n")
    indexer.run()
    assert all("tagged_text" not in h for h in tagged(client, "exam"))


def test_the_same_tag_on_the_page_and_on_a_quote_means_the_whole_section(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf#page=1]] #hard\n\n[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    (hit,) = tagged(client, "hard")
    assert "tagged_text" not in hit


def test_two_quotes_with_the_same_tag_combine(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "conditional probability and how it is defined" #hard\n\n[[Spans.pdf]] "law of total probability" #hard\n')
    indexer.run()
    (hit,) = tagged(client, "hard")
    assert hit["tagged_text"].startswith(P0) and hit["tagged_text"].endswith(P2) and P1 not in hit["tagged_text"]


def test_a_quote_over_two_paragraphs_marks_both(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "what that means. The third paragraph covers" #hard\n')
    indexer.run()
    (hit,) = tagged(client, "hard")
    assert hit["tagged_text"] == f"{P1}\n\n{P2}"


def test_each_tag_keeps_its_own_span(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n\n[[Spans.pdf]] "law of total probability" #easy\n\n[[Spans.pdf#page=1]] #whole\n')
    indexer.run()
    assert tagged(client, "hard")[0]["tagged_text"] == P1
    assert tagged(client, "easy")[0]["tagged_text"] == P2
    assert "tagged_text" not in tagged(client, "whole")[0]
    # with no tag filter, a result carrying a whole-section tag is simply tagged as a whole
    assert "tagged_text" not in client.call("search_academic_memory", query="independence of events", file_type="pdf", top_k=1)["results"][0]


def test_a_tagged_search_with_a_query_marks_the_span_too(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    (hit,) = client.call("search_academic_memory", query="probability", tags="hard", file_type="pdf")["results"]
    assert hit["tagged_text"] == P1


def test_get_source_chunk_marks_it_too(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    chunk_id = tagged(client, "hard")[0]["chunk_id"]
    assert client.call("get_source_chunk", chunk_id=chunk_id)["chunk"]["tagged_text"] == P1


def test_removing_the_line_removes_the_span(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    assert indexer.index.tag_span_rows(chunk_ids=[tagged(client, "hard")[0]["chunk_id"]])
    note(vault, "no longer tagged\n")
    indexer.run()
    assert tagged(client, "hard") == []
    assert VectorIndex(indexer.config).tag_spans_table.count_rows() == 0


def test_editing_the_quote_moves_the_span(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    note(vault, '[[Spans.pdf]] "law of total probability" #hard\n')
    indexer.run()
    assert tagged(client, "hard")[0]["tagged_text"] == P2


def test_the_span_survives_reindexing_the_pdf_and_a_rebuild(world, pdf_factory):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    pdf_factory(vault / PDF, [{"heading": "Topic", "body": [P0, P0B, "", "", P1, "", "", P2, "", "", "A new fourth paragraph."]}], header=None, footer=None)
    assert PDF in indexer.run().modified
    assert tagged(client, "hard")[0]["tagged_text"] == P1
    indexer.run(full=True)
    assert tagged(client, "hard")[0]["tagged_text"] == P1


def test_deleting_the_note_or_the_pdf_leaves_no_stale_spans(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    assert indexer.index.tag_spans_table.count_rows() == 1
    (vault / PDF).unlink()
    indexer.run()
    assert indexer.index.tag_spans_table.count_rows() == 0


def test_browse_mode_lists_the_span(world):
    indexer, client, vault = world
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    out = client.call("search_academic_memory", tags="hard")
    assert out["mode"] == "browse"
    assert [r["tagged_text"] for r in out["results"] if r["file_type"] == "pdf"] == [P1]


def test_the_write_tool_reports_exactly_which_paragraph_it_tagged(world):
    indexer, client, vault = world
    res = client.call("tag_pdf_passage", pdf="Spans", tags="hard", quote="independence of events")
    (section,) = res["sections"]
    assert section["tagged_text"] == P1
    assert tagged(client, "hard")[0]["tagged_text"] == P1


def test_the_write_tool_omits_tagged_text_for_a_page_tag(world):
    indexer, client, vault = world
    (section,) = client.call("tag_pdf_passage", pdf="Spans", tags="revisit", page=1)["sections"]
    assert "tagged_text" not in section


# --------------------------------------------------------------------------- nothing about retrieval changes
def test_chunks_their_text_and_their_vectors_are_untouched_by_a_span_tag(world):
    indexer, client, vault = world
    before = sorted(indexer.index.scan(f"source_path = '{PDF}'", columns=["chunk_id", "text", "heading_path", "page_start"]), key=lambda r: r["chunk_id"])
    vectors = {r["chunk_id"]: list(r["vector"]) for r in indexer.index.chunks.search().where(f"source_path = '{PDF}'").limit(10).to_list()}
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    after = sorted(indexer.index.scan(f"source_path = '{PDF}'", columns=["chunk_id", "text", "heading_path", "page_start"]), key=lambda r: r["chunk_id"])
    assert after == before
    assert {r["chunk_id"]: list(r["vector"]) for r in indexer.index.chunks.search().where(f"source_path = '{PDF}'").limit(10).to_list()} == vectors


def test_the_chunk_table_schema_is_unchanged(world):
    indexer, client, vault = world
    assert indexer.index.chunks.schema.names == chunks_schema(indexer.embedder.dimension).names


def test_an_index_made_before_spans_existed_still_works(world):
    indexer, client, vault = world
    indexer.index.db.drop_table("tag_spans")
    fresh = VectorIndex(indexer.config)
    assert fresh.tag_span_rows(chunk_ids=["x"]) == []  # created on demand, no rebuild needed
    note(vault, '[[Spans.pdf]] "independence of events" #hard\n')
    indexer.run()
    assert tagged(client, "hard")[0]["tagged_text"] == P1
