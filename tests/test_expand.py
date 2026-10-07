from __future__ import annotations

import pytest

from acadmem.expand import ContextExpander, ExpansionSpec
from acadmem.index import CHUNK_COLUMNS, VectorIndex

LONG = " ".join(f"word{i}" for i in range(400))  # forces the chunker to split a single section

NOTE = f"""# Course

Course overview paragraph that is long enough to stand alone as its own chunk in the index.

## Alpha

Alpha section text describing the first topic with enough words to stay a separate chunk here.

### Alpha detail

Alpha detail text describing a refinement of the first topic with enough words to be separate.

## Beta

{LONG}

## Gamma

Gamma section text describing the last topic with enough words to remain a separate chunk too.
"""


@pytest.fixture
def built(indexer, vault):
    (vault / "3B1" / "Notes" / "course.md").write_text(NOTE, encoding="utf-8")
    indexer.run(full=True)
    return indexer


@pytest.fixture
def expander(built, config):
    return ContextExpander(VectorIndex(config), config)


def rows_for(config, source_path):
    index = VectorIndex(config)
    rows = index.scan(f"source_path = '{source_path}'", columns=CHUNK_COLUMNS)
    return sorted(rows, key=lambda r: r["chunk_index"])


def by_heading(rows, heading):
    return next(r for r in rows if r["heading_path"] and r["heading_path"][-1] == heading)


def test_spec_parsing():
    assert ExpansionSpec.parse("neighbors:2") == ExpansionSpec(neighbors=2)
    assert ExpansionSpec.parse("Neighbours:3") == ExpansionSpec(neighbors=3)
    assert ExpansionSpec.parse("section+parent") == ExpansionSpec(section=True, parent=True)
    assert ExpansionSpec.parse("neighbors:1+pages:2+section") == ExpansionSpec(neighbors=1, pages=2, section=True)
    assert ExpansionSpec.parse({"neighbors": 2, "parent": True}) == ExpansionSpec(neighbors=2, parent=True)
    assert ExpansionSpec.parse("none").describe() == "none"
    assert ExpansionSpec.parse("neighbors:1+parent").describe() == "neighbors:1+parent"
    assert ExpansionSpec.parse(None, None) == ExpansionSpec(neighbors=1)


@pytest.mark.parametrize("bad", ["neighbours:x", "wibble", "neighbors:99", "neighbors:-1"])
def test_bad_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        ExpansionSpec.parse(bad)
    with pytest.raises(ValueError):
        ExpansionSpec.parse({"bogus": 1})


def test_defaults_come_from_config(config):
    assert ExpansionSpec.parse(None, config) == ExpansionSpec(neighbors=config.context_neighbors, pages=config.context_pages)


def test_neighbors_marks_the_hit_and_orders_by_reading_order(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    hit = by_heading(rows, "Alpha")
    ctx = expander.expand(hit, "neighbors:1")
    assert [p.chunk_index for p in ctx.pieces] == sorted(p.chunk_index for p in ctx.pieces)
    assert [p.chunk_index for p in ctx.pieces] == [hit["chunk_index"] - 1, hit["chunk_index"], hit["chunk_index"] + 1]
    assert [p.is_hit for p in ctx.pieces] == [False, True, False]
    assert {p.role for p in ctx.pieces} == {"hit", "neighbor"}
    assert ctx.spec == "neighbors:1" and not ctx.truncated


def test_neighbors_stay_inside_the_document_and_clip_at_the_ends(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    first = expander.expand(rows[0], "neighbors:2")
    assert first.pieces[0].is_hit and {p.source_path for p in first.pieces} == {"3B1/Notes/course.md"}
    assert len(first.pieces) == 3
    last = expander.expand(rows[-1], "neighbors:5")
    assert last.pieces[-1].is_hit and len(last.pieces) == min(len(rows), 6)


def test_section_returns_every_chunk_of_a_split_section(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    beta = [r for r in rows if r["heading_path"] and r["heading_path"][-1] == "Beta"]
    assert len(beta) >= 2, "Beta is long enough to be split"
    ctx = expander.expand(beta[0], "section")
    assert {p.chunk_id for p in ctx.pieces} == {r["chunk_id"] for r in beta}
    assert [p.role for p in ctx.pieces if p.chunk_id == beta[0]["chunk_id"]] == ["hit"]
    assert all(p.role == "section" for p in ctx.pieces if p.chunk_id != beta[0]["chunk_id"])


def test_parent_adds_the_enclosing_section(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    detail = by_heading(rows, "Alpha detail")
    ctx = expander.expand(detail, "parent")
    parent_ids = {p.chunk_id for p in ctx.pieces if p.role == "parent"}
    assert parent_ids and by_heading(rows, "Alpha")["chunk_id"] in parent_ids
    assert ctx.pieces[-1].is_hit


def test_parent_of_a_top_level_chunk_adds_nothing(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    ctx = expander.expand(rows[0], "parent")
    assert [p.chunk_id for p in ctx.pieces] == [rows[0]["chunk_id"]]


def test_pages_expands_a_pdf_by_page_window(expander, config):
    rows = rows_for(config, "3B1/Handouts/Week 5 - Probability.pdf")
    indep = next(r for r in rows if "independent" in r["text"].lower() and r["page_start"] == 3)
    ctx = expander.expand(indep, "pages:1")
    pages = {p for piece in ctx.pieces for p in range(piece.page_start, piece.page_end + 1)}
    assert {2, 3, 4} <= pages
    assert {p.source_path for p in ctx.pieces} == {"3B1/Handouts/Week 5 - Probability.pdf"}
    assert all(p.page_start is not None for p in ctx.pieces)


def test_pages_is_a_noop_for_markdown(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    assert len(expander.expand(rows[1], "pages:2").pieces) == 1


def test_combined_spec_dedupes_and_keeps_most_specific_role(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    detail = by_heading(rows, "Alpha detail")
    ctx = expander.expand(detail, "neighbors:1+parent")
    ids = [p.chunk_id for p in ctx.pieces]
    assert len(ids) == len(set(ids))
    alpha = next(p for p in ctx.pieces if p.chunk_id == by_heading(rows, "Alpha")["chunk_id"])
    assert alpha.role == "parent"  # adjacent *and* the parent: the parent reason wins over plain adjacency
    assert next(p for p in ctx.pieces if p.chunk_id == detail["chunk_id"]).role == "hit"


def test_every_piece_keeps_provenance(expander, config):
    rows = rows_for(config, "3B1/Handouts/Week 5 - Probability.pdf")
    ctx = expander.expand(rows[0], "neighbors:1+pages:1")
    for piece in ctx.pieces:
        d = piece.to_dict()
        assert d["source_path"].endswith(".pdf") and d["page_start"] and d["chunk_id"].startswith(piece.document_id)
        assert piece.location.startswith("3B1/Handouts/Week 5 - Probability.pdf p.")
        assert d["text"]
    assert "(match)" in ctx.text and ctx.to_dict()["pieces"][0]["location"]


def test_expand_accepts_search_results_and_chunk_ids(expander, config, fake_embedder):
    from acadmem.search import Searcher

    hit = Searcher(config, embedder=fake_embedder).search("memoryless property", top_k=1)[0]
    by_result = expander.expand(hit, "neighbors:1")
    by_id = expander.expand(hit.chunk_id, "neighbors:1")
    assert [p.chunk_id for p in by_result.pieces] == [p.chunk_id for p in by_id.pieces]
    assert by_result.hit_chunk_id == hit.chunk_id
    with pytest.raises(KeyError):
        expander.expand("nope#nothing")
    with pytest.raises(TypeError):
        expander.expand(42)


def test_max_chunks_truncates_around_the_hit(built, config):
    small = ContextExpander(VectorIndex(config), config, max_chunks=3)
    rows = rows_for(config, "3B1/Notes/course.md")
    mid = by_heading(rows, "Alpha detail")
    ctx = small.expand(mid, "neighbors:4")
    assert ctx.truncated and len(ctx.pieces) == 3
    assert any(p.is_hit for p in ctx.pieces)
    assert [p.chunk_index for p in ctx.pieces] == sorted(p.chunk_index for p in ctx.pieces)


def test_expand_many_unions_and_keeps_hit_roles(expander, config):
    rows = rows_for(config, "3B1/Notes/course.md")
    a, b = by_heading(rows, "Alpha"), by_heading(rows, "Alpha detail")
    pieces = expander.expand_many([a, b], "neighbors:1")
    ids = [p.chunk_id for p in pieces]
    assert len(ids) == len(set(ids))
    roles = {p.chunk_id: p.role for p in pieces}
    assert roles[a["chunk_id"]] == "hit" and roles[b["chunk_id"]] == "hit"
