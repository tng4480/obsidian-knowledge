"""Paragraph-level tags: an inline #tag belongs to the block (paragraph, list, table ...) it is written in.

Frontmatter tags stay note-level (every chunk). A tag filter must return exactly the tagged passage, never the
paragraphs next to it.
"""

from __future__ import annotations

import pytest

from acadmem.ingest.links import is_tag_only
from acadmem.ingest.markdown import parse_markdown
from acadmem.mcp_server import AcademicMemoryService
from acadmem.search import Searcher

NOTE = """---
tags: [revision]
---
# 3F7 revision

## Entropy

Entropy measures the average surprise of a source.

## Huffman coding

Huffman codes are built by merging the two least likely symbols.

I do not understand why the tree is optimal. #revisit

Kraft's inequality bounds the codeword lengths.

## Channel capacity

Capacity is the maximum mutual information over input distributions.
"""


def blocks(parsed):
    return [(b.text.split("\n")[0][:40], b.tags) for s in parsed.sections for b in s.blocks]


def tags_by_chunk(chunks):
    return {c.text.split("\n")[0][:40]: c.tags for c in chunks}


# ------------------------------------------------------------------ parsing
def test_tag_belongs_to_its_own_paragraph_only():
    got = dict(blocks(parse_markdown(NOTE)))
    key = "I do not understand why the tree is opti"
    assert got[key] == ("revisit",)
    others = {k: v for k, v in got.items() if k != key}
    assert all(v == () for v in others.values())


def test_document_tags_still_list_every_tag_and_note_tags_are_frontmatter_only():
    parsed = parse_markdown(NOTE)
    assert parsed.tags == ["revision", "revisit"]  # what list_indexed_sources / get_document report
    assert parsed.note_tags == ["revision"]  # what every chunk gets


def test_a_tag_line_below_a_paragraph_tags_that_paragraph():
    text = "# A\n\nFirst paragraph about poles.\n\n#exam\n\nSecond paragraph about zeros.\n"
    got = blocks(parse_markdown(text))
    assert got[0] == ("First paragraph about poles.", ("exam",))
    assert got[1] == ("#exam", ("exam",))  # the tag line stays with it (same chunk)
    assert got[2] == ("Second paragraph about zeros.", ())


def test_tag_line_with_several_tags():
    got = dict(blocks(parse_markdown("# A\n\nSome claim.\n\n#exam #revisit\n\nUnrelated.\n")))
    assert got["Some claim."] == ("exam", "revisit")
    assert got["Unrelated."] == ()


def test_a_tag_line_at_the_top_of_a_section_tags_the_paragraph_below():
    got = dict(blocks(parse_markdown("# A\n\n#exam\n\nThe tagged paragraph.\n\nNot tagged.\n")))
    assert got["The tagged paragraph."] == ("exam",)
    assert got["Not tagged."] == ()


def test_a_tag_on_a_heading_line_tags_the_first_paragraph_below():
    got = dict(blocks(parse_markdown("# A\n\n## Huffman #revisit\n\nFirst under it.\n\nSecond under it.\n")))
    assert got["First under it."] == ("revisit",)
    assert got["Second under it."] == ()


def test_tag_does_not_leak_into_the_next_section():
    got = dict(blocks(parse_markdown("# A\n\nTagged. #x\n\n# B\n\nFresh section.\n")))
    assert got["Fresh section."] == ()


def test_tags_are_case_insensitive_and_nested_tags_work():
    got = dict(blocks(parse_markdown("# A\n\nA line. #ReVisit #3F7/Huffman\n")))
    assert got["A line. #ReVisit #3F7/Huffman"] == ("revisit", "3f7/huffman")


def test_a_tag_inside_a_list_or_table_tags_that_whole_block():
    got = blocks(parse_markdown("# A\n\n- point one #exam\n- point two\n\nPlain.\n"))
    assert got[0][1] == ("exam",) and got[1][1] == ()


@pytest.mark.parametrize(
    "text",
    [
        "# A\n\n```python\nx = 1  # not a tag\n#alsonot\n```\n\nPlain.\n",
        "# A\n\nInline `#code` and math $a #b$ here.\n",
        "# A\n\nA link [text](https://example.com/page#anchor) and a colour &#123;.\n",
        "# A\n\nIssue #123 and a bare # sign.\n",
        "# A\n\nA double ## hash and ##notatag and a trailing #\n",
    ],
)
def test_things_that_only_look_like_tags_are_not_tags(text):
    parsed = parse_markdown(text)
    assert parsed.tags == []
    assert all(b.tags == () for s in parsed.sections for b in s.blocks)


@pytest.mark.parametrize("tag", ["3F7", "3f7/huffman", "1a", "_private", "exam-2024", "a/b/c"])
def test_module_style_and_nested_tags_are_recognised(tag):
    parsed = parse_markdown(f"# A\n\nA claim worth keeping. #{tag}\n")
    assert parsed.tags == [tag.lower()]


def test_is_tag_only():
    assert is_tag_only("#exam")
    assert is_tag_only("#exam #revisit")
    assert is_tag_only("  #exam,  #revisit.  ")
    assert not is_tag_only("A sentence #exam")
    assert not is_tag_only("no tags here")
    assert not is_tag_only("`#code`")


# ------------------------------------------------------------------ chunking and retrieval
@pytest.fixture
def note_path(vault):
    path = vault / "3F7" / "Notes" / "Revision.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NOTE, encoding="utf-8")
    return path


@pytest.fixture
def chunks(note_path, indexer):
    indexer.run(full=True)
    rows = indexer.index.scan("source_path = '3F7/Notes/Revision.md'")
    return sorted(rows, key=lambda r: r["chunk_index"])


def test_only_the_tagged_paragraph_carries_the_tag_and_it_is_its_own_chunk(chunks):
    tagged = [c for c in chunks if "revisit" in c["tags"]]
    assert len(tagged) == 1
    assert "I do not understand why the tree is optimal" in tagged[0]["text"]
    assert "Huffman codes are built" not in tagged[0]["text"]  # the neighbours are not dragged in
    assert "Kraft" not in tagged[0]["text"]


def test_frontmatter_tags_still_apply_to_every_chunk(chunks):
    assert len(chunks) == 3  # head (small sections merged), the tagged paragraph on its own, tail
    assert all("revision" in c["tags"] for c in chunks)


def test_untagged_neighbours_still_merge_normally(chunks):
    untagged = [c for c in chunks if "revisit" not in c["tags"]]
    assert any("Huffman codes are built" in c["text"] for c in untagged)
    assert any("Kraft" in c["text"] for c in untagged)


def test_search_by_tag_returns_just_the_tagged_passage(note_path, indexer, config, fake_embedder):
    indexer.run(full=True)
    hits = Searcher(config, embedder=fake_embedder).search("Huffman", top_k=10, filters={"tags": "revisit"})
    assert len(hits) == 1
    assert "I do not understand" in hits[0].text
    assert hits[0].heading_path[-1] == "Huffman coding"  # provenance: which heading it sits under
    assert "revisit" in hits[0].tags


def test_tag_filter_accepts_hash_and_any_case(note_path, indexer, config, fake_embedder):
    indexer.run(full=True)
    searcher = Searcher(config, embedder=fake_embedder)
    for spelling in ("#revisit", "REVISIT", "#ReVisit"):
        assert len(searcher.search("Huffman", top_k=10, filters={"tags": spelling})) == 1, spelling


def test_note_level_tag_returns_every_chunk_of_the_note(note_path, indexer, config, fake_embedder):
    indexer.run(full=True)
    hits = Searcher(config, embedder=fake_embedder).search("entropy", top_k=20, filters={"tags": "revision"})
    assert {h.source_path for h in hits} == {"3F7/Notes/Revision.md"} and len(hits) == 3


def test_list_indexed_sources_finds_notes_containing_a_paragraph_tag(note_path, indexer, config, fake_embedder):
    indexer.run(full=True)
    service = AcademicMemoryService(config, embedder=fake_embedder)
    listed = service.list_indexed_sources(tags="revisit")
    assert [s["source"] for s in listed["sources"]] == ["3F7/Notes/Revision.md"]
    assert "revisit" in service.get_document("3F7/Notes/Revision.md")["tags"]


def test_adding_and_removing_a_tag_updates_the_index(note_path, indexer, config, fake_embedder):
    indexer.run(full=True)
    searcher = Searcher(config, embedder=fake_embedder)
    assert len(searcher.search("Kraft", top_k=10, filters={"tags": "exam"})) == 0

    note_path.write_text(NOTE.replace("bounds the codeword lengths.", "bounds the codeword lengths. #exam"), encoding="utf-8")
    assert indexer.run().modified == ["3F7/Notes/Revision.md"]
    hits = searcher.search("Kraft", top_k=10, filters={"tags": "exam"})
    assert len(hits) == 1 and "Kraft" in hits[0].text and "Huffman codes are built" not in hits[0].text

    note_path.write_text(NOTE, encoding="utf-8")
    indexer.run()
    assert len(searcher.search("Kraft", top_k=10, filters={"tags": "exam"})) == 0


def test_a_long_tagged_paragraph_is_split_but_every_piece_keeps_the_tag(vault, indexer):
    long_para = " ".join(f"sentence{i} explains one more detail of the proof." for i in range(120)) + " #proof"
    (vault / "3F7" / "Notes").mkdir(parents=True, exist_ok=True)
    (vault / "3F7" / "Notes" / "Long.md").write_text(f"# Long\n\n{long_para}\n\nAn untagged closing remark.\n", encoding="utf-8")
    indexer.run(full=True)
    rows = indexer.index.scan("source_path = '3F7/Notes/Long.md'")
    tagged = [r for r in rows if "proof" in r["tags"]]
    assert len(tagged) >= 2
    assert all("untagged closing remark" not in r["text"] for r in tagged)
    assert [r for r in rows if "untagged closing remark" in r["text"]][0]["tags"] == []


def test_pdf_chunks_are_untouched_by_tag_handling(indexer):
    indexer.run(full=True)
    pdf_rows = indexer.index.scan("file_type = 'pdf'")
    assert pdf_rows and all(r["tags"] == [] for r in pdf_rows)


def test_ai_notes_can_tag_paragraphs_too(indexer, config, fake_embedder):
    from acadmem.vault_write import VaultWriter

    VaultWriter(config, indexer).create_note(
        "3F7/Notes/AI tags.md", "# Summary\n\nPlain intro.\n\nKey theorem: the source coding bound. #exam\n", tags=["ai"]
    )
    hits = Searcher(config, embedder=fake_embedder).search("theorem", top_k=10, filters={"tags": "exam"})
    assert [h.source_path for h in hits] == ["3F7/Notes/AI tags.md"] and "Key theorem" in hits[0].text
