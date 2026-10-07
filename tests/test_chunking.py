"""Chunker tests. Uses a whitespace token counter so no model is needed."""

from __future__ import annotations

import re

import pytest

from acadmem.chunking import Chunker
from acadmem.ingest.markdown import parse_markdown
from acadmem.models import (
    CODE,
    MATH,
    PARAGRAPH,
    TABLE,
    Block,
    Document,
    ParsedDocument,
    Section,
)


def count(text: str) -> int:
    return len(text.split())


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def make_doc(sections: list[Section], source="3B1/Questions/x.md", file_type="md") -> ParsedDocument:
    doc = Document(
        id="doc1", source_path=source, file_type=file_type, module="3B1", document_type="Questions",
        modification_time=1.0, title="X", tags=["t"],
    )
    return ParsedDocument(document=doc, sections=sections)


def sec(sid, title, level, path, parent, blocks) -> Section:
    return Section(sid, title, level, tuple(path), parent, list(blocks))


def chunker(**kw) -> Chunker:
    kw.setdefault("max_tokens", 50)
    kw.setdefault("min_tokens", 10)
    kw.setdefault("overlap_tokens", 5)
    return Chunker(count, **kw)


def test_small_blocks_pack_into_one_chunk_and_keep_structure():
    s = sec("s1", "Bayes", 1, ["Bayes"], None, [Block("alpha beta", PARAGRAPH, 3), Block("gamma delta", PARAGRAPH, 3)])
    chunks = chunker().chunk_document(make_doc([s]))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == "alpha beta\n\ngamma delta"
    assert c.heading_path == ["Bayes"] and c.section_id == "doc1#s1" and c.parent_section_id is None
    assert (c.page_number, c.page_start, c.page_end) == (3, 3, 3)


def test_provenance_fields_copied_from_document():
    s = sec("s1", "H", 1, ["H"], None, [Block(words(5), PARAGRAPH)])
    c = chunker().chunk_document(make_doc([s]))[0]
    assert (c.source_path, c.file_type, c.module, c.document_type) == ("3B1/Questions/x.md", "md", "3B1", "Questions")
    assert c.document_id == "doc1" and c.chunk_id == "doc1:00000" and c.tags == ["t"]
    assert c.modification_time == 1.0 and c.page_number is None


def test_chunks_never_exceed_max_tokens():
    blocks = [Block(words(30, f"a{i}_"), PARAGRAPH, 1 + i) for i in range(12)]
    blocks.append(Block(words(400, "huge"), PARAGRAPH, 20))
    s = sec("s1", "Big", 1, ["Big"], None, blocks)
    chunks = chunker().chunk_document(make_doc([s]))
    assert len(chunks) > 5
    assert all(count(c.text) <= 50 for c in chunks)


def test_oversized_prose_is_split_with_overlap_only_there():
    s = sec("s1", "Long", 1, ["Long"], None, [Block(words(180), PARAGRAPH)])
    chunks = chunker(overlap_tokens=6).chunk_document(make_doc([s]))
    assert len(chunks) >= 4
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text.split()[-3:]
        assert tail[-1] in b.text.split()[:10], "forced splits should overlap"
    # Without any oversized block there is no overlap between chunks
    blocks = [Block(words(40, f"p{i}_"), PARAGRAPH) for i in range(3)]
    chunks = chunker().chunk_document(make_doc([sec("s1", "H", 1, ["H"], None, blocks)]))
    seen: set[str] = set()
    for c in chunks:
        toks = set(c.text.split())
        assert not (seen & toks)
        seen |= toks


def test_atomic_math_block_is_not_split_when_it_fits():
    math = "$$\n" + "x + y = z\n" * 5 + "$$"
    s = sec("s1", "Derivation", 1, ["Derivation"], None, [Block(words(30), PARAGRAPH), Block(math, MATH), Block(words(30, "z"), PARAGRAPH)])
    chunks = chunker().chunk_document(make_doc([s]))
    holder = [c for c in chunks if "$$" in c.text]
    assert len(holder) == 1 and holder[0].text.count("$$") == 2


def test_oversized_table_splits_by_rows_and_repeats_header():
    rows = [f"|r{i}|v{i}|" for i in range(60)]
    table = "\n".join(["|key|value|", "|---|---|", *rows])
    s = sec("s1", "T", 1, ["T"], None, [Block(table, TABLE)])
    chunks = chunker(max_tokens=40).chunk_document(make_doc([s]))
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("|key|value|")
        assert count(c.text) <= 40
    body_rows = [line for c in chunks for line in c.text.splitlines() if line.startswith("|r")]
    assert body_rows == rows  # nothing lost, nothing duplicated


def test_oversized_code_block_splits_on_lines():
    code = "\n".join(f"line_{i} = {i}" for i in range(100))
    s = sec("s1", "C", 1, ["C"], None, [Block(code, CODE)])
    chunks = chunker(max_tokens=40).chunk_document(make_doc([s]))
    assert len(chunks) > 1
    assert "\n".join(c.text for c in chunks).count("line_") == 100


def test_big_sections_are_not_mixed_and_keep_parent_links():
    parent = sec("s1", "Parent", 1, ["Parent"], None, [Block(words(45, "p"), PARAGRAPH, 1)])
    child = sec("s2", "Child", 2, ["Parent", "Child"], "s1", [Block(words(45, "c"), PARAGRAPH, 2)])
    sib = sec("s3", "Sibling", 1, ["Sibling"], None, [Block(words(45, "s"), PARAGRAPH, 3)])
    chunks = chunker().chunk_document(make_doc([parent, child, sib]))
    assert [c.section_id for c in chunks] == ["doc1#s1", "doc1#s2", "doc1#s3"]
    assert chunks[1].parent_section_id == "doc1#s1"
    assert chunks[1].heading_path == ["Parent", "Child"]
    assert chunks[0].page_number == 1 and chunks[1].page_number == 2 and chunks[2].page_number == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_tiny_related_sections_merge_with_inline_heading_and_page_range():
    a = sec("s1", "Intro", 1, ["Intro"], None, [Block("short note one", PARAGRAPH, 4)])
    b = sec("s2", "Detail", 2, ["Intro", "Detail"], "s1", [Block("short note two", PARAGRAPH, 5)])
    chunks = chunker().chunk_document(make_doc([a, b]))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section_id == "doc1#s1" and c.heading_path == ["Intro"]  # structural parent = first piece
    assert "Detail" in c.text and "short note two" in c.text  # inline heading not lost
    assert (c.page_start, c.page_end, c.page_number) == (4, 5, 4)


def test_unrelated_small_sections_do_not_merge():
    a = sec("s1", "A", 1, ["A"], None, [Block("tiny a", PARAGRAPH)])
    inner = sec("s2", "A1", 2, ["A", "A1"], "s1", [Block(words(45, "x"), PARAGRAPH)])
    b_parent = sec("s8", "B", 1, ["B"], None, [])
    other = sec("s3", "B1", 2, ["B", "B1"], "s8", [Block("tiny b", PARAGRAPH)])
    # `other` has a different parent than `inner` and is not its descendant -> must stay separate
    chunks = chunker().chunk_document(make_doc([a, inner, b_parent, other]))
    assert any(c.text == "tiny b" for c in chunks)
    assert not any("tiny b" in c.text and "x0" in c.text for c in chunks)
    assert any("tiny a" in c.text and "x0" in c.text for c in chunks)  # parent+child DO merge


def test_empty_sections_produce_no_chunks():
    heading_only = sec("s1", "Only a heading", 1, ["Only a heading"], None, [])
    assert chunker().chunk_document(make_doc([heading_only])) == []


def test_no_content_is_lost():
    md = (
        "# Title\n\nIntro words here.\n\n## One\n\n" + words(120, "one") + "\n\n## Two\n\ntiny\n\n## Three\n\n"
        + words(60, "three") + "\n\n```\ncode line\n```\n"
    )
    parsed = make_doc(parse_markdown(md).sections)
    chunks = chunker().chunk_document(parsed)
    got = " ".join(c.text for c in chunks)
    for token in re.findall(r"\w+", md):
        if token in {"Title", "One", "Three"}:  # heading text lives in heading_path
            continue
        assert token in got, token


def test_embedding_text_has_breadcrumb():
    s = sec("s1", "Parent", 1, ["Parent", "Child"], None, [Block("body text", PARAGRAPH)])
    c = chunker().chunk_document(make_doc([s]))[0]
    assert c.embedding_text("Doc").startswith("Doc > Parent > Child")
    assert c.embedding_text("Doc").endswith("body text")


def test_links_are_extracted_per_chunk():
    s = sec("s1", "H", 1, ["H"], None, [Block("see [[Other note]] and ![[x.pdf#page=3]]", PARAGRAPH)])
    c = chunker().chunk_document(make_doc([s]))[0]
    assert c.links == ["Other note", "x.pdf#page=3"]


@pytest.mark.parametrize("bad", [0, 4])
def test_rejects_tiny_budgets(bad):
    with pytest.raises(ValueError):
        Chunker(count, max_tokens=bad)
