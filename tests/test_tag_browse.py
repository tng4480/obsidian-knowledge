"""Asking Codex "show me everything I tagged #revisit": an empty query plus a tag lists the tagged passages."""

from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from acadmem.mcp_server import AcademicMemoryService, build_server

NOTE_A = """# 3F7 revision

## Huffman

Huffman codes merge the two least likely symbols.

Why is the tree optimal? #revisit

## Entropy

Entropy is average surprise. #exam
"""

NOTE_B = """# 3F4 revision

Viterbi keeps the best path into each state. #revisit

Trellis diagrams unroll the state machine.
"""


@pytest.fixture
def service(vault, indexer, config, fake_embedder):
    for rel, text in {"3F7/Notes/Revision.md": NOTE_A, "3F4/Notes/Revision.md": NOTE_B}.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    indexer.run(full=True)
    return AcademicMemoryService(config, embedder=fake_embedder)


def call(service, **arguments):
    result = asyncio.run(build_server(service).call_tool("search_academic_memory", arguments))
    assert not result.is_error
    return result.structured_content


def test_empty_query_with_a_tag_lists_every_tagged_passage_in_vault_order(service):
    out = call(service, tags="revisit")
    assert out["mode"] == "browse" and out["count"] == out["total_matches"] == 2
    assert [r["source"] for r in out["results"]] == ["3F4/Notes/Revision.md", "3F7/Notes/Revision.md"]
    assert [r["rank"] for r in out["results"]] == [1, 2]
    assert "Viterbi keeps the best path" in out["results"][0]["text"]
    assert "Why is the tree optimal" in out["results"][1]["text"]
    assert "Huffman codes merge" not in out["results"][1]["text"]  # the neighbouring paragraph is not included
    assert out["results"][1]["section"].endswith("Huffman")  # provenance survives


def test_browse_respects_other_filters(service):
    out = call(service, tags="#revisit", module="3F7")
    assert [r["source"] for r in out["results"]] == ["3F7/Notes/Revision.md"]


def test_browse_limit_and_hint(service):
    out = call(service, tags="revisit", top_k=1)
    assert out["count"] == 1 and out["total_matches"] == 2 and "Showing 1 of 2" in out["hint"]


def test_browse_with_expansion_attaches_neighbours(service):
    out = call(service, tags="revisit", module="3F4", expand="neighbors:1")
    pieces = out["results"][0]["context"]
    assert any(p["is_match"] for p in pieces) and any("Trellis" in p["text"] for p in pieces)


def test_unknown_tag_explains_itself(service):
    out = call(service, tags="nonexistent")
    assert out["count"] == 0 and "Nothing carries that tag" in out["hint"]


def test_a_query_with_a_tag_still_ranks_within_the_tag(service):
    out = call(service, query="Viterbi trellis", tags="revisit")
    assert out["mode"] != "browse"
    assert {r["source"] for r in out["results"]} == {"3F4/Notes/Revision.md", "3F7/Notes/Revision.md"}
    assert all("revisit" in r["tags"] for r in out["results"])


def test_an_empty_query_without_a_tag_is_still_an_error(service):
    with pytest.raises(ToolError, match="empty"):
        asyncio.run(build_server(service).call_tool("search_academic_memory", {"module": "3F7"}))


def test_browse_without_an_index_is_an_explained_error(config, fake_embedder):
    service = AcademicMemoryService(config, embedder=fake_embedder)
    with pytest.raises(ToolError, match="No index yet"):
        asyncio.run(build_server(service).call_tool("search_academic_memory", {"tags": "revisit"}))
