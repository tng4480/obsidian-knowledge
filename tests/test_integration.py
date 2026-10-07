"""Whole-system lifecycle on a temporary vault, through the same tool surface Codex uses.

index -> semantic query -> user edits a note in Obsidian -> verify -> user deletes it -> verify ->
AI writes a note -> verify -> throw the index away and rebuild -> same answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from pathlib import Path

import pytest

from acadmem.indexer import Indexer
from acadmem.mcp_server import AcademicMemoryService, build_server


@pytest.fixture
def system(vault, config, fake_embedder):
    service = AcademicMemoryService(config, embedder=fake_embedder)
    server = build_server(service)

    def call(name, **arguments):
        result = asyncio.run(server.call_tool(name, arguments))
        assert not result.is_error
        return result.structured_content

    def sources(query, **kw):
        return [r["source"] for r in call("search_academic_memory", query=query, top_k=5, **kw)["results"]]

    service.indexer.run(full=True)
    return service, call, sources


def test_full_lifecycle(system, vault, config, fake_embedder):
    service, call, sources = system
    note = vault / "3F1" / "Notes" / "Lifecycle.md"

    # 1. semantic query over what was indexed (markdown and pdf)
    assert sources("posterior probability of disease given a positive test")[0] == "3B1/Questions/Bayes.md"
    assert sources("independence of events", file_type="pdf")[0].endswith("Week 5 - Probability.pdf")
    assert sources("memoryless property next state depends only on current state")[0] == "3F1/Notes/deep/er/Markov.md"

    # 2. a new note appears in Obsidian: invisible until the next sync, then found
    note.write_text("# Lifecycle\n\nThe zygomorphic frobnicator converges quadratically.\n", encoding="utf-8")
    assert "3F1/Notes/Lifecycle.md" not in sources("zygomorphic frobnicator converges")
    assert service.indexer.run().added == ["3F1/Notes/Lifecycle.md"]
    assert sources("zygomorphic frobnicator converges")[0] == "3F1/Notes/Lifecycle.md"

    # 3. the user edits it: the new wording is found, the old wording is gone (no stale chunks)
    note.write_text("# Lifecycle\n\nThe quixotic reticulator diverges linearly.\n", encoding="utf-8")
    report = service.indexer.run()
    assert report.modified == ["3F1/Notes/Lifecycle.md"] and not report.added and not report.deleted
    assert sources("quixotic reticulator diverges")[0] == "3F1/Notes/Lifecycle.md"
    assert "3F1/Notes/Lifecycle.md" not in sources("zygomorphic frobnicator converges")
    stale = [c for c in service.index.scan(chunk_filter(service, "3F1/Notes/Lifecycle.md"))]
    assert len(stale) == 1 and "zygomorphic" not in stale[0]["text"]

    # 4. unrelated notes were not re-embedded by any of that
    assert not service.indexer.run().changed

    # 5. the user deletes the note: it disappears from search, listing and chunk lookup
    note.unlink()
    assert service.indexer.run().deleted == ["3F1/Notes/Lifecycle.md"]
    assert "3F1/Notes/Lifecycle.md" not in sources("quixotic reticulator diverges")
    listed = {s["source"] for s in call("list_indexed_sources")["sources"]}
    assert "3F1/Notes/Lifecycle.md" not in listed
    assert service.index.count_chunks() == sum(s["chunks"] for s in call("list_indexed_sources")["sources"])

    # 6. the AI writes a note: searchable at once, flagged, and the human notes are byte-identical
    human_before = tree_digest(vault)
    call("create_note", path="3F1/Notes/AI Lifecycle.md", content="# AI\n\nThe flibbertigibbet operator is self-adjoint.")
    assert sources("flibbertigibbet operator self-adjoint", ai_generated=True)[0] == "3F1/Notes/AI Lifecycle.md"
    assert tree_digest(vault) == {**human_before, "3F1/Notes/AI Lifecycle.md": tree_digest(vault)["3F1/Notes/AI Lifecycle.md"]}

    # 7. Obsidian is the source of truth: delete the whole index, rebuild, get the same answers
    before = {q: sources(q) for q in ("posterior probability of disease", "flibbertigibbet operator self-adjoint", "stationary distribution")}
    del service
    shutil.rmtree(config.index_dir)
    rebuilt = AcademicMemoryService(config, embedder=fake_embedder)
    rebuilt_server = build_server(rebuilt)
    Indexer(config, embedder=fake_embedder, index=rebuilt.index).run(full=True)

    def rebuilt_sources(query):
        result = asyncio.run(rebuilt_server.call_tool("search_academic_memory", {"query": query, "top_k": 5}))
        return [r["source"] for r in result.structured_content["results"]]

    assert {q: rebuilt_sources(q) for q in before} == before


def test_a_broken_file_never_stops_the_rest_being_searchable(system, vault):
    service, call, sources = system
    (vault / "3B1" / "Handouts" / "Corrupt.pdf").write_bytes(b"%PDF-1.7 this is not really a pdf")
    (vault / "3B1" / "Notes" / "Fine.md").write_text("# Fine\n\nThe wibbly eigenfunction is orthogonal.\n", encoding="utf-8")
    report = service.indexer.run()
    assert [e["source_path"] for e in report.errors] == ["3B1/Handouts/Corrupt.pdf"]
    assert report.added == ["3B1/Notes/Fine.md"]
    assert sources("wibbly eigenfunction orthogonal")[0] == "3B1/Notes/Fine.md"
    failed = call("index_status")["failed_files"]
    assert any(f["source"].endswith("Corrupt.pdf") for f in failed)
    assert sources("posterior probability of disease")[0] == "3B1/Questions/Bayes.md"


def chunk_filter(service, source: str) -> str:
    from acadmem.index import sql_quote

    return f"source_path = {sql_quote(source)}"


def tree_digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
