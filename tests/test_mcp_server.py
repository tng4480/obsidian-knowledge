"""The MCP layer: tool surface, result shape (provenance), error reporting and the write guard rails.

Tools are driven through ``MCPServer.call_tool`` (the same path a Codex request takes, minus stdio framing).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from acadmem.mcp_server import AcademicMemoryService, build_server

READ_TOOLS = {
    "search_academic_memory",
    "get_source_chunk",
    "get_context",
    "get_document",
    "list_indexed_sources",
    "index_status",
}
WRITE_TOOLS = {"create_note", "append_to_note", "replace_section"}


class Client:
    """Tiny synchronous wrapper so tests read like tool calls."""

    def __init__(self, server) -> None:
        self.server = server

    def call(self, name: str, **arguments):
        result = asyncio.run(self.server.call_tool(name, arguments))
        assert not result.is_error
        return result.structured_content

    def error(self, name: str, **arguments) -> str:
        """Call a tool that must fail; return the message the model would read."""
        with pytest.raises(ToolError) as info:
            asyncio.run(self.server.call_tool(name, arguments))
        return str(info.value)

    def tools(self):
        return {t.name: t for t in asyncio.run(self.server.list_tools())}


@pytest.fixture
def service(indexer, config, fake_embedder):
    indexer.run(full=True)
    return AcademicMemoryService(config, embedder=fake_embedder)


@pytest.fixture
def client(service) -> Client:
    return Client(build_server(service))


def digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ------------------------------------------------------------------ surface
def test_all_tools_are_listed(client):
    assert set(client.tools()) == READ_TOOLS | WRITE_TOOLS


def test_read_only_mode_registers_no_write_tools(config, fake_embedder, indexer):
    indexer.run(full=True)
    read_only = dataclasses.replace(config, writes_enabled=False)
    server_client = Client(build_server(AcademicMemoryService(read_only, embedder=fake_embedder)))
    assert set(server_client.tools()) == READ_TOOLS


def test_annotations_tell_the_client_which_tools_mutate(client):
    tools = client.tools()
    for name in READ_TOOLS:
        assert tools[name].annotations.read_only_hint is True, name
    for name in WRITE_TOOLS:
        assert tools[name].annotations.read_only_hint is False, name
        assert tools[name].annotations.destructive_hint is (name == "replace_section"), name


def test_no_tool_can_delete(client):
    assert not [n for n in client.tools() if "delete" in n or "remove" in n or "move" in n or "rename" in n]


def test_schemas_expose_filters_and_required_fields(client):
    search = client.tools()["search_academic_memory"].input_schema
    assert search["required"] == ["query"]
    for field in ("top_k", "module", "document_type", "file_type", "tags", "modified_after", "ai_generated", "mode", "expand"):
        assert field in search["properties"], field
    create = client.tools()["create_note"].input_schema
    assert set(create["required"]) == {"path", "content"}


def test_every_tool_has_a_description(client):
    for name, tool in client.tools().items():
        assert tool.description and len(tool.description) > 20, name


# ------------------------------------------------------------------ search
def test_search_results_carry_full_provenance(client):
    out = client.call("search_academic_memory", query="posterior probability of disease", top_k=3)
    assert out["count"] == len(out["results"]) >= 1
    top = out["results"][0]
    for field in ("rank", "score", "source", "section", "heading_path", "text", "chunk_id", "file_type", "module"):
        assert field in top, field
    assert top["source"] == "3B1/Questions/Bayes.md"
    assert top["module"] == "3B1"
    assert "posterior" in top["text"].lower()


@pytest.mark.parametrize("mode", ["vector", "fts", "hybrid"])
def test_search_score_is_a_cosine_similarity_in_every_mode(client, mode):
    out = client.call("search_academic_memory", query="posterior probability of disease", top_k=4, mode=mode)
    assert out["mode"] == mode and out["results"]
    for r in out["results"]:
        assert -1.0 <= r["score"] <= 1.0
        assert "rank_score" not in r  # the fusion/BM25 value is an internal ordering detail, not shown to the model
    assert out["results"][0]["score"] > 0.1  # not the ~0.016-0.033 reciprocal-rank-fusion value
    assert [r["rank"] for r in out["results"]] == list(range(1, len(out["results"]) + 1))


def test_search_tool_description_explains_score_without_apologising(client):
    desc = client.tools()["search_academic_memory"].description
    assert "cosine similarity" in desc
    assert "rank-fusion" not in desc and "does not mean a poor match" not in desc


def test_pdf_results_report_the_page(client):
    out = client.call("search_academic_memory", query="independence of events", file_type="pdf", top_k=3)
    assert out["results"], out
    assert all(r["file_type"] == "pdf" and r["page"] is not None for r in out["results"])


def test_filters_are_applied(client):
    out = client.call("search_academic_memory", query="probability", module="3F1", top_k=10)
    assert out["results"] and {r["module"] for r in out["results"]} == {"3F1"}
    assert out["filters"] == {"module": "3F1"}


def test_filters_accept_a_list(client):
    out = client.call("search_academic_memory", query="probability", module=["3F1", "3B1"], top_k=10)
    assert {r["module"] for r in out["results"]} <= {"3F1", "3B1"}


def test_no_match_returns_a_hint_not_an_error(client):
    out = client.call("search_academic_memory", query="probability", module="9Z9")
    assert out["count"] == 0 and "hint" in out


def test_search_with_expansion_attaches_context(client):
    out = client.call("search_academic_memory", query="pairwise independence", top_k=1, expand="neighbors:1")
    assert out["results"][0]["context"]
    assert any(p["is_match"] for p in out["results"][0]["context"])


def test_search_modes(client):
    for mode in ("vector", "fts", "hybrid"):
        out = client.call("search_academic_memory", query="stationary distribution", mode=mode, top_k=2)
        assert out["mode"] == mode and out["results"], mode


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"query": ""}, "empty"),
        ({"query": "   "}, "empty"),
        ({"query": "x", "mode": "nope"}, "mode"),
        ({"query": "x", "expand": "bogus:1"}, "expan"),
        ({"query": "x", "modified_after": "not-a-date"}, "date"),
    ],
)
def test_bad_search_arguments_explain_themselves(client, arguments, fragment):
    message = client.error("search_academic_memory", **arguments)
    assert fragment in message.lower(), message


def test_wrong_argument_types_are_rejected_by_the_schema(client):
    message = client.error("search_academic_memory", query="x", top_k="many")
    assert "top_k" in message


# ------------------------------------------------------------------ chunk / context / document
def first_hit(client, query="posterior probability of disease"):
    return client.call("search_academic_memory", query=query, top_k=1)["results"][0]


def test_get_source_chunk_round_trips(client):
    hit = first_hit(client)
    chunk = client.call("get_source_chunk", chunk_id=hit["chunk_id"])["chunk"]
    assert chunk["chunk_id"] == hit["chunk_id"] and chunk["text"] == hit["text"] and chunk["source"] == hit["source"]


def test_unknown_chunk_id_is_an_explained_error(client):
    assert "No chunk with id" in client.error("get_source_chunk", chunk_id="missing:00000")
    assert "No chunk with id" in client.error("get_context", chunk_id="missing:00000")


def test_get_context_returns_ordered_pieces_and_joined_text(client):
    hit = first_hit(client, "pairwise independence")
    ctx = client.call("get_context", chunk_id=hit["chunk_id"], neighbors=1, section=True)
    assert ctx["pieces"] and ctx["text"]
    assert sum(p["is_match"] for p in ctx["pieces"]) == 1
    assert all(p["source"] == hit["source"] for p in ctx["pieces"])


def test_get_context_pages_for_pdf(client):
    hit = client.call("search_academic_memory", query="pairwise independence", file_type="pdf", top_k=1)["results"][0]
    ctx = client.call("get_context", chunk_id=hit["chunk_id"], neighbors=0, pages=1)
    assert len(ctx["pieces"]) >= 1 and ctx["expansion"]


def test_get_document_outline_and_text(client):
    doc = client.call("get_document", source_path="3B1/Questions/Bayes.md", include_text=True)
    assert doc["file_type"] == "md" and doc["module"] == "3B1" and "probability" in doc["tags"]
    assert any(o["heading_path"][-1] == "Worked example" for o in doc["outline"])
    assert doc["chunks"] and "posterior" in " ".join(c["text"] for c in doc["chunks"])


def test_get_document_accepts_windows_separators_and_limits_text(client):
    doc = client.call("get_document", source_path="3B1\\Handouts\\Week 5 - Probability.pdf", include_text=True, max_chars=500)
    assert doc["file_type"] == "pdf" and doc["page_count"] == 4


def test_get_document_unknown_is_an_explained_error(client):
    assert "Not in the index" in client.error("get_document", source_path="nope.md")


def test_list_indexed_sources_and_filters(client):
    everything = client.call("list_indexed_sources")
    sources = {s["source"] for s in everything["sources"]}
    assert {"3B1/Questions/Bayes.md", "3B1/Handouts/Week 5 - Probability.pdf", "loose.md"} <= sources
    assert everything["total"] == len(sources)
    only_pdf = client.call("list_indexed_sources", file_type="pdf")
    assert {s["file_type"] for s in only_pdf["sources"]} == {"pdf"}
    limited = client.call("list_indexed_sources", limit=1)
    assert limited["returned"] == 1 and limited["total"] == everything["total"]


def test_index_status(client, config):
    status = client.call("index_status")
    assert status["indexed"] is True and status["documents"] >= 4 and status["chunks"] >= status["documents"]
    assert status["writes_enabled"] is True and status["failed_files"] == []
    assert status["documents_by_file_type"]["pdf"] == 1


# ------------------------------------------------------------------ missing index
def test_missing_index_is_reported_not_crashed(config, fake_embedder):
    client = Client(build_server(AcademicMemoryService(config, embedder=fake_embedder)))
    status = client.call("index_status")
    assert status["indexed"] is False and "message" in status
    assert client.error("search_academic_memory", query="anything")  # explained error, not a hang or crash


# ------------------------------------------------------------------ warm-up
def test_warm_up_loads_the_embedder_before_the_first_question(service, fake_embedder, monkeypatch):
    calls = []
    original = fake_embedder.embed_query
    monkeypatch.setattr(fake_embedder, "embed_query", lambda text: calls.append(text) or original(text))
    service.warm_up().join(timeout=30)
    assert calls == ["warm up"]


def test_warm_up_never_raises_without_an_index(config, fake_embedder):
    service = AcademicMemoryService(config, embedder=fake_embedder)  # nothing indexed
    service.warm_up().join(timeout=30)  # must swallow the error: the real call will explain it


# ------------------------------------------------------------------ writes
def test_create_note_is_marked_ai_generated_indexed_and_searchable(client, vault):
    res = client.call("create_note", path="3F1/Notes/AI summary.md", content="# Summary\n\nErgodic theorem for chains.", tags="ai, markov")
    assert res["ok"] and res["action"] == "created" and res["indexed"] == "indexed" and res["ai_generated"] is True
    text = (vault / "3F1" / "Notes" / "AI summary.md").read_text(encoding="utf-8")
    assert "ai_generated: true" in text
    found = client.call("search_academic_memory", query="ergodic theorem for chains", ai_generated=True, top_k=3)
    assert found["results"][0]["source"] == "3F1/Notes/AI summary.md"
    assert found["results"][0]["ai_generated"] is True
    assert client.call("get_document", source_path="3F1/Notes/AI summary.md")["tags"] == ["ai", "markov"]
    human = client.call("search_academic_memory", query="ergodic theorem for chains", ai_generated=False, top_k=5)
    assert "3F1/Notes/AI summary.md" not in {r["source"] for r in human["results"]}


def test_append_leaves_existing_text_intact_and_is_searchable(client, vault):
    path = vault / "loose.md"
    before = path.read_text(encoding="utf-8")
    res = client.call("append_to_note", path="loose.md", content="Little's law relates queue length and wait.")
    assert res["ok"] and res["indexed"] == "indexed"
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before.rstrip("\n"))
    assert "Little's law" in after
    found = client.call("search_academic_memory", query="Little's law queue length wait", top_k=3)
    assert "loose.md" in {r["source"] for r in found["results"]}


def test_replace_section_only_changes_that_section(client, vault):
    path = vault / "3B1" / "Questions" / "Bayes.md"
    before = path.read_text(encoding="utf-8")
    res = client.call("replace_section", path="3B1/Questions/Bayes.md", heading="What I don't understand", content="Resolved: it normalises.")
    assert res["ok"] and res["backup"]
    after = path.read_text(encoding="utf-8")
    assert "Resolved: it normalises." in after and "I struggled with the denominator" not in after
    assert before.split("## What I don't understand")[0] in after
    assert "## Worked example" in after and "prior prevalence" in after
    assert Path(res["backup"]).read_text(encoding="utf-8") == before  # the old text is recoverable, outside the vault
    assert vault not in Path(res["backup"]).parents


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "../escape.md", "content": "x"},
        {"path": "C:/Windows/evil.md", "content": "x"},
        {"path": "/etc/evil.md", "content": "x"},
        {"path": ".obsidian/plugins/evil.md", "content": "x"},
        {"path": "notes/evil.txt", "content": "x"},
        {"path": "notes/evil.py", "content": "x"},
        {"path": "notes/CON.md", "content": "x"},
        {"path": "3B1/Notes/empty.md", "content": ""},
    ],
)
def test_create_note_refuses_dangerous_input_and_leaves_vault_untouched(client, vault, tmp_path, arguments):
    before = digest(vault)
    outside = {p.name for p in tmp_path.iterdir()}
    message = client.error("create_note", **arguments)
    assert message  # the model is told why
    assert digest(vault) == before
    assert {p.name for p in tmp_path.iterdir()} == outside


def test_create_never_overwrites_an_existing_note(client, vault):
    before = digest(vault)
    assert "NoteExists" in client.error("create_note", path="loose.md", content="clobber")
    assert digest(vault) == before


def test_append_and_replace_need_an_existing_note_and_section(client, vault):
    before = digest(vault)
    assert "NoteNotFound" in client.error("append_to_note", path="ghost.md", content="x")
    assert "NoteNotFound" in client.error("replace_section", path="ghost.md", heading="H", content="x")
    assert "SectionNotFound" in client.error("replace_section", path="loose.md", heading="No such heading", content="x")
    assert digest(vault) == before


def test_pdfs_cannot_be_written(client, vault):
    before = digest(vault)
    for tool, args in (
        ("append_to_note", {"path": "3B1/Handouts/Week 5 - Probability.pdf", "content": "x"}),
        ("create_note", {"path": "3B1/Handouts/new.pdf", "content": "x"}),
    ):
        assert client.error(tool, **args)
    assert digest(vault) == before


def test_oversize_content_is_refused(client, vault, config):
    before = digest(vault)
    assert "ContentRejected" in client.error("create_note", path="3B1/Notes/huge.md", content="x" * (config.write_max_bytes + 1))
    assert digest(vault) == before


def test_write_that_cannot_reindex_still_reports_success_with_the_error(service, client, monkeypatch):
    monkeypatch.setattr(service.indexer, "sync_path", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index offline")))
    res = client.call("create_note", path="3B1/Notes/offline.md", content="# Offline\n\nstill saved")
    assert res["ok"] and res["indexed"] == "error" and "index offline" in res["index_error"]


def test_a_real_crash_is_hidden_from_the_client(service, client, monkeypatch):
    monkeypatch.setattr(service.searcher, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret /internal/path")))
    message = client.error("search_academic_memory", query="x")
    assert "secret" not in message and "search_academic_memory" in message
