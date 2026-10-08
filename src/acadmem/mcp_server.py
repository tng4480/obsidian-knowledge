"""MCP server exposing the academic memory to Codex (or any MCP client) over stdio.

Read tools
    search_academic_memory   semantic / keyword / hybrid search with metadata filters and optional context expansion
    get_source_chunk         one chunk, in full, with provenance
    get_context              the material around a chunk (neighbours, its section, its parent, nearby pages)
    get_document             a document's metadata, outline and (optionally) text
    list_indexed_sources     what is in the index
    index_status             health: counts, model, files that failed to index

Write tools (Markdown notes only; guarded, see ``vault_write``)
    create_note, append_to_note, replace_section

Every result carries provenance (source file, page, section) all the way from the original document.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from .config import Config, load_config
from .expand import ContextExpander, ContextPiece, ExpansionSpec
from .index import CHUNK_COLUMNS, IndexMismatchError, VectorIndex, sql_quote
from .indexer import Indexer
from .search import SearchFilters, SearchResult, Searcher
from .vault_write import VaultWriter, WriteError

log = logging.getLogger("acadmem.mcp")

SERVER_INSTRUCTIONS = """\
Semantic memory over the user's Obsidian study vault (lecture handouts, worked solutions, their own notes).

* For any question about the user's coursework or notes, call search_academic_memory first. Do not ask the user
  which file to open. Use filters (module, document_type, file_type, path_prefix, tags, dates, ai_generated) when the
  question names a course, paper type or year.
* Tags: the user marks passages in Obsidian with inline #tags (a tag tags the paragraph it is written in; frontmatter
  tags cover a whole note). When they ask about something they tagged or marked, pass it as the `tags` filter.
* Always cite sources as `source` + `page` (PDF) or `source` + `section` (Markdown). Use the retrieved `text` verbatim
  when quoting. If the results do not answer the question, say so rather than guessing.
* If a hit looks truncated or only half of an answer, call get_context with its chunk_id.
* Writing: create_note / append_to_note / replace_section only ever write Markdown notes inside the vault. Notes you
  create are tagged `ai_generated: true`. Never invent paths outside the vault's module folders (e.g. `3F7/Notes/...`).
  Writes are re-indexed immediately, so the new content is searchable straight away.
"""


# --------------------------------------------------------------------------- formatting
def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pages(start: int | None, end: int | None) -> str | None:
    if not start:
        return None
    return str(start) if not end or end == start else f"{start}-{end}"


def format_hit(hit: SearchResult) -> dict[str, Any]:
    """The shape Codex sees for one search result."""
    return {
        "rank": hit.rank,
        "score": round(hit.score, 3),
        "source": hit.source_path,
        "page": hit.page_start,
        "pages": _pages(hit.page_start, hit.page_end),
        "section": hit.heading,
        "heading_path": hit.heading_path,
        "text": hit.text,
        "chunk_id": hit.chunk_id,
        "file_type": hit.file_type,
        "module": hit.module,
        "document_type": hit.document_type,
        "ai_generated": hit.ai_generated,
        "tags": hit.tags,
        "modified": _iso(hit.modification_time),
        "location": hit.location,
    }


def format_row(row: dict[str, Any]) -> dict[str, Any]:
    """A stored chunk (no score) in the same shape as a hit."""
    return {
        "source": row["source_path"],
        "page": row.get("page_start"),
        "pages": _pages(row.get("page_start"), row.get("page_end")),
        "section": row.get("heading") or "",
        "heading_path": list(row.get("heading_path") or []),
        "text": row.get("text") or "",
        "chunk_id": row["chunk_id"],
        "chunk_index": row.get("chunk_index"),
        "file_type": row.get("file_type"),
        "module": row.get("module"),
        "document_type": row.get("document_type"),
        "ai_generated": bool(row.get("ai_generated")),
        "tags": list(row.get("tags") or []),
        "links": list(row.get("links") or []),
        "modified": _iso(row.get("modification_time")),
    }


def format_piece(piece: ContextPiece) -> dict[str, Any]:
    return {
        "role": piece.role,
        "is_match": piece.is_hit,
        "source": piece.source_path,
        "page": piece.page_start,
        "pages": _pages(piece.page_start, piece.page_end),
        "section": piece.heading,
        "text": piece.text,
        "chunk_id": piece.chunk_id,
    }


class _LockedEmbedder:
    """Serialise embedding calls: the watcher thread and the request thread share one model."""

    def __init__(self, inner: Any) -> None:
        self._inner, self._lock = inner, threading.Lock()
        self.model_name = inner.model_name

    @property
    def dimension(self) -> int:
        with self._lock:
            return self._inner.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._inner.embed_query(text)

    def count_tokens(self, text: str) -> int:
        with self._lock:
            return self._inner.count_tokens(text)


# --------------------------------------------------------------------------- service
class AcademicMemoryService:
    """The logic behind every tool, independent of the MCP transport (so it is unit-testable)."""

    def __init__(
        self,
        config: Config,
        embedder: Any = None,
        searcher: Searcher | None = None,
        indexer: Indexer | None = None,
        writer: VaultWriter | None = None,
    ) -> None:
        self.config = config
        if embedder is None:
            from .embeddings import get_embedder

            embedder = _LockedEmbedder(get_embedder(config))
        self.embedder = embedder
        self.index = VectorIndex(config)
        self.searcher = searcher or Searcher(config, embedder=embedder, index=self.index)
        self.indexer = indexer or Indexer(config, embedder=embedder, index=self.index)
        self.writer = writer or VaultWriter(config, self.indexer)
        self.expander = ContextExpander(self.index, config)
        self.watching = False

    # ------------------------------------------------------------------ read tools
    def search(
        self,
        query: str,
        top_k: int | None = None,
        module: str | list[str] | None = None,
        document_type: str | list[str] | None = None,
        file_type: str | list[str] | None = None,
        path_prefix: str | None = None,
        tags: str | list[str] | None = None,
        modified_after: str | None = None,
        modified_before: str | None = None,
        ai_generated: bool | None = None,
        mode: str | None = None,
        expand: str | None = None,
    ) -> dict[str, Any]:
        filters = {
            k: v
            for k, v in {
                "module": module,
                "document_type": document_type,
                "file_type": file_type,
                "path_prefix": path_prefix,
                "tags": tags,
                "modified_after": modified_after,
                "modified_before": modified_before,
                "ai_generated": ai_generated,
            }.items()
            if v not in (None, "", [])
        }
        if not (query or "").strip() and "tags" in filters:
            return self._browse_tagged(filters, top_k, expand)
        try:
            hits = self.searcher.search(query, top_k=top_k, filters=filters, mode=mode)
        except IndexMismatchError as exc:
            raise ValueError(str(exc)) from exc
        results = [format_hit(h) for h in hits]
        if expand:
            for hit, item in zip(hits, results):
                item["context"] = [format_piece(p) for p in self.expander.expand(hit, expand).pieces]
        used_mode = (mode or self.config.search_mode).lower()
        out: dict[str, Any] = {"query": query, "mode": used_mode, "filters": filters, "count": len(results), "results": results}
        if not results:
            out["hint"] = "No matches. Try broader wording, drop a filter, or check list_indexed_sources / index_status."
        return out

    BROWSE_DEFAULT = 20
    BROWSE_MAX = 100

    def _browse_tagged(self, filters: dict[str, Any], top_k: int | None, expand: str | None) -> dict[str, Any]:
        """List the passages matching a tag filter (no query), in vault order, instead of ranking them by similarity.

        This answers "show me everything I tagged #revisit", where there is nothing to rank by and a similarity
        search would cap the answer at the few nearest chunks.
        """
        if not self.index.exists():
            raise ValueError("No index yet. Run `acadmem index` (or wait for the server's first sync) and retry.")
        limit = max(1, min(int(top_k) if top_k is not None else self.BROWSE_DEFAULT, self.BROWSE_MAX))
        try:
            rows = self.index.scan(SearchFilters.from_mapping(filters).to_sql(), columns=CHUNK_COLUMNS)
        except IndexMismatchError as exc:
            raise ValueError(str(exc)) from exc
        rows.sort(key=lambda r: (r["source_path"], r.get("chunk_index") or 0))
        results = []
        for rank, row in enumerate(rows[:limit], 1):
            item = {"rank": rank, **format_row(row)}
            if expand:
                item["context"] = [format_piece(p) for p in self.expander.expand(row, expand).pieces]
            results.append(item)
        out: dict[str, Any] = {
            "query": "",
            "mode": "browse",
            "filters": filters,
            "count": len(results),
            "total_matches": len(rows),
            "results": results,
        }
        if len(rows) > limit:
            out["hint"] = f"Showing {limit} of {len(rows)} matching passages in vault order; raise top_k (max {self.BROWSE_MAX}) or add filters such as module."
        if not rows:
            out["hint"] = "Nothing carries that tag. Tags are case-insensitive and written without spaces (e.g. #revisit); check list_indexed_sources(tags=...)."
        return out

    def get_source_chunk(self, chunk_id: str) -> dict[str, Any]:
        row = self._chunk_or_error(chunk_id)
        return {"chunk": format_row(row)}

    def get_context(
        self,
        chunk_id: str,
        neighbors: int = 1,
        section: bool = False,
        parent: bool = False,
        pages: int = 0,
    ) -> dict[str, Any]:
        self._chunk_or_error(chunk_id)
        spec = ExpansionSpec.parse({"neighbors": neighbors, "section": section, "parent": parent, "pages": pages})
        ctx = self.expander.expand(chunk_id, spec)
        return {
            "chunk_id": chunk_id,
            "expansion": ctx.spec,
            "truncated": ctx.truncated,
            "pieces": [format_piece(p) for p in ctx.pieces],
            "text": ctx.text,
        }

    def get_document(self, source_path: str, include_text: bool = False, max_chars: int = 20000) -> dict[str, Any]:
        normalized = source_path.replace("\\", "/").strip().lstrip("/")
        doc = self.index.get_document(source_path=normalized)
        if doc is None:
            raise ValueError(f"Not in the index: {source_path!r}. Use list_indexed_sources to see what is available.")
        rows = sorted(
            self.index.scan(f"document_id = {sql_quote(doc['id'])}", columns=CHUNK_COLUMNS), key=lambda r: r["chunk_index"]
        )
        outline: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for r in rows:
            key = tuple(r["heading_path"] or [])
            if key and key not in seen:
                seen.add(key)
                outline.append({"heading_path": list(key), "page": r.get("page_start"), "chunk_id": r["chunk_id"]})
        out: dict[str, Any] = {
            "source": doc["source_path"],
            "title": doc.get("title"),
            "file_type": doc.get("file_type"),
            "module": doc.get("module"),
            "document_type": doc.get("document_type"),
            "page_count": doc.get("page_count"),
            "chunk_count": doc.get("chunk_count"),
            "ai_generated": bool(doc.get("ai_generated")),
            "tags": list(doc.get("tags") or []),
            "links": list(doc.get("links") or []),
            "modified": _iso(doc.get("modification_time")),
            "outline": outline,
        }
        if include_text:
            budget = max(500, int(max_chars))
            parts, used = [], 0
            for r in rows:
                piece = format_row(r)
                if used + len(piece["text"]) > budget and parts:
                    out["truncated"] = True
                    break
                parts.append(piece)
                used += len(piece["text"])
            out["chunks"] = parts
        return out

    def list_indexed_sources(
        self,
        module: str | list[str] | None = None,
        document_type: str | list[str] | None = None,
        file_type: str | list[str] | None = None,
        path_prefix: str | None = None,
        tags: str | list[str] | None = None,
        ai_generated: bool | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        filters = SearchFilters.from_mapping(
            {
                k: v
                for k, v in {
                    "module": module,
                    "document_type": document_type,
                    "file_type": file_type,
                    "path_prefix": path_prefix,
                    "tags": tags,
                    "ai_generated": ai_generated,
                }.items()
                if v not in (None, "", [])
            }
        )
        try:
            docs = self.index.list_documents(where=filters.to_sql())
        except IndexMismatchError as exc:
            raise ValueError(str(exc)) from exc
        limit = max(1, int(limit))
        return {
            "total": len(docs),
            "returned": min(len(docs), limit),
            "sources": [
                {
                    "source": d["source_path"],
                    "title": d.get("title"),
                    "file_type": d.get("file_type"),
                    "module": d.get("module"),
                    "document_type": d.get("document_type"),
                    "pages": d.get("page_count"),
                    "chunks": d.get("chunk_count"),
                    "ai_generated": bool(d.get("ai_generated")),
                    "modified": _iso(d.get("modification_time")),
                }
                for d in docs[:limit]
            ],
        }

    def index_status(self) -> dict[str, Any]:
        base = {
            "vault": str(self.config.vault_path),
            "index_dir": str(self.config.index_dir),
            "embedding_model": self.config.embedding_model,
            "default_search_mode": self.config.search_mode,
            "writes_enabled": self.config.writes_enabled,
            "watching": self.watching,
        }
        if not self.index.exists():
            return {**base, "indexed": False, "message": "No index yet. It is being built if the watcher is on; otherwise run `acadmem index`."}
        docs = self.index.list_documents()
        by_module: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for d in docs:
            by_module[d.get("module") or "(root)"] = by_module.get(d.get("module") or "(root)", 0) + 1
            by_type[d.get("file_type") or "?"] = by_type.get(d.get("file_type") or "?", 0) + 1
        meta = self.index.read_meta() or {}
        return {
            **base,
            "indexed": True,
            "documents": len(docs),
            "chunks": self.index.count_chunks(),
            "documents_by_module": dict(sorted(by_module.items())),
            "documents_by_file_type": by_type,
            "index_created": _iso(meta.get("created_at")),
            "last_indexed": _iso(max((d.get("indexed_at") or 0 for d in docs), default=0)),
            "failed_files": [
                {"source": e["source_path"], "error": f"{e['error_type']}: {e['message']}"[:300]} for e in self.index.errors()
            ],
        }

    # ------------------------------------------------------------------ write tools
    def create_note(self, path: str, content: str, tags: str | list[str] | None = None) -> dict[str, Any]:
        if isinstance(tags, str):
            tags = [t for t in tags.replace(",", " ").split() if t]
        return {"ok": True, **self.writer.create_note(path, content, tags=tags).to_dict()}

    def append_to_note(self, path: str, content: str) -> dict[str, Any]:
        return {"ok": True, **self.writer.append_to_note(path, content).to_dict()}

    def replace_section(self, path: str, heading: str, content: str, create_if_missing: bool = False) -> dict[str, Any]:
        return {"ok": True, **self.writer.replace_section(path, heading, content, create_if_missing).to_dict()}

    # ------------------------------------------------------------------ helpers
    def _chunk_or_error(self, chunk_id: str) -> dict[str, Any]:
        try:
            row = self.index.get_chunk(chunk_id)
        except IndexMismatchError as exc:
            raise ValueError(str(exc)) from exc
        if row is None:
            raise ValueError(f"No chunk with id {chunk_id!r}. Chunk ids come from search results and can change when a file is re-indexed.")
        return row

    def warm_up(self) -> threading.Thread:
        """Load the embedding model and open the index in the background.

        The first search otherwise pays this one-off cost (~15 s on CPU for the bge model), which is long enough
        for a client to give up. A query that arrives while warming up simply waits for the same load (the embedder
        is lock-protected), so there is no double work.
        """

        def run() -> None:
            started = time.perf_counter()
            try:
                self.searcher.search("warm up", top_k=1)
                log.info("warm-up finished in %.1fs", time.perf_counter() - started)
            except Exception as exc:  # no index yet / schema change: the real call will explain it
                log.info("warm-up skipped: %s", exc)

        thread = threading.Thread(target=run, name="acadmem-warmup", daemon=True)
        thread.start()
        return thread

    def start_watcher(self) -> threading.Thread:
        """Keep the index in sync with the vault in the background (initial catch-up first)."""
        from .watch import VaultWatcher

        watcher = VaultWatcher(self.indexer, debounce=self.config.watch_debounce_seconds, initial_sync=True)

        def run() -> None:
            try:
                watcher.start()
                self.watching = True
                log.info("watching %s", self.config.vault_path)
            except Exception:
                log.exception("watcher failed to start; the index will not update automatically")

        thread = threading.Thread(target=run, name="acadmem-watcher", daemon=True)
        thread.start()
        self.watcher = watcher
        return thread


# --------------------------------------------------------------------------- MCP wiring
#: Filters accept one value ("3F1") or several (["3F1", "3F7"]); models send both forms.
Names = str | list[str] | None

EXPECTED_ERRORS = (ValueError, KeyError, TypeError, WriteError, IndexMismatchError)


def _tool_errors(fn):
    """Turn anticipated failures (bad input, refused writes, no index) into MCP ``ToolError``s.

    The model then sees the real reason ("path resolves outside the vault", "heading is ambiguous ...") and can
    correct itself. Anything else is a genuine crash: the SDK hides its details from the client and logs it.
    """
    import functools

    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except EXPECTED_ERRORS as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise ToolError(f"{type(exc).__name__}: {message}") from exc

    return wrapper


def build_server(service: AcademicMemoryService):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("academic-memory", instructions=SERVER_INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @server.tool(annotations=read_only)
    @_tool_errors
    def search_academic_memory(
        query: str = "",
        top_k: int | None = None,
        module: Names = None,
        document_type: Names = None,
        file_type: Names = None,
        path_prefix: str | None = None,
        tags: Names = None,
        modified_after: str | None = None,
        modified_before: str | None = None,
        ai_generated: bool | None = None,
        mode: Literal["vector", "fts", "hybrid"] | None = None,
        expand: str | None = None,
    ) -> dict[str, Any]:
        """Search the user's study vault (PDF handouts, worked solutions, Markdown notes) by meaning.

        Returns the best-matching chunks with provenance: `source` (vault path), `page` (PDFs), `section`
        (heading breadcrumb), `text`, `score` and a `chunk_id` for follow-up calls. `rank` is the order (best first).

        `score` is the cosine similarity between the query and the chunk (the same measure in every `mode`).
        In this vault, chunks that really answer a question typically score 0.70-0.87, and queries about something
        the vault does not cover rarely get a best hit above 0.68. It is not a probability: neighbouring chunks of
        the right topic score alike, and because the order also uses keyword matching, scores need not strictly
        decrease down the list. Decide relevance by reading `text`, not by the number.

        Filters (all optional, combined with AND): `module` (e.g. ["3F7"]), `document_type` (the folder under the
        module, e.g. "Questions", "Notes", "Handouts"), `file_type` ("pdf" or "md"), `path_prefix` (a folder or file
        prefix such as "3F4/Questions/3F4_Tripos_2023"), `tags`, `modified_after` / `modified_before` (ISO dates),
        `ai_generated` (true = only AI-written notes, false = only human material).
        `tags` (e.g. "revisit", "#3F7/huffman", case-insensitive, no spaces) returns the passages the user marked:
        an inline #tag written in a paragraph (or list, table, ...) tags just that paragraph, while a tag in a
        note's frontmatter tags every passage of the note. For "show me everything I tagged #revisit" leave `query`
        empty: the tagged passages are then listed in vault order (default 20, up to 100 with `top_k`) instead of
        being ranked. With a `query` as well, the tagged passages are ranked by relevance to it.
        `expand` adds surrounding context to every hit, e.g. "neighbors:1", "section", "parent", "pages:1" ("+" joins them).
        """
        return service.search(
            query, top_k, module, document_type, file_type, path_prefix, tags, modified_after, modified_before, ai_generated, mode, expand
        )

    @server.tool(annotations=read_only)
    @_tool_errors
    def get_source_chunk(chunk_id: str) -> dict[str, Any]:
        """Fetch one chunk by `chunk_id` (from a search result), with full text and provenance."""
        return service.get_source_chunk(chunk_id)

    @server.tool(annotations=read_only)
    @_tool_errors
    def get_context(chunk_id: str, neighbors: int = 1, section: bool = False, parent: bool = False, pages: int = 0) -> dict[str, Any]:
        """Get the material around a chunk when a hit is cut off or only part of an answer.

        `neighbors`: that many chunks before and after (reading order). `section`: every chunk of the same
        section. `parent`: the enclosing section (e.g. the question statement above a worked sub-part). `pages`:
        all chunks within that many pages of the hit (PDFs). Each piece is marked `is_match` (the original hit) or context.
        """
        return service.get_context(chunk_id, neighbors, section, parent, pages)

    @server.tool(annotations=read_only)
    @_tool_errors
    def get_document(source_path: str, include_text: bool = False, max_chars: int = 20000) -> dict[str, Any]:
        """Metadata and outline (headings with pages) of one indexed document; `include_text` adds its chunks (capped by `max_chars`)."""
        return service.get_document(source_path, include_text, max_chars)

    @server.tool(annotations=read_only)
    @_tool_errors
    def list_indexed_sources(
        module: Names = None,
        document_type: Names = None,
        file_type: Names = None,
        path_prefix: str | None = None,
        tags: Names = None,
        ai_generated: bool | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """List indexed documents (optionally filtered), e.g. to see which papers exist for a module."""
        return service.list_indexed_sources(module, document_type, file_type, path_prefix, tags, ai_generated, limit)

    @server.tool(annotations=read_only)
    @_tool_errors
    def index_status() -> dict[str, Any]:
        """Index health: document/chunk counts, embedding model, last update and any files that failed to index."""
        return service.index_status()

    if service.config.writes_enabled:

        @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
        @_tool_errors
        def create_note(path: str, content: str, tags: Names = None) -> dict[str, Any]:
            """Create a NEW Markdown note in the vault (never overwrites; fails if it exists).

            `path` is relative to the vault root, e.g. "3F7/Notes/Entropy summary.md" (".md" is added if missing;
            folders are created). The note is saved with `ai_generated: true` frontmatter and re-indexed immediately.
            Only .md files inside the vault can be written. `tags` are note-level (frontmatter, apply to every passage);
            to tag just one paragraph, write `#tag` inside that paragraph's text in `content`.
            """
            return service.create_note(path, content, tags)

        @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
        @_tool_errors
        def append_to_note(path: str, content: str) -> dict[str, Any]:
            """Append Markdown to the end of an EXISTING note. Existing text is never changed; a backup is kept.

            Text added to a note the user wrote is preceded by an invisible `<!-- ai:append DATE -->` marker.
            """
            return service.append_to_note(path, content)

        @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
        @_tool_errors
        def replace_section(path: str, heading: str, content: str, create_if_missing: bool = False) -> dict[str, Any]:
            """Replace the body of one section of an existing note (everything under `heading` up to the next
            heading of the same or higher level, including its sub-sections). The heading line and all other
            sections are untouched; a backup of the whole note is kept.

            `heading` is the section title or "Parent > Title" if titles repeat. With `create_if_missing`
            a new "## heading" section is appended instead of failing.
            """
            return service.replace_section(path, heading, content, create_if_missing)

    return server


# --------------------------------------------------------------------------- entry point
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acadmem-mcp", description="MCP server for the academic memory (stdio).")
    parser.add_argument("--vault", help="Obsidian vault path (or ACADMEM_VAULT)")
    parser.add_argument("--index-dir", help="Index location (or ACADMEM_INDEX_DIR)")
    parser.add_argument("--config", help="TOML config file (or ACADMEM_CONFIG)")
    parser.add_argument("--no-watch", action="store_true", help="Do not keep the index in sync with the vault")
    parser.add_argument("--read-only", action="store_true", help="Do not expose the Markdown write tools")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # stdout is the protocol channel: all logging goes to stderr.
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    overrides: dict[str, Any] = {"vault_path": args.vault, "index_dir": args.index_dir}
    if args.read_only:
        overrides["writes_enabled"] = False
    config = load_config(args.config, **overrides)
    if not config.vault_path or not Path(config.vault_path).is_dir():
        print(f"acadmem-mcp: vault not found: {config.vault_path!r} (set --vault or ACADMEM_VAULT)", file=sys.stderr)
        return 2

    service = AcademicMemoryService(config)
    service.warm_up()
    if not args.no_watch:
        service.start_watcher()
    build_server(service).run(args.transport)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
