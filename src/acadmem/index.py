"""LanceDB-backed derived index.

Everything here is *disposable*: the vault is the source of truth, and ``reset()`` followed by
a full ingest recreates the entire index. Nothing in the index is needed to reconstruct a note
or PDF.

Tables
------
``chunks``     one row per chunk: provenance + text + embedding vector
``documents``  one row per indexed file (also the incremental-indexing manifest)
``errors``     last failure per file (cleared when the file next indexes cleanly)
``pdf_tags``   tagged lines in notes that point into PDFs (their tags are applied to those PDFs' chunks afterwards)
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from .config import Config
from .models import Chunk, Document, PdfRef

SCHEMA_VERSION = 2  # 2: added chunks.search_text (heading breadcrumb + body) and moved the FTS index onto it
FTS_COLUMN = "search_text"
CHUNKS = "chunks"
DOCUMENTS = "documents"
ERRORS = "errors"
PDF_TAGS = "pdf_tags"


class IndexMismatchError(RuntimeError):
    """The existing index was built with a different embedding model / schema."""


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def chunks_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
            pa.field("file_type", pa.string()),
            pa.field("module", pa.string()),
            pa.field("document_type", pa.string()),
            pa.field("modification_time", pa.float64()),
            pa.field("chunk_index", pa.int32()),
            pa.field("section_id", pa.string()),
            pa.field("parent_section_id", pa.string()),
            pa.field("heading_path", pa.list_(pa.string())),
            pa.field("heading", pa.string()),
            pa.field("text", pa.string()),
            pa.field("search_text", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("ai_generated", pa.bool_()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("links", pa.list_(pa.string())),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


DOCUMENTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("file_type", pa.string()),
        pa.field("module", pa.string()),
        pa.field("document_type", pa.string()),
        pa.field("modification_time", pa.float64()),
        pa.field("content_hash", pa.string()),
        pa.field("size", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("page_count", pa.int32()),
        pa.field("ai_generated", pa.bool_()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("links", pa.list_(pa.string())),
        pa.field("chunk_count", pa.int32()),
        pa.field("indexed_at", pa.float64()),
    ]
)

ERRORS_SCHEMA = pa.schema(
    [
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("error_type", pa.string()),
        pa.field("message", pa.string()),
        pa.field("occurred_at", pa.float64()),
    ]
)


PDF_TAGS_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string(), nullable=False),  # the note holding the tagged line
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("target", pa.string()),  # the PDF as written in the note
        pa.field("page_start", pa.int32()),
        pa.field("page_end", pa.int32()),
        pa.field("quote", pa.string()),
        pa.field("quote_is_weak", pa.bool_()),
        pa.field("tags", pa.list_(pa.string())),
    ]
)


def chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    if chunk.embedding is None:
        raise ValueError(f"chunk {chunk.chunk_id} has no embedding")
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_path": chunk.source_path,
        "file_type": chunk.file_type,
        "module": chunk.module,
        "document_type": chunk.document_type,
        "modification_time": chunk.modification_time,
        "chunk_index": chunk.chunk_index,
        "section_id": chunk.section_id,
        "parent_section_id": chunk.parent_section_id,
        "heading_path": list(chunk.heading_path),
        "heading": chunk.heading,
        "text": chunk.text,
        "search_text": chunk.search_text or chunk.text,
        "page_number": chunk.page_number,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "token_count": chunk.token_count,
        "ai_generated": chunk.ai_generated,
        "tags": list(chunk.tags),
        "links": list(chunk.links),
        "vector": [float(x) for x in chunk.embedding],
    }


def document_to_row(doc: Document, size: int = 0) -> dict[str, Any]:
    d = asdict(doc)
    d["size"] = int(size)
    return d


class VectorIndex:
    """Thin, vault-agnostic wrapper around the LanceDB tables."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._db: lancedb.DBConnection | None = None
        self._chunks = None
        self._documents = None
        self._errors = None
        self._pdf_tags = None

    # ---------------------------------------------------------------- lifecycle
    @property
    def meta_path(self) -> Path:
        return self.config.index_dir / "index_meta.json"

    @property
    def db(self) -> lancedb.DBConnection:
        if self._db is None:
            self.config.db_path.mkdir(parents=True, exist_ok=True)
            # Always read the latest committed version so a long-lived MCP server sees writes
            # made by the CLI / watcher process.
            self._db = lancedb.connect(str(self.config.db_path), read_consistency_interval=timedelta(0))
        return self._db

    def _table_names(self) -> set[str]:
        return set(self.db.list_tables().tables)

    def exists(self) -> bool:
        return self.meta_path.is_file() and CHUNKS in self._table_names()

    def read_meta(self) -> dict[str, Any] | None:
        if not self.meta_path.is_file():
            return None
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def ensure(self, model_name: str, dimension: int) -> None:
        """Create tables if needed; refuse to mix embeddings from different models."""
        meta = self.read_meta()
        if meta is not None:
            if meta.get("embedding_model") != model_name or int(meta.get("dimension", -1)) != dimension:
                raise IndexMismatchError(
                    f"Index was built with {meta.get('embedding_model')!r} (dim {meta.get('dimension')}), "
                    f"but the configured model is {model_name!r} (dim {dimension}). "
                    "The index is disposable: run `acadmem rebuild`."
                )
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise IndexMismatchError("Index schema version changed; run `acadmem rebuild`.")
        names = self._table_names()
        if CHUNKS not in names:
            self.db.create_table(CHUNKS, schema=chunks_schema(dimension))
        if DOCUMENTS not in names:
            self.db.create_table(DOCUMENTS, schema=DOCUMENTS_SCHEMA)
        if ERRORS not in names:
            self.db.create_table(ERRORS, schema=ERRORS_SCHEMA)
        if PDF_TAGS not in names:  # added after schema v2: an additive table, so older indexes just gain it
            self.db.create_table(PDF_TAGS, schema=PDF_TAGS_SCHEMA)
        if meta is None:
            self.config.index_dir.mkdir(parents=True, exist_ok=True)
            self.meta_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "embedding_model": model_name,
                        "dimension": dimension,
                        "vault_path": str(self.config.vault_path),
                        "created_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._chunks = self.db.open_table(CHUNKS)
        self._documents = self.db.open_table(DOCUMENTS)
        self._errors = self.db.open_table(ERRORS)
        self._pdf_tags = self.db.open_table(PDF_TAGS)

    def reset(self) -> None:
        """Delete the entire derived index (full rebuild starts from here)."""
        self._chunks = self._documents = self._errors = self._pdf_tags = None
        self._db = None
        if self.config.db_path.exists():
            shutil.rmtree(self.config.db_path, ignore_errors=True)
        if self.meta_path.exists():
            self.meta_path.unlink()

    def _open(self, name: str):
        attr = {CHUNKS: "_chunks", DOCUMENTS: "_documents", ERRORS: "_errors"}[name]
        table = getattr(self, attr)
        if table is None:
            if name not in self._table_names():
                raise IndexMismatchError("No index found. Run `acadmem rebuild` (or `acadmem index`).")
            table = self.db.open_table(name)
            setattr(self, attr, table)
        return table

    @property
    def chunks(self):
        return self._open(CHUNKS)

    @property
    def documents(self):
        return self._open(DOCUMENTS)

    @property
    def errors_table(self):
        return self._open(ERRORS)

    @property
    def pdf_tags_table(self):
        if self._pdf_tags is None:
            if PDF_TAGS not in self._table_names():  # an index created before PDF tagging existed
                self.db.create_table(PDF_TAGS, schema=PDF_TAGS_SCHEMA)
            self._pdf_tags = self.db.open_table(PDF_TAGS)
        return self._pdf_tags

    # ------------------------------------------------------------------ writes
    def write_document(self, doc: Document, chunks: list[Chunk], size: int = 0, pdf_refs: list[PdfRef] | None = None) -> None:
        """Atomically-ish replace everything stored for ``doc`` (delete old rows, add new)."""
        self.chunks.delete(f"document_id = {sql_quote(doc.id)}")
        if chunks:
            self.chunks.add([chunk_to_row(c) for c in chunks])
        self.documents.delete(f"id = {sql_quote(doc.id)}")
        doc.chunk_count = len(chunks)
        doc.indexed_at = time.time()
        self.documents.add([document_to_row(doc, size)])
        self.set_pdf_refs(doc.id, doc.source_path, pdf_refs or [])
        self.clear_error(doc.source_path)

    def set_pdf_refs(self, document_id: str, source_path: str, refs: list[PdfRef]) -> None:
        """Replace the tagged PDF lines recorded for one note."""
        table = self.pdf_tags_table
        table.delete(f"document_id = {sql_quote(document_id)}")
        if refs:
            table.add(
                [
                    {
                        "document_id": document_id,
                        "source_path": source_path,
                        "target": r.target,
                        "page_start": r.page_start,
                        "page_end": r.page_end,
                        "quote": r.quote,
                        "quote_is_weak": r.quote_is_weak,
                        "tags": list(r.tags),
                    }
                    for r in refs
                ]
            )

    def pdf_ref_rows(self) -> list[dict[str, Any]]:
        rows = self.pdf_tags_table.search().limit(1_000_000).to_list()
        return sorted(rows, key=lambda r: (r["source_path"], r["target"] or "", r["page_start"] or 0, r["quote"] or ""))

    def replace_chunk_tags(self, new_tags: dict[str, list[str]]) -> int:
        """Set ``tags`` on the given chunks (``chunk_id -> tags``) in place, keeping vectors and everything else."""
        if not new_tags:
            return 0
        ids = list(new_tags)
        rows: list[dict[str, Any]] = []
        for start in range(0, len(ids), 200):
            part = ids[start : start + 200]
            where = "chunk_id IN (" + ", ".join(sql_quote(i) for i in part) + ")"
            rows.extend(self.chunks.search().where(where, prefilter=True).limit(len(part)).to_list())
        for row in rows:
            row["tags"] = list(new_tags[row["chunk_id"]])
            row.pop("_rowid", None)
        schema = self.chunks.schema
        data = pa.Table.from_pylist([{k: r.get(k) for k in schema.names} for r in rows], schema=schema)
        self.chunks.merge_insert("chunk_id").when_matched_update_all().execute(data)
        return len(rows)

    def replace_document_tags(self, new_tags: dict[str, list[str]]) -> None:
        """Set ``tags`` on the given documents (``document_id -> tags``) in place."""
        if not new_tags:
            return
        where = "id IN (" + ", ".join(sql_quote(i) for i in new_tags) + ")"
        rows = self.documents.search().where(where, prefilter=True).limit(len(new_tags)).to_list()
        for row in rows:
            row["tags"] = list(new_tags[row["id"]])
        schema = self.documents.schema
        data = pa.Table.from_pylist([{k: r.get(k) for k in schema.names} for r in rows], schema=schema)
        self.documents.merge_insert("id").when_matched_update_all().execute(data)

    def delete_document(self, document_id: str, source_path: str | None = None) -> None:
        self.chunks.delete(f"document_id = {sql_quote(document_id)}")
        self.documents.delete(f"id = {sql_quote(document_id)}")
        self.pdf_tags_table.delete(f"document_id = {sql_quote(document_id)}")
        if source_path:
            self.clear_error(source_path)

    def record_error(self, source_path: str, error: BaseException | str) -> None:
        self.clear_error(source_path)
        etype = type(error).__name__ if isinstance(error, BaseException) else "Error"
        self.errors_table.add(
            [{"source_path": source_path, "error_type": etype, "message": str(error)[:2000], "occurred_at": time.time()}]
        )

    def clear_error(self, source_path: str) -> None:
        self.errors_table.delete(f"source_path = {sql_quote(source_path)}")

    def refresh_fts(self) -> None:
        """(Re)build the BM25 full-text index on chunk text (only used by fts/hybrid modes)."""
        if self.chunks.count_rows() == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.chunks.create_index(FTS_COLUMN, config=FTS(stem=True, remove_stop_words=False), replace=True)

    # ------------------------------------------------------------------- reads
    def manifest(self) -> dict[str, dict[str, Any]]:
        rows = self.documents.search().select(
            ["id", "source_path", "content_hash", "modification_time", "size", "chunk_count"]
        ).limit(1_000_000).to_list()
        return {r["source_path"]: r for r in rows}

    def errors(self) -> list[dict[str, Any]]:
        return self.errors_table.search().limit(100_000).to_list()

    def count_chunks(self) -> int:
        return int(self.chunks.count_rows())

    def count_documents(self) -> int:
        return int(self.documents.count_rows())

    def scan(self, where: str | None = None, limit: int = 1_000_000, columns: list[str] | None = None) -> list[dict[str, Any]]:
        q = self.chunks.search()
        if where:
            q = q.where(where, prefilter=True)
        if columns:
            q = q.select(columns)
        return q.limit(limit).to_list()

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        rows = self.scan(f"chunk_id = {sql_quote(chunk_id)}", limit=1, columns=CHUNK_COLUMNS)
        return rows[0] if rows else None

    def get_document(self, document_id: str | None = None, source_path: str | None = None) -> dict[str, Any] | None:
        if document_id:
            where = f"id = {sql_quote(document_id)}"
        elif source_path:
            where = f"source_path = {sql_quote(source_path)}"
        else:
            raise ValueError("document_id or source_path required")
        rows = self.documents.search().where(where, prefilter=True).limit(1).to_list()
        return rows[0] if rows else None

    def list_documents(self, where: str | None = None, include_empty: bool = False) -> list[dict[str, Any]]:
        q = self.documents.search()
        clauses = [c for c in (where, None if include_empty else "chunk_count > 0") if c]
        if clauses:
            q = q.where(" AND ".join(f"({c})" for c in clauses), prefilter=True)
        rows = q.limit(1_000_000).to_list()
        return sorted(rows, key=lambda r: r["source_path"])


CHUNK_COLUMNS = [
    "chunk_id",
    "document_id",
    "source_path",
    "file_type",
    "module",
    "document_type",
    "modification_time",
    "chunk_index",
    "section_id",
    "parent_section_id",
    "heading_path",
    "heading",
    "text",
    "page_number",
    "page_start",
    "page_end",
    "token_count",
    "ai_generated",
    "tags",
    "links",
]
