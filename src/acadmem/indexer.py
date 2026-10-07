"""Indexing pipeline: discover -> classify -> extract -> chunk -> embed -> LanceDB.

Two entry points share one code path:

* ``run(full=True)``   - delete the derived index and rebuild everything from the vault
* ``run(full=False)``  - incremental: new -> index, modified -> re-index, deleted -> remove,
                         unchanged -> do nothing (decided by content hash, never by index state alone)

Errors in an individual file are recorded (``errors`` table + report) and never abort the run.
A single writer is enforced with a file lock, so the watcher, CLI and MCP write tools can coexist.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock

from .chunking import Chunker
from .config import Config
from .embeddings import Embedder, get_embedder
from .index import VectorIndex
from .ingest.base import ExtractorRegistry
from .ingest.discovery import (
    DiscoveredFile,
    discover,
    is_indexable,
    read_bytes,
    sha256_hex,
    to_source_path,
)
from .ingest.markdown import is_ai_generated
from .ingest.registry import build_registry
from .models import Document, ParsedDocument, infer_path_metadata, stable_document_id

log = logging.getLogger("acadmem")

ProgressFn = Callable[[str, str], None]  # (event, source_path)


@dataclass
class IndexReport:
    mode: str = "incremental"
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    chunks_written: int = 0
    seconds: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return (
            f"{self.mode}: +{len(self.added)} new, ~{len(self.modified)} modified, -{len(self.deleted)} deleted, "
            f"={len(self.unchanged)} unchanged, {len(self.errors)} errors, "
            f"{self.chunks_written} chunks written in {self.seconds:.1f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "unchanged_count": len(self.unchanged),
            "errors": self.errors,
            "chunks_written": self.chunks_written,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class ScanResult:
    new: list[DiscoveredFile] = field(default_factory=list)
    modified: list[DiscoveredFile] = field(default_factory=list)
    unchanged: list[DiscoveredFile] = field(default_factory=list)
    deleted: list[tuple[str, str]] = field(default_factory=list)  # (source_path, document_id)
    hashes: dict[str, str] = field(default_factory=dict)  # source_path -> sha256 for files hashed while scanning


class Indexer:
    def __init__(
        self,
        config: Config,
        embedder: Embedder | None = None,
        registry: ExtractorRegistry | None = None,
        index: VectorIndex | None = None,
        lock_timeout: float = 600.0,
    ) -> None:
        self.config = config
        self.embedder = embedder or get_embedder(config)
        self.registry = registry or build_registry(config)
        self.index = index or VectorIndex(config)
        self.chunker = Chunker(
            self.embedder.count_tokens,
            max_tokens=config.max_chunk_tokens,
            min_tokens=config.min_chunk_tokens,
            overlap_tokens=config.overlap_tokens,
        )
        self._lock = FileLock(str(config.lock_path), timeout=lock_timeout)

    # ------------------------------------------------------------------ setup
    def _prepare(self) -> None:
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        self.index.ensure(self.embedder.model_name, self.embedder.dimension)

    # ------------------------------------------------------------------- scan
    def scan(self) -> ScanResult:
        """Classify every vault file against the manifest (no extraction / embedding)."""
        manifest = self.index.manifest()
        result = ScanResult()
        seen: set[str] = set()
        for df in discover(self.config):
            seen.add(df.source_path)
            row = manifest.get(df.source_path)
            if row is None:
                result.new.append(df)
                continue
            if row["size"] == df.size and abs(float(row["modification_time"]) - df.mtime) < 1e-3:
                result.unchanged.append(df)  # fast path: size + mtime identical
                continue
            try:
                digest = sha256_hex(read_bytes(df.path))
            except OSError:
                result.modified.append(df)  # let index_file record the read error
                continue
            result.hashes[df.source_path] = digest
            if digest == row["content_hash"]:
                result.unchanged.append(df)  # touched (e.g. by cloud sync) but identical content
                self.index.documents.update(
                    where=f"id = '{row['id']}'", values={"modification_time": df.mtime, "size": df.size}
                )
                # The file matches what is indexed again (e.g. a failed edit was reverted), so any
                # error recorded for the failed attempt is stale.
                self.index.clear_error(df.source_path)
            else:
                result.modified.append(df)
        for source_path, row in manifest.items():
            if source_path not in seen:
                result.deleted.append((source_path, row["id"]))
        return result

    # ------------------------------------------------------------- one file
    def build_parsed(self, df: DiscoveredFile, content_hash: str | None = None) -> ParsedDocument:
        extractor = self.registry.for_path(df.path)
        if extractor is None:
            raise ValueError(f"No extractor registered for {df.path.suffix!r}")
        if content_hash is None:
            content_hash = sha256_hex(read_bytes(df.path))
        content = extractor.parse(df.path)
        module, document_type = infer_path_metadata(df.source_path)
        document = Document(
            id=stable_document_id(df.source_path),
            source_path=df.source_path,
            file_type=extractor.file_type,
            module=module,
            document_type=document_type,
            modification_time=df.mtime,
            content_hash=content_hash,
            title=content.title,
            page_count=content.page_count,
            ai_generated=is_ai_generated(content),
            tags=list(content.tags),
            links=list(content.links),
        )
        return ParsedDocument(document=document, sections=content.sections, frontmatter=content.frontmatter)

    def index_file(self, df: DiscoveredFile, content_hash: str | None = None) -> int:
        """Extract, chunk, embed and write one file. Returns the number of chunks written."""
        parsed = self.build_parsed(df, content_hash)
        chunks = self.chunker.chunk_document(parsed)
        if chunks:
            texts = [c.embedding_text(parsed.document.title) for c in chunks]
            vectors = self.embedder.embed_documents(texts)
            for chunk, text, vector in zip(chunks, texts, vectors):
                chunk.embedding = vector
                chunk.search_text = text
        self.index.write_document(parsed.document, chunks, size=df.size)
        return len(chunks)

    def _safe_index_file(self, df: DiscoveredFile, report: IndexReport, hashes: dict[str, str] | None = None) -> bool:
        try:
            report.chunks_written += self.index_file(df, (hashes or {}).get(df.source_path))
            return True
        except Exception as exc:  # per-file isolation: record and continue
            log.warning("failed to index %s: %s", df.source_path, exc)
            report.errors.append({"source_path": df.source_path, "error_type": type(exc).__name__, "message": str(exc)})
            try:
                self.index.record_error(df.source_path, exc)
            except Exception:  # never let error bookkeeping kill the run
                log.exception("could not record error for %s", df.source_path)
            return False

    # ------------------------------------------------------------------ runs
    def run(self, full: bool = False, progress: ProgressFn | None = None) -> IndexReport:
        report = IndexReport(mode="full" if full else "incremental")
        started = time.time()
        emit = progress or (lambda event, path: None)
        with self._lock:
            if full:
                self.index.reset()
            self._prepare()
            scan = self.scan()
            report.unchanged = [df.source_path for df in scan.unchanged]

            for source_path, doc_id in scan.deleted:
                self.index.delete_document(doc_id, source_path)
                report.deleted.append(source_path)
                emit("deleted", source_path)
            for df in scan.new:
                if self._safe_index_file(df, report, scan.hashes):
                    report.added.append(df.source_path)
                    emit("added", df.source_path)
            for df in scan.modified:
                if self._safe_index_file(df, report, scan.hashes):
                    report.modified.append(df.source_path)
                    emit("modified", df.source_path)

            if report.changed:
                self.index.refresh_fts()
        report.seconds = time.time() - started
        log.info(report.summary())
        return report

    def sync_path(self, path: str | Path) -> str:
        """Bring a single path in line with the vault (used by the watcher and MCP write tools).

        Returns one of ``"indexed" | "deleted" | "ignored" | "error"``.
        """
        path = Path(path)
        if not path.is_absolute():
            path = self.config.vault_path / path
        try:
            source_path = to_source_path(self.config.vault_path, path)
        except ValueError:
            return "ignored"  # outside the vault
        if not is_indexable(self.config, source_path):
            return "ignored"
        with self._lock:
            self._prepare()
            doc_id = stable_document_id(source_path)
            if not path.exists():
                if self.index.get_document(document_id=doc_id) is None:
                    return "ignored"
                self.index.delete_document(doc_id, source_path)
                self.index.refresh_fts()
                return "deleted"
            st = path.stat()
            df = DiscoveredFile(path, source_path, st.st_size, st.st_mtime)
            report = IndexReport(mode="sync")
            existing = self.index.get_document(document_id=doc_id)
            if existing is not None:
                try:
                    if sha256_hex(read_bytes(path)) == existing["content_hash"]:
                        return "ignored"  # content unchanged
                except OSError:
                    pass
            ok = self._safe_index_file(df, report)
            if ok:
                self.index.refresh_fts()
            return "indexed" if ok else "error"
