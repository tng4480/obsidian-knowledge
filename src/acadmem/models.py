"""Common cross-format representation.

Every supported file type (Markdown, PDF, future formats) is reduced to the same
``ParsedDocument``: a flat, document-ordered list of ``Section`` objects, each
carrying its heading path, parent link, and ``Block``s that remember the page
they came from. The chunker, indexer and retriever only ever see these types, so
provenance (file -> page -> section -> text) survives every stage.

The index is *derived* data: nothing in these models is needed to reconstruct the
vault.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

# Block kinds. "atomic" kinds are never split in the middle by the chunker.
PARAGRAPH = "paragraph"
LIST = "list"
TABLE = "table"
CODE = "code"
MATH = "math"
QUOTE = "quote"
CALLOUT = "callout"  # Obsidian "> [!note]" blocks
ATOMIC_KINDS = frozenset({TABLE, CODE, MATH, CALLOUT})


def stable_document_id(source_path: str) -> str:
    """Stable id derived from the vault-relative POSIX path."""
    return hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:16]


def infer_path_metadata(source_path: str) -> tuple[str | None, str | None]:
    """Infer ``(module, document_type)`` from a vault-relative path.

    ``3B1/Questions/foo.md`` -> ``("3B1", "Questions")``. Nothing is hard-coded: the
    first folder is the module and the second folder is the document type. Files
    that sit at the vault root, or directly inside a module folder, get ``None``
    for the levels that do not exist.
    """
    parts = PurePosixPath(source_path).parts
    folders = parts[:-1]
    module = folders[0] if len(folders) >= 1 else None
    document_type = folders[1] if len(folders) >= 2 else None
    return module, document_type


@dataclass(frozen=True)
class Block:
    """A contiguous piece of body text inside a section."""

    text: str
    kind: str = PARAGRAPH
    page: int | None = None  # 1-based; None for formats without pages

    @property
    def atomic(self) -> bool:
        return self.kind in ATOMIC_KINDS


@dataclass
class Section:
    """A heading-delimited region of a document."""

    section_id: str
    title: str
    level: int  # 0 = document root / preamble, 1.. = heading depth
    heading_path: tuple[str, ...]
    parent_id: str | None
    blocks: list[Block] = field(default_factory=list)

    @property
    def page_start(self) -> int | None:
        pages = [b.page for b in self.blocks if b.page is not None]
        return min(pages) if pages else None

    @property
    def page_end(self) -> int | None:
        pages = [b.page for b in self.blocks if b.page is not None]
        return max(pages) if pages else None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass
class ParsedContent:
    """What an extractor returns for one file (before path metadata is attached)."""

    title: str
    sections: list[Section]
    links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    page_count: int | None = None


@dataclass
class Document:
    """Document-level record (one row per indexed file)."""

    id: str
    source_path: str  # vault-relative, POSIX separators
    file_type: str  # "md" | "pdf" | ...
    module: str | None
    document_type: str | None
    modification_time: float  # POSIX seconds
    content_hash: str = ""
    title: str = ""
    page_count: int | None = None
    ai_generated: bool = False
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    chunk_count: int = 0
    indexed_at: float = 0.0


@dataclass
class ParsedDocument:
    document: Document
    sections: list[Section]
    frontmatter: dict[str, Any] = field(default_factory=dict)

    def section(self, section_id: str) -> Section | None:
        for s in self.sections:
            if s.section_id == section_id:
                return s
        return None


@dataclass
class Chunk:
    """The unit that is embedded and retrieved."""

    chunk_id: str
    document_id: str
    source_path: str
    file_type: str
    module: str | None
    document_type: str | None
    modification_time: float
    chunk_index: int  # position within the document, document order
    section_id: str
    parent_section_id: str | None
    heading_path: list[str]
    text: str  # raw text shown to the user / LLM
    page_number: int | None  # first page the chunk touches
    page_start: int | None
    page_end: int | None
    token_count: int = 0
    ai_generated: bool = False
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    # Breadcrumb + body: exactly what is embedded, and what the full-text index covers, so a query can match a
    # heading that its body never repeats ("## Before using the final value theorem" -> "Look at the poles ...").
    search_text: str = ""

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    def embedding_text(self, document_title: str = "") -> str:
        """Text actually embedded: breadcrumb + body, so the vector knows where it lives."""
        crumbs = [c for c in ([document_title] if document_title else []) + self.heading_path if c]
        # avoid repeating identical neighbours
        dedup: list[str] = []
        for c in crumbs:
            if not dedup or dedup[-1] != c:
                dedup.append(c)
        prefix = " > ".join(dedup)
        return f"{prefix}\n\n{self.text}" if prefix else self.text
