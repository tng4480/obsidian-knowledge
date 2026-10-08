"""Structure-aware chunking.

Input is a ``ParsedDocument`` (flat, ordered sections with heading paths and page-tagged
blocks); output is a list of :class:`~acadmem.models.Chunk`. The algorithm:

1. **Sections are the primary boundary.** Blocks of one section are packed greedily into
   chunks of at most ``max_tokens``; a chunk never silently straddles two sections.
2. **Atomic blocks stay whole** (math, code, tables, callouts) unless a single one exceeds
   ``max_tokens`` - then it is split on line boundaries (tables repeat their header row).
3. **Oversized prose blocks** are split on semantic boundaries with ``semchunk`` and consecutive
   pieces overlap by ``overlap_tokens``. This is the *only* place overlap is used.
4. **Small adjacent pieces are combined** (``min_tokens``) when they are structurally related
   (same section, parent/child, or siblings). When pieces from different sections are merged, the
   later section's heading is written inline so no heading text is lost; the chunk keeps the first
   piece's section as its structural parent.
5. Every chunk carries its section id, parent section id, heading path, page range and the
   document provenance, so a retrieved chunk is traceable to file -> page -> section -> text.

The chunker takes a ``count_tokens`` callable (the embedding model's tokenizer in production, a
whitespace counter in tests) and has no other dependencies, so it is independently testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import semchunk

from .ingest.links import extract_links
from .models import TABLE, Block, Chunk, ParsedDocument, Section

TokenCounter = Callable[[str], int]


@dataclass
class _Piece:
    """A contiguous run of text from a single section, with its page range."""

    section: Section
    texts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    heading_inline: list[tuple[str, int]] = field(default_factory=list)  # (title, offset in texts)
    tokens: int = 0
    # Inline #tags of the blocks this piece is made of. Blocks with different tags are never packed together, so a
    # tag filter returns exactly the tagged passage and not the paragraphs that happen to sit next to it.
    tags: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n\n".join(self.texts)


class Chunker:
    def __init__(
        self,
        count_tokens: TokenCounter,
        max_tokens: int = 384,
        min_tokens: int = 64,
        overlap_tokens: int = 48,
    ) -> None:
        if max_tokens < 8:
            raise ValueError("max_tokens must be >= 8")
        self.count = count_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min(min_tokens, max_tokens)
        self.overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))
        self._splitter = semchunk.chunkerify(count_tokens, max_tokens)

    # ------------------------------------------------------------------ public
    def chunk_document(self, parsed: ParsedDocument) -> list[Chunk]:
        doc = parsed.document
        by_id = {s.section_id: s for s in parsed.sections}
        note_tags = list(parsed.note_tags if parsed.note_tags is not None else doc.tags)

        pieces: list[_Piece] = []
        for section in parsed.sections:
            pieces.extend(self._pack_section(section))
        pieces = self._merge_small(pieces, by_id)

        chunks: list[Chunk] = []
        for idx, piece in enumerate(pieces):
            section = piece.section
            pages = [p for p in piece.pages if p is not None]
            text = self._render(piece)
            parent = f"{doc.id}#{section.parent_id}" if section.parent_id else None
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.id}:{idx:05d}",
                    document_id=doc.id,
                    source_path=doc.source_path,
                    file_type=doc.file_type,
                    module=doc.module,
                    document_type=doc.document_type,
                    modification_time=doc.modification_time,
                    chunk_index=idx,
                    section_id=f"{doc.id}#{section.section_id}",
                    parent_section_id=parent,
                    heading_path=list(section.heading_path),
                    text=text,
                    page_number=min(pages) if pages else None,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    token_count=self.count(text),
                    ai_generated=doc.ai_generated,
                    tags=list(dict.fromkeys([*note_tags, *piece.tags])),
                    links=extract_links(text),
                )
            )
        return chunks

    # ------------------------------------------------------------- packing
    def _expand_block(self, block: Block) -> list[tuple[str, int | None]]:
        """Return (text, page) units no larger than ``max_tokens`` (overlap only on forced splits)."""
        text = block.text.strip()
        if not text:
            return []
        if self.count(text) <= self.max_tokens:
            return [(text, block.page)]
        if block.atomic:
            return [(t, block.page) for t in self._split_atomic(text, block.kind)]
        return [(t, block.page) for t in self._split_prose(text)]  # every unit of a block inherits its tags

    def _split_prose(self, text: str) -> list[str]:
        overlap = self.overlap_tokens if self.overlap_tokens > 0 else None
        pieces = self._splitter(text, overlap=overlap) if overlap else self._splitter(text)
        return [p.strip() for p in pieces if p.strip()]

    def _split_atomic(self, text: str, kind: str) -> list[str]:
        lines = text.split("\n")
        prefix: list[str] = []
        if kind == TABLE and len(lines) > 3:
            prefix, lines = lines[:2], lines[2:]  # header row + separator row, repeated per piece
        prefix_text = "\n".join(prefix)
        budget = self.max_tokens - (self.count(prefix_text) if prefix else 0)
        budget = max(budget, self.max_tokens // 2)

        out: list[str] = []
        cur: list[str] = []
        cur_tokens = 0
        for line in lines:
            lt = self.count(line)
            if lt > budget:  # a single over-long line: fall back to semantic splitting
                if cur:
                    out.append("\n".join(cur))
                    cur, cur_tokens = [], 0
                out.extend(self._split_prose(line))
                continue
            if cur and cur_tokens + lt > budget:
                out.append("\n".join(cur))
                cur, cur_tokens = [], 0
            cur.append(line)
            cur_tokens += lt
        if cur:
            out.append("\n".join(cur))
        if prefix:
            out = [f"{prefix_text}\n{o}" if not o.startswith(prefix[0]) else o for o in out]
        return out

    def _pack_section(self, section: Section) -> list[_Piece]:
        units: list[tuple[str, int | None, tuple[str, ...]]] = []
        for block in section.blocks:
            units.extend((text, page, block.tags) for text, page in self._expand_block(block))

        pieces: list[_Piece] = []
        cur: _Piece | None = None
        for text, page, tags in units:
            t = self.count(text)
            if cur is not None and cur.tags == tags and cur.tokens + t <= self.max_tokens:
                cur.texts.append(text)
                if page is not None:
                    cur.pages.append(page)
                cur.tokens = self.count(cur.text)
                continue
            if cur is not None:
                pieces.append(cur)
            cur = _Piece(section=section, texts=[text], pages=[page] if page is not None else [], tokens=t, tags=tags)
        if cur is not None:
            pieces.append(cur)
        return pieces

    # ------------------------------------------------------------- merging
    @staticmethod
    def _related(prev: Section, nxt: Section, by_id: dict[str, Section]) -> bool:
        if prev.section_id == nxt.section_id:
            return True
        if prev.parent_id == nxt.parent_id:
            return True  # siblings (including top-level sections, whose parent is None)
        cur: Section | None = nxt  # is nxt a descendant of prev (child content after the parent's text)?
        while cur is not None and cur.parent_id is not None:
            if cur.parent_id == prev.section_id:
                return True
            cur = by_id.get(cur.parent_id)
        return False

    def _merge_small(self, pieces: list[_Piece], by_id: dict[str, Section]) -> list[_Piece]:
        merged: list[_Piece] = []
        for piece in pieces:
            if merged:
                prev = merged[-1]
                if (
                    prev.tags == piece.tags
                    and (prev.tokens < self.min_tokens or piece.tokens < self.min_tokens)
                    and self._related(prev.section, piece.section, by_id)
                ):
                    candidate = self._combine(prev, piece)
                    if candidate.tokens <= self.max_tokens:
                        merged[-1] = candidate
                        continue
            merged.append(piece)
        return merged

    def _combine(self, a: _Piece, b: _Piece) -> _Piece:
        texts = list(a.texts)
        heading_inline = list(a.heading_inline)
        if b.section.section_id != a.section.section_id and b.section.title:
            heading_inline.append((b.section.title, len(texts)))
        texts.extend(b.texts)
        out = _Piece(section=a.section, texts=texts, pages=a.pages + b.pages, heading_inline=heading_inline, tags=a.tags)
        out.tokens = self.count(self._render(out))
        return out

    @staticmethod
    def _render(piece: _Piece) -> str:
        """Join texts; inline headings of merged-in sections precede their first text."""
        inline = dict((offset, title) for title, offset in piece.heading_inline)
        parts: list[str] = []
        for i, text in enumerate(piece.texts):
            if i in inline:
                parts.append(inline[i])
            parts.append(text)
        return "\n\n".join(parts)
