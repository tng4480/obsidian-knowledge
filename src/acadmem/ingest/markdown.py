"""Obsidian-flavoured Markdown -> ``ParsedContent``.

* headings become the section tree (``heading_path`` preserved)
* YAML frontmatter becomes metadata (``tags``, ``ai_generated``, ``title``...)
* wikilinks / embeds (``![[file.pdf#page=3]]``) / Markdown links are recorded
* ``$$`` math blocks, fenced code, tables and ``> [!callout]`` blocks are atomic
  blocks that the chunker never splits in the middle
* block text is sliced from the *original* source by line range, so what is
  indexed is exactly what is in the note
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import frontmatter
from markdown_it import MarkdownIt

from ..models import CALLOUT, CODE, LIST, MATH, PARAGRAPH, QUOTE, TABLE, Block, ParsedContent
from .links import clean_inline, extract_inline_tags, extract_links, is_tag_only, strip_code_and_math
from .pdf_refs import collect_pdf_refs
from .sections import SectionBuilder

_FENCE_OPEN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_CALLOUT = re.compile(r"^\s{0,3}>\s*\[!")

_MD = MarkdownIt("commonmark").enable("table")


def _split_segments(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split body lines into ("text"|"math", lines) segments.

    ``$$`` math blocks are lifted out so blank lines inside a derivation can never
    split it; fenced code is tracked so ``$$`` inside code is not treated as math.
    """
    segments: list[tuple[str, list[str]]] = []
    text_buf: list[str] = []
    i = 0
    fence: str | None = None
    n = len(lines)

    def flush_text() -> None:
        nonlocal text_buf
        if text_buf:
            segments.append(("text", text_buf))
            text_buf = []

    while i < n:
        line = lines[i]
        if fence is not None:
            text_buf.append(line)
            if re.match(rf"^\s{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$", line):
                fence = None
            i += 1
            continue
        m = _FENCE_OPEN.match(line)
        if m:
            fence = m.group(1)
            text_buf.append(line)
            i += 1
            continue
        stripped = line.strip()
        if stripped.startswith("$$"):
            # single-line block: $$ ... $$
            if len(stripped) > 2 and stripped.endswith("$$") and stripped.count("$$") >= 2:
                flush_text()
                segments.append(("math", [line]))
                i += 1
                continue
            # multi-line block: find the closing $$
            j = i + 1
            while j < n and "$$" not in lines[j]:
                j += 1
            if j < n:
                flush_text()
                segments.append(("math", lines[i : j + 1]))
                i = j + 1
                continue
            # unterminated: treat as ordinary text
        text_buf.append(line)
        i += 1
    flush_text()
    return segments


HEADING = "heading"


def iter_markdown_blocks(body: str) -> Iterator[tuple[str, str, int | None]]:
    """Yield ``(kind, text, heading_level)`` in document order.

    ``kind`` is :data:`HEADING` (``text`` is the cleaned title, ``heading_level`` 1-6)
    or one of the block kinds from :mod:`acadmem.models` (``heading_level`` None).
    Block text is sliced verbatim from ``body``. Shared by the Markdown extractor and
    the pymupdf4llm PDF adapter, whose per-page output is Markdown.
    """
    for kind, seg_lines in _split_segments(body.splitlines()):
        if kind == "math":
            yield MATH, "\n".join(seg_lines).strip("\n"), None
            continue
        tokens = _MD.parse("\n".join(seg_lines))
        for idx, tok in enumerate(tokens):
            if tok.level != 0 or tok.nesting == -1:
                continue
            if tok.type == "heading_open":
                title = clean_inline(tokens[idx + 1].content)
                yield HEADING, title or "(untitled)", int(tok.tag[1:])
                continue
            if tok.map is None:
                continue
            raw = "\n".join(seg_lines[tok.map[0] : tok.map[1]]).strip("\n")
            if not raw.strip():
                continue
            if tok.type == "paragraph_open":
                bkind = PARAGRAPH
            elif tok.type in ("bullet_list_open", "ordered_list_open"):
                bkind = LIST
            elif tok.type in ("fence", "code_block"):
                bkind = CODE
            elif tok.type == "table_open":
                bkind = TABLE
            elif tok.type == "blockquote_open":
                bkind = CALLOUT if _CALLOUT.match(raw) else QUOTE
            elif tok.type == "html_block":
                bkind = PARAGRAPH
            else:
                continue  # hr etc.
            yield bkind, raw, None


def _normalize_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = re.split(r"[,\s]+", raw)
    tags: list[str] = []
    for t in raw if isinstance(raw, (list, tuple)) else [raw]:
        t = str(t).strip().lstrip("#").lower()  # Obsidian tags are case-insensitive
        if t and t not in tags:
            tags.append(t)
    return tags


def _union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Order-preserving union of tag tuples."""
    return tuple(dict.fromkeys(t for g in groups for t in g))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def parse_markdown(text: str, fallback_title: str = "") -> ParsedContent:
    """Parse Markdown text into a section tree."""
    try:
        post = frontmatter.loads(text)
        meta: dict[str, Any] = dict(post.metadata)
        body = post.content
    except Exception:  # malformed frontmatter: index the file as plain Markdown
        meta, body = {}, text

    builder = SectionBuilder(root_title=str(meta.get("title") or fallback_title))
    # Inline #tags belong to the block (paragraph, list, table, ...) they are written in. Two conveniences:
    #   * a line holding only tags ("#revisit") tags the block directly above it;
    #   * a tag on a heading line, or a tag line at the very top of a section, tags the first block below it.
    pending: tuple[str, ...] = ()
    for kind, text_or_title, level in iter_markdown_blocks(body):
        if kind == HEADING:
            builder.open(text_or_title, level or 1)
            pending = tuple(_normalize_tags(extract_inline_tags(text_or_title)))
            continue
        own = tuple(_normalize_tags(extract_inline_tags(text_or_title)))
        if own and kind == PARAGRAPH and is_tag_only(text_or_title):
            blocks = builder.current.blocks
            if blocks and not pending:
                blocks[-1] = replace(blocks[-1], tags=_union(blocks[-1].tags, own))
                builder.add(Block(text_or_title, kind, tags=blocks[-1].tags))  # same tags: stays in the same chunk
            else:
                pending = _union(pending, own)
                builder.add(Block(text_or_title, kind, tags=pending))
            continue
        builder.add(Block(text_or_title, kind, tags=_union(own, pending)))
        pending = ()

    sections = builder.build()

    title = str(meta.get("title") or "").strip()
    if not title:
        for s in sections:
            if s.level == 1:
                title = s.title
                break
    title = title or fallback_title

    prose = strip_code_and_math(body)
    note_tags = _normalize_tags(meta.get("tags"))
    tags = _normalize_tags([*note_tags, *extract_inline_tags(body)])
    return ParsedContent(
        title=title,
        sections=sections,
        links=extract_links(prose),
        tags=tags,
        frontmatter=meta,
        note_tags=note_tags,
        pdf_refs=collect_pdf_refs(sections),
    )


class MarkdownExtractor:
    extensions: ClassVar[tuple[str, ...]] = (".md",)
    file_type: ClassVar[str] = "md"

    def parse(self, path: Path) -> ParsedContent:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        return parse_markdown(text, fallback_title=Path(path).stem)


def is_ai_generated(parsed: ParsedContent) -> bool:
    return _truthy(parsed.frontmatter.get("ai_generated"))
