"""Tagged lines in a note that point into a PDF.

A PDF cannot carry tags, so the tag is written in a note next to a link to the PDF::

    [[Week 5.pdf#page=3]] #revisit                                  every section on page 3
    [[Week 5.pdf]] "posterior is proportional" #revisit              the section containing those words
    > selected text                                                  PDF++ "copy as quote":
    [[Week 5.pdf#page=3&selection=4,0,5,20|Week 5, page 3]] #revisit   the quote above the link picks the section
    > [!PDF|yellow] [[Week 5.pdf#page=3&selection=4,0,5,20]]         PDF++ callout: its body is the quote
    > selected text

The ``selection=`` numbers are positions in the PDF viewer's internal text list, not words, so they cannot be
mapped onto our own extraction reliably. They are ignored; the page and the quoted words are used instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from urllib.parse import unquote

from ..models import CALLOUT, QUOTE, Block, PdfRef
from .links import _TAG, strip_code_and_math

_WIKI_PDF = re.compile(r"!?\[\[([^\]\n|#]+?\.pdf)(#[^\]\n|]*)?(?:\|([^\]\n]*))?\]\]", re.I)
_MD_PDF = re.compile(r"!?\[([^\]\n]*)\]\(<?([^)\n>#]+?\.pdf)(#[^)\n>]*)?>?\)", re.I)
_PAGE = re.compile(r"(?:^|[&#])page=(\d+)(?:-(\d+))?", re.I)
_SELECTION = re.compile(r"(?:^|[&#])selection=", re.I)
# PDF++'s default display text is "<file>, page 3" (Obsidian's own uses the same shape).
_DEFAULT_ALIAS = re.compile(r"^.*,\s*(?:page|p\.?)\s*\d+\s*$", re.I)
_QUOTED = re.compile(r'"([^"]{3,500}?)"|\u201c([^\u201d]{3,500}?)\u201d', re.S)
_CALLOUT_MARK = re.compile(r"\[![^\]\n]*\][+-]?")


def _iter_links(prose: str) -> Iterator[tuple[str, str, str | None]]:
    """Yield ``(target, fragment, display_text)`` for every PDF link."""
    for m in _WIKI_PDF.finditer(prose):
        yield m.group(1).strip(), m.group(2) or "", m.group(3)
    for m in _MD_PDF.finditer(prose):
        yield unquote(m.group(2)).strip(), m.group(3) or "", m.group(1) or None


def _strip_links(prose: str) -> str:
    return _MD_PDF.sub(" ", _WIKI_PDF.sub(" ", prose))


def _unquote_block(text: str) -> str:
    """Drop the ``>`` markers of a blockquote / callout."""
    return "\n".join(re.sub(r"^\s{0,3}>[ \t]?", "", line) for line in text.splitlines())


def _body_text(text: str) -> str:
    """The text of a quote or callout block, without its links, callout marker and tags."""
    body = _CALLOUT_MARK.sub(" ", _strip_links(_unquote_block(text)))
    body = _TAG.sub(" ", body)
    return re.sub(r"\s+", " ", body).strip()


def _quoted_strings(prose: str) -> list[str]:
    return [re.sub(r"\s+", " ", (a or b)).strip() for a, b in _QUOTED.findall(prose)]


def extract_pdf_refs(block: Block, previous: Block | None = None) -> list[PdfRef]:
    """The PDF passages a tagged block refers to (empty when the block has no tags or no PDF link)."""
    if not block.tags:
        return []
    prose = strip_code_and_math(block.text)
    links = list(_iter_links(prose))
    if not links:
        return []

    # Where the quote comes from, strongest first: the block's own quote / callout body, words in "double quotes",
    # the blockquote just above (PDF++ "copy as quote").
    quotes: list[str] = []
    if block.kind in (QUOTE, CALLOUT):
        body = _body_text(block.text)
        quotes = [body] if body else []
    else:
        quotes = _quoted_strings(_strip_links(prose))
        if not quotes and previous is not None and previous.kind == QUOTE:
            body = _body_text(previous.text)
            quotes = [body] if body else []

    refs: list[PdfRef] = []
    for target, fragment, alias in links:
        page = _PAGE.search(fragment)
        start = int(page.group(1)) if page else None
        end = int(page.group(2)) if page and page.group(2) else start
        if start is not None and end is not None and end < start:
            start, end = end, start
        if quotes:
            refs.extend(PdfRef(target, start, end, q, False, block.tags) for q in quotes)
            continue
        # A PDF++ selection link may carry the selected words as its display text.
        shown = (alias or "").strip()
        if shown and _SELECTION.search(fragment) and not _DEFAULT_ALIAS.match(shown):
            refs.append(PdfRef(target, start, end, shown, True, block.tags))
        else:
            refs.append(PdfRef(target, start, end, None, False, block.tags))
    return refs


def collect_pdf_refs(sections) -> list[PdfRef]:
    """Every PDF reference in a parsed note, in document order, without duplicates."""
    found: dict[PdfRef, None] = {}
    for section in sections:
        previous: Block | None = None
        for block in section.blocks:
            for ref in extract_pdf_refs(block, previous):
                found.setdefault(ref, None)
            previous = block
    return list(found)
