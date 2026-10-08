"""Tags on PDF passages.

A PDF cannot be edited, so its tags live in the user's notes (see ``ingest/pdf_refs.py``). The indexer records
every tagged line that points into a PDF, and :func:`reconcile_pdf_tags` turns those into ``tags`` on the PDF's
chunks. The chunk tags are therefore a pure function of (the notes, the PDF's chunks): re-indexing a PDF, editing a
note or deleting one all converge to the right tags, and an index rebuild loses nothing.

Matching works on the indexed sections (chunks), which is the smallest unit a tag can address:

* no page, no quote      every section of the PDF
* page(s) only           every section that touches those pages
* quote (+ optional page) the section(s) containing those words (heading breadcrumb included), compared on words
                          only, so line breaks, hyphenation, case, punctuation and ``**bold**`` markers do not matter
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .index import VectorIndex, sql_quote

# A quote that matches more sections than this is too vague to be what the user meant: report it instead of
# tagging half a handout.
MAX_QUOTE_MATCHES = 5
MIN_COVERAGE = 0.7  # share of the quote's word-groups that must be found for a non-verbatim match

_WORDS = re.compile(r"[^\W_]+")
_HYPHEN_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")

_SCAN_COLUMNS = ["chunk_id", "document_id", "source_path", "chunk_index", "heading_path", "text", "page_start", "page_end", "tags"]


_HTML_TAG = re.compile(r"</?(?:sup|sub|b|i|u|em|strong|br|span)\b[^>]*>", re.I)  # what the extractor writes for formatting
MIN_SQUASHED_CHARS = 12  # a quote shorter than this is only ever compared word by word


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")  # ligatures, sub/superscript digits (C₁ -> C1), full-width forms
    return _HYPHEN_BREAK.sub("", _HTML_TAG.sub("", text))


def normalize_words(text: str) -> list[str]:
    """Lower-case words of ``text`` with typographic noise removed (the form quotes are compared in)."""
    return _WORDS.findall(_clean(text).lower())


def squash(text: str) -> str:
    """Letters and digits only. Robust to what spacing does to formulas and tables (``C ₁`` vs ``C₁``, ``|a|b|``)."""
    return "".join(_WORDS.findall(_clean(text).lower()))


def _shingles(words: list[str], size: int) -> set[tuple[str, ...]]:
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


# --------------------------------------------------------------------------- resolving the PDF
def resolve_pdf(target: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """Find the indexed PDF a note's link names. Returns ``(document, None)`` or ``(None, reason)``.

    Like Obsidian, a bare file name is enough when it is unique in the vault; a path (or the tail of one) is needed
    when two PDFs share a name.
    """
    wanted = target.replace("\\", "/").strip().strip("/").lower()
    while wanted.startswith("./"):
        wanted = wanted[2:]
    wanted = wanted.replace("../", "")
    if not wanted:
        return None, "empty PDF name"
    if not wanted.endswith(".pdf"):
        wanted += ".pdf"  # tolerate "Week 5 - Probability" when asked by name rather than written as a link
    pdfs = [d for d in docs if (d.get("file_type") or "") == "pdf"]
    exact = [d for d in pdfs if d["source_path"].lower() == wanted]
    if exact:
        return exact[0], None
    tail = [d for d in pdfs if d["source_path"].lower().endswith("/" + wanted) or d["source_path"].lower() == wanted]
    if len(tail) == 1:
        return tail[0], None
    if len(tail) > 1:
        names = ", ".join(sorted(d["source_path"] for d in tail)[:5])
        return None, f"ambiguous: several PDFs match ({names}); write the folder in the link"
    return None, "no indexed PDF with that name"


# --------------------------------------------------------------------------- matching
@dataclass
class Passage:
    chunk_id: str
    chunk_index: int
    heading: str
    page_start: int | None
    page_end: int | None


@dataclass
class Match:
    passages: list[Passage] = field(default_factory=list)
    reason: str | None = None  # set when nothing was matched: why

    @property
    def ok(self) -> bool:
        return bool(self.passages) and self.reason is None


def _to_passage(row: dict[str, Any]) -> Passage:
    return Passage(
        row["chunk_id"],
        int(row.get("chunk_index") or 0),
        " > ".join(row.get("heading_path") or []),
        row.get("page_start"),
        row.get("page_end"),
    )


def _on_pages(row: dict[str, Any], start: int | None, end: int | None) -> bool:
    if start is None:
        return True
    lo = row.get("page_start")
    hi = row.get("page_end") or lo
    if lo is None:
        return False
    return lo <= (end or start) and hi >= start


def find_passages(rows: list[dict[str, Any]], page_start: int | None, page_end: int | None, quote: str | None) -> Match:
    """Pick the chunks of one PDF that a page / quote refers to."""
    rows = sorted(rows, key=lambda r: r.get("chunk_index") or 0)
    candidates = [r for r in rows if _on_pages(r, page_start, page_end)]
    where = ""
    if page_start is not None:
        where = f" on page {page_start}" if (page_end or page_start) == page_start else f" on pages {page_start}-{page_end}"
    if not candidates:
        return Match(reason=f"the PDF has no indexed text{where}")
    if not quote:
        return Match([_to_passage(r) for r in candidates])

    words = normalize_words(quote)
    if not words:
        return Match(reason="the quoted text has no words")
    needle = " " + " ".join(words) + " "
    texts = {r["chunk_id"]: normalize_words(" ".join(list(r.get("heading_path") or []) + [r.get("text") or ""])) for r in candidates}

    exact = [r for r in candidates if needle in " " + " ".join(texts[r["chunk_id"]]) + " "]
    squashed_quote = squash(quote)
    use_squash = len(squashed_quote) >= MIN_SQUASHED_CHARS
    if not exact and use_squash:
        # Same words once spacing is ignored: formulas, tables and subscripts are spaced differently by the PDF
        # viewer and by our extractor.
        exact = [r for r in candidates if squashed_quote in squash(" ".join(list(r.get("heading_path") or []) + [r.get("text") or ""]))]
    if not exact:
        # A selection that runs over a section boundary is in neither section alone, but is verbatim in two
        # neighbours read one after the other.
        bodies = {r["chunk_id"]: normalize_words(r.get("text") or "") for r in candidates}
        for a, b in zip(candidates, candidates[1:]):
            if (b.get("chunk_index") or 0) != (a.get("chunk_index") or 0) + 1:
                continue
            if needle in " " + " ".join(bodies[a["chunk_id"]] + bodies[b["chunk_id"]]) + " " or (
                use_squash and squashed_quote in squash((a.get("text") or "") + " " + (b.get("text") or ""))
            ):
                exact.extend(r for r in (a, b) if r not in exact)
    if not exact:
        # Not verbatim (extraction differences, a quote that runs over a section boundary): fall back to how much
        # of the quote's word groups each section contains, and accept a small set that together covers most of it.
        size = 3 if len(words) >= 3 else len(words)
        wanted = _shingles(words, size)
        scored = [(len(wanted & _shingles(texts[r["chunk_id"]], size)), r) for r in candidates]
        scored = [(n, r) for n, r in scored if n > 0]
        scored.sort(key=lambda s: (-s[0], s[1].get("chunk_index") or 0))
        remaining, chosen = set(wanted), []
        for _, r in scored:
            gain = remaining & _shingles(texts[r["chunk_id"]], size)
            if len(gain) >= max(1, 0.2 * len(wanted)):
                chosen.append(r)
                remaining -= gain
            if len(remaining) <= (1 - MIN_COVERAGE) * len(wanted):
                break
        if not chosen or len(remaining) > (1 - MIN_COVERAGE) * len(wanted) + 1e-9:
            return Match(reason=f"the quoted words were not found in the PDF's indexed text{where}")
        exact = chosen
    if len(exact) > MAX_QUOTE_MATCHES:
        return Match(reason=f"the quote matches {len(exact)} sections{where}; quote a longer or more distinctive passage")
    return Match([_to_passage(r) for r in sorted(exact, key=lambda r: r.get("chunk_index") or 0)])


def load_chunks(index: VectorIndex, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The stored chunks of one indexed PDF (without their vectors)."""
    return index.scan(f"document_id = {sql_quote(doc['id'])}", columns=_SCAN_COLUMNS)


# --------------------------------------------------------------------------- reconciling
@dataclass
class PdfTagEntry:
    note: str  # the note holding the tagged line
    target: str  # the PDF as written
    pdf: str | None  # the resolved indexed PDF
    page: str | None
    quote: str | None
    tags: list[str]
    status: str  # "applied" | "page_fallback" | "unresolved" | "unmatched"
    detail: str = ""
    sections: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("applied", "page_fallback")

    def to_dict(self) -> dict[str, Any]:
        return {
            "note": self.note,
            "pdf": self.pdf or self.target,
            "page": self.page,
            "quote": self.quote,
            "tags": self.tags,
            "status": self.status,
            "sections": self.sections,
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass
class PdfTagReport:
    entries: list[PdfTagEntry] = field(default_factory=list)
    chunks_changed: int = 0

    @property
    def problems(self) -> list[PdfTagEntry]:
        return [e for e in self.entries if not e.ok]

    def to_dict(self, max_problems: int = 20) -> dict[str, Any]:
        return {
            "tag_lines": len(self.entries),
            "applied": sum(1 for e in self.entries if e.ok),
            "problems": [e.to_dict() for e in self.problems[:max_problems]],
            "problem_count": len(self.problems),
        }


def pages_text(start: int | None, end: int | None) -> str | None:
    if start is None:
        return None
    return str(start) if not end or end == start else f"{start}-{end}"


def evaluate_pdf_tags(index: VectorIndex) -> tuple[PdfTagReport, dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Work out what every PDF chunk's tags should be. Read-only.

    Returns ``(report, desired, chunk_rows)`` where ``desired`` maps ``chunk_id -> tags`` for every chunk of every PDF
    that is referenced or currently carries tags, and ``chunk_rows`` holds those chunks as they are stored now.
    """
    refs = index.pdf_ref_rows()
    pdf_docs = index.list_documents("file_type = 'pdf'", include_empty=False)
    report = PdfTagReport()
    chunk_cache: dict[str, list[dict[str, Any]]] = {}

    def chunks_of(doc: dict[str, Any]) -> list[dict[str, Any]]:
        if doc["id"] not in chunk_cache:
            chunk_cache[doc["id"]] = load_chunks(index, doc)
        return chunk_cache[doc["id"]]

    # Documents to examine: the ones some note points at, plus the ones that carry tags now (so removed tags go away).
    for doc in pdf_docs:
        if doc.get("tags"):
            chunks_of(doc)
    desired_sets: dict[str, dict[str, None]] = {}

    for ref in refs:
        entry = PdfTagEntry(
            note=ref["source_path"],
            target=ref["target"] or "",
            pdf=None,
            page=pages_text(ref["page_start"], ref["page_end"]),
            quote=ref["quote"],
            tags=list(ref["tags"] or []),
            status="unresolved",
        )
        report.entries.append(entry)
        doc, reason = resolve_pdf(entry.target, pdf_docs)
        if doc is None:
            entry.detail = reason or ""
            continue
        entry.pdf = doc["source_path"]
        rows = chunks_of(doc)
        match = find_passages(rows, ref["page_start"], ref["page_end"], ref["quote"])
        if not match.ok and ref["quote"] and ref["quote_is_weak"]:
            fallback = find_passages(rows, ref["page_start"], ref["page_end"], None)
            if fallback.ok:
                entry.status, entry.detail = "page_fallback", "the link text was not found in the PDF; tagged the page instead"
                match = fallback
        if not match.ok:
            entry.status, entry.detail = "unmatched", match.reason or "nothing matched"
            continue
        if entry.status != "page_fallback":
            entry.status = "applied"
        entry.sections = len(match.passages)
        for passage in match.passages:
            bucket = desired_sets.setdefault(passage.chunk_id, {})
            for tag in entry.tags:
                bucket.setdefault(tag, None)

    desired: dict[str, list[str]] = {}
    current: dict[str, dict[str, Any]] = {}
    for rows in chunk_cache.values():
        for row in rows:
            current[row["chunk_id"]] = row
            desired[row["chunk_id"]] = list(desired_sets.get(row["chunk_id"], {}))
    return report, desired, current


def reconcile_pdf_tags(index: VectorIndex) -> PdfTagReport:
    """Make every PDF chunk's tags match what the notes say, writing only the chunks that differ."""
    report, desired, current = evaluate_pdf_tags(index)
    changed = {
        cid: tags for cid, tags in desired.items() if sorted(tags) != sorted(current[cid].get("tags") or [])
    }
    if changed:
        index.replace_chunk_tags(changed)
        by_doc: dict[str, dict[str, None]] = {}
        for cid, row in current.items():
            for tag in desired[cid]:
                by_doc.setdefault(row["document_id"], {}).setdefault(tag, None)
        touched = {current[cid]["document_id"] for cid in changed}
        index.replace_document_tags({doc_id: sorted(by_doc.get(doc_id, {})) for doc_id in touched})
    report.chunks_changed = len(changed)
    return report


__all__ = [
    "MAX_QUOTE_MATCHES",
    "Match",
    "Passage",
    "PdfTagEntry",
    "PdfTagReport",
    "evaluate_pdf_tags",
    "find_passages",
    "load_chunks",
    "normalize_words",
    "reconcile_pdf_tags",
    "resolve_pdf",
    "squash",
]
