"""Context expansion: given a retrieved chunk, fetch the surrounding material that makes it understandable.

Specs are small strings so they work from the CLI, benchmark files and MCP tool arguments alike::

    neighbors:2          the 2 chunks before and after (within the same document)
    section              every chunk of the hit's own section
    parent               the hit's parent section (e.g. the chapter intro above a subsection)
    pages:1              every chunk touching the hit's pages +/- 1 (PDF only)
    neighbors:1+parent   combine any of the above

Each returned piece keeps its full provenance and is flagged as the retrieved hit or as added context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .config import Config
from .index import CHUNK_COLUMNS, VectorIndex, sql_quote

DEFAULT_MAX_CHUNKS = 12
HIT, NEIGHBOR, SECTION, PARENT, PAGE = "hit", "neighbor", "section", "parent", "page"
_ROLE_PRIORITY = {HIT: 0, SECTION: 1, PARENT: 2, NEIGHBOR: 3, PAGE: 4}  # most specific reason wins


@dataclass(frozen=True)
class ExpansionSpec:
    neighbors: int = 0
    section: bool = False
    parent: bool = False
    pages: int = 0

    @classmethod
    def parse(cls, spec: "str | dict[str, Any] | ExpansionSpec | None", config: Config | None = None) -> "ExpansionSpec":
        if isinstance(spec, ExpansionSpec):
            return spec
        if spec is None or spec == "":
            if config is None:
                return cls(neighbors=1)
            return cls(neighbors=config.context_neighbors, pages=config.context_pages)
        if isinstance(spec, dict):
            unknown = set(spec) - {"neighbors", "section", "parent", "pages"}
            if unknown:
                raise ValueError(f"Unknown expansion option(s): {sorted(unknown)}")
            return cls(
                neighbors=_non_negative(spec.get("neighbors", 0), "neighbors"),
                section=bool(spec.get("section", False)),
                parent=bool(spec.get("parent", False)),
                pages=_non_negative(spec.get("pages", 0), "pages"),
            )
        kwargs: dict[str, Any] = {}
        for part in str(spec).replace(",", "+").split("+"):
            part = part.strip().lower()
            if not part:
                continue
            name, _, arg = part.partition(":")
            name = name.strip()
            if name in ("neighbors", "neighbours", "neighbor"):
                kwargs["neighbors"] = _non_negative(arg or 1, "neighbors")
            elif name == "pages":
                kwargs["pages"] = _non_negative(arg or 1, "pages")
            elif name in ("section", "parent"):
                kwargs[name] = True
            elif name == "none":
                continue
            else:
                raise ValueError(f"Unknown expansion {name!r}. Use neighbors:N, section, parent, pages:N (join with '+').")
        return cls(**kwargs)

    def describe(self) -> str:
        parts = []
        if self.neighbors:
            parts.append(f"neighbors:{self.neighbors}")
        if self.section:
            parts.append("section")
        if self.parent:
            parts.append("parent")
        if self.pages:
            parts.append(f"pages:{self.pages}")
        return "+".join(parts) or "none"


def _non_negative(value: Any, name: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if n < 0 or n > 20:
        raise ValueError(f"{name} must be between 0 and 20, got {n}")
    return n


@dataclass
class ContextPiece:
    chunk_id: str
    document_id: str
    source_path: str
    file_type: str
    chunk_index: int
    text: str
    role: str
    heading: str = ""
    heading_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    section_id: str | None = None

    @property
    def is_hit(self) -> bool:
        return self.role == HIT

    @property
    def location(self) -> str:
        loc = self.source_path
        if self.page_start:
            loc += f" p.{self.page_start}" + (f"-{self.page_end}" if self.page_end and self.page_end != self.page_start else "")
        if self.heading_path:
            loc += " > " + " > ".join(self.heading_path)
        return loc

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_hit"] = self.is_hit
        d["location"] = self.location
        return d


@dataclass
class ExpandedContext:
    hit_chunk_id: str
    spec: str
    pieces: list[ContextPiece]
    truncated: bool = False

    @property
    def text(self) -> str:
        """The pieces stitched together in reading order, each introduced by its location."""
        return "\n\n".join(f"[{p.location}]{' (match)' if p.is_hit else ''}\n{p.text}" for p in self.pieces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_chunk_id": self.hit_chunk_id,
            "spec": self.spec,
            "truncated": self.truncated,
            "pieces": [p.to_dict() for p in self.pieces],
        }


def _piece(row: dict[str, Any], role: str) -> ContextPiece:
    return ContextPiece(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        source_path=row["source_path"],
        file_type=row.get("file_type") or "",
        chunk_index=int(row.get("chunk_index") or 0),
        text=row.get("text") or "",
        role=role,
        heading=row.get("heading") or "",
        heading_path=list(row.get("heading_path") or []),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        section_id=row.get("section_id"),
    )


class ContextExpander:
    def __init__(self, index: VectorIndex, config: Config | None = None, max_chunks: int = DEFAULT_MAX_CHUNKS) -> None:
        self.index = index
        self.config = config
        self.max_chunks = max_chunks

    # ------------------------------------------------------------------ single hit
    def expand(self, hit: Any, spec: "str | dict | ExpansionSpec | None" = None) -> ExpandedContext:
        """``hit`` is a SearchResult, a chunk row (dict) or a chunk id."""
        parsed = ExpansionSpec.parse(spec, self.config)
        row = self._as_row(hit)
        pieces: dict[str, ContextPiece] = {row["chunk_id"]: _piece(row, HIT)}

        def add(rows: Iterable[dict[str, Any]], role: str) -> None:
            for r in rows:
                existing = pieces.get(r["chunk_id"])
                if existing is None:
                    pieces[r["chunk_id"]] = _piece(r, role)
                elif _ROLE_PRIORITY[role] < _ROLE_PRIORITY[existing.role]:
                    existing.role = role

        doc = sql_quote(row["document_id"])
        if parsed.neighbors:
            idx = int(row["chunk_index"])
            add(self._scan(f"document_id = {doc} AND chunk_index >= {idx - parsed.neighbors} AND chunk_index <= {idx + parsed.neighbors}"), NEIGHBOR)
        if parsed.section and row.get("section_id"):
            add(self._scan(f"document_id = {doc} AND section_id = {sql_quote(row['section_id'])}"), SECTION)
        if parsed.parent and row.get("parent_section_id"):
            add(self._scan(f"document_id = {doc} AND section_id = {sql_quote(row['parent_section_id'])}"), PARENT)
        if parsed.pages and row.get("page_start"):
            lo, hi = int(row["page_start"]) - parsed.pages, int(row.get("page_end") or row["page_start"]) + parsed.pages
            add(self._scan(f"document_id = {doc} AND page_end >= {lo} AND page_start <= {hi}"), PAGE)

        ordered = sorted(pieces.values(), key=lambda p: p.chunk_index)
        truncated = False
        if len(ordered) > self.max_chunks:
            truncated = True
            # Keep the hit and the pieces closest to it, then restore reading order.
            centre = int(row["chunk_index"])
            ordered = sorted(sorted(ordered, key=lambda p: (not p.is_hit, abs(p.chunk_index - centre)))[: self.max_chunks], key=lambda p: p.chunk_index)
        return ExpandedContext(hit_chunk_id=row["chunk_id"], spec=parsed.describe(), pieces=ordered, truncated=truncated)

    # ------------------------------------------------------------------ several hits
    def expand_many(self, hits: Sequence[Any], spec: "str | dict | ExpansionSpec | None" = None) -> list[ContextPiece]:
        """Union of the expansions of several hits: deduplicated, hits keep their ``hit`` role."""
        merged: dict[str, ContextPiece] = {}
        for hit in hits:
            for piece in self.expand(hit, spec).pieces:
                have = merged.get(piece.chunk_id)
                if have is None or _ROLE_PRIORITY[piece.role] < _ROLE_PRIORITY[have.role]:
                    merged[piece.chunk_id] = piece
        return sorted(merged.values(), key=lambda p: (p.source_path, p.chunk_index))

    # ------------------------------------------------------------------ helpers
    def _scan(self, where: str) -> list[dict[str, Any]]:
        return self.index.scan(where, limit=1000, columns=CHUNK_COLUMNS)

    def _as_row(self, hit: Any) -> dict[str, Any]:
        if isinstance(hit, str):
            row = self.index.get_chunk(hit)
            if row is None:
                raise KeyError(f"No such chunk: {hit}")
            return row
        if isinstance(hit, dict):
            return hit
        needed = ("chunk_id", "document_id", "chunk_index")
        if all(hasattr(hit, a) for a in needed):  # SearchResult
            return {c: getattr(hit, c, None) for c in CHUNK_COLUMNS}
        raise TypeError(f"Cannot expand {type(hit).__name__}")


__all__ = [
    "ContextExpander",
    "ContextPiece",
    "ExpandedContext",
    "ExpansionSpec",
    "HIT",
    "NEIGHBOR",
    "SECTION",
    "PARENT",
    "PAGE",
]
