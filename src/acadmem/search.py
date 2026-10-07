"""Retrieval: vector / full-text / hybrid search with metadata prefiltering.

Filters are compiled into a SQL predicate that LanceDB applies *before* ranking (``prefilter=True``),
so ``top_k`` always means "the k best chunks *among those allowed*", never "k best, then thrown away".
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable

from .config import Config
from .embeddings import Embedder, get_embedder
from .index import CHUNK_COLUMNS, FTS_COLUMN, VectorIndex, sql_quote

MODES = ("vector", "fts", "hybrid")
_WORD = re.compile(r"\w+", re.UNICODE)


# --------------------------------------------------------------------------- filters
def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def to_epoch(value: float | int | str | date | datetime | None) -> float | None:
    """Accept epoch seconds, ``datetime``/``date`` objects or ISO strings (``2026-01-31``). Naive = UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"Not a date/time: {value!r} (use e.g. 2026-01-31 or 2026-01-31T09:00)") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


@dataclass
class SearchFilters:
    """All fields optional; unset means "don't filter". Lists mean "any of".

    ``module`` / ``document_type`` / ``file_type`` / ``tags`` match case-insensitively.
    ``source_path`` is an exact vault-relative path; ``path_prefix`` is a folder or filename prefix.
    ``modified_after`` / ``modified_before`` compare against the file's modification time.
    ``ai_generated`` = True/False keeps only AI-written / only human-written notes.
    """

    module: str | list[str] | None = None
    document_type: str | list[str] | None = None
    file_type: str | list[str] | None = None
    source_path: str | list[str] | None = None
    path_prefix: str | None = None
    tags: str | list[str] | None = None
    modified_after: float | str | date | datetime | None = None
    modified_before: float | str | date | datetime | None = None
    ai_generated: bool | None = None

    @classmethod
    def from_mapping(cls, data: "dict[str, Any] | SearchFilters | None") -> "SearchFilters":
        if data is None:
            return cls()
        if isinstance(data, SearchFilters):
            return data
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown filter(s): {sorted(unknown)}. Valid filters: {sorted(known)}")
        return cls(**{k: v for k, v in data.items() if v not in (None, "", [])})

    def is_empty(self) -> bool:
        return self.to_sql() is None

    def to_sql(self) -> str | None:
        clauses: list[str] = []

        def any_of_ci(column: str, values: list[str]) -> None:
            if values:
                clauses.append(f"lower({column}) IN ({', '.join(sql_quote(v.lower()) for v in values)})")

        any_of_ci("module", _as_list(self.module))
        any_of_ci("document_type", _as_list(self.document_type))
        any_of_ci("file_type", [v.lower().lstrip(".") for v in _as_list(self.file_type)])

        paths = [p.replace("\\", "/") for p in _as_list(self.source_path)]
        if paths:
            clauses.append(f"source_path IN ({', '.join(sql_quote(p) for p in paths)})")
        if self.path_prefix:
            prefix = self.path_prefix.replace("\\", "/").lstrip("/")
            clauses.append(f"starts_with(lower(source_path), {sql_quote(prefix.lower())})")

        tags = [t.lstrip("#").lower() for t in _as_list(self.tags)]
        if tags:
            clauses.append(f"array_has_any(tags, make_array({', '.join(sql_quote(t) for t in tags)}))")

        after, before = to_epoch(self.modified_after), to_epoch(self.modified_before)
        if after is not None:
            clauses.append(f"modification_time >= {after!r}")
        if before is not None:
            clauses.append(f"modification_time < {before!r}")
        if self.ai_generated is not None:
            clauses.append("ai_generated = true" if self.ai_generated else "ai_generated = false")

        return " AND ".join(f"({c})" for c in clauses) if clauses else None


# --------------------------------------------------------------------------- results
@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    source_path: str
    file_type: str
    module: str | None
    document_type: str | None
    text: str
    #: Cosine similarity between the query and this chunk's embedding, in every mode (FTS-only hits included).
    #: Same meaning and scale whatever ``mode`` was used. It does NOT determine the order of a hybrid result list.
    score: float
    #: 1-based position in the result list (this, not ``score``, is what the ranking decided).
    rank: int
    mode: str
    #: The value the ranking was actually based on: reciprocal-rank-fusion (hybrid, <= ~0.033), BM25 (fts) or cosine
    #: (vector). Only meaningful for ordering within one list; kept for debugging and evaluation.
    rank_score: float = 0.0
    heading: str = ""
    heading_path: list[str] = field(default_factory=list)
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int = 0
    section_id: str | None = None
    parent_section_id: str | None = None
    modification_time: float | None = None
    ai_generated: bool = False
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    token_count: int = 0

    @property
    def location(self) -> str:
        """Human-readable provenance, e.g. ``3B1/Handouts/x.pdf p.3-4 > Probability > Independence``."""
        loc = self.source_path
        if self.page_start:
            loc += f" p.{self.page_start}" + (f"-{self.page_end}" if self.page_end and self.page_end != self.page_start else "")
        if self.heading_path:
            loc += " > " + " > ".join(self.heading_path)
        return loc

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["location"] = self.location
        return d


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity of two vectors (pure Python: 384 floats per hit is nothing next to the embedding call)."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / math.sqrt(na * nb) if na and nb else 0.0


def row_to_result(row: dict[str, Any], rank_score: float, rank: int, mode: str, similarity: float) -> SearchResult:
    """Build a result: ``similarity`` becomes ``score``; ``rank_score`` is the mode's native ranking value."""

    def g(key, default=None):
        value = row.get(key)
        return default if value is None else value

    return SearchResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        source_path=row["source_path"],
        file_type=g("file_type", ""),
        module=row.get("module"),
        document_type=row.get("document_type"),
        text=g("text", ""),
        score=float(similarity),
        rank=rank,
        mode=mode,
        rank_score=float(rank_score),
        heading=g("heading", ""),
        heading_path=list(g("heading_path", [])),
        page_number=row.get("page_number"),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        chunk_index=int(g("chunk_index", 0)),
        section_id=row.get("section_id"),
        parent_section_id=row.get("parent_section_id"),
        modification_time=row.get("modification_time"),
        ai_generated=bool(g("ai_generated", False)),
        tags=list(g("tags", [])),
        links=list(g("links", [])),
        token_count=int(g("token_count", 0)),
    )


# --------------------------------------------------------------------------- searcher
class Searcher:
    def __init__(self, config: Config, embedder: Embedder | None = None, index: VectorIndex | None = None) -> None:
        self.config = config
        self.index = index or VectorIndex(config)
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: "SearchFilters | dict[str, Any] | None" = None,
        mode: str | None = None,
    ) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        mode = (mode or self.config.search_mode).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        k = int(self.config.default_top_k if top_k is None else top_k)
        if k < 1:
            raise ValueError("top_k must be >= 1")
        where = SearchFilters.from_mapping(filters).to_sql()

        if self.index.chunks.count_rows() == 0:
            return []
        if mode == "vector":
            return self._vector(query, k, where)
        words = " ".join(_WORD.findall(query))
        if mode == "fts" or not words:
            # Nothing tokenisable (e.g. a query of pure symbols): fall back to vector search.
            return self._fts(query, words, k, where) if words else self._vector(query, k, where)
        return self._hybrid(query, words, k, where)

    # -- modes
    @staticmethod
    def _similarity(row: dict[str, Any], qvec: list[float]) -> float:
        """Cosine(query, stored chunk vector): the one relevance number every mode reports."""
        vector = row.get("vector")
        if vector is None:  # pragma: no cover - LanceDB returns the vector column for every query type we use
            raise RuntimeError("LanceDB did not return the chunk vector; cannot compute similarity")
        return cosine_similarity(qvec, vector)

    def _vector(self, query: str, k: int, where: str | None) -> list[SearchResult]:
        qvec = self.embedder.embed_query(query)
        q = self.index.chunks.search(qvec).metric("cosine")
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(k).to_list()
        # LanceDB reports cosine *distance*; the ranking value is the similarity 1 - distance (higher = better).
        return [
            row_to_result(r, 1.0 - float(r["_distance"]), i + 1, "vector", self._similarity(r, qvec)) for i, r in enumerate(rows)
        ]

    def _fts(self, query: str, words: str, k: int, where: str | None) -> list[SearchResult]:
        self._ensure_fts()
        qvec = self.embedder.embed_query(query)
        q = self.index.chunks.search(words, query_type="fts", fts_columns=FTS_COLUMN)
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(k).to_list()
        # BM25 only decides the order; the reported score is the cosine similarity of the same hit.
        return [
            row_to_result(r, float(r.get("_score", 0.0)), i + 1, "fts", self._similarity(r, qvec)) for i, r in enumerate(rows)
        ]

    def _hybrid(self, query: str, words: str, k: int, where: str | None) -> list[SearchResult]:
        from lancedb.rerankers import RRFReranker

        self._ensure_fts()
        qvec = self.embedder.embed_query(query)
        q = (
            self.index.chunks.search(query_type="hybrid", fts_columns=FTS_COLUMN)
            .vector(qvec)
            .text(words)
            .distance_type("cosine")
            .rerank(RRFReranker())
        )
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(k).to_list()
        # Reciprocal-rank fusion decides the order (unchanged); the reported score is the cosine similarity.
        return [
            row_to_result(r, float(r.get("_relevance_score", 0.0)), i + 1, "hybrid", self._similarity(r, qvec))
            for i, r in enumerate(rows)
        ]

    def _ensure_fts(self) -> None:
        """Create the BM25 index if the index was built without one (e.g. by an older build)."""
        try:
            names = {i.name for i in self.index.chunks.list_indices()}
        except Exception:  # pragma: no cover - API differences between lancedb versions
            names = set()
        if not any(FTS_COLUMN in n for n in names):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.index.refresh_fts()


__all__ = ["MODES", "SearchFilters", "SearchResult", "Searcher", "CHUNK_COLUMNS", "to_epoch", "row_to_result", "cosine_similarity"]
