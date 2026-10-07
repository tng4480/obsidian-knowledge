"""Benchmark runner for the academic-memory retrieval system.

The benchmark (``eval/benchmark.yaml``) is a list of realistic study queries, each with the sources / pages /
headings that should be retrieved. For every search mode we report:

* **Recall@k**: per query, the fraction of expected items found in the top-k (``match: all``) or 1/0 for
  "found at least one" (``match: any``). Averaged over queries.
* **Source recall@k**: the same but only the source file has to match (separates "wrong document" from
  "right document, wrong page/section").
* **MRR**: reciprocal rank of the first relevant hit.
* **Filter correctness**: share of returned hits that actually satisfy the query's metadata filters.
* **Expansion lift** (context-expansion queries): recall of the hits alone vs after context expansion.
* Latency, and per-category breakdowns, plus the queries that failed and what was retrieved instead.
"""

from __future__ import annotations

import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import Config
from .expand import ContextExpander, ContextPiece
from .index import VectorIndex
from .search import SearchFilters, SearchResult, Searcher, to_epoch

CATEGORIES = (
    "conceptual",
    "paraphrase",
    "pdf",
    "markdown",
    "cross_document",
    "metadata",
    "context_expansion",
)


# --------------------------------------------------------------------------- benchmark model
@dataclass
class ExpectedItem:
    source: str
    pages: list[int] | None = None
    heading: str | None = None
    evidence: str | None = None


@dataclass
class BenchQuery:
    id: str
    category: str
    query: str
    expected: list[ExpectedItem]
    filters: dict[str, Any] = field(default_factory=dict)
    match: str = "any"
    expand: str | None = None
    expect_empty: bool = False
    notes: str = ""


@dataclass
class Benchmark:
    queries: list[BenchQuery]
    version: int = 1
    path: Path | None = None


def load_benchmark(path: str | Path) -> Benchmark:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(f"{path}: expected a mapping with a 'queries' list")
    queries: list[BenchQuery] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["queries"]):
        where = f"{path.name} query #{i + 1} ({raw.get('id', '?')})"
        for key in ("id", "category", "query"):
            if not raw.get(key):
                raise ValueError(f"{where}: missing '{key}'")
        if raw["id"] in seen:
            raise ValueError(f"{where}: duplicate id")
        seen.add(raw["id"])
        if raw["category"] not in CATEGORIES:
            raise ValueError(f"{where}: category must be one of {CATEGORIES}")
        match = raw.get("match", "any")
        if match not in ("any", "all"):
            raise ValueError(f"{where}: match must be 'any' or 'all'")
        expected = [
            ExpectedItem(
                source=str(e["source"]).replace("\\", "/"),
                pages=[int(p) for p in e["pages"]] if e.get("pages") else None,
                heading=e.get("heading"),
                evidence=e.get("evidence"),
            )
            for e in raw.get("expected") or []
        ]
        if not expected and not raw.get("expect_empty"):
            raise ValueError(f"{where}: needs 'expected' (or expect_empty: true)")
        queries.append(
            BenchQuery(
                id=raw["id"],
                category=raw["category"],
                query=raw["query"],
                expected=expected,
                filters=dict(raw.get("filters") or {}),
                match=match,
                expand=raw.get("expand"),
                expect_empty=bool(raw.get("expect_empty", False)),
                notes=raw.get("notes", ""),
            )
        )
    return Benchmark(queries=queries, version=int(data.get("version", 1)), path=path)


# --------------------------------------------------------------------------- matching
def item_matches(item: ExpectedItem, source_path: str, page_start: int | None, page_end: int | None, heading_path: list[str], heading: str = "", *, source_only: bool = False) -> bool:
    if source_path != item.source:
        return False
    if source_only:
        return True
    if item.pages:
        lo = page_start if page_start is not None else None
        hi = page_end if page_end is not None else lo
        if lo is None:
            return False
        if not any(lo <= p <= (hi if hi is not None else lo) for p in item.pages):
            return False
    if item.heading:
        haystack = " > ".join([*heading_path, heading]).lower()
        if item.heading.lower() not in haystack:
            return False
    return True


def _hit_matches(item: ExpectedItem, h: SearchResult | ContextPiece, source_only: bool = False) -> bool:
    return item_matches(item, h.source_path, h.page_start, h.page_end, h.heading_path, h.heading, source_only=source_only)


def satisfies_filters(filters: SearchFilters, hit: SearchResult) -> bool:
    """Python-side re-check of the metadata filters on a returned hit (independent of the SQL path)."""

    def listed(v: Any) -> list[str]:
        return [] if v is None else ([v] if isinstance(v, str) else list(v))

    if (mods := listed(filters.module)) and (hit.module or "").lower() not in {m.lower() for m in mods}:
        return False
    if (dts := listed(filters.document_type)) and (hit.document_type or "").lower() not in {d.lower() for d in dts}:
        return False
    if (fts := listed(filters.file_type)) and hit.file_type.lower() not in {f.lower().lstrip(".") for f in fts}:
        return False
    if (paths := listed(filters.source_path)) and hit.source_path not in {p.replace("\\", "/") for p in paths}:
        return False
    if filters.path_prefix and not hit.source_path.lower().startswith(filters.path_prefix.replace("\\", "/").lstrip("/").lower()):
        return False
    if (tags := listed(filters.tags)) and not ({t.lstrip("#").lower() for t in tags} & set(hit.tags)):
        return False
    after, before = to_epoch(filters.modified_after), to_epoch(filters.modified_before)
    if after is not None and (hit.modification_time or 0) < after:
        return False
    if before is not None and (hit.modification_time or 0) >= before:
        return False
    if filters.ai_generated is not None and hit.ai_generated != filters.ai_generated:
        return False
    return True


# --------------------------------------------------------------------------- scoring one query
def _recall(q: BenchQuery, candidates: list[Any], source_only: bool = False) -> tuple[float, list[bool]]:
    found = [any(_hit_matches(item, c, source_only) for c in candidates) for item in q.expected]
    if not found:
        return 1.0, []
    return (sum(found) / len(found) if q.match == "all" else (1.0 if any(found) else 0.0)), found


def evaluate_query(searcher: Searcher, expander: ContextExpander, q: BenchQuery, mode: str, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    hits = searcher.search(q.query, top_k=top_k, filters=q.filters or None, mode=mode)
    latency_ms = (time.perf_counter() - started) * 1000

    filters = SearchFilters.from_mapping(q.filters)
    filter_ok = None if not q.filters else (all(satisfies_filters(filters, h) for h in hits) if hits else True)

    if q.expect_empty:
        recall = 1.0 if not hits else 0.0
        return {
            "id": q.id, "category": q.category, "recall": recall, "hit_recall": recall, "source_recall": recall,
            "rank": None, "rr": 1.0 if recall else 0.0, "filter_ok": filter_ok, "latency_ms": latency_ms,
            "retrieved": [h.location for h in hits], "found": [], "distinct_sources": len({h.source_path for h in hits}),
        }

    hit_recall, found = _recall(q, hits)
    source_recall, _ = _recall(q, hits, source_only=True)
    rank = next((h.rank for h in hits if any(_hit_matches(item, h) for item in q.expected)), None)

    recall = hit_recall
    expanded_pieces: list[ContextPiece] = []
    if q.expand:
        expanded_pieces = expander.expand_many(hits, q.expand)
        recall, found = _recall(q, expanded_pieces)
        source_recall, _ = _recall(q, expanded_pieces, source_only=True)

    return {
        "id": q.id,
        "category": q.category,
        "recall": recall,
        "hit_recall": hit_recall,
        "source_recall": source_recall,
        "rank": rank,
        "rr": 1.0 / rank if rank else 0.0,
        "filter_ok": filter_ok,
        "latency_ms": latency_ms,
        "found": found,
        "retrieved": [("* " if any(_hit_matches(item, h) for item in q.expected) else "  ") + h.location for h in hits],
        "context_size": len(expanded_pieces),
        "distinct_sources": len({h.source_path for h in hits}),
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if r.get("recall") is not None]
    with_rank = [r for r in scored if "rr" in r]
    lat = [r["latency_ms"] for r in results]
    return {
        "n": len(results),
        "recall": _mean(r["recall"] for r in scored),
        "hit_recall": _mean(r["hit_recall"] for r in scored),
        "source_recall": _mean(r["source_recall"] for r in scored),
        "mrr": _mean(r["rr"] for r in with_rank),
        "filter_correctness": _mean(1.0 if r["filter_ok"] else 0.0 for r in results if r["filter_ok"] is not None),
        "avg_latency_ms": _mean(lat),
        "p95_latency_ms": (statistics.quantiles(lat, n=20)[-1] if len(lat) >= 2 else (lat[0] if lat else None)),
        "avg_distinct_sources": _mean(r["distinct_sources"] for r in results),
    }


# --------------------------------------------------------------------------- running
def run_benchmark(
    searcher: Searcher,
    benchmark: Benchmark,
    modes: Iterable[str] = ("vector", "fts", "hybrid"),
    top_k: int = 5,
    categories: Iterable[str] | None = None,
) -> dict[str, Any]:
    index = searcher.index
    available = {d["source_path"] for d in index.list_documents()}
    expander = ContextExpander(index, searcher.config)

    selected, skipped = [], []
    for q in benchmark.queries:
        if categories and q.category not in set(categories):
            continue
        missing = sorted({e.source for e in q.expected} - available)
        (skipped if missing else selected).append((q, missing))
    queries = [q for q, _ in selected]

    report: dict[str, Any] = {
        "top_k": top_k,
        "n_queries": len(queries),
        "skipped": [{"id": q.id, "missing_sources": missing} for q, missing in skipped],
        "modes": {},
    }
    for mode in modes:
        results = [evaluate_query(searcher, expander, q, mode, top_k) for q in queries]
        by_cat = {c: aggregate([r for r in results if r["category"] == c]) for c in CATEGORIES if any(r["category"] == c for r in results)}
        report["modes"][mode] = {"overall": aggregate(results), "by_category": by_cat, "queries": results}
    return report


# --------------------------------------------------------------------------- overlay (fixture notes)
def copy_vault_with_fixtures(vault_path: Path, fixtures_dir: Path, target: Path) -> Path:
    """Copy the vault (minus dot-directories) to ``target`` and lay the fixture notes over it."""
    if target.exists():
        shutil.rmtree(target)

    def ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if n.startswith(".")]

    shutil.copytree(vault_path, target, ignore=ignore, copy_function=shutil.copy2)
    if fixtures_dir.is_dir():
        shutil.copytree(fixtures_dir, target, dirs_exist_ok=True, copy_function=shutil.copy2)
    return target


def prepare_overlay(config: Config, fixtures_dir: Path, workdir: Path | None = None, embedder: Any = None) -> tuple[Config, Path]:
    """Copy the vault + index to a scratch dir, add the fixture notes, and bring the copied index up to date.

    The real vault and index are never touched. Because document ids/hashes depend only on relative path and
    content, the incremental indexer only embeds the *new* fixture notes.
    """
    from .indexer import Indexer

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="acadmem-eval-"))
    vault_copy, index_copy = work / "vault", work / "index"
    if index_copy.exists():
        shutil.rmtree(index_copy)
    copy_vault_with_fixtures(config.vault_path, fixtures_dir, vault_copy)
    if config.index_dir.exists():
        shutil.copytree(config.index_dir, index_copy, ignore=shutil.ignore_patterns("*.lock"))
    overlay = config.with_overrides(vault_path=vault_copy, index_dir=index_copy)
    Indexer(overlay, embedder=embedder).run()
    return overlay, work


# --------------------------------------------------------------------------- label verification
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def verify_labels(benchmark: Benchmark, vault_path: Path, fixtures_dir: Path | None = None) -> list[str]:
    """Check every label against the actual source text. Returns a list of problems (empty = all good)."""
    import pymupdf

    problems: list[str] = []
    page_text: dict[str, list[str]] = {}

    def pdf_pages(path: Path) -> list[str]:
        key = str(path)
        if key not in page_text:
            with pymupdf.open(path) as doc:
                page_text[key] = [_norm(p.get_text()) for p in doc]
        return page_text[key]

    for q in benchmark.queries:
        for item in q.expected:
            path = vault_path / item.source
            if not path.exists() and fixtures_dir is not None:
                path = fixtures_dir / item.source
            if not path.exists():
                problems.append(f"{q.id}: source not found: {item.source}")
                continue
            if path.suffix.lower() == ".pdf":
                pages = pdf_pages(path)
                for p in item.pages or []:
                    if not 1 <= p <= len(pages):
                        problems.append(f"{q.id}: {item.source} has {len(pages)} pages, label says page {p}")
                if item.evidence:
                    scope = [pages[p - 1] for p in item.pages if 1 <= p <= len(pages)] if item.pages else pages
                    if not any(_norm(item.evidence) in s for s in scope):
                        where = f"pages {item.pages}" if item.pages else "any page"
                        problems.append(f"{q.id}: evidence {item.evidence!r} not found on {where} of {item.source}")
            else:
                text = _norm(path.read_text(encoding="utf-8"))
                if item.pages:
                    problems.append(f"{q.id}: pages given for non-PDF {item.source}")
                if item.evidence and _norm(item.evidence) not in text:
                    problems.append(f"{q.id}: evidence {item.evidence!r} not found in {item.source}")
                if item.heading and not re.search(rf"^#+\s*{re.escape(item.heading)}\s*$", path.read_text(encoding="utf-8"), re.I | re.M):
                    problems.append(f"{q.id}: heading {item.heading!r} not found in {item.source}")
    return problems


# --------------------------------------------------------------------------- reporting
def _pct(v: float | None) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}"


def format_report(report: dict[str, Any], details: bool = False) -> str:
    k = report["top_k"]
    modes = list(report["modes"])
    lines: list[str] = [f"Benchmark: {report['n_queries']} queries, top_k={k}"]
    if report["skipped"]:
        lines.append(f"Skipped {len(report['skipped'])} queries whose expected sources are not indexed: " + ", ".join(s["id"] for s in report["skipped"]))

    def table(title: str, rows: list[tuple[str, dict[str, Any]]], fields: list[tuple[str, str]]) -> None:
        lines.append("")
        lines.append(title)
        head = f"  {'':<22}" + "".join(f"{m:>10}" for m in modes)
        lines.append(head)
        for label, getter in fields:
            vals = []
            for mode in modes:
                data = report["modes"][mode]
                for key in getter.split("."):
                    data = (data or {}).get(key)
                vals.append(data)
            lines.append(f"  {label:<22}" + "".join(f"{_pct(v) if label.endswith('%') else (f'{v:10.1f}' if v is not None else '       n/a'):>10}" for v in vals))

    overall_fields = [
        (f"Recall@{k} %", "overall.recall"),
        (f"Source recall@{k} %", "overall.source_recall"),
        ("MRR x100 %", "overall.mrr"),
        ("Filter correct %", "overall.filter_correctness"),
        ("avg latency ms", "overall.avg_latency_ms"),
        ("p95 latency ms", "overall.p95_latency_ms"),
    ]
    table("Overall", [], overall_fields)
    lines.append("")
    lines.append(f"Recall@{k} % by category (n)")
    lines.append(f"  {'':<22}" + "".join(f"{m:>10}" for m in modes))
    cats = [c for c in CATEGORIES if any(c in report["modes"][m]["by_category"] for m in modes)]
    for c in cats:
        n = next(report["modes"][m]["by_category"][c]["n"] for m in modes if c in report["modes"][m]["by_category"])
        lines.append(f"  {f'{c} ({n})':<22}" + "".join(f"{_pct(report['modes'][m]['by_category'].get(c, {}).get('recall')):>10}" for m in modes))
    if "context_expansion" in cats:
        lines.append("")
        lines.append("Context expansion: recall of hits alone -> with expansion")
        for m in modes:
            agg = report["modes"][m]["by_category"].get("context_expansion")
            if agg:
                lines.append(f"  {m:<10} {_pct(agg['hit_recall'])} -> {_pct(agg['recall'])}")

    best = max(modes, key=lambda m: (report["modes"][m]["overall"]["recall"] or 0, report["modes"][m]["overall"]["mrr"] or 0))
    misses = [r for r in report["modes"][best]["queries"] if r["recall"] < 1.0]
    lines.append("")
    lines.append(f"Queries not fully satisfied by '{best}' ({len(misses)}):")
    for r in misses:
        lines.append(f"  - {r['id']} [{r['category']}] recall={r['recall']:.2f} found={r['found']}")
        for loc in r["retrieved"]:
            lines.append(f"        {loc}")
    if details:
        lines.append("")
        lines.append("Per query: rank of first relevant hit (- = not in top-k)")
        lines.append(f"  {'id':<34}" + "".join(f"{m:>10}" for m in modes))
        for i, r in enumerate(report["modes"][modes[0]]["queries"]):
            row = [report["modes"][m]["queries"][i] for m in modes]
            lines.append(f"  {r['id']:<34}" + "".join(f"{(str(x['rank']) if x['rank'] else '-'):>10}" for x in row))
    return "\n".join(lines)


__all__ = [
    "Benchmark", "BenchQuery", "ExpectedItem", "CATEGORIES",
    "load_benchmark", "run_benchmark", "format_report", "prepare_overlay", "verify_labels",
    "evaluate_query", "item_matches", "satisfies_filters", "aggregate",
]
