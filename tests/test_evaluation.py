from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acadmem.evaluation import (
    BenchQuery,
    ExpectedItem,
    aggregate,
    evaluate_query,
    format_report,
    item_matches,
    load_benchmark,
    prepare_overlay,
    run_benchmark,
    satisfies_filters,
    verify_labels,
)
from acadmem.expand import ContextExpander
from acadmem.search import SearchFilters, Searcher

PDF = "3B1/Handouts/Week 5 - Probability.pdf"


# ---------------------------------------------------------------- matching
def test_item_matching_rules():
    item = ExpectedItem(source=PDF, pages=[3], heading="independence")
    assert item_matches(item, PDF, 3, 3, ["Probability", "Independence"])
    assert item_matches(item, PDF, 2, 4, ["Independence"])  # page range overlaps
    assert not item_matches(item, PDF, 4, 5, ["Independence"])  # page outside
    assert not item_matches(item, PDF, 3, 3, ["Conditional probability"])  # heading missing
    assert not item_matches(item, "other.pdf", 3, 3, ["Independence"])  # wrong source
    assert item_matches(item, PDF, 9, 9, [], source_only=True)
    assert not item_matches(ExpectedItem(source=PDF, pages=[1]), PDF, None, None, [])  # page label but no page info
    assert item_matches(ExpectedItem(source="a.md"), "a.md", None, None, [])  # source-only label


def test_filter_recheck_is_independent_of_sql(searcher_hits):
    hits = searcher_hits("probability", {"module": "3B1", "file_type": "pdf"})
    assert hits
    f = SearchFilters(module="3B1", file_type="pdf")
    assert all(satisfies_filters(f, h) for h in hits)
    assert not satisfies_filters(SearchFilters(module="3F1"), hits[0])
    assert not satisfies_filters(SearchFilters(file_type="md"), hits[0])
    assert not satisfies_filters(SearchFilters(path_prefix="3F1/"), hits[0])
    assert not satisfies_filters(SearchFilters(ai_generated=True), hits[0])
    assert not satisfies_filters(SearchFilters(tags=["nope"]), hits[0])
    assert not satisfies_filters(SearchFilters(modified_after="2999-01-01"), hits[0])
    assert satisfies_filters(SearchFilters(), hits[0])


@pytest.fixture
def searcher_hits(indexer, config, fake_embedder):
    indexer.run(full=True)
    searcher = Searcher(config, embedder=fake_embedder)
    return lambda q, filters=None, k=10: searcher.search(q, top_k=k, filters=filters)


# ---------------------------------------------------------------- loading
def write_bench(tmp_path: Path, queries: list[dict]) -> Path:
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "queries": queries}), encoding="utf-8")
    return path


def q(**kw):
    base = {"id": "a", "category": "pdf", "query": "x", "expected": [{"source": PDF}]}
    base.update(kw)
    return base


def test_load_benchmark_parses_all_fields(tmp_path):
    b = load_benchmark(write_bench(tmp_path, [q(filters={"module": "3B1"}, match="all", expand="neighbors:1", expected=[{"source": PDF, "pages": [3], "heading": "h", "evidence": "e"}])]))
    only = b.queries[0]
    assert only.filters == {"module": "3B1"} and only.match == "all" and only.expand == "neighbors:1"
    assert only.expected[0] == ExpectedItem(source=PDF, pages=[3], heading="h", evidence="e")


@pytest.mark.parametrize(
    "bad, message",
    [
        (q(category="vibes"), "category"),
        (q(match="most"), "match"),
        (q(expected=[]), "expected"),
        (q(query=""), "query"),
    ],
)
def test_load_benchmark_rejects_bad_entries(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message):
        load_benchmark(write_bench(tmp_path, [bad]))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark(write_bench(tmp_path, [q(), q()]))


def test_expect_empty_needs_no_expected(tmp_path):
    b = load_benchmark(write_bench(tmp_path, [q(expected=[], expect_empty=True)]))
    assert b.queries[0].expect_empty


def test_shipped_benchmark_is_well_formed():
    path = Path(__file__).resolve().parents[1] / "eval" / "benchmark.yaml"
    b = load_benchmark(path)
    cats = {x.category for x in b.queries}
    assert cats == {"conceptual", "paraphrase", "pdf", "markdown", "cross_document", "metadata", "context_expansion"}
    assert len(b.queries) >= 30
    assert all(x.expected or x.expect_empty for x in b.queries)
    assert all(x.expand for x in b.queries if x.category == "context_expansion")
    assert all(x.match == "all" for x in b.queries if x.category == "cross_document")


# ---------------------------------------------------------------- scoring
def run_one(config, fake_embedder, indexer, bench_query: BenchQuery, mode="hybrid", top_k=5):
    indexer.run(full=True)
    s = Searcher(config, embedder=fake_embedder)
    return evaluate_query(s, ContextExpander(s.index, config), bench_query, mode, top_k)


def test_recall_any_vs_all(config, fake_embedder, indexer):
    good = ExpectedItem(source="3F1/Notes/deep/er/Markov.md".replace("3F1", "3F1"))
    good = ExpectedItem(source="3F1/Notes/deep/er/Markov.md")
    absent = ExpectedItem(source="3F1/Notes/Nope.md")
    any_q = BenchQuery(id="a", category="markdown", query="memoryless markov chain", expected=[good, absent], match="any")
    all_q = BenchQuery(id="b", category="markdown", query="memoryless markov chain", expected=[good, absent], match="all")
    r_any = run_one(config, fake_embedder, indexer, any_q)
    r_all = run_one(config, fake_embedder, indexer, all_q)
    assert r_any["recall"] == 1.0
    assert r_all["recall"] == 0.5 and r_all["found"] == [True, False]
    assert r_any["rank"] == 1 and r_any["rr"] == 1.0
    assert r_any["retrieved"][0].startswith("* ")  # relevant hits are marked


def test_source_recall_is_more_lenient_than_page_recall(config, fake_embedder, indexer):
    wrong_page = ExpectedItem(source=PDF, pages=[99])
    bq = BenchQuery(id="a", category="pdf", query="independent events probability", expected=[wrong_page])
    r = run_one(config, fake_embedder, indexer, bq)
    assert r["recall"] == 0.0 and r["source_recall"] == 1.0 and r["rank"] is None and r["rr"] == 0.0


def test_filter_correctness_and_expect_empty(config, fake_embedder, indexer):
    ok = BenchQuery(id="a", category="metadata", query="probability", expected=[ExpectedItem(source=PDF)], filters={"module": "3B1"})
    r = run_one(config, fake_embedder, indexer, ok)
    assert r["filter_ok"] is True
    empty = BenchQuery(id="b", category="metadata", query="probability", expected=[], expect_empty=True, filters={"modified_after": "2999-01-01"})
    r = run_one(config, fake_embedder, indexer, empty)
    assert r["recall"] == 1.0 and r["retrieved"] == []
    leaky = BenchQuery(id="c", category="metadata", query="probability", expected=[], expect_empty=True)
    assert run_one(config, fake_embedder, indexer, leaky)["recall"] == 0.0


def test_expansion_lift_is_measured(config, fake_embedder, indexer):
    # The query matches the "Independence" section (pages 3-4); the label additionally needs the preceding
    # "Conditional probability" section (page 1), which is only reachable through context expansion.
    items = [ExpectedItem(source=PDF, pages=[3]), ExpectedItem(source=PDF, pages=[1])]
    base = dict(id="a", category="context_expansion", query="two events are independent when the occurrence of one does not change the probability", expected=items, match="all")
    plain = run_one(config, fake_embedder, indexer, BenchQuery(**base), top_k=1)
    expanded = run_one(config, fake_embedder, indexer, BenchQuery(**base, expand="neighbors:1"), top_k=1)
    assert plain["recall"] < 1.0
    assert expanded["recall"] == 1.0 and expanded["hit_recall"] == plain["hit_recall"] and expanded["context_size"] >= 1


def test_aggregate_handles_empty_and_mixed():
    assert aggregate([])["recall"] is None
    rows = [
        {"recall": 1.0, "hit_recall": 1.0, "source_recall": 1.0, "rr": 1.0, "filter_ok": True, "latency_ms": 10, "distinct_sources": 2},
        {"recall": 0.0, "hit_recall": 0.0, "source_recall": 1.0, "rr": 0.0, "filter_ok": None, "latency_ms": 30, "distinct_sources": 1},
    ]
    agg = aggregate(rows)
    assert agg["recall"] == 0.5 and agg["source_recall"] == 1.0 and agg["mrr"] == 0.5
    assert agg["filter_correctness"] == 1.0 and agg["avg_latency_ms"] == 20


def test_run_benchmark_and_report(config, fake_embedder, indexer, tmp_path):
    indexer.run(full=True)
    bench = load_benchmark(
        write_bench(
            tmp_path,
            [
                q(id="md", category="markdown", query="memoryless markov chain", expected=[{"source": "3F1/Notes/deep/er/Markov.md"}]),
                q(id="pdf", category="pdf", query="independent events", expected=[{"source": PDF, "pages": [3]}]),
                q(id="needs-fixture", category="markdown", query="x", expected=[{"source": "nowhere.md"}]),
            ],
        )
    )
    report = run_benchmark(Searcher(config, embedder=fake_embedder), bench, modes=["vector", "hybrid"], top_k=3)
    assert report["n_queries"] == 2 and [s["id"] for s in report["skipped"]] == ["needs-fixture"]
    assert set(report["modes"]) == {"vector", "hybrid"}
    for mode in report["modes"].values():
        assert mode["overall"]["recall"] == 1.0 and set(mode["by_category"]) == {"markdown", "pdf"}
    text = format_report(report, details=True)
    assert "Recall@3" in text and "vector" in text and "hybrid" in text and "needs-fixture" in text
    only_pdf = run_benchmark(Searcher(config, embedder=fake_embedder), bench, modes=["vector"], categories=["pdf"])
    assert only_pdf["n_queries"] == 1


# ---------------------------------------------------------------- labels
def test_verify_labels_catches_bad_labels(vault, tmp_path):
    good = q(expected=[{"source": PDF, "pages": [3], "evidence": "independent"}])
    bad_page = q(id="p", expected=[{"source": PDF, "pages": [99]}])
    bad_ev = q(id="e", expected=[{"source": PDF, "pages": [1], "evidence": "this sentence is not in the pdf"}])
    gone = q(id="g", expected=[{"source": "3B1/Missing.pdf"}])
    md_ok = q(id="m", expected=[{"source": "3F1/Notes/deep/er/Markov.md", "heading": "Stationary distribution", "evidence": "memoryless"}])
    md_bad = q(id="mb", expected=[{"source": "3F1/Notes/deep/er/Markov.md", "heading": "No such heading"}])
    problems = verify_labels(load_benchmark(write_bench(tmp_path, [good, bad_page, bad_ev, gone, md_ok, md_bad])), vault)
    joined = "\n".join(problems)
    assert "p: " in joined and "99" in joined
    assert "e: evidence" in joined
    assert "g: source not found" in joined
    assert "mb: heading" in joined
    assert not any(line.startswith(("a:", "m:")) for line in problems)


# ---------------------------------------------------------------- overlay
def test_overlay_adds_fixtures_without_touching_the_real_vault_or_index(config, indexer, fake_embedder, tmp_path):
    indexer.run(full=True)
    fixtures = tmp_path / "fx"
    (fixtures / "3B1" / "Notes").mkdir(parents=True)
    (fixtures / "3B1" / "Notes" / "fixture.md").write_text("# Fixture\n\nA unique fixture note about zygomorphic petals.\n", encoding="utf-8")
    before_files = sorted(p.name for p in config.vault_path.rglob("*") if p.is_file())
    from acadmem.index import VectorIndex

    docs_before = VectorIndex(config).count_documents()
    overlay, work = prepare_overlay(config, fixtures, tmp_path / "work", embedder=fake_embedder)
    assert overlay.vault_path != config.vault_path and overlay.index_dir != config.index_dir
    idx = VectorIndex(overlay)
    assert idx.get_document(source_path="3B1/Notes/fixture.md") is not None
    assert idx.count_documents() == docs_before + 1
    assert idx.get_document(source_path=PDF) is not None  # the pre-existing index content came along
    assert sorted(p.name for p in config.vault_path.rglob("*") if p.is_file()) == before_files
    assert VectorIndex(config).get_document(source_path="3B1/Notes/fixture.md") is None
    assert VectorIndex(config).count_documents() == docs_before
