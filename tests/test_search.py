from __future__ import annotations

import math
import os
import time

import pytest

from acadmem.index import FTS_COLUMN
from acadmem.search import MODES, SearchFilters, Searcher, cosine_similarity, to_epoch


@pytest.fixture
def built(indexer):
    indexer.run(full=True)
    return indexer


@pytest.fixture
def searcher(built, config, fake_embedder):
    return Searcher(config, embedder=fake_embedder)


@pytest.mark.parametrize("mode", MODES)
def test_each_mode_finds_the_relevant_chunk(searcher, mode):
    hits = searcher.search("memoryless property of a markov chain", top_k=3, mode=mode)
    assert hits and hits[0].source_path.endswith("Markov.md")
    assert hits[0].mode == mode
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_results_carry_full_provenance(searcher):
    top = searcher.search("independent events probability of the other", top_k=3)[0]
    assert top.source_path == "3B1/Handouts/Week 5 - Probability.pdf"
    assert top.file_type == "pdf"
    assert top.page_start and top.page_start <= top.page_end
    assert top.heading_path and top.text and top.chunk_id.startswith(top.document_id)
    assert "Week 5 - Probability.pdf p." in top.location
    d = top.to_dict()
    assert d["location"] == top.location and d["module"] == "3B1"


def _stored_cosine(searcher, query: str, chunk_id: str) -> float:
    """Reference similarity computed independently of the search code path (query embedding vs stored vector)."""
    row = searcher.index.scan(f"chunk_id = '{chunk_id}'", limit=1, columns=["chunk_id", "vector"])[0]
    q = searcher.embedder.embed_query(query)
    dot = sum(a * b for a, b in zip(q, row["vector"]))
    return dot / (math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in row["vector"])))


QUERIES = ["stationary distribution pi P = pi", "probability", "queueing", "independent events probability of the other"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("query", QUERIES)
def test_score_is_cosine_similarity_in_every_mode(searcher, mode, query):
    """`score` means the same thing whatever the mode: cosine(query, chunk), including for keyword-only hits."""
    hits = searcher.search(query, top_k=5, mode=mode)
    assert hits
    for h in hits:
        assert -1.0001 <= h.score <= 1.0001
        assert h.score == pytest.approx(_stored_cosine(searcher, query, h.chunk_id), abs=1e-5)


def test_same_chunk_has_the_same_score_in_every_mode(searcher):
    by_mode = {m: {h.chunk_id: h.score for h in searcher.search("markov chain stationary distribution", top_k=20, mode=m)} for m in MODES}
    shared = set.intersection(*(set(d) for d in by_mode.values()))
    assert shared
    for chunk_id in shared:
        assert by_mode["vector"][chunk_id] == pytest.approx(by_mode["fts"][chunk_id], abs=1e-6)
        assert by_mode["vector"][chunk_id] == pytest.approx(by_mode["hybrid"][chunk_id], abs=1e-6)


def test_keyword_only_hits_still_get_a_similarity(searcher):
    # "queueing" is a rare word: FTS finds it, the hash embedder barely ranks it. Its score must still be a cosine,
    # not a BM25 value (BM25 is unbounded and >1 here) and not the hybrid fusion value (<= 0.033).
    fts = searcher.search("queueing", top_k=3, mode="fts")
    assert fts[0].source_path == "loose.md"
    assert fts[0].rank_score > 1.0 >= fts[0].score  # BM25 lives in rank_score; score is the cosine
    assert all(-1.0001 <= h.score <= 1.0001 for h in fts)
    vector_top = {h.chunk_id for h in searcher.search("queueing", top_k=1, mode="vector")}
    keyword_only = [h for h in searcher.search("queueing", top_k=10, mode="hybrid") if h.chunk_id not in vector_top]
    assert keyword_only
    for h in keyword_only:
        assert h.score == pytest.approx(_stored_cosine(searcher, "queueing", h.chunk_id), abs=1e-5)


def test_vector_mode_ranking_value_equals_its_score(searcher):
    hits = searcher.search("stationary distribution pi P = pi", top_k=5, mode="vector")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    for h in hits:
        assert h.rank_score == pytest.approx(h.score, abs=1e-5)


def test_fts_order_is_bm25_and_hybrid_order_is_unchanged_rrf(searcher):
    """Reporting cosine as `score` must not change who comes first: ranking still follows BM25 / RRF."""
    from lancedb.rerankers import RRFReranker

    for query in ("markov chain stationary distribution", "probability of the other event", "queueing"):
        words = " ".join(query.split())
        qvec = searcher.embedder.embed_query(query)

        fts = searcher.search(query, top_k=6, mode="fts")
        expected_fts = searcher.index.chunks.search(words, query_type="fts", fts_columns=FTS_COLUMN).limit(6).to_list()
        assert [h.chunk_id for h in fts] == [r["chunk_id"] for r in expected_fts]
        assert [h.rank_score for h in fts] == sorted((h.rank_score for h in fts), reverse=True)

        hybrid = searcher.search(query, top_k=6, mode="hybrid")
        expected = (
            searcher.index.chunks.search(query_type="hybrid", fts_columns=FTS_COLUMN)
            .vector(qvec).text(words).distance_type("cosine").rerank(RRFReranker()).limit(6).to_list()
        )
        assert [h.chunk_id for h in hybrid] == [r["chunk_id"] for r in expected]
        assert [h.rank_score for h in hybrid] == pytest.approx([r["_relevance_score"] for r in expected])
        assert all(0 < h.rank_score <= 2 / 60 + 1e-9 for h in hybrid)
        assert [h.rank for h in hybrid] == list(range(1, len(hybrid) + 1))


def test_hybrid_score_is_not_the_fusion_value(searcher):
    hits = searcher.search("independent events probability of the other", top_k=5, mode="hybrid")
    assert hits[0].score > 0.1  # a cosine of a clearly matching chunk, not ~0.016-0.033
    d = hits[0].to_dict()
    assert d["score"] == hits[0].score and d["rank_score"] == hits[0].rank_score and d["rank_score"] <= 2 / 60 + 1e-9


def test_cosine_similarity_helper():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([2, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_top_k_is_respected(searcher):
    assert len(searcher.search("probability", top_k=2)) == 2
    assert len(searcher.search("probability", top_k=50)) <= 50


def test_default_top_k_and_mode_come_from_config(searcher, config):
    assert len(searcher.search("probability")) <= config.default_top_k
    assert searcher.search("probability")[0].mode == config.search_mode


@pytest.mark.parametrize("mode", MODES)
def test_filters_are_applied_before_ranking(searcher, mode):
    # The best global match for this query lives in 3F1; with the filter we must still get 3B1 results,
    # i.e. the filter is a prefilter, not a post-filter that leaves us with nothing.
    unfiltered = searcher.search("markov chain stationary distribution", top_k=3, mode=mode)
    assert unfiltered[0].module == "3F1"
    filtered = searcher.search("markov chain stationary distribution", top_k=3, mode=mode, filters={"module": "3B1"})
    assert filtered and {h.module for h in filtered} == {"3B1"}
    # filtered hits still carry a real cosine similarity
    for h in filtered:
        assert h.score == pytest.approx(_stored_cosine(searcher, "markov chain stationary distribution", h.chunk_id), abs=1e-5)


def test_metadata_filters(searcher):
    q = "probability"
    assert {h.file_type for h in searcher.search(q, top_k=10, filters={"file_type": "pdf"})} == {"pdf"}
    assert {h.file_type for h in searcher.search(q, top_k=10, filters={"file_type": ".md"})} == {"md"}
    assert {h.document_type for h in searcher.search(q, top_k=10, filters={"document_type": "questions"})} == {"Questions"}
    assert {h.module for h in searcher.search(q, top_k=10, filters={"module": ["3b1", "3f1"]})} <= {"3B1", "3F1"}
    only = searcher.search(q, top_k=10, filters={"source_path": "3B1/Questions/Bayes.md"})
    assert only and {h.source_path for h in only} == {"3B1/Questions/Bayes.md"}


def test_path_prefix_filter(searcher):
    hits = searcher.search("probability", top_k=10, filters={"path_prefix": "3b1/handouts"})
    assert hits and all(h.source_path.startswith("3B1/Handouts") for h in hits)
    hits = searcher.search("probability", top_k=10, filters={"path_prefix": "3F1\\Notes"})  # Windows separator
    assert hits and all(h.source_path.startswith("3F1/Notes") for h in hits)
    assert searcher.search("probability", filters={"path_prefix": "nope/"}) == []


def test_tag_filter_is_case_insensitive_and_accepts_hash(searcher):
    for tag in ("probability", "#Probability", ["PROBABILITY"]):
        hits = searcher.search("belief", top_k=10, filters={"tags": tag})
        assert hits and {h.source_path for h in hits} == {"3B1/Questions/Bayes.md"}
    assert searcher.search("belief", filters={"tags": "unknown-tag"}) == []


def test_date_filters(searcher, vault):
    bayes = vault / "3B1" / "Questions" / "Bayes.md"
    past = time.time() - 10 * 86400
    os.utime(bayes, (past, past))
    # (Index not refreshed: filter on the indexed mtime, which is "now" for everything.)
    now = time.time()
    assert searcher.search("probability", top_k=5, filters={"modified_after": now - 3600})
    assert searcher.search("probability", filters={"modified_after": now + 3600}) == []
    assert searcher.search("probability", filters={"modified_before": now - 3600}) == []
    assert searcher.search("probability", filters={"modified_before": "2999-01-01"})
    assert searcher.search("probability", filters={"modified_after": "2999-01-01"}) == []


def test_ai_generated_filter(searcher, indexer, vault):
    (vault / "3B1" / "Notes" / "ai-summary.md").write_text(
        "---\nai_generated: true\n---\n# Summary\n\nA machine written summary of Bayes theorem and priors.\n",
        encoding="utf-8",
    )
    indexer.run()
    ai = searcher.search("bayes theorem", top_k=10, filters={"ai_generated": True})
    assert ai and {h.source_path for h in ai} == {"3B1/Notes/ai-summary.md"} and all(h.ai_generated for h in ai)
    human = searcher.search("bayes theorem", top_k=10, filters={"ai_generated": False})
    assert human and not any(h.ai_generated for h in human)
    assert "3B1/Notes/ai-summary.md" in {h.source_path for h in searcher.search("bayes theorem", top_k=10)}


def test_combined_filters_and(searcher):
    hits = searcher.search("probability", top_k=10, filters={"module": "3B1", "file_type": "md", "tags": "probability"})
    assert hits and {h.source_path for h in hits} == {"3B1/Questions/Bayes.md"}
    assert searcher.search("probability", filters={"module": "3F1", "file_type": "pdf"}) == []


def test_filter_values_are_quoted_safely(searcher):
    assert searcher.search("probability", filters={"module": "3B1' OR 1=1 --"}) == []
    assert searcher.search("probability", filters={"path_prefix": "a%_b'"}) == []


def test_filters_object_and_mapping_equivalent():
    f = SearchFilters(module="3B1", ai_generated=False)
    assert SearchFilters.from_mapping({"module": "3B1", "ai_generated": False}).to_sql() == f.to_sql()
    assert SearchFilters.from_mapping({"module": None, "tags": []}).is_empty()
    assert SearchFilters.from_mapping(None).to_sql() is None


def test_invalid_input_is_rejected_clearly(searcher):
    with pytest.raises(ValueError, match="Unknown filter"):
        searcher.search("x", filters={"modul": "3B1"})
    with pytest.raises(ValueError, match="mode"):
        searcher.search("x", mode="magic")
    with pytest.raises(ValueError, match="empty"):
        searcher.search("   ")
    with pytest.raises(ValueError, match="top_k"):
        searcher.search("x", top_k=0)
    with pytest.raises(ValueError, match="date"):
        to_epoch("last tuesday")


def test_to_epoch_accepts_several_forms():
    assert to_epoch(None) is None and to_epoch(12.5) == 12.5
    assert to_epoch("1970-01-02") == 86400.0
    assert to_epoch("1970-01-01T01:00:00+00:00") == 3600.0


@pytest.mark.parametrize("mode", ["fts", "hybrid"])
def test_awkward_query_syntax_does_not_break_text_search(searcher, mode):
    assert searcher.search('P(A|B) = "P(A and B)" / P(B) AND OR NOT *', top_k=3, mode=mode)
    assert searcher.search("???", top_k=3, mode=mode) is not None  # symbols only: falls back, never raises


def test_fts_finds_exact_term_vector_could_miss(searcher):
    hits = searcher.search("queueing", top_k=3, mode="fts")
    assert hits and hits[0].source_path == "loose.md"


@pytest.mark.parametrize("mode", ["fts", "hybrid"])
def test_text_search_matches_headings_not_repeated_in_the_body(searcher, indexer, vault, mode):
    # Real notes put the topic in the heading and never repeat it; lexical search must still find them.
    (vault / "3B1" / "Notes" / "checklist.md").write_text(
        "# Checklist\n\n## Before using the quokka theorem\n\nLook at the poles first. Then check the limit.\n",
        encoding="utf-8",
    )
    indexer.run()
    hits = searcher.search("quokka theorem", top_k=3, mode=mode)
    assert hits and hits[0].source_path == "3B1/Notes/checklist.md"
    assert "quokka" not in hits[0].text.lower()  # the returned text stays verbatim (no injected breadcrumb)


def test_search_reflects_incremental_updates(searcher, indexer, vault):
    assert not [h for h in searcher.search("zygomorphic", top_k=3, mode="fts") if "zygomorphic" in h.text.lower()]
    (vault / "loose.md").write_text("# Loose\n\nA note about zygomorphic flowers.\n", encoding="utf-8")
    indexer.run()
    for mode in MODES:
        hits = searcher.search("zygomorphic flowers", top_k=1, mode=mode)
        assert hits[0].source_path == "loose.md", mode
    (vault / "loose.md").unlink()
    indexer.run()
    assert all(h.source_path != "loose.md" for h in searcher.search("zygomorphic flowers", top_k=5, mode="hybrid"))


def test_empty_index_returns_nothing(indexer, config, fake_embedder):
    for p in config.vault_path.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".pdf"}:
            p.unlink()
    indexer.run(full=True)
    assert Searcher(config, embedder=fake_embedder).search("anything") == []
