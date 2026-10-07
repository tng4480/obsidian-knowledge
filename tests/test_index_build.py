"""Embedding / LanceDB integration and full-rebuild tests (fake embedder; one slow real-model test)."""

from __future__ import annotations

from pathlib import Path

import pytest

from acadmem.config import Config
from acadmem.index import IndexMismatchError, VectorIndex
from acadmem.indexer import Indexer
from acadmem.ingest.discovery import discover, is_indexable


def test_discovery_is_recursive_and_ignores_hidden_dirs(config: Config):
    found = sorted(df.source_path for df in discover(config))
    assert found == [
        "3B1/Handouts/Week 5 - Probability.pdf",
        "3B1/Notes/_folder.md",
        "3B1/Questions/Bayes.md",
        "3F1/Notes/deep/er/Markov.md",
        "loose.md",
    ]
    assert not is_indexable(config, ".obsidian/app.json")
    assert not is_indexable(config, "3B1/photo.png")


def test_full_rebuild_indexes_markdown_and_pdf_into_one_table(indexer: Indexer):
    report = indexer.run(full=True)
    assert report.errors == []
    assert sorted(report.added) == [
        "3B1/Handouts/Week 5 - Probability.pdf",
        "3B1/Notes/_folder.md",
        "3B1/Questions/Bayes.md",
        "3F1/Notes/deep/er/Markov.md",
        "loose.md",
    ]
    idx = indexer.index
    types = {r["file_type"] for r in idx.scan(columns=["file_type"])}
    assert types == {"md", "pdf"}
    assert idx.count_chunks() > 5
    # the empty file is recorded (so it's "unchanged" next time) but has no chunks and is not listed
    assert idx.get_document(source_path="3B1/Notes/_folder.md")["chunk_count"] == 0
    assert "3B1/Notes/_folder.md" not in [d["source_path"] for d in idx.list_documents()]


def test_rows_keep_full_provenance(indexer: Indexer):
    indexer.run(full=True)
    rows = indexer.index.scan("source_path = '3B1/Questions/Bayes.md'")
    dont = next(r for r in rows if "evidence term" in r["text"])
    assert dont["module"] == "3B1" and dont["document_type"] == "Questions"
    assert dont["file_type"] == "md"
    assert dont["heading_path"] == ["Bayesian inference", "What I don't understand"]
    assert dont["heading"] == "Bayesian inference > What I don't understand"
    assert dont["page_number"] is None
    assert dont["section_id"].startswith(dont["document_id"] + "#")
    assert dont["parent_section_id"] is not None
    assert "probability" in dont["tags"]

    pdf_rows = indexer.index.scan("source_path = '3B1/Handouts/Week 5 - Probability.pdf'")
    indep = next(r for r in pdf_rows if "occurrence of one" in r["text"])
    assert indep["file_type"] == "pdf" and indep["document_type"] == "Handouts"
    assert indep["page_number"] == 3
    assert indep["heading_path"] == ["Independence"]
    # the "Independence" bookmark section continues onto page 4; its chunk reports the full page range
    assert (indep["page_start"], indep["page_end"]) == (3, 4)
    assert "Pairwise independence" in indep["text"]


def test_rebuild_recreates_the_index_from_scratch(indexer: Indexer, config: Config):
    indexer.run(full=True)
    before = sorted(r["chunk_id"] for r in indexer.index.scan(columns=["chunk_id"]))
    # blow the derived index away completely; the vault is untouched
    VectorIndex(config).reset()
    assert not config.db_path.exists()
    fresh = Indexer(config, embedder=indexer.embedder)
    report = fresh.run(full=True)
    after = sorted(r["chunk_id"] for r in fresh.index.scan(columns=["chunk_id"]))
    assert before == after and len(report.added) == 5


def test_error_in_one_file_does_not_abort_the_run(vault: Path, config: Config, fake_embedder):
    class Boom(Exception):
        pass

    indexer = Indexer(config, embedder=fake_embedder)
    real_parse = indexer.registry.for_path(Path("x.md")).parse

    class Flaky:
        extensions = (".md",)
        file_type = "md"

        def parse(self, path: Path):
            if path.name == "Markov.md":
                raise Boom("cannot parse this file")
            return real_parse(path)

    indexer.registry.register(Flaky())
    report = indexer.run(full=True)
    assert [e["source_path"] for e in report.errors] == ["3F1/Notes/deep/er/Markov.md"]
    assert report.errors[0]["error_type"] == "Boom"
    # other files were still indexed
    assert "3B1/Questions/Bayes.md" in report.added and "loose.md" in report.added
    stored = indexer.index.errors()
    assert [e["source_path"] for e in stored] == ["3F1/Notes/deep/er/Markov.md"]
    assert "cannot parse" in stored[0]["message"]


def test_corrupt_pdf_is_recorded_not_fatal(vault: Path, config: Config, fake_embedder):
    (vault / "3B1" / "Handouts" / "broken.pdf").write_bytes(b"this is not a pdf")
    report = Indexer(config, embedder=fake_embedder).run(full=True)
    assert any(e["source_path"].endswith("broken.pdf") for e in report.errors)
    assert "3B1/Questions/Bayes.md" in report.added


def test_model_mismatch_requires_rebuild(indexer: Indexer, config: Config, fake_embedder):
    indexer.run(full=True)

    class OtherModel(type(fake_embedder)):
        def __init__(self):
            super().__init__()
            self.model_name = "some-other-model"

    with pytest.raises(IndexMismatchError, match="acadmem rebuild"):
        Indexer(config, embedder=OtherModel()).run()


def test_vectors_have_embedding_dimension(indexer: Indexer):
    indexer.run(full=True)
    row = indexer.index.chunks.search().limit(1).to_list()[0]
    assert len(row["vector"]) == indexer.embedder.dimension


@pytest.mark.slow
def test_real_bge_model_end_to_end_index(vault: Path, tmp_path: Path):
    from acadmem.embeddings import get_embedder

    cfg = Config(vault_path=vault, index_dir=tmp_path / "idx_real")
    embedder = get_embedder(cfg)
    assert embedder.dimension == 384
    report = Indexer(cfg, embedder=embedder).run(full=True)
    # production chunk limits (max 384 / min 64) merge this vault's tiny sections into few chunks
    assert report.errors == [] and report.chunks_written >= 3
    q = embedder.embed_query("how does a prior combine with data")
    assert len(q) == 384 and abs(sum(x * x for x in q) - 1.0) < 1e-3  # normalised
