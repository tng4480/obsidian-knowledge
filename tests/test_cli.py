"""The command line: index / rebuild / status / search, driven through ``main`` with the fake embedder."""

from __future__ import annotations

import json

import pytest

from acadmem.cli import main


@pytest.fixture(autouse=True)
def fake_models(monkeypatch, fake_embedder):
    monkeypatch.setattr("acadmem.indexer.get_embedder", lambda config: fake_embedder)
    monkeypatch.setattr("acadmem.search.get_embedder", lambda config: fake_embedder)


@pytest.fixture
def args(vault, config):
    return ["--vault", str(vault), "--index-dir", str(config.index_dir)]


def run(capsys, command, args, *extra) -> tuple[int, str, str]:
    code = main([command, *args, *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_status_before_any_index(capsys, args):
    code, out, _ = run(capsys, "status", args)
    assert code == 1 and "No index yet" in out


def test_index_then_status_counts_searchable_documents_not_empty_placeholders(capsys, args):
    code, out, _ = run(capsys, "index", args)
    assert code == 0 and "new" in out and "0 errors" in out

    code, out, _ = run(capsys, "status", args)
    assert code == 0
    # the fixture vault has 4 real documents plus an empty `_folder.md` placeholder
    assert "documents: 4 " in out and "1 empty placeholder" in out
    assert "dim " in out and "schema v" in out and "failed:    0" in out


def test_second_index_is_incremental_and_rebuild_starts_over(capsys, args):
    run(capsys, "index", args)
    _, out, _ = run(capsys, "index", args)
    assert out.startswith("incremental") and "+0 new" in out
    _, out, _ = run(capsys, "rebuild", args)
    assert "full:" in out and "+5 new" in out


def test_search_prints_provenance(capsys, args):
    run(capsys, "index", args)
    code, out, _ = run(capsys, "search", args, "posterior probability of disease")
    assert code == 0 and "3B1/Questions/Bayes.md" in out
    first = out.splitlines()[0]
    assert first.startswith("[1] sim ")  # cosine similarity, not the ~0.016 fusion value
    assert 0.1 < float(first.split()[2]) <= 1.0


def test_search_json_has_similarity_score_and_rank_score(capsys, args):
    run(capsys, "index", args)
    code, out, _ = run(capsys, "search", args, "posterior probability of disease", "--json", "-k", "3")
    payload = json.loads(out)
    assert code == 0 and payload
    for h in payload:
        assert -1.0 <= h["score"] <= 1.0 and 0 < h["rank_score"] <= 2 / 60 + 1e-9 and h["mode"] == "hybrid"


def test_search_json_filters_and_expansion(capsys, args):
    run(capsys, "index", args)
    code, out, _ = run(capsys, "search", args, "probability", "--module", "3B1", "--file-type", "pdf", "--json", "--expand", "neighbors:1", "-k", "3")
    payload = json.loads(out)
    assert code == 0 and payload and len(payload) <= 3
    assert all(h["source_path"].endswith(".pdf") and h["module"] == "3B1" for h in payload)
    assert all(h["context"]["pieces"] for h in payload)


@pytest.mark.parametrize("mode", ["vector", "fts", "hybrid"])
def test_search_modes(capsys, args, mode):
    run(capsys, "index", args)
    code, out, _ = run(capsys, "search", args, "stationary distribution", "--mode", mode)
    assert code == 0 and "Markov.md" in out


def test_search_without_matches_says_so(capsys, args):
    run(capsys, "index", args)
    code, out, _ = run(capsys, "search", args, "probability", "--module", "9Z9")
    assert code == 0 and "No results" in out


def test_missing_vault_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit) as info:
        main(["status", "--vault", str(tmp_path / "nope"), "--index-dir", str(tmp_path / "i")])
    assert "Vault not found" in str(info.value)


def test_corrupt_file_is_reported_without_failing_the_run(capsys, args, vault):
    (vault / "3B1" / "Handouts" / "Broken.pdf").write_bytes(b"not a pdf")
    code, out, err = run(capsys, "index", args)
    assert code == 0 and "1 errors" in out and "Broken.pdf" in err
    _, out, _ = run(capsys, "status", args)
    assert "failed:    1" in out and "Broken.pdf" in out
