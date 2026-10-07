"""Incremental indexing: new / modified / deleted / unchanged, plus the watchdog watcher."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from acadmem.indexer import Indexer
from acadmem.watch import VaultWatcher

NEW_NOTE = "# Queueing\n\nLittle's law relates the average number in a queue to arrival rate and waiting time.\n"


def paths(idx: Indexer) -> set[str]:
    return {d["source_path"] for d in idx.index.list_documents()}


def texts(idx: Indexer, source_path: str) -> str:
    return "\n".join(r["text"] for r in idx.index.scan(f"source_path = '{source_path}'"))


def test_second_run_with_no_changes_does_nothing(indexer: Indexer):
    indexer.run(full=True)
    before = indexer.index.count_chunks()
    stamp = {d["id"]: d["indexed_at"] for d in indexer.index.list_documents(include_empty=True)}
    report = indexer.run()
    assert not report.changed and report.errors == []
    assert len(report.unchanged) == 5 and report.chunks_written == 0
    assert indexer.index.count_chunks() == before
    # nothing was re-written: indexed_at is untouched
    assert stamp == {d["id"]: d["indexed_at"] for d in indexer.index.list_documents(include_empty=True)}


def test_new_file_is_indexed(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    (vault / "3F1" / "Notes" / "Queueing.md").write_text(NEW_NOTE, encoding="utf-8")
    report = indexer.run()
    assert report.added == ["3F1/Notes/Queueing.md"] and report.modified == [] and report.deleted == []
    assert "Little's law" in texts(indexer, "3F1/Notes/Queueing.md")
    row = indexer.index.scan("source_path = '3F1/Notes/Queueing.md'")[0]
    assert row["module"] == "3F1" and row["document_type"] == "Notes"


def test_modified_file_is_reindexed_and_stale_chunks_removed(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    note = vault / "3F1" / "Notes" / "deep" / "er" / "Markov.md"
    assert "memoryless" in texts(indexer, "3F1/Notes/deep/er/Markov.md")
    note.write_text("# Markov chains\n\nCompletely rewritten text about ergodicity and mixing time.\n", encoding="utf-8")
    report = indexer.run()
    assert report.modified == ["3F1/Notes/deep/er/Markov.md"] and report.added == []
    new_text = texts(indexer, "3F1/Notes/deep/er/Markov.md")
    assert "ergodicity" in new_text
    assert "memoryless" not in new_text  # old content is gone, not duplicated
    assert len(indexer.index.list_documents()) == 4


def test_deleted_file_disappears_from_index(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    (vault / "3B1" / "Questions" / "Bayes.md").unlink()
    report = indexer.run()
    assert report.deleted == ["3B1/Questions/Bayes.md"]
    assert indexer.index.scan("source_path = '3B1/Questions/Bayes.md'") == []
    assert "3B1/Questions/Bayes.md" not in paths(indexer)
    assert indexer.index.get_document(source_path="3B1/Questions/Bayes.md") is None


def test_deleted_pdf_disappears_from_index(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    (vault / "3B1" / "Handouts" / "Week 5 - Probability.pdf").unlink()
    report = indexer.run()
    assert report.deleted == ["3B1/Handouts/Week 5 - Probability.pdf"]
    assert indexer.index.scan("file_type = 'pdf'") == []


def test_touched_but_identical_file_is_not_reindexed(indexer: Indexer, vault: Path):
    """Cloud sync touches mtimes; content hash must keep this 'unchanged'."""
    indexer.run(full=True)
    note = vault / "loose.md"
    future = time.time() + 3600
    os.utime(note, (future, future))
    report = indexer.run()
    assert not report.changed and "loose.md" in report.unchanged and report.chunks_written == 0


def test_moved_file_is_delete_plus_add(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    shutil.move(str(vault / "loose.md"), str(vault / "3F1" / "moved.md"))
    report = indexer.run()
    assert report.deleted == ["loose.md"] and report.added == ["3F1/moved.md"]
    assert indexer.index.scan("source_path = 'loose.md'") == []


def test_failed_modification_keeps_running_and_is_reported(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    pdf = vault / "3B1" / "Handouts" / "Week 5 - Probability.pdf"
    pdf.write_bytes(b"corrupted now")
    report = indexer.run()
    assert [e["source_path"] for e in report.errors] == ["3B1/Handouts/Week 5 - Probability.pdf"]
    assert report.modified == []
    assert indexer.index.errors()[0]["source_path"].endswith("Probability.pdf")


def test_error_is_cleared_after_file_is_fixed(indexer: Indexer, vault: Path, pdf_factory):
    indexer.run(full=True)
    pdf = vault / "3B1" / "Handouts" / "Week 5 - Probability.pdf"
    good = pdf.read_bytes()
    pdf.write_bytes(b"corrupted now")
    indexer.run()
    assert indexer.index.errors()
    pdf.write_bytes(good)
    report = indexer.run()
    assert report.errors == [] and indexer.index.errors() == []


def test_sync_path_handles_create_modify_delete(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    f = vault / "3B1" / "Notes" / "Single.md"
    f.write_text("# Single\n\nStochastic gradient descent notes.\n", encoding="utf-8")
    assert indexer.sync_path(f) == "indexed"
    assert indexer.sync_path(f) == "ignored"  # unchanged content
    f.write_text("# Single\n\nAdam optimiser momentum notes.\n", encoding="utf-8")
    assert indexer.sync_path(f) == "indexed"
    assert "Adam optimiser" in texts(indexer, "3B1/Notes/Single.md")
    assert "Stochastic" not in texts(indexer, "3B1/Notes/Single.md")
    f.unlink()
    assert indexer.sync_path(f) == "deleted"
    assert indexer.sync_path(f) == "ignored"
    assert indexer.sync_path(vault.parent / "outside.md") == "ignored"
    assert indexer.sync_path(vault / ".obsidian" / "app.json") == "ignored"


def test_watcher_picks_up_create_modify_delete(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    target = vault / "3F1" / "Notes" / "Watched.md"
    with VaultWatcher(indexer, debounce=0.3) as watcher:
        target.write_text("# Watched\n\nFirst version about eigenvalues of symmetric matrices.\n", encoding="utf-8")
        assert wait_for(lambda: "eigenvalues" in texts(indexer, "3F1/Notes/Watched.md")), "new file not indexed"

        target.write_text("# Watched\n\nSecond version about singular value decomposition.\n", encoding="utf-8")
        assert wait_for(lambda: "singular value" in texts(indexer, "3F1/Notes/Watched.md")), "modification not indexed"
        assert "eigenvalues" not in texts(indexer, "3F1/Notes/Watched.md")

        target.unlink()
        assert wait_for(lambda: indexer.index.scan("source_path = '3F1/Notes/Watched.md'") == []), "deletion not applied"
        assert watcher.wait_idle(timeout=15)


def test_watcher_catches_up_on_changes_made_while_stopped(indexer: Indexer, vault: Path):
    indexer.run(full=True)
    (vault / "offline.md").write_text("# Offline\n\nWritten while the watcher was not running.\n", encoding="utf-8")
    (vault / "loose.md").unlink()
    with VaultWatcher(indexer, debounce=0.2):
        assert "offline.md" in paths(indexer) and "loose.md" not in paths(indexer)


def test_writers_are_serialised_by_the_lock(indexer: Indexer, config, fake_embedder):
    import pytest
    from filelock import Timeout

    indexer.run(full=True)
    other = Indexer(config, embedder=fake_embedder, lock_timeout=0.3)
    with indexer._lock:  # a writer (watcher / CLI / MCP write tool) holds the lock...
        with pytest.raises(Timeout):
            other.run()  # ...so a second writer must wait (and here, time out) rather than interleave
    assert not other.run().changed  # once released, it proceeds normally


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False
