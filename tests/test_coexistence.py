"""Living next to the separate `obsidian` MCP server (mcp-obsidian-vault).

That server keeps tasks, decisions and daily notes at the vault root and rewrites `DASHBOARD.md` on every task change.
None of that is study material: it must not be indexed, searched or written by this server, while a folder that merely
shares a name *inside* a module stays ordinary study material.
"""

from __future__ import annotations

import dataclasses

import pytest

from acadmem.indexer import Indexer
from acadmem.ingest.discovery import discover, is_indexable
from acadmem.search import Searcher
from acadmem.vault_write import PathNotAllowed, VaultWriter

BOOKKEEPING = {
    "Tasks/Revise Viterbi.md": "# Task\n\nzorblax revise the Viterbi decoder before Friday.\n",
    "Decisions/Use bge.md": "# Decision\n\nzorblax chose the bge embedding model.\n",
    "Daily Notes/2026-10-07.md": "# Today\n\nzorblax worked on the retrieval server.\n",
    "DASHBOARD.md": "# Dashboard\n\nzorblax open tasks: 3\n",
}


@pytest.fixture
def busy_vault(vault):
    for rel, text in BOOKKEEPING.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return vault


def test_bookkeeping_is_not_discovered(config, busy_vault):
    found = {f.source_path for f in discover(config)}
    assert not found & set(BOOKKEEPING)
    assert "3B1/Questions/Bayes.md" in found  # real study material still is


@pytest.mark.parametrize("rel", sorted(BOOKKEEPING))
def test_bookkeeping_is_not_indexable_even_via_the_watcher_path(config, rel):
    assert not is_indexable(config, rel)


def test_matching_is_case_insensitive(config):
    assert not is_indexable(config, "tasks/x.md")
    assert not is_indexable(config, "daily notes/x.md")
    assert not is_indexable(config, "dashboard.md")


def test_same_names_inside_a_module_are_study_material(config, vault):
    nested = vault / "3F1" / "Notes" / "Tasks"
    nested.mkdir(parents=True)
    (nested / "Exercises.md").write_text("# Exercises\n\nSample the signal below Nyquist.\n", encoding="utf-8")
    assert is_indexable(config, "3F1/Notes/Tasks/Exercises.md")
    assert "3F1/Notes/Tasks/Exercises.md" in {f.source_path for f in discover(config)}


def test_a_note_merely_named_like_a_folder_is_still_indexed(config):
    assert is_indexable(config, "Tasks of the week.md")  # only the exact entry names are reserved


def test_searches_never_return_bookkeeping(busy_vault, config, fake_embedder):
    Indexer(config, embedder=fake_embedder).run(full=True)
    hits = Searcher(config, embedder=fake_embedder).search("zorblax", top_k=10)
    assert not {h.source_path for h in hits} & set(BOOKKEEPING)


def test_bookkeeping_indexed_by_an_older_version_is_removed_on_the_next_sync(busy_vault, config, fake_embedder):
    permissive = dataclasses.replace(config, ignored_root_entries=())
    Indexer(permissive, embedder=fake_embedder).run(full=True)
    assert {h.source_path for h in Searcher(permissive, embedder=fake_embedder).search("zorblax", top_k=10)} >= set(BOOKKEEPING)

    report = Indexer(config, embedder=fake_embedder).run()
    assert set(report.deleted) == set(BOOKKEEPING)
    remaining = {h.source_path for h in Searcher(config, embedder=fake_embedder).search("zorblax", top_k=10)}
    assert not remaining & set(BOOKKEEPING)


def test_dashboard_rewrites_do_not_reindex_anything(busy_vault, config, fake_embedder):
    indexer = Indexer(config, embedder=fake_embedder)
    indexer.run(full=True)
    (busy_vault / "DASHBOARD.md").write_text("# Dashboard\n\nopen tasks: 4\n", encoding="utf-8")
    assert not indexer.run().changed
    assert indexer.sync_path(busy_vault / "DASHBOARD.md") == "ignored"


@pytest.mark.parametrize("path", ["Tasks/new.md", "tasks/new.md", "Decisions/x.md", "Daily Notes/2026-10-08.md", "DASHBOARD.md"])
def test_this_server_will_not_write_into_the_other_servers_area(indexer, config, vault, path):
    writer = VaultWriter(config, indexer)
    before = sorted(p.relative_to(vault).as_posix() for p in vault.rglob("*") if p.is_file())
    with pytest.raises(PathNotAllowed, match="reserved for the obsidian server"):
        writer.create_note(path, "# x\n\nbody")
    assert sorted(p.relative_to(vault).as_posix() for p in vault.rglob("*") if p.is_file()) == before


def test_the_ignore_list_is_configurable(config):
    custom = dataclasses.replace(config, ignored_root_entries=("Scratch",))
    assert not is_indexable(custom, "Scratch/a.md")
    assert is_indexable(custom, "Tasks/a.md")
