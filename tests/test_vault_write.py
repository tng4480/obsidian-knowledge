from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import frontmatter
import pytest

from acadmem.search import Searcher
from acadmem.vault_write import (
    AmbiguousSection,
    ContentRejected,
    NoteExists,
    NoteNotFound,
    PathNotAllowed,
    SectionNotFound,
    VaultWriter,
    WriteConflict,
    WriteError,
    find_section,
    scan_headings,
)

PDF = "3B1/Handouts/Week 5 - Probability.pdf"


@pytest.fixture
def writer(indexer, config):
    indexer.run(full=True)
    return VaultWriter(config, indexer)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def no_temp_files(root: Path) -> bool:
    return not [p for p in root.rglob("*") if p.name.startswith(".acadmem-")]


# ---------------------------------------------------------------- path guards
@pytest.mark.parametrize(
    "bad",
    [
        "../escape.md",
        "..\\escape.md",
        "3B1/../../escape.md",
        "a/../../escape.md",
        "/abs/escape.md",
        "\\\\server\\share\\x.md",
        "C:/Windows/x.md",
        "C:\\x.md",
        ".obsidian/plugins/x.md",
        ".obsidian/app.md",
        ".git/config.md",
        ".trash/x.md",
        "3B1/.hidden/x.md",
        "3B1/.secret.md",
        "node_modules/x.md",
        "3B1/CON.md",
        "3B1/nul.md",
        "3B1/what?.md",
        "3B1/a<b>.md",
        "3B1/trailingdot..md.",
        "3B1/x.pdf",
        "3B1/x.txt",
        "3B1/x.md.exe",
        PDF,
        "",
        "   ",
        "///",
        "3B1/\x00evil.md",
    ],
)
def test_paths_that_leave_the_vault_or_are_not_markdown_are_refused(writer, vault, bad):
    before = snapshot(vault)
    with pytest.raises(PathNotAllowed):
        writer.create_note(bad, "hello")
    with pytest.raises(WriteError):
        writer.append_to_note(bad, "hello")
    with pytest.raises(WriteError):
        writer.replace_section(bad, "H", "hello")
    assert snapshot(vault) == before


def test_traversal_never_creates_files_outside_the_vault(writer, vault, tmp_path):
    outside_before = sorted(p.name for p in tmp_path.iterdir())
    for bad in ("../outside.md", "..\\outside.md", "3B1/../../outside.md"):
        with pytest.raises(PathNotAllowed):
            writer.create_note(bad, "x")
    assert sorted(p.name for p in tmp_path.iterdir()) == outside_before
    assert not (vault.parent / "outside.md").exists()


def test_symlink_escape_is_refused(writer, vault, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = vault / "3B1" / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Symlinks need privileges on Windows, but directory junctions (the same escape route) do not.
        import subprocess

        made = os.name == "nt" and subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True).returncode == 0
        if not made:
            pytest.skip("cannot create symlinks or junctions on this machine")
    with pytest.raises(PathNotAllowed, match="outside"):
        writer.create_note("3B1/link/x.md", "hello")
    assert not list(outside.iterdir())


def test_extension_is_optional_but_only_md_is_ever_written(writer, vault):
    res = writer.create_note("3B1/Notes/no extension", "# T\n\nbody text here")
    assert res.path == "3B1/Notes/no extension.md" and (vault / "3B1/Notes/no extension.md").is_file()
    assert writer.create_note("3B1/Notes/Upper.MD", "# T\n\nbody").path == "3B1/Notes/Upper.MD"


def test_existing_casing_is_reused_so_one_note_has_one_identity(writer, vault):
    res = writer.create_note("3b1/notes/CaseTest.md", "# Case\n\ntext")
    assert res.path == "3B1/Notes/CaseTest.md"
    assert (vault / "3B1" / "Notes" / "CaseTest.md").is_file()
    again = writer.append_to_note("3B1/NOTES/casetest.md", "more")
    assert again.path == "3B1/Notes/CaseTest.md"


def test_cannot_use_a_file_as_a_folder(writer):
    with pytest.raises(PathNotAllowed, match="is a file"):
        writer.create_note("loose.md/child.md", "x")


# ---------------------------------------------------------------- create
def test_create_note_writes_frontmatter_indexes_and_is_filterable(writer, vault, config, fake_embedder):
    res = writer.create_note("3F7/Notes/New Idea.md", "# New idea\n\nThe zygomorphic coding gain is interesting.", tags=["#entropy", "idea"])
    assert (res.action, res.indexed, res.ai_generated, res.backup) == ("created", "indexed", True, None)
    post = frontmatter.load(vault / "3F7" / "Notes" / "New Idea.md")
    assert post.metadata["ai_generated"] is True and post.metadata["tags"] == ["entropy", "idea"]
    assert post.content.startswith("# New idea")
    searcher = Searcher(config, embedder=fake_embedder)
    hit = searcher.search("zygomorphic coding gain", top_k=1)[0]
    assert hit.source_path == "3F7/Notes/New Idea.md" and hit.ai_generated is True
    assert {"entropy", "idea"} <= set(hit.tags)
    assert [h.source_path for h in searcher.search("zygomorphic", filters={"ai_generated": True})] == ["3F7/Notes/New Idea.md"]
    assert all(h.source_path != "3F7/Notes/New Idea.md" for h in searcher.search("zygomorphic", filters={"ai_generated": False}))
    assert no_temp_files(vault)


def test_create_note_creates_nested_folders_inside_the_vault(writer, vault):
    res = writer.create_note("3F9/Brand new/Deeper/Note.md", "# N\n\ntext")
    assert (vault / "3F9" / "Brand new" / "Deeper" / "Note.md").is_file() and res.indexed == "indexed"


def test_create_note_forces_ai_generated_even_if_the_model_claims_otherwise(writer, vault):
    writer.create_note("3B1/Notes/Claim.md", "---\nai_generated: false\ntags: [mine]\nstatus: draft\n---\n# Claim\n\nbody", tags=["extra"])
    post = frontmatter.load(vault / "3B1" / "Notes" / "Claim.md")
    assert post.metadata["ai_generated"] is True
    assert post.metadata["tags"] == ["mine", "extra"] and post.metadata["status"] == "draft"


def test_create_note_never_overwrites(writer, vault):
    original = sha(vault / "3B1" / "Questions" / "Bayes.md")
    with pytest.raises(NoteExists, match="append_to_note"):
        writer.create_note("3B1/Questions/Bayes.md", "clobber!")
    with pytest.raises(NoteExists):
        writer.create_note("3b1/questions/bayes", "clobber!")
    assert sha(vault / "3B1" / "Questions" / "Bayes.md") == original
    writer.create_note("3B1/Notes/Once.md", "first")
    with pytest.raises(NoteExists):
        writer.create_note("3B1/Notes/Once.md", "second")
    assert "first" in (vault / "3B1" / "Notes" / "Once.md").read_text(encoding="utf-8")
    assert no_temp_files(vault)


@pytest.mark.parametrize("content", ["", "   \n  ", "x\x00y", None, 42])
def test_create_rejects_bad_content(writer, vault, content):
    before = snapshot(vault)
    with pytest.raises(ContentRejected):
        writer.create_note("3B1/Notes/Bad.md", content)  # type: ignore[arg-type]
    assert snapshot(vault) == before


def test_size_limit(writer, config):
    big = "x" * (config.write_max_bytes + 1)
    with pytest.raises(ContentRejected, match="limit"):
        writer.create_note("3B1/Notes/Big.md", big)
    with pytest.raises(ContentRejected, match="limit"):
        writer.append_to_note("loose.md", big)


def test_malformed_frontmatter_rejected(writer):
    with pytest.raises(ContentRejected):
        writer.create_note("3B1/Notes/Mal.md", "---\nkey: [unclosed\n---\nbody")


def test_writes_can_be_disabled(indexer, config):
    off = VaultWriter(config.with_overrides(writes_enabled=False), indexer)
    with pytest.raises(WriteError, match="disabled"):
        off.create_note("3B1/Notes/X.md", "x")
    with pytest.raises(WriteError, match="disabled"):
        off.append_to_note("loose.md", "x")
    with pytest.raises(WriteError, match="disabled"):
        off.replace_section("loose.md", "Loose", "x")


# ---------------------------------------------------------------- append
def test_append_to_human_note_keeps_original_bytes_marks_the_addition_and_reindexes(writer, vault, config, fake_embedder):
    path = vault / "3B1" / "Questions" / "Bayes.md"
    original = path.read_bytes()
    res = writer.append_to_note("3B1/Questions/Bayes.md", "## AI follow-up\n\nThe quokka prior is conjugate.")
    new = path.read_bytes()
    assert new.startswith(original.rstrip(b"\r\n"))
    tail = new[len(original.rstrip(b"\r\n")):].decode("utf-8")
    assert "<!-- ai:append " in tail and "## AI follow-up" in tail
    assert (res.action, res.indexed, res.ai_generated) == ("appended", "indexed", False)
    hit = Searcher(config, embedder=fake_embedder).search("quokka prior conjugate", top_k=1)[0]
    assert hit.source_path == "3B1/Questions/Bayes.md" and hit.ai_generated is False  # human note stays human
    assert no_temp_files(vault)


def test_append_to_an_ai_note_adds_no_marker(writer, vault):
    writer.create_note("3B1/Notes/Mine.md", "# Mine\n\nfirst")
    writer.append_to_note("3B1/Notes/Mine.md", "second")
    text = (vault / "3B1" / "Notes" / "Mine.md").read_text(encoding="utf-8")
    assert "<!-- ai" not in text and text.rstrip().endswith("second")


def test_append_preserves_crlf_line_endings(writer, vault):
    path = vault / "3B1" / "Notes" / "crlf.md"
    path.write_bytes(b"# Windows note\r\n\r\nline one\r\n")
    writer.append_to_note("3B1/Notes/crlf.md", "added\nacross lines")
    raw = path.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert raw.endswith(b"across lines\r\n")


def test_append_saves_a_backup_outside_the_vault(writer, vault, config):
    path = vault / "3F1" / "Notes" / "x.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# X\n\noriginal text\n", encoding="utf-8")
    res = writer.append_to_note("3F1/Notes/x.md", "more")
    backup = Path(res.backup)
    assert backup.read_text(encoding="utf-8") == "# X\n\noriginal text\n"
    assert vault not in backup.parents and config.backup_dir in backup.parents


def test_backup_retention(writer, vault, config):
    small = VaultWriter(config.with_overrides(backup_keep=3), writer.indexer)
    for i in range(6):
        small.append_to_note("loose.md", f"edit {i}")
    assert len(list(config.backup_dir.iterdir())) == 3


def test_append_errors(writer, vault):
    with pytest.raises(NoteNotFound, match="create_note"):
        writer.append_to_note("3B1/Notes/missing.md", "x")
    with pytest.raises(ContentRejected, match="frontmatter"):
        writer.append_to_note("loose.md", "---\nai_generated: false\n---\nx")
    with pytest.raises(PathNotAllowed):
        writer.append_to_note(PDF, "x")


def test_non_utf8_notes_are_refused_not_corrupted(writer, vault):
    path = vault / "3B1" / "Notes" / "latin.md"
    path.write_bytes("# Caf\xe9\n\ncaf\xe9".encode("latin-1"))
    before = path.read_bytes()
    with pytest.raises(WriteError, match="UTF-8"):
        writer.append_to_note("3B1/Notes/latin.md", "x")
    assert path.read_bytes() == before


# ---------------------------------------------------------------- replace_section
DOC = """---
tags: [demo]
---
# Title

Intro text.

## Alpha

alpha body one
alpha body two

### Alpha child

child body

## Beta

beta body

```python
## not a heading
```

## Alpha

second alpha

$$
## also not a heading
$$
"""


@pytest.fixture
def doc(vault):
    path = vault / "3B1" / "Notes" / "doc.md"
    path.write_text(DOC, encoding="utf-8")
    return path


def test_scan_headings_ignores_frontmatter_code_and_math():
    heads = scan_headings(DOC.split("\n"))
    assert [(h.level, h.title) for h in heads] == [(1, "Title"), (2, "Alpha"), (3, "Alpha child"), (2, "Beta"), (2, "Alpha")]
    assert heads[2].path == ("Title", "Alpha", "Alpha child")


def test_find_section_by_title_path_and_case():
    heads = scan_headings(DOC.split("\n"))
    assert find_section(heads, "beta").title == "Beta"
    assert find_section(heads, "Title > Beta").path == ("Title", "Beta")
    assert find_section(heads, "alpha > alpha child").title == "Alpha child"
    with pytest.raises(AmbiguousSection, match="Alpha"):
        find_section(heads, "Alpha")
    with pytest.raises(SectionNotFound, match="Headings in this note"):
        find_section(heads, "Gamma")


def test_replace_section_touches_only_that_section(writer, doc):
    before = doc.read_text(encoding="utf-8")
    res = writer.replace_section("3B1/Notes/doc.md", "Beta", "brand new beta text")
    after = doc.read_text(encoding="utf-8")
    assert res.action == "replaced_section" and res.indexed == "indexed" and Path(res.backup).read_text(encoding="utf-8") == before
    assert "brand new beta text" in after and "beta body" not in after
    assert "## Beta\n" in after and "<!-- ai:replace " in after
    # everything outside Beta (including the sibling sections, the fenced '## not a heading', the math block) is untouched
    for kept in ("Intro text.", "alpha body one", "child body", "second alpha", "## also not a heading", "tags: [demo]"):
        assert kept in after
    assert "```python\n## not a heading\n```" not in after  # that fence belonged to the Beta section, so it was replaced


@pytest.mark.parametrize("heading", ["Alpha", "Title > Alpha"])
def test_replace_section_with_ambiguous_heading_is_refused_and_changes_nothing(writer, doc, heading):
    before = doc.read_bytes()
    with pytest.raises(AmbiguousSection):
        writer.replace_section("3B1/Notes/doc.md", heading, "x")
    assert doc.read_bytes() == before
    assert not list(writer.config.backup_dir.glob("*")) if writer.config.backup_dir.exists() else True


def test_replace_section_removes_child_sections_with_their_parent(writer, doc):
    path = doc.parent / "simple.md"
    path.write_text("# T\n\n## A\n\na body\n\n### A1\n\nchild\n\n## B\n\nb body\n", encoding="utf-8")
    writer.replace_section("3B1/Notes/simple.md", "A", "new a")
    text = path.read_text(encoding="utf-8")
    assert "child" not in text and "### A1" not in text and "new a" in text and "b body" in text
    assert text.index("## A") < text.index("new a") < text.index("## B")


def test_replace_last_section_and_heading_path(writer, doc):
    path = doc.parent / "simple.md"
    path.write_text("# T\n\n## A\n\nfirst\n\n## B\n\nlast\n", encoding="utf-8")
    writer.replace_section("3B1/Notes/simple.md", "T > B", "tail")
    text = path.read_text(encoding="utf-8")
    assert re.search(r"## B\n\n<!-- ai:replace \d{4}-\d{2}-\d{2} -->\ntail\n$", text)
    assert "first" in text and "last" not in text


def test_replace_section_missing_heading(writer, doc):
    before = doc.read_bytes()
    with pytest.raises(SectionNotFound, match="Headings in this note"):
        writer.replace_section("3B1/Notes/doc.md", "Gamma", "x")
    assert doc.read_bytes() == before


def test_replace_section_create_if_missing(writer, doc):
    res = writer.replace_section("3B1/Notes/doc.md", "Gamma", "gamma text", create_if_missing=True)
    text = doc.read_text(encoding="utf-8")
    assert res.action == "added_section" and "\n## Gamma\n" in text and text.rstrip().endswith("gamma text")
    assert "second alpha" in text  # nothing else lost


def test_replace_section_in_ai_note_adds_no_marker(writer, vault):
    writer.create_note("3B1/Notes/Gen.md", "# Gen\n\n## Part\n\nold")
    writer.replace_section("3B1/Notes/Gen.md", "Part", "fresh")
    text = (vault / "3B1" / "Notes" / "Gen.md").read_text(encoding="utf-8")
    assert "fresh" in text and "old" not in text and "<!-- ai" not in text


def test_replacing_a_section_updates_search(writer, doc, config, fake_embedder):
    writer.replace_section("3B1/Notes/doc.md", "Beta", "all about wombats and marsupials")
    hits = Searcher(config, embedder=fake_embedder).search("wombats marsupials", top_k=1)
    assert hits[0].source_path == "3B1/Notes/doc.md"


# ---------------------------------------------------------------- safety nets
def test_concurrent_edit_is_detected_and_not_clobbered(writer, vault, monkeypatch):
    path = vault / "3B1" / "Notes" / "race.md"
    path.write_text("# Race\n\nbefore\n", encoding="utf-8")
    original_backup = writer._backup

    def edit_in_obsidian(absolute, source_path):  # runs between our read and our write
        result = original_backup(absolute, source_path)
        absolute.write_text("# Race\n\nbefore\n\nTHE USER TYPED THIS\n", encoding="utf-8")
        return result

    monkeypatch.setattr(writer, "_backup", edit_in_obsidian)
    with pytest.raises(WriteConflict, match="changed"):
        writer.append_to_note("3B1/Notes/race.md", "ai text")
    assert "THE USER TYPED THIS" in path.read_text(encoding="utf-8") and "ai text" not in path.read_text(encoding="utf-8")
    assert no_temp_files(vault)


def test_a_failing_reindex_does_not_lose_the_write(writer, vault, monkeypatch):
    def boom(_path):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(writer.indexer, "sync_path", boom)
    res = writer.create_note("3B1/Notes/Safe.md", "# Safe\n\nbody")
    assert (vault / "3B1" / "Notes" / "Safe.md").is_file()
    assert res.indexed == "error" and "index exploded" in res.index_error


def test_without_an_indexer_the_write_still_works(config, vault):
    res = VaultWriter(config, None).create_note("3B1/Notes/Plain.md", "# P\n\nbody")
    assert res.indexed == "skipped" and (vault / "3B1" / "Notes" / "Plain.md").is_file()


def test_pdfs_and_other_notes_are_never_touched(writer, vault):
    before = snapshot(vault)
    writer.create_note("3F7/Notes/Fresh.md", "# Fresh\n\nbody")
    writer.append_to_note("loose.md", "extra")
    after = snapshot(vault)
    changed = {k for k in after if before.get(k) != after[k]}
    assert changed == {"3F7/Notes/Fresh.md".replace("/", os.sep), "loose.md"}
    assert after[PDF.replace("/", os.sep)] == before[PDF.replace("/", os.sep)]


def test_there_is_no_delete_or_overwrite_operation():
    public = {n for n in dir(VaultWriter) if not n.startswith("_")}
    assert public == {"create_note", "append_to_note", "replace_section", "resolve"}


def test_the_watcher_does_not_double_index_our_writes(writer, vault, config):
    from acadmem.index import VectorIndex

    writer.create_note("3B1/Notes/Once.md", "# Once\n\nbody")
    report = writer.indexer.run()
    assert "3B1/Notes/Once.md" in report.unchanged and not report.added
    assert VectorIndex(config).get_document(source_path="3B1/Notes/Once.md") is not None
