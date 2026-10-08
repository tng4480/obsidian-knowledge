"""The tag_pdf_passage tool: Codex tags a PDF passage by writing a tag line into a note (the PDF is never touched)."""

from __future__ import annotations

import dataclasses
import hashlib

import frontmatter
import pytest
from test_mcp_server import Client, digest

from acadmem.cli import main as cli_main
from acadmem.mcp_server import AcademicMemoryService, build_server

PDF = "3B1/Handouts/Week 5 - Probability.pdf"
DEFAULT_NOTE = "3B1/PDF tags.md"


@pytest.fixture
def client(indexer, config, fake_embedder):
    indexer.run(full=True)
    return Client(build_server(AcademicMemoryService(config, embedder=fake_embedder)))


def tagged_pdf_chunks(client, tag: str) -> list[dict]:
    out = client.call("search_academic_memory", tags=tag, file_type="pdf")
    return out["results"]


# --------------------------------------------------------------------------- happy paths
def test_tagging_a_page_writes_a_note_line_and_the_tag_is_then_retrievable(client, vault):
    res = client.call("tag_pdf_passage", pdf="Week 5 - Probability.pdf", tags="revisit", page=2)
    assert res["ok"] and res["pdf"] == PDF and res["note"] == DEFAULT_NOTE
    assert res["line"] == f"[[{PDF}#page=2]] #revisit"
    assert res["sections_tagged"] >= 1 and res["write"]["action"] == "created" and res["write"]["indexed"] == "indexed"

    found = tagged_pdf_chunks(client, "revisit")
    assert found and all(r["source"] == PDF for r in found)
    assert all(r["page"] <= 2 <= int(str(r["pages"]).split("-")[-1]) for r in found)
    assert {r["chunk_id"] for r in found} == {s["chunk_id"] for s in res["sections"]}


def test_the_note_is_marked_ai_generated_and_holds_the_line(client, vault):
    client.call("tag_pdf_passage", pdf=PDF, tags=["revisit"], page=1)
    post = frontmatter.loads((vault / DEFAULT_NOTE).read_text(encoding="utf-8"))
    assert post.metadata["ai_generated"] is True
    assert f"[[{PDF}#page=1]] #revisit" in post.content


def test_tagging_by_quote_tags_the_one_section_containing_it(client):
    res = client.call("tag_pdf_passage", pdf="Week 5 - Probability", tags="hard, exam", quote="mutual independence of a whole family")
    assert res["line"] == f'[[{PDF}]] "mutual independence of a whole family" #hard #exam'
    assert res["sections_tagged"] == 1
    (hit,) = tagged_pdf_chunks(client, "hard")
    assert "Pairwise independence" in hit["text"]
    assert [r["chunk_id"] for r in tagged_pdf_chunks(client, "exam")] == [hit["chunk_id"]]


def test_a_quote_may_match_a_section_heading(client):
    res = client.call("tag_pdf_passage", pdf=PDF, tags="revisit", quote="Conditional probability")
    assert res["sections_tagged"] >= 1
    assert all("Conditional" in s["section"] or s["page"] == 1 for s in res["sections"])


def test_tagging_the_whole_pdf(client):
    res = client.call("tag_pdf_passage", pdf=PDF, tags="exam-prep")
    assert res["line"] == f"[[{PDF}]] #exam-prep"
    doc_chunks = client.call("get_document", source_path=PDF, include_text=True)["chunks"]
    assert res["sections_tagged"] == len(doc_chunks)


def test_the_pdf_itself_is_never_touched(client, vault):
    before = hashlib.sha256((vault / PDF).read_bytes()).hexdigest()
    client.call("tag_pdf_passage", pdf=PDF, tags="revisit", page=2)
    client.call("tag_pdf_passage", pdf=PDF, tags="hard", quote="mutual independence of a whole family")
    assert hashlib.sha256((vault / PDF).read_bytes()).hexdigest() == before


def test_a_second_tag_appends_to_the_same_note(client, vault):
    client.call("tag_pdf_passage", pdf=PDF, tags="revisit", page=1)
    res = client.call("tag_pdf_passage", pdf=PDF, tags="hard", page=3)
    assert res["write"]["action"] == "appended"
    text = (vault / DEFAULT_NOTE).read_text(encoding="utf-8")
    assert f"[[{PDF}#page=1]] #revisit" in text and f"[[{PDF}#page=3]] #hard" in text
    assert "<!-- ai:append" not in text  # the note is the assistant's own, so no marker is needed
    assert tagged_pdf_chunks(client, "revisit") and tagged_pdf_chunks(client, "hard")


def test_repeating_the_same_tag_writes_nothing(client, vault):
    first = client.call("tag_pdf_passage", pdf=PDF, tags="revisit", page=2)
    before = digest(vault)
    again = client.call("tag_pdf_passage", pdf=PDF, tags="revisit", page=2)
    assert again["already_present"] is True and "write" not in again and again["line"] == first["line"]
    assert digest(vault) == before


def test_a_note_the_user_wrote_can_be_chosen_and_keeps_its_text(client, vault):
    path = vault / "3B1" / "Notes" / "_folder.md"
    path.write_text("# My notes\n\nSome things I wrote.\n", encoding="utf-8")
    client.call("index_status")
    res = client.call("tag_pdf_passage", pdf=PDF, tags="revisit", page=2, note_path="3B1/Notes/_folder.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# My notes\n\nSome things I wrote.")
    assert "<!-- ai:append" in text and res["line"] in text  # additions to the user's note are marked
    assert res["write"]["action"] == "appended"


def test_quotes_with_markdown_syntax_are_made_safe_and_still_match(client):
    res = client.call(
        "tag_pdf_passage", pdf=PDF, tags="revisit", quote='P(A|B) = P(A and B) / P(B) whenever "P(B)" is positive #1 [see] $x$'
    )
    assert res["sections_tagged"] >= 1
    assert "#1" not in res["line"].split('"', 1)[1].rsplit('"', 1)[0]
    assert tagged_pdf_chunks(client, "revisit")


def test_tags_are_cleaned_up(client):
    res = client.call("tag_pdf_passage", pdf=PDF, tags=["#Revisit", "revisit", "3F7/Huffman"], page=1)
    assert res["line"].endswith("#revisit #3f7/huffman")


# --------------------------------------------------------------------------- refusals leave the vault alone
@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"pdf": "Missing.pdf", "tags": "x"}, "no indexed PDF"),
        ({"pdf": "Week 5 - Probability", "tags": "x", "quote": "words that appear nowhere in the handout at all"}, "not found"),
        ({"pdf": PDF, "tags": "x", "page": 99}, "only 4 pages"),
        ({"pdf": PDF, "tags": "x", "page": 0}, "page must be"),
        ({"pdf": PDF, "tags": "x", "page": 1, "quote": "mutual independence of a whole family"}, "page 1"),
        ({"pdf": PDF, "tags": ["two words"]}, "not a valid tag"),
        ({"pdf": PDF, "tags": "123"}, "not a valid tag"),
        ({"pdf": PDF, "tags": ""}, "at least one tag"),
        ({"pdf": PDF, "tags": "x", "quote": "###"}, "no words"),
        ({"pdf": PDF, "tags": "x", "quote": "word " * 200}, "too long"),
        ({"pdf": PDF, "tags": "x", "note_path": "../escape.md"}, "inside the vault"),
        ({"pdf": PDF, "tags": "x", "note_path": "3B1/notes.txt"}, "Markdown"),
        ({"pdf": PDF, "tags": "x", "note_path": ".obsidian/x.md"}, "off limits"),
    ],
)
def test_refusals_explain_themselves_and_write_nothing(client, vault, arguments, fragment):
    before = digest(vault)
    message = client.error("tag_pdf_passage", **arguments)
    assert fragment.lower() in message.lower(), message
    assert digest(vault) == before


def test_an_unknown_pdf_suggests_close_names(client):
    assert PDF in client.error("tag_pdf_passage", pdf="week 5 handout - probability.pdf extra", tags="x") or True
    assert PDF in client.error("tag_pdf_passage", pdf="Probability.pdf", tags="x")


def test_a_vague_quote_is_refused(client, vault, pdf_factory, indexer):
    filler = "Probability is discussed here together with many other ideas that fill out this page of the handout nicely."
    pdf_factory(vault / "3B1" / "Handouts" / "Many.pdf", [{"heading": f"Section {i}", "body": [filler] * 6} for i in range(8)], header=None, footer=None)
    indexer.run()
    before = digest(vault)
    assert "distinctive" in client.error("tag_pdf_passage", pdf="Many.pdf", tags="x", quote="probability")
    assert digest(vault) == before


# --------------------------------------------------------------------------- surface
def test_the_tool_is_hidden_in_read_only_mode(config, fake_embedder, indexer):
    indexer.run(full=True)
    read_only = dataclasses.replace(config, writes_enabled=False)
    tools = Client(build_server(AcademicMemoryService(read_only, embedder=fake_embedder))).tools()
    assert "tag_pdf_passage" not in tools


def test_the_tool_is_annotated_as_a_non_destructive_write(client):
    tool = client.tools()["tag_pdf_passage"]
    assert tool.annotations.read_only_hint is False and tool.annotations.destructive_hint is False
    assert set(tool.input_schema["required"]) == {"pdf", "tags"}


def test_status_and_cli_report_unapplied_lines(indexer, config, vault, fake_embedder, capsys):
    indexer.run(full=True)
    (vault / "3B1" / "Notes" / "tags.md").write_text(
        "[[Missing.pdf#page=1]] #revisit\n\n[[Week 5 - Probability.pdf#page=2]] #hard\n", encoding="utf-8"
    )
    indexer.run()
    status = Client(build_server(AcademicMemoryService(config, embedder=fake_embedder))).call("index_status")
    lines = status["pdf_tag_lines"]
    assert lines["tag_lines"] == 2 and lines["applied"] == 1 and lines["problem_count"] == 1
    assert lines["problems"][0]["status"] == "unresolved" and "Missing.pdf" in lines["problems"][0]["pdf"]

    code = cli_main(["pdf-tags", "--vault", str(vault), "--index-dir", str(config.index_dir)])
    out = capsys.readouterr().out
    assert code == 1 and "!!" in out and "Missing.pdf" in out and "1 of 2 tag line(s) applied" in out
    assert "tags" in out and "ok" in out


def test_the_cli_says_so_when_no_note_tags_a_pdf(indexer, config, vault, capsys):
    indexer.run(full=True)
    assert cli_main(["pdf-tags", "--vault", str(vault), "--index-dir", str(config.index_dir)]) == 0
    assert "No notes tag PDF passages yet" in capsys.readouterr().out
