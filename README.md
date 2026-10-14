# acadmem

Local, single-user **academic memory** over an Obsidian vault: Codex can search your PDFs and notes by meaning,
read the surrounding context, and write new Markdown notes back into the vault.

Obsidian is the source of truth. The LanceDB index is a disposable derived artefact: delete it and
`acadmem rebuild` recreates it; `acadmem index` (or the watcher) only re-processes what changed.

```
Vault -> ingest (md/pdf) -> section tree -> structure-aware chunks -> local embeddings (bge-small)
      -> LanceDB -> search (vector | fts | hybrid + metadata filters) -> context expansion -> MCP -> Codex
```

Nothing leaves the machine: the embedding model runs locally and there are no network calls at query time.

## Setup

```powershell
uv sync
$env:ACADMEM_VAULT = 'C:\Users\tejas\OneDrive\University work\Work\Part IIA\Studies'
uv run acadmem rebuild      # first build (~10 min for ~100 PDFs; later runs are incremental)
uv run acadmem status
uv run acadmem search "why does the evidence term normalise the posterior" --module 3F8 --expand neighbors:1
```

The index lives in `%LOCALAPPDATA%\acadmem` (not in OneDrive, so sync cannot corrupt it). Override with
`--index-dir` / `ACADMEM_INDEX_DIR`. Every setting can also come from a TOML file (`--config` / `ACADMEM_CONFIG`) or
an `ACADMEM_<FIELD>` environment variable; see `src/acadmem/config.py`.

## Commands

| command | what it does |
| --- | --- |
| `acadmem index` | incremental update: new / modified / deleted files only |
| `acadmem rebuild` | drop the index and rebuild from the vault |
| `acadmem watch` | keep the index current while you work in Obsidian |
| `acadmem status` | counts per module and type, plus any files that failed |
| `acadmem pdf-tags` | the tag lines in your notes that tag PDF passages, and which could not be applied (and why) |
| `acadmem search` | query with filters (`--module --document-type --file-type --path-prefix --tag --modified-after --ai-generated`), `--mode vector\|fts\|hybrid`, `--expand`, `--json` |
| `acadmem eval` | run the retrieval benchmark (`uv run python eval/run.py` is the same thing) |

A file that cannot be parsed never aborts indexing: it is recorded (see `status`) and retried when it changes.

## Using it from Codex (MCP)

Everything Codex needs is in the repo's `codex/` folder, so a fresh clone can be set up in one step:

```powershell
uv sync
powershell -ExecutionPolicy Bypass -File codex\install.ps1 -Vault 'C:\path\to\your\vault'
```

It adds the server entry to `~/.codex/config.toml` (an existing entry is never overwritten) and puts the guidance from
`codex/AGENTS.md` into `~/.codex/AGENTS.md` between marker comments, so it can be re-run to update it and anything
else in that file is left alone. Edit the guidance in `codex/AGENTS.md`, not in your home folder. Restart Codex
afterwards. The entry it adds looks like this (`codex/config.example.toml`; single-quoted TOML strings keep the
backslashes literal):

```toml
[mcp_servers.academic_memory]
command = 'C:\Users\tejas\Projects\obsidian-knowledge\.venv\Scripts\python.exe'
args = ['-m', 'acadmem.mcp_server']
startup_timeout_sec = 120

[mcp_servers.academic_memory.env]
ACADMEM_VAULT = 'C:\Users\tejas\OneDrive\University work\Work\Part IIA\Studies'
```

The server is started through the virtual environment's `python.exe` rather than `acadmem-mcp.exe` because Windows
Application Control can block the generated `.exe` launchers (the same applies to `acadmem` itself: use
`python -m acadmem.cli ...` if `uv run acadmem` is refused). Add `'--read-only'` to `args` to expose no write tools, or
`'--no-watch'` to stop the server updating the index in the background. By default the server catches up on changes at start and then watches the vault.

### Tools

| tool | purpose |
| --- | --- |
| `search_academic_memory` | semantic search; filters, `mode`, and `expand` for context. Returns `source`, `page`, `section`, `text`, `score` (cosine similarity, see below), `chunk_id` |
| `get_source_chunk` | one chunk by id, full text and provenance |
| `get_context` | neighbours / whole section / parent section / surrounding pages of a hit |
| `get_document` | metadata and heading outline of a document (optionally its text) |
| `list_indexed_sources` | what is indexed, optionally filtered (e.g. every 3F7 paper) |
| `index_status` | counts, model, last update, files that failed |
| `create_note` | create a new `.md` note (always marked `ai_generated: true`) |
| `append_to_note` | append to an existing note |
| `replace_section` | replace the body of one heading in an existing note |
| `tag_pdf_passage` | tag a PDF page / passage by writing a tag line into a note (the PDF is untouched; see *Tagging PDFs*) |

### Tagging passages

Mark any part of a note with an ordinary Obsidian `#tag` and Codex can retrieve exactly those passages later.

```markdown
## Huffman coding

Huffman codes are built by merging the two least likely symbols.

I still don't see why the tree is optimal. #revisit      <- tags this paragraph only
```

* **A tag covers the paragraph it is written in.** The same goes for a list, table, callout or quote: the whole block.
  Neighbouring paragraphs are *not* tagged, and a tagged paragraph is kept as its own passage, so a tag search returns it
  alone with its note, heading and page-free provenance.
* **A tag on a line by itself tags the paragraph above it:** put `#exam` (or `#exam #revisit`) on its own line under the
  paragraph. At the very top of a section, a tag line tags the paragraph below it, and a tag on a heading line
  (`## Huffman #revisit`) tags the first paragraph under that heading.
* **Tags in the note's frontmatter (`tags: [revision]`) cover the whole note.**
* Tags are case-insensitive, have no spaces, and may nest (`#3F7/huffman`) or look like module codes (`#3F7`).
  Purely numeric ones (`#123`) and anything inside code or math are not tags.
* Ask Codex things like *"show me everything I tagged #revisit in 3F7"* (it lists the tagged passages in vault order) or
  *"explain the passages I tagged #exam about entropy"* (it ranks the tagged passages by relevance). From the CLI,
  `acadmem search "entropy" --tag exam`.
* The index follows your edits: add or remove a tag in Obsidian and it is picked up within a second or two while the
  server (or `acadmem watch`) is running. Notes indexed before this feature existed keep their old note-level tags until
  you run `acadmem rebuild`.

#### Tagging PDFs

A PDF has no place to hold a tag, and this system never edits your PDFs. So a PDF is tagged by a line in one of your
**notes** that points at it. The tag lives in your vault (it survives index rebuilds and PDF re-indexing), and Codex
retrieves the PDF section through the same `tags` filter as everything else.

| write this in a note | what gets tagged |
| --- | --- |
| `[[Week 5.pdf]] #exam-prep` | every section of the PDF |
| `[[Week 5.pdf#page=3]] #revisit` | the sections on page 3 (`#page=2-4` for a range) |
| `[[Week 5.pdf#page=3]] "posterior is proportional" #revisit` | the section on page 3 containing those words (drop the page to search the whole PDF) |
| `[[Week 5.pdf]] "Bayes' theorem" #revisit` | words may also match a section's heading |

* **The smallest unit is an indexed section** (usually a paragraph or a few). Quoting one sentence tags the section around it.
* **Quotes** are compared on words only: case, punctuation, line breaks, hyphenation, `C â‚` vs `Câ‚`, table pipes and
  `<sup>` tags do not matter. A quote that runs over a section boundary tags both sections. A quote that matches more than
  5 sections is refused as too vague rather than tagging half a handout.
* **PDF++ users:** the plugin's "copy as quote" format works as is. Select text in the PDF, copy the link, add a tag:

  ```markdown
  > the occurrence of one does not change the probability of the other

  [[Week 5.pdf#page=3&selection=4,0,5,20|Week 5, page 3]] #revisit
  ```

  The page comes from the link and the words from the quote above it. A PDF++ callout (`> [!PDF|yellow] [[...]]` with
  the selected text as its body) works too. The `selection=` numbers are positions in the viewer's internal text list,
  not words, so they cannot be mapped onto our extraction reliably and are ignored. If only the link's display text holds
  the selected words, they are used, and if they are not found the page is tagged instead. (PDF++ is not installed in
  the vault this was developed against, so these formats follow its documentation and are covered by tests with those
  exact link shapes; they have not been exercised against the live plugin.)
* **Ask Codex to do it:** *"tag the part of the Week 5 probability handout about independence as #revisit"*. The
  `tag_pdf_passage` tool checks the PDF and the words against the index first, then writes the line into
  `<module>/PDF tags.md` (created if needed) and tells Codex which sections it tagged. If nothing matches it writes
  nothing and says why.
* **Which part of the section:** a quote tags the paragraph(s) it actually hit, not the whole section. Search results
  still return the whole section as `text` (so the passage is read in context) and add `tagged_text` holding exactly
  the tagged paragraph(s). No `tagged_text` means the whole section is tagged (page tags, whole-PDF tags, notes, or a
  quote that only matches a heading). Chunking, embeddings and ranking are unchanged; spans live in a small
  `tag_spans` table and are recomputed with the tags.
* **See what happened:** `acadmem pdf-tags` lists every such line and whether it was applied (and why not): PDF not
  found, words not found on that page, page beyond the end. `index_status` shows the same summary.
* Measured on 150 random 12-word selections from 6 real vault PDFs (raw PDF text, as a viewer would copy it; page + quote):
  93% of body-text selections matched, 98% of those to a single section, none to the wrong section (`eval/smoke_pdf_tags.py`).
  What fails is text that is not in the extracted text at all: chart axis numbers, the running page header (stripped
  as boilerplate on purpose), and some formula glyphs the extractor encodes differently. Scanned PDFs without a text layer
  cannot be quoted (they are not indexed either).

### Reading `score`

Every hit carries the same `score` in every mode (`vector`, `fts`, `hybrid`): the **cosine similarity between the
query embedding and the chunk's embedding**, in -1..1 (in practice ~0.4-0.9 for `bge-small`; it is a compressed scale,
so 0.7 is not "70%"). Keyword-only hits get one too, so the number is always comparable. What *ranks* the list is
separate: reciprocal-rank fusion in `hybrid` (at most ~0.033), BM25 in `fts`, cosine in `vector`. That value is kept as
`rank_score` in `acadmem search --json` and `SearchResult`, but is deliberately not shown to the model: it is only an ordering
device, and read as a relevance it looks like a terrible match (the bug this replaces). Consequence: in `hybrid` the
`score`s need not strictly decrease down the list.

What the number can and cannot tell you (`uv run python eval/score_calibration.py`; 44 labelled benchmark queries,
top 10 hybrid hits each, plus 10 obviously off-topic and 6 engineering-but-absent queries; draft labels, small samples):

| | cosine `score` (hybrid) |
| --- | --- |
| hit that matches a benchmark label | median 0.74 (p10 0.67, p90 0.81) |
| other top-10 hits (mostly neighbouring chunks of the same topic) | median 0.71 (p10 0.61, p90 0.78) |
| best hit for the answerable queries (top 1) | median 0.79 (p10 0.72) |
| best hit for 10 off-topic queries (recipes, football, ...) | 0.52-0.68 (median 0.59) |
| best hit for 6 engineering topics the vault lacks (thermodynamics, concrete, ...) | 0.64-0.77 |

* It separates "this query is about something the vault covers" (top hit usually >= 0.70; 9% of the answerable queries fell
  below) from "obviously unrelated" (best hit <= 0.68 for all 10 off-topic queries).
* It does **not** separate a relevant chunk from an irrelevant one *within* a result list (AUC 0.66: the relevant page
  and the pages next to it score alike), nor topics adjacent to the vault's from topics inside it. So there is no
  `confidence` flag and no rescaling: a threshold would be invented, not supported. Read the text.
* One-sided only: short or vague, filtered queries can score low even when answered ("independence" with a tag filter: 0.54; "the things I did not understand" with a Markdown filter: 0.52).

### What the AI can and cannot write

Writes are deliberately narrow. The server **cannot** delete, rename, move or overwrite anything, cannot write
anything but `.md`, and cannot leave the vault (absolute paths, `..`, drive letters, dot-folders such as `.obsidian`,
reserved Windows names and symlink/junction escapes are all refused; a test asserts the vault is byte-identical after
every refused call). Additional guarantees:

* **Atomic**: a note is written to a temp file and then linked or replaced into place; Obsidian never sees half a file.
* **No lost edits**: if the note changed on disk (e.g. you typed in Obsidian) between read and write, the write is refused.
* **Recoverable**: before `append_to_note` / `replace_section` changes a note, the previous version is copied to
  `<index dir>\backups` (outside the vault; the newest 200 are kept).
* **Labelled, with limits**: `create_note` forces `ai_generated: true` frontmatter, so notes *created by this server* can
  be filtered (`ai_generated=true` finds them, `ai_generated=false` excludes them). The flag is **not** a guarantee that
  `ai_generated=false` is "only my own writing":
  * appends and replacements inside *your* notes leave your frontmatter alone and add only an invisible
    `<!-- ai:append DATE -->` comment, so those notes still count as `ai_generated=false`;
  * notes written by any other tool (for example the separate `obsidian` MCP server) carry no flag at all.
* **Immediately searchable**: the touched file is re-indexed before the call returns.

### Running next to the `obsidian` MCP server

If `~/.codex/config.toml` also has the `obsidian` server (`mcp-obsidian-vault`), the two are complementary: it does
note housekeeping (read/create/update/delete, daily notes, tags, wikilinks and backlinks, git sync, a task and
decision queue) with plain-text search over `.md` files only, so it **cannot see PDFs**; this server does semantic search
over PDFs and notes. They share the vault safely (my watcher picks up its writes within a second, and notes it deletes
go to `.trash/`, which is ignored).

* Its bookkeeping lives at the vault root (`Tasks/`, `Decisions/`, `Daily Notes/`, `DASHBOARD.md`). Those entries are
  **neither indexed nor writable here** (`ignored_root_entries` in the config; root level only, so a `Tasks` folder
  inside a module is still study material).
* Its write tools (`delete_note`, replace-mode `update_note`, `create_note(overwrite=true)`) are **not** covered by the
  guarantees above: no backup, no refusal to overwrite, no AI flag. Restricting them is a choice for the Codex config.
* Neither server tells the model about the other. `codex/AGENTS.md` does (which one to use for study questions and
  which for housekeeping, plus the tagging, maths-formatting and databook rules); `codex/install.ps1` installs it.

## Design decisions

* **PDF extraction**: `pymupdf4llm` by default (fast, keeps page numbers and heading structure); Docling is available
  behind the same `PDFExtractor` interface. Chosen from a bake-off on real vault PDFs (`eval/pdf_bakeoff.py`).
  Scanned pages (no text layer) are **not OCR'd by default**, with one opt-in exception: PDFs under the folders in
  `ocr_paths` (default `Databook`) that contain scanned pages go through Docling with OCR and formula recognition
  instead. That gives section headings, prose, lists and equations as LaTeX, but it is slow (roughly 5 s a page plus a
  few seconds per formula, so a 26-page scan takes several minutes, once per file change), diagrams and text inside
  figures are still lost, and printed scans read far better than handwriting. A PDF in those folders that already has
  text on every page keeps the fast path. Set `ocr_paths` to a comma-separated list of vault folders, or empty to turn
  it off. A file already indexed is only re-processed when its content changes, so after changing `ocr_paths` clear
  that document from the index (or `acadmem rebuild`).
* **Embeddings**: `BAAI/bge-small-en-v1.5` on CPU, with the bge query instruction on queries only.
* **Chunking**: follows the document's heading tree; small sections are merged and large ones split on token counts,
  with provenance (file, pages, heading breadcrumb) kept on every chunk.
* **Hybrid by default**: vector search and BM25 (LanceDB native FTS) fused by reciprocal rank fusion. On the benchmark each
  single method had weak spots (vector: PDF, Markdown and cross-document queries at 80-86% Recall@5; FTS: paraphrased
  queries at 86%), and the fusion had none. The FTS index covers the heading breadcrumb
  as well as the text, so searching a section title works.
* **Filters are prefilters** (SQL `WHERE` before the vector search), so a filtered query returns the best matches *within*
  the filter instead of the best matches overall minus whatever the filter removed.

## Tests and benchmark

```powershell
uv run pytest                  # everything (~4 min); includes two tests that load the real model / Docling
uv run pytest -m "not slow"    # fast subset
uv run acadmem eval --details  # Recall@5 per mode and category, failing queries
uv run acadmem eval --check-labels
```

`eval/score_calibration.py` measures how well `score` separates relevant from irrelevant hits (see *Reading `score`*).

`eval/benchmark.yaml` holds 45 queries over seven categories (conceptual, paraphrased, PDF, Markdown, cross-document,
metadata-constrained, context-expansion). **The labels are a draft written from the PDFs' text and need your review**:
each is a claim that "this page answers this query", and every `evidence` phrase is machine-checked against the source
(`--check-labels`), but whether a page is the *best* answer is your call. The vault has no Markdown notes yet, so
`eval/fixtures/notes/` are overlaid on a *copy* of the vault and index when evaluating; the real vault and index are
never touched.

Metrics: Recall@k, source-level recall, MRR, metadata-filter correctness, context-expansion lift, latency, per category.

### Results (real vault: 96 PDFs, 2,840 chunks; 45 draft queries)

| | vector | fts | hybrid (default) |
| --- | --- | --- | --- |
| Recall@5 | 93.3% | 92.2% | **100%** |
| Source recall@5 | 95.6% | 94.4% | **100%** |
| MRR | 0.777 | 0.823 | **0.846** |
| Metadata filter correctness | 100% | 100% | 100% |
| Avg latency | ~110 ms | ~45 ms | ~55 ms |

Read these with care. There are only 45 queries (one query is 2.2 points), the labels were written by the same author as
the system and are unreviewed, and several hybrid hits sit at rank 3-5, so 100% is a thin margin rather than headroom.
On the context-expansion queries, expanding the hits lifted Recall from 91.7% to 100% for hybrid (75% to 100% for vector). Chunk sizes were
compared across four settings (`eval/tune_chunking.py`); the defaults (min 64 / max 384 / overlap 48 tokens) did best, and
smaller chunks never beat them clearly, so they were kept.

Known limitation: there is no reranker, so very chatty phrasing ("what does my material say about ...") ranks the best
chunk lower than the terse form ("Huffman coding"). Cold start is ~20 s the first time the model loads; the MCP server
warms it up in the background at launch so the first question is not slow.

`uv run python eval/smoke_mcp.py` launches `acadmem-mcp` the way Codex does and asks real questions of the real index
(read-only: it never calls a write tool).
