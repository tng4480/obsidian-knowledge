## Study notes and material (Obsidian vault)

For questions about my university study material (lectures, handouts, worked solutions, Tripos papers, my notes),
use the academic_memory server: search_academic_memory first, then get_context / get_document for more. It is the
only tool that can read the PDFs that are inside the vault.

This does NOT apply to a file I attach or point you to that is outside the vault (a lab handout in Downloads or the
project folder, a path I give you, an attachment): read that file directly with your normal file and PDF tools. Never
stop or ask me to add a file to the vault or paste its contents just because it is not in the index. If the file is
not in the index, read it from its path; use academic_memory only to look up related course material.

When I ask for things I tagged (e.g. "everything I tagged #revisit in 3F7"), call search_academic_memory with the
`tags` filter (and the module filter if I named one). With no search words it lists the tagged paragraphs; with search
words it searches only inside them.

PDFs cannot hold tags themselves. To tag part of a PDF (a page, or a passage by a few of its words), call
tag_pdf_passage; it writes a tag line into a note and the PDF is never changed. Tagged PDF sections come back through
the same `tags` filter as tagged notes. A tagged PDF result has the whole section in `text`; if only part of it was
tagged, `tagged_text` is exactly that part.

Use the obsidian server only for vault housekeeping: links and backlinks, tags, daily notes, tasks and decisions,
git sync. Never use obsidian's search_vault for study questions.

## Writing notes into the vault

This section applies ONLY to text you save into a vault file with a write tool (create_note, append_to_note, replace_section, or the obsidian server's note tools). It does NOT apply to your replies to me in chat: in chat, format maths exactly as you normally would, and never use `$...$` or `$$...$$` there.

Inside a vault note, maths uses Obsidian's delimiters: $...$ inline and $$...$$ on their own lines for display equations. Never use `\(...\)` or `\[...\]` in a note; Obsidian does not render them reliably. Write each LaTeX backslash once (`\frac`, `\partial`), never doubled, and re-read the saved note to check. Keep tables simple (no maths or line breaks inside cells; use a list for anything complicated). Link a source with `[[file.pdf#page=N]]`.

## Databook (formulae, data and methods)

The CUED Engineering Databook is indexed as module `Databook` (file `Databook/Handouts/CUED_Full_Databook.pdf`, 227 pages: materials, thermofluids, linear algebra and so on). When a question involves a formula, a method, a table of properties or a constant, search it first with search_academic_memory (module "Databook"), then also search the course material as usual.

- If the databook has the method, formula or data, use that version, not one recalled from memory, and keep its notation and units.
- Always say where it came from: "CUED Databook, PDF p. N" plus the section heading in the result (and the databook's own page number or sheet title if it appears in the text, e.g. "THERMOFLUIDS_DATA ... 32"). Quote the relevant line from the retrieved text.
- If the databook has nothing relevant, say so before using other sources. Never cite a databook page you did not retrieve.
- About 38 pages (mostly pp. 4-26, 48-50 and 215-227) are images with no text layer and cannot be searched; if the answer might be there, tell me to check the printed databook.
