from pathlib import Path

from acadmem.ingest.links import clean_inline, extract_inline_tags, extract_links
from acadmem.ingest.markdown import MarkdownExtractor, is_ai_generated, parse_markdown
from acadmem.models import CALLOUT, CODE, LIST, MATH, PARAGRAPH, TABLE

NOTE = """---
title: Bayes notes
tags: [probability, "#revision"]
ai_generated: false
---

Intro paragraph before any heading.

# Bayesian inference

Prior times likelihood. See [[Probability basics]] and ![[Week 5.pdf#page=17]].

## What I don't understand

Why is the evidence term a normaliser?

$$
p(\\theta|x) = \\frac{p(x|\\theta)p(\\theta)}{p(x)}

\\text{blank line above must not split this}
$$

### Worked example

- coin flip
- die roll

| a | b |
|---|---|
| 1 | 2 |

```python
# not a heading
x = 1
```

> [!note] Remember
> The posterior is proportional.

## Next topic #stats

Final words with a [link](https://example.com/page).
"""


def test_headings_form_a_section_tree_with_paths():
    parsed = parse_markdown(NOTE, "fallback")
    by_title = {s.title: s for s in parsed.sections}
    assert parsed.title == "Bayes notes"
    root = parsed.sections[0]
    assert root.level == 0 and root.heading_path == ()
    assert root.blocks[0].text.startswith("Intro paragraph")

    assert by_title["Bayesian inference"].heading_path == ("Bayesian inference",)
    dont = by_title["What I don't understand"]
    assert dont.heading_path == ("Bayesian inference", "What I don't understand")
    assert dont.parent_id == by_title["Bayesian inference"].section_id
    ex = by_title["Worked example"]
    assert ex.heading_path == ("Bayesian inference", "What I don't understand", "Worked example")
    assert ex.parent_id == dont.section_id
    # sibling at level 2 returns to level-1 parent
    nxt = by_title["Next topic #stats"]
    assert nxt.heading_path == ("Bayesian inference", "Next topic #stats")


def test_math_code_table_callout_are_atomic_blocks():
    parsed = parse_markdown(NOTE)
    blocks = [b for s in parsed.sections for b in s.blocks]
    kinds = [b.kind for b in blocks]
    assert kinds.count(MATH) == 1
    assert kinds.count(CODE) == 1
    assert kinds.count(TABLE) == 1
    assert kinds.count(CALLOUT) == 1
    assert LIST in kinds and PARAGRAPH in kinds
    math = next(b for b in blocks if b.kind == MATH)
    assert "blank line above must not split this" in math.text  # blank line did not split it
    assert math.text.startswith("$$") and math.text.rstrip().endswith("$$")
    code = next(b for b in blocks if b.kind == CODE)
    assert "# not a heading" in code.text
    assert all(s.title != "not a heading" for s in parsed.sections)


def test_links_embeds_and_tags_preserved():
    parsed = parse_markdown(NOTE)
    assert "Probability basics" in parsed.links
    assert "Week 5.pdf#page=17" in parsed.links
    assert "https://example.com/page" in parsed.links
    assert "probability" in parsed.tags and "revision" in parsed.tags
    assert "stats" in parsed.tags
    assert not is_ai_generated(parsed)


def test_block_text_is_verbatim_source_slice():
    parsed = parse_markdown(NOTE)
    para = next(b for s in parsed.sections for b in s.blocks if b.text.startswith("Prior times"))
    assert para.text == "Prior times likelihood. See [[Probability basics]] and ![[Week 5.pdf#page=17]]."


def test_dollar_in_code_fence_is_not_math():
    parsed = parse_markdown("# T\n\n```\n$$\nnot math\n$$\n```\n\nafter\n")
    kinds = [b.kind for s in parsed.sections for b in s.blocks]
    assert MATH not in kinds and CODE in kinds


def test_single_line_math_and_no_frontmatter_title_falls_back_to_h1():
    parsed = parse_markdown("# Heading Title\n\n$$ a^2 + b^2 = c^2 $$\n\ntext\n", "stem")
    assert parsed.title == "Heading Title"
    assert [b.kind for b in parsed.sections[0].blocks] == [MATH, PARAGRAPH]


def test_empty_and_heading_only_files():
    assert parse_markdown("", "e").sections[0].blocks == []
    p = parse_markdown("# Only a heading\n")
    assert [s.title for s in p.sections] == ["Only a heading"]
    assert p.sections[0].blocks == []


def test_malformed_frontmatter_does_not_crash():
    p = parse_markdown("---\ntitle: [unclosed\n---\n# H\n\nbody\n", "x")
    assert any(b.text == "body" for s in p.sections for b in s.blocks)


def test_ai_generated_flag():
    p = parse_markdown("---\nai_generated: true\n---\n# H\n\ntext\n")
    assert is_ai_generated(p)


def test_extractor_reads_file_with_bom(tmp_path: Path):
    f = tmp_path / "n.md"
    f.write_bytes(b"\xef\xbb\xbf# Title\n\nHello\n")
    parsed = MarkdownExtractor().parse(f)
    assert parsed.title == "Title"


def test_link_helpers():
    assert extract_links("[[A|alias]] and [[B#sec]] and ![[c.png]] [x](#local)") == ["A", "B#sec", "c.png"]
    assert extract_inline_tags("text #one and `#code` and $#math$ and [[x#y]]") == ["one"]
    assert clean_inline("**Bold** [[Page|shown]] `code` my_var") == "Bold shown code my_var"
