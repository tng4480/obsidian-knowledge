"""Link and tag extraction (Obsidian wikilinks, embeds, Markdown links, #tags)."""

from __future__ import annotations

import re

_WIKILINK = re.compile(r"!?\[\[([^\]\n]+?)\]\]")
_MDLINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# Obsidian tags: letters, digits, "_", "-" and "/" (nesting), with at least one non-digit, so "#3F7" and "#3f7/huffman"
# are tags but "#123" (an issue number) is not. The lookbehind skips URL anchors, "&#123;" and "##".
_TAG = re.compile(r"(?<![\w/#&\[(])#((?=[\w/-]*[^\W\d])[\w/-]+)")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,}).*?(?:^\s*\2[`~]*\s*$|\Z)", re.M | re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_MATH_BLOCK = re.compile(r"\$\$.*?\$\$", re.S)
_MATH_INLINE = re.compile(r"(?<!\\)\$[^$\n]+\$")


def strip_code_and_math(text: str) -> str:
    """Blank out fenced code, inline code and math so syntax inside them is ignored."""
    text = _FENCE.sub(" ", text)
    text = _MATH_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    return _MATH_INLINE.sub(" ", text)


def extract_links(text: str) -> list[str]:
    """Return link targets in order of first appearance.

    ``[[Note|alias]]`` -> ``Note``; ``![[file.pdf#page=3]]`` keeps its anchor so a
    note can be traced to an exact PDF page. External ``http`` links are kept as-is.
    """
    found: list[str] = []
    for m in _WIKILINK.finditer(text):
        target = m.group(1).split("|", 1)[0].strip()
        if target:
            found.append(target)
    for m in _MDLINK.finditer(text):
        target = m.group(1).strip()
        if target and not target.startswith("#"):
            found.append(target)
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_inline_tags(text: str) -> list[str]:
    """Inline ``#tags`` outside code and math."""
    stripped = strip_code_and_math(text)
    stripped = _WIKILINK.sub(" ", stripped)
    stripped = _MDLINK.sub(" ", stripped)
    seen: dict[str, None] = {}
    for m in _TAG.finditer(stripped):
        seen.setdefault(m.group(1), None)
    return list(seen)


def is_tag_only(text: str) -> bool:
    """True when ``text`` is nothing but one or more ``#tags`` (a "tag line" such as ``#revisit #exam``)."""
    stripped = strip_code_and_math(text)
    if not _TAG.search(stripped):
        return False
    return not _TAG.sub(" ", stripped).strip(" \t\r\n,;.")


_WIKI_TEXT = re.compile(r"\[\[([^\]|\n]+)\|([^\]\n]+)\]\]")
_WIKI_PLAIN = re.compile(r"\[\[([^\]\n]+)\]\]")
_MDLINK_TEXT = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_EMPHASIS = re.compile(r"(\*\*|\*|`|==|~~)")


def clean_inline(text: str) -> str:
    """Strip Markdown/Obsidian inline syntax for use in heading titles."""
    text = _WIKI_TEXT.sub(r"\2", text)
    text = _WIKI_PLAIN.sub(lambda m: m.group(1).split("#")[0] or m.group(1), text)
    text = _MDLINK_TEXT.sub(r"\1", text)
    text = _EMPHASIS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
