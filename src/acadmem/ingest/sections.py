"""Format-independent construction of the section tree.

Both the Markdown parser and the PDF structure builder drive a ``SectionBuilder``,
so the resulting hierarchy (``heading_path`` + ``parent_id``) has identical
semantics regardless of the source format.
"""

from __future__ import annotations

from ..models import Block, Section


class SectionBuilder:
    """Builds a flat, document-ordered section list with parent links.

    Section 0 is a root/preamble section (level 0, empty ``heading_path``) that
    receives any content appearing before the first heading.
    """

    def __init__(self, root_title: str = "") -> None:
        self._sections: list[Section] = []
        self._stack: list[Section] = []
        root = self._new(root_title, 0, parent=None)
        self._stack.append(root)

    def _new(self, title: str, level: int, parent: Section | None) -> Section:
        section = Section(
            section_id=f"s{len(self._sections)}",
            title=title,
            level=level,
            heading_path=(parent.heading_path + (title,)) if parent is not None else (),
            parent_id=parent.section_id if parent is not None else None,
        )
        self._sections.append(section)
        return section

    @property
    def current(self) -> Section:
        return self._stack[-1]

    def open(self, title: str, level: int) -> Section:
        """Open a new section at ``level`` (>=1), closing any siblings/deeper sections."""
        level = max(1, level)
        while len(self._stack) > 1 and self._stack[-1].level >= level:
            self._stack.pop()
        section = self._new(title, level, parent=self._stack[-1])
        self._stack.append(section)
        return section

    def add(self, block: Block) -> None:
        if block.text.strip():
            self.current.blocks.append(block)

    def build(self) -> list[Section]:
        """Return sections; the root is dropped if it is empty and has children."""
        sections = list(self._sections)
        root = sections[0]
        if not root.blocks and len(sections) > 1:
            # Re-parent first-level children to None so they have no dangling parent.
            for s in sections[1:]:
                if s.parent_id == root.section_id:
                    s.parent_id = None
            sections = sections[1:]
        return sections
