"""Ingestion interfaces.

An ``Extractor`` turns one file into :class:`~acadmem.models.ParsedContent`. New
formats are added by registering another extractor keyed by file extension; the
rest of the pipeline (chunking, embedding, indexing, retrieval) is format-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from ..models import ParsedContent


@runtime_checkable
class Extractor(Protocol):
    extensions: ClassVar[tuple[str, ...]]
    file_type: ClassVar[str]

    def parse(self, path: Path) -> ParsedContent:  # pragma: no cover - protocol
        ...


class ExtractorRegistry:
    """Maps file extensions to extractors."""

    def __init__(self) -> None:
        self._by_ext: dict[str, Extractor] = {}

    def register(self, extractor: Extractor) -> None:
        for ext in extractor.extensions:
            self._by_ext[ext.lower()] = extractor

    def for_path(self, path: Path | str) -> Extractor | None:
        return self._by_ext.get(Path(path).suffix.lower())

    @property
    def extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_ext))
