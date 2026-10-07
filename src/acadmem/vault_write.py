"""Guarded Markdown writes into the Obsidian vault (used by the MCP write tools).

The vault is the source of truth and belongs to the user, so every write goes through these guard rails:

* **Confinement**: paths are vault-relative; absolute paths, drive letters, ``..``, NULs, hidden/ignored folders
  (``.obsidian``, ``.git``, ``.trash`` ...), symlinks/junctions that leave the vault and Windows-illegal names are refused.
* **Markdown only**: only ``.md`` files can be created or edited. PDFs and every other file type are untouchable.
  There is no delete operation, and ``create_note`` never overwrites.
* **Atomic**: content is written to a temp file in the target folder and moved into place, so Obsidian (and
  OneDrive) never observe a half-written note.
* **No lost updates**: if the note changes on disk between reading and writing (you were typing in Obsidian),
  the write is refused rather than clobbering your edit.
* **Recoverable**: before touching an existing note, a copy is saved *outside* the vault (``<index>/backups``).
* **Marked**: AI-created notes carry ``ai_generated: true`` frontmatter so they can be filtered in or out. Text
  an AI adds to a note you wrote is preceded by an invisible ``<!-- ai ... -->`` marker.
* **Immediately searchable**: the touched file is re-indexed before the call returns.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from .config import Config
from .ingest.discovery import FileUnavailableError, read_bytes, to_source_path

if TYPE_CHECKING:  # pragma: no cover
    from .indexer import Indexer

log = logging.getLogger(__name__)

_ILLEGAL_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_ATX = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
MAX_PATH_CHARS = 240
MAX_PART_CHARS = 120


# --------------------------------------------------------------------------- errors / results
class WriteError(Exception):
    """A refused or failed write. The message is safe and useful to show to the model."""


class PathNotAllowed(WriteError): ...


class NoteExists(WriteError): ...


class NoteNotFound(WriteError): ...


class SectionNotFound(WriteError): ...


class AmbiguousSection(WriteError): ...


class WriteConflict(WriteError): ...


class ContentRejected(WriteError): ...


@dataclass
class WriteResult:
    path: str  # vault-relative source path
    action: str  # created | appended | replaced_section | added_section
    bytes_written: int
    indexed: str  # indexed | ignored | error | skipped
    ai_generated: bool
    backup: str | None = None
    index_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Heading:
    line: int
    level: int
    title: str
    path: tuple[str, ...]


# --------------------------------------------------------------------------- markdown helpers
def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("**", "").replace("__", "")).strip().casefold()


def scan_headings(lines: list[str]) -> list[_Heading]:
    """ATX headings outside frontmatter, fenced code and ``$$`` math, with their ancestor path."""
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                start = i + 1
                break
    out: list[_Heading] = []
    stack: list[tuple[int, str]] = []
    fence: str | None = None
    in_math = False
    for i in range(start, len(lines)):
        line = lines[i]
        m = _FENCE.match(line)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        if line.strip() == "$$":
            in_math = not in_math
            continue
        if in_math:
            continue
        h = _ATX.match(line)
        if not h:
            continue
        level, title = len(h.group(1)), h.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        out.append(_Heading(line=i, level=level, title=title, path=tuple(t for _, t in stack)))
    return out


def find_section(headings: list[_Heading], wanted: str) -> _Heading:
    """Resolve ``"Title"`` or ``"Parent > Title"`` to exactly one heading."""
    parts = [_norm_title(p) for p in re.split(r"\s*>\s*", wanted.strip()) if p.strip()]
    if not parts:
        raise SectionNotFound("heading must not be empty")
    matches = [
        h
        for h in headings
        if len(h.path) >= len(parts) and [_norm_title(p) for p in h.path[-len(parts):]] == parts
    ]
    if not matches:
        available = "; ".join(" > ".join(h.path) for h in headings) or "(the note has no headings)"
        raise SectionNotFound(f"No heading matches {wanted!r}. Headings in this note: {available}")
    if len(matches) > 1:
        options = "; ".join(f"line {h.line + 1}: {' > '.join(h.path)}" for h in matches)
        raise AmbiguousSection(f"{len(matches)} headings match {wanted!r} ({options}). Give the full path, e.g. 'Parent > Title'.")
    return matches[0]


def _detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


# --------------------------------------------------------------------------- writer
class VaultWriter:
    def __init__(self, config: Config, indexer: "Indexer | None" = None) -> None:
        self.config = config
        self.indexer = indexer
        self.vault = Path(config.vault_path)

    # ---------------------------------------------------------------- guards
    def _check_enabled(self) -> None:
        if not self.config.writes_enabled:
            raise WriteError("Writing to the vault is disabled (writes_enabled = false).")

    def _check_content(self, content: str, what: str = "content") -> str:
        if not isinstance(content, str):
            raise ContentRejected(f"{what} must be text")
        if "\x00" in content:
            raise ContentRejected(f"{what} contains a NUL character")
        if not content.strip():
            raise ContentRejected(f"{what} must not be empty")
        if len(content.encode("utf-8")) > self.config.write_max_bytes:
            raise ContentRejected(f"{what} is larger than the {self.config.write_max_bytes}-byte limit")
        return content

    def resolve(self, path: str, *, for_create: bool = False) -> tuple[Path, str]:
        """Validate a vault-relative path; return ``(absolute path, canonical source path)``."""
        if not isinstance(path, str) or not path.strip():
            raise PathNotAllowed("path must be a non-empty vault-relative path such as '3F7/Notes/Entropy.md'")
        raw = path.strip().replace("\\", "/")
        if "\x00" in raw or len(raw) > MAX_PATH_CHARS:
            raise PathNotAllowed("invalid path")
        if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
            raise PathNotAllowed("path must be relative to the vault root (no absolute paths or drive letters)")
        parts = [p for p in raw.split("/") if p not in ("", ".")]
        if not parts:
            raise PathNotAllowed("path must name a note")
        ignored = {d.casefold() for d in self.config.ignored_dirs}
        if parts[0].casefold() in {e.casefold() for e in self.config.ignored_root_entries}:
            raise PathNotAllowed(
                f"{parts[0]!r} is reserved for the obsidian server's task/decision/daily-note bookkeeping and is not "
                "indexed here; put study notes under a module folder such as '3F7/Notes/'"
            )
        for part in parts:
            if part == "..":
                raise PathNotAllowed("path must stay inside the vault ('..' is not allowed)")
            if part.startswith("."):
                raise PathNotAllowed(f"hidden/system folders and files are off limits: {part!r}")
            if part.casefold() in ignored:
                raise PathNotAllowed(f"{part!r} is an ignored folder")
            if _ILLEGAL_CHARS.search(part) or part.endswith((".", " ")):
                raise PathNotAllowed(f"illegal character or name: {part!r}")
            if len(part) > MAX_PART_CHARS:
                raise PathNotAllowed(f"path component too long: {part[:30]!r}...")
            if part.split(".")[0].upper() in _RESERVED:
                raise PathNotAllowed(f"reserved name: {part!r}")

        filename = parts[-1]
        suffix = Path(filename).suffix
        if suffix == "":
            filename += ".md"
        elif suffix.lower() != ".md":
            raise PathNotAllowed(f"only Markdown (.md) notes can be written, not {suffix!r}")
        parts[-1] = filename

        # Re-use the on-disk casing of folders/files that already exist, so one note never gets two identities.
        current = self.vault
        canonical: list[str] = []
        for i, part in enumerate(parts):
            actual = part
            if current.is_dir():
                try:
                    for entry in os.scandir(current):
                        if entry.name.casefold() == part.casefold():
                            actual = entry.name
                            break
                except OSError:
                    pass
            canonical.append(actual)
            current = current / actual
            if i < len(parts) - 1 and current.exists() and not current.is_dir():
                raise PathNotAllowed(f"{'/'.join(canonical)!r} is a file, not a folder")

        absolute = self.vault.joinpath(*canonical)
        vault_real = os.path.normcase(os.path.realpath(self.vault))
        target_real = os.path.normcase(os.path.realpath(absolute))
        try:
            inside = os.path.commonpath([vault_real, target_real]) == vault_real and target_real != vault_real
        except ValueError:  # different drives
            inside = False
        if not inside:
            raise PathNotAllowed("path resolves outside the vault")
        if absolute.exists() and not absolute.is_file():
            raise PathNotAllowed("path is not a file")
        return absolute, to_source_path(self.vault, absolute)

    # ---------------------------------------------------------------- IO helpers
    @staticmethod
    def _stat_key(path: Path) -> tuple[int, int]:
        st = path.stat()
        return st.st_size, st.st_mtime_ns

    def _read_existing(self, absolute: Path, source_path: str) -> tuple[str, tuple[int, int]]:
        if not absolute.is_file():
            raise NoteNotFound(f"No such note: {source_path}. Use create_note to make a new one.")
        before = self._stat_key(absolute)
        try:
            data = read_bytes(absolute)
        except FileUnavailableError as exc:
            raise WriteError(f"{source_path} is not available locally yet (cloud placeholder): {exc}") from exc
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise WriteError(f"{source_path} is not valid UTF-8 text; refusing to modify it") from exc
        return text, before

    def _backup(self, absolute: Path, source_path: str) -> str:
        directory = self.config.backup_dir
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target = directory / f"{stamp}__{source_path.replace('/', '__')}"
        shutil.copy2(absolute, target)
        keep = max(1, self.config.backup_keep)
        old = sorted(directory.iterdir(), key=lambda p: p.name)[:-keep]
        for p in old:
            try:
                p.unlink()
            except OSError:  # pragma: no cover
                pass
        return str(target)

    def _atomic_replace(self, absolute: Path, text: str, expected: tuple[int, int]) -> int:
        data = text.encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(prefix=".acadmem-", suffix=".tmp", dir=absolute.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if self._stat_key(absolute) != expected:
                raise WriteConflict(
                    f"{absolute.name} changed while it was being edited (probably in Obsidian). Nothing was written; "
                    "re-read the note and try again."
                )
            os.replace(tmp, absolute)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass
        return len(data)

    def _atomic_create(self, absolute: Path, text: str) -> int:
        data = text.encode("utf-8")
        absolute.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".acadmem-", suffix=".tmp", dir=absolute.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp, absolute)  # atomic and fails if the target exists: never overwrites
            except FileExistsError as exc:
                raise NoteExists(f"{absolute.name} already exists") from exc
            except OSError:
                # Filesystem without hard links: fall back to check-then-rename (still never replaces a file we saw).
                if absolute.exists():
                    raise NoteExists(f"{absolute.name} already exists")
                os.replace(tmp, absolute)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass
        return len(data)

    def _reindex(self, source_path: str) -> tuple[str, str | None]:
        if self.indexer is None:
            return "skipped", None
        try:
            return self.indexer.sync_path(source_path), None
        except Exception as exc:  # the note is safely written; indexing can be retried by the watcher / `acadmem index`
            log.exception("re-index after write failed for %s", source_path)
            return "error", f"{type(exc).__name__}: {exc}"

    def _result(self, source_path: str, action: str, nbytes: int, ai: bool, backup: str | None = None) -> WriteResult:
        outcome, err = self._reindex(source_path)
        return WriteResult(path=source_path, action=action, bytes_written=nbytes, indexed=outcome, ai_generated=ai, backup=backup, index_error=err)

    @staticmethod
    def _is_ai_note(text: str) -> bool:
        try:
            return bool(frontmatter.loads(text).metadata.get("ai_generated"))
        except Exception:
            return False

    @staticmethod
    def _marker(action: str) -> str:
        return f"<!-- ai:{action} {datetime.now(timezone.utc).strftime('%Y-%m-%d')} -->"

    # ---------------------------------------------------------------- operations
    def create_note(self, path: str, content: str, tags: list[str] | None = None) -> WriteResult:
        """Create a new note (refuses if it exists). Frontmatter gets ``ai_generated: true`` and the given tags."""
        self._check_enabled()
        self._check_content(content)
        absolute, source_path = self.resolve(path, for_create=True)
        if absolute.exists():
            raise NoteExists(f"{source_path} already exists. Use append_to_note or replace_section to change it.")

        text = content.replace("\r\n", "\n")
        try:
            post = frontmatter.loads(text)
        except Exception as exc:
            raise ContentRejected(f"content has malformed YAML frontmatter: {exc}") from exc
        meta = dict(post.metadata)
        meta["ai_generated"] = True
        if tags:
            existing = meta.get("tags") or []
            existing = [existing] if isinstance(existing, str) else list(existing)
            meta["tags"] = existing + [t.lstrip("#") for t in tags if t.lstrip("#") not in existing]
        post.metadata = meta
        body = post.content.strip("\n")
        if not body.strip():
            raise ContentRejected("content has frontmatter but no body")
        out = frontmatter.dumps(post).rstrip("\n") + "\n"
        if self.config.write_max_bytes and len(out.encode("utf-8")) > self.config.write_max_bytes:
            raise ContentRejected("content is too large")

        nbytes = self._atomic_create(absolute, out)
        return self._result(source_path, "created", nbytes, ai=True)

    def append_to_note(self, path: str, content: str) -> WriteResult:
        """Add text to the end of an existing note."""
        self._check_enabled()
        self._check_content(content)
        if _has_frontmatter(content.lstrip("\ufeff")):
            raise ContentRejected("appended content must not start with YAML frontmatter")
        absolute, source_path = self.resolve(path)
        text, stat_key = self._read_existing(absolute, source_path)
        nl = _detect_newline(text)
        is_ai = self._is_ai_note(text)

        addition = content.replace("\r\n", "\n").strip("\n")
        if not is_ai:
            addition = f"{self._marker('append')}\n{addition}"
        base = text.rstrip("\r\n")
        new_text = (base + "\n\n" if base else "") + addition + "\n"
        if nl == "\r\n":
            new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")

        backup = self._backup(absolute, source_path)
        nbytes = self._atomic_replace(absolute, new_text, stat_key)
        return self._result(source_path, "appended", nbytes, ai=is_ai, backup=backup)

    def replace_section(self, path: str, heading: str, content: str, create_if_missing: bool = False) -> WriteResult:
        """Replace everything under ``heading`` (up to the next heading of the same or higher level).

        The heading line itself and all other sections are left untouched. ``heading`` may be ``"Title"`` or
        ``"Parent > Title"`` to disambiguate.
        """
        self._check_enabled()
        self._check_content(content)
        if not isinstance(heading, str) or not heading.strip():
            raise SectionNotFound("heading must not be empty")
        absolute, source_path = self.resolve(path)
        text, stat_key = self._read_existing(absolute, source_path)
        nl = _detect_newline(text)
        is_ai = self._is_ai_note(text)
        work = text.replace("\r\n", "\n")
        lines = work.split("\n")
        body_lines = content.replace("\r\n", "\n").strip("\n").split("\n")
        marker = [] if is_ai else [self._marker("replace")]

        headings = scan_headings(lines)
        try:
            target = find_section(headings, heading)
        except SectionNotFound:
            if not create_if_missing:
                raise
            title = re.split(r"\s*>\s*", heading.strip())[-1].strip()
            new_text = work.rstrip("\n") + f"\n\n## {title}\n\n" + "\n".join(marker + body_lines) + "\n"
            action = "added_section"
        else:
            end = next((h.line for h in headings if h.line > target.line and h.level <= target.level), len(lines))
            block = ["", *marker, *body_lines, ""]
            rebuilt = lines[: target.line + 1] + block + lines[end:]
            new_text = "\n".join(rebuilt).rstrip("\n") + "\n"
            action = "replaced_section"
        if nl == "\r\n":
            new_text = new_text.replace("\n", "\r\n")

        backup = self._backup(absolute, source_path)
        nbytes = self._atomic_replace(absolute, new_text, stat_key)
        return self._result(source_path, action, nbytes, ai=is_ai, backup=backup)


__all__ = [
    "VaultWriter",
    "WriteResult",
    "WriteError",
    "PathNotAllowed",
    "NoteExists",
    "NoteNotFound",
    "SectionNotFound",
    "AmbiguousSection",
    "WriteConflict",
    "ContentRejected",
    "scan_headings",
    "find_section",
]
