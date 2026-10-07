"""Recursive vault discovery and safe file reading (OneDrive-aware)."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

from ..config import Config

# Windows cloud-file attributes (OneDrive Files On-Demand placeholders)
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
_PLACEHOLDER_MASK = FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS


class FileUnavailableError(OSError):
    """A cloud placeholder could not be hydrated in time (e.g. OneDrive offline)."""


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path  # absolute
    source_path: str  # vault-relative, POSIX separators
    size: int
    mtime: float


def to_source_path(vault: Path, path: Path) -> str:
    return path.resolve().relative_to(vault.resolve()).as_posix()


def is_ignored_root_entry(config: Config, name: str) -> bool:
    """True for a top-level vault folder/file reserved for another tool (see ``DEFAULT_IGNORED_ROOT_ENTRIES``)."""
    return name.casefold() in {e.casefold() for e in config.ignored_root_entries}


def is_indexable(config: Config, rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    if is_ignored_root_entry(config, parts[0]):
        return False
    if any(p in config.ignored_dirs or (p.startswith(".") and p not in (".", "..")) for p in parts[:-1]):
        return False
    name = parts[-1]
    if name.startswith(".") or name.startswith("~$"):
        return False
    return Path(name).suffix.lower() in config.extensions


def discover(config: Config) -> Iterator[DiscoveredFile]:
    """Yield every indexable file under the vault (any nesting depth), in sorted order."""
    root = config.vault_path
    for dirpath, dirnames, filenames in os.walk(root):
        at_root = Path(dirpath) == Path(root)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in config.ignored_dirs and not d.startswith(".") and not (at_root and is_ignored_root_entry(config, d))
        )
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            if not is_indexable(config, rel):
                continue
            try:
                st = full.stat()
            except OSError:
                continue
            yield DiscoveredFile(full, rel, st.st_size, st.st_mtime)


def is_placeholder(path: Path) -> bool:
    attrs = getattr(os.stat(path), "st_file_attributes", 0)
    return bool(attrs & _PLACEHOLDER_MASK) and not (attrs & stat.FILE_ATTRIBUTE_DIRECTORY)


def read_bytes(path: Path, timeout: float = 120.0) -> bytes:
    """Read a file; if it is an un-hydrated cloud placeholder, bound the wait instead of hanging."""
    if not is_placeholder(path):
        return path.read_bytes()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(path.read_bytes)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout as exc:
        raise FileUnavailableError(f"{path.name}: cloud placeholder not available after {timeout:.0f}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
