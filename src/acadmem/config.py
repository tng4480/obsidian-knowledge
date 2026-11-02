"""Configuration.

Resolution order (later wins): defaults < TOML config file < environment variables
< explicit keyword arguments.

The index directory deliberately lives *outside* the vault (default under
``%LOCALAPPDATA%``) so that OneDrive/Obsidian sync never touches the derived,
disposable index.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

# bge models expect this instruction on *queries* only (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_IGNORED_DIRS = (".obsidian", ".trash", ".git", ".smart-env", "node_modules", "__pycache__")

# Top-level vault entries that belong to the separate `obsidian` MCP server (mcp-obsidian-vault): its task queue,
# decision log, daily notes and the dashboard it rewrites on every task change. They are agent bookkeeping, not study
# material, so they are neither indexed nor writable here. Matched at the vault root only (case-insensitive), so a
# `Tasks` folder inside a module is still ordinary study material.
DEFAULT_IGNORED_ROOT_ENTRIES = ("Tasks", "Decisions", "Daily Notes", "DASHBOARD.md")


def default_index_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "acadmem"


@dataclass(frozen=True)
class Config:
    # --- vault / storage -------------------------------------------------
    vault_path: Path = Path(".")
    index_dir: Path = field(default_factory=default_index_dir)
    extensions: tuple[str, ...] = (".md", ".pdf")
    ignored_dirs: tuple[str, ...] = DEFAULT_IGNORED_DIRS
    ignored_root_entries: tuple[str, ...] = DEFAULT_IGNORED_ROOT_ENTRIES

    # --- embeddings -------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    query_prefix: str | None = BGE_QUERY_PREFIX  # None disables the prefix
    document_prefix: str = ""

    # --- PDF --------------------------------------------------------------
    pdf_extractor: str = "pymupdf"  # "pymupdf" | "docling"
    # Vault folders (top level or nested, relative to the vault) whose PDFs get OCR + formula recognition on any scanned
    # pages. Slow (about 5 s a page, more with formulas), so it is opt-in per folder. Empty tuple turns it off.
    ocr_paths: tuple[str, ...] = ("Databook",)

    # --- chunking (tokens measured with the embedding model's tokenizer) --
    max_chunk_tokens: int = 384  # model max is 512; leave room for the heading breadcrumb
    min_chunk_tokens: int = 64
    overlap_tokens: int = 48  # only applied when a single block must be force-split

    # --- retrieval --------------------------------------------------------
    # "vector" | "fts" | "hybrid". Hybrid (vector + BM25 fused with RRF) was chosen by the benchmark: it had the best
    # Recall@5 in every chunking configuration tried (96.7-100% vs 91-93% vector-only, 92-96% FTS-only).
    search_mode: str = "hybrid"
    default_top_k: int = 5
    context_neighbors: int = 1
    context_pages: int = 0

    # --- watcher ----------------------------------------------------------
    watch_debounce_seconds: float = 3.0
    #: Codex starts one MCP server per chat plus short-lived probes. Only one server at a time watches the vault
    #: (holder of ``watcher.lock``); the others retry every ``watch_retry_seconds`` and take over if it exits.
    #: The delay keeps a probe that lives for a second from starting an indexing pass.
    watch_start_delay_seconds: float = 5.0
    watch_retry_seconds: float = 30.0

    # --- MCP server ---------------------------------------------------------
    #: Load the embedding model at startup (~560 MB extra per server). Off by default: the model loads on the first
    #: search, so chats and probes that never search stay small.
    warm_up_on_start: bool = False

    # --- guarded Markdown writes (MCP create_note / append_to_note / replace_section) ----
    writes_enabled: bool = True
    write_max_bytes: int = 200_000  # per call; a note is never going to need more than this in one go
    backup_keep: int = 200  # copies of notes (stored OUTSIDE the vault) kept from before an edit

    # ---------------------------------------------------------------------
    @property
    def backup_dir(self) -> Path:
        return self.index_dir / "backups"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "lancedb"

    @property
    def lock_path(self) -> Path:
        return self.index_dir / "index.lock"

    @property
    def watcher_lock_path(self) -> Path:
        return self.index_dir / "watcher.lock"

    def with_overrides(self, **kwargs: Any) -> "Config":
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **_coerce(kwargs))


_PATH_FIELDS = {"vault_path", "index_dir"}
_TUPLE_FIELDS = {"extensions", "ignored_dirs", "ignored_root_entries", "ocr_paths"}
_INT_FIELDS = {
    "embedding_batch_size",
    "max_chunk_tokens",
    "min_chunk_tokens",
    "overlap_tokens",
    "default_top_k",
    "context_neighbors",
    "context_pages",
    "write_max_bytes",
    "backup_keep",
}
_FLOAT_FIELDS = {"watch_debounce_seconds", "watch_start_delay_seconds", "watch_retry_seconds"}
_BOOL_FIELDS = {"writes_enabled", "warm_up_on_start"}


def _coerce(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    valid = {f.name for f in fields(Config)}
    for key, value in values.items():
        if key not in valid:
            raise KeyError(f"Unknown config key: {key}")
        if key in _PATH_FIELDS:
            value = Path(str(value)).expanduser()
        elif key in _TUPLE_FIELDS:
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            value = tuple(value)
        elif key in _INT_FIELDS:
            value = int(value)
        elif key in _FLOAT_FIELDS:
            value = float(value)
        elif key in _BOOL_FIELDS:
            value = value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
        elif key == "query_prefix" and isinstance(value, str) and value.lower() in {"none", "null", ""}:
            value = None
        out[key] = value
    return out


def load_config(
    config_file: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    **overrides: Any,
) -> Config:
    """Build a :class:`Config` from file, environment and overrides."""
    env = dict(os.environ if env is None else env)
    values: dict[str, Any] = {}

    path = config_file or env.get("ACADMEM_CONFIG")
    if path and Path(path).is_file():
        with open(path, "rb") as fh:
            values.update(tomllib.load(fh))

    # ACADMEM_VAULT, ACADMEM_INDEX_DIR, ACADMEM_EMBEDDING_MODEL, ...
    for f in fields(Config):
        env_key = f"ACADMEM_{f.name.upper()}"
        if env_key in env:
            values[f.name] = env[env_key]
    if "ACADMEM_VAULT" in env:
        values["vault_path"] = env["ACADMEM_VAULT"]

    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config().with_overrides(**values)
