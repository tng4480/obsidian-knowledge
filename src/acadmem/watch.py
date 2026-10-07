"""Filesystem watching via ``watchdog`` (no custom watcher).

Events are debounced per path: editors and cloud sync (OneDrive) emit bursts of
create/modify events for a single save, and re-indexing on each would be wasteful and racy.
A path is processed once it has been quiet for ``debounce`` seconds. Directory deletions/moves
(for which watchdog does not always emit per-file events) trigger a full incremental rescan.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .indexer import Indexer
from .ingest.discovery import is_indexable

log = logging.getLogger("acadmem.watch")

_RESCAN = "<rescan>"


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(self, config: Config, indexer: Indexer, debounce: float) -> None:
        self.config = config
        self.indexer = indexer
        self.debounce = debounce
        self._pending: dict[str, float] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._worker, name="acadmem-watch-worker", daemon=True)
        self._busy = False

    # ---------------------------------------------------------------- events
    def _rel_ok(self, path: str) -> bool:
        try:
            rel = Path(path).resolve().relative_to(self.config.vault_path.resolve()).as_posix()
        except (ValueError, OSError):
            return False
        return is_indexable(self.config, rel)

    def _schedule(self, key: str) -> None:
        with self._cv:
            self._pending[key] = time.monotonic() + self.debounce
            self._cv.notify()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type == "opened" or event.event_type == "closed":
            return
        paths = [str(event.src_path)]
        dest = getattr(event, "dest_path", "")
        if dest:
            paths.append(str(dest))
        if event.is_directory:
            if event.event_type in ("deleted", "moved"):
                self._schedule(_RESCAN)
            elif event.event_type == "created":
                self._schedule(_RESCAN)  # a copied-in folder may contain many files
            return
        for p in paths:
            if self._rel_ok(p):
                self._schedule(p)

    # ---------------------------------------------------------------- worker
    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._stop:
                    now = time.monotonic()
                    due = [k for k, t in self._pending.items() if t <= now]
                    if due:
                        for k in due:
                            del self._pending[k]
                        self._busy = True
                        break
                    timeout = (min(self._pending.values()) - now) if self._pending else None
                    self._cv.wait(timeout)
                if self._stop:
                    return
            try:
                if _RESCAN in due:
                    report = self.indexer.run(full=False)
                    log.info("rescan: %s", report.summary())
                for key in due:
                    if key != _RESCAN:
                        outcome = self.indexer.sync_path(key)
                        log.info("%s -> %s", key, outcome)
            except Exception:  # keep watching no matter what
                log.exception("watcher failed while syncing %s", due)
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=10)

    def idle(self) -> bool:
        with self._cv:
            return not self._pending and not self._busy


class VaultWatcher:
    """Run an initial incremental pass, then keep the index in sync with the vault."""

    def __init__(self, indexer: Indexer, debounce: float | None = None, initial_sync: bool = True) -> None:
        self.indexer = indexer
        self.config = indexer.config
        self.debounce = self.config.watch_debounce_seconds if debounce is None else debounce
        self.initial_sync = initial_sync
        self._observer: Observer | None = None
        self._handler: _DebouncedHandler | None = None

    def start(self) -> None:
        if self.initial_sync:
            self.indexer.run(full=False)  # catch up on changes made while we were not watching
        self._handler = _DebouncedHandler(self.config, self.indexer, self.debounce)
        self._handler.start()
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.config.vault_path), recursive=True)
        self._observer.start()
        log.info("watching %s", self.config.vault_path)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=10)
        if self._handler is not None:
            self._handler.stop()

    def wait_idle(self, timeout: float = 30.0, settle: float = 0.5) -> bool:
        """Block until no events are pending/being processed (used by tests and the CLI)."""
        deadline = time.monotonic() + timeout
        quiet_since: float | None = None
        while time.monotonic() < deadline:
            if self._handler is not None and self._handler.idle():
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= settle:
                    return True
            else:
                quiet_since = None
            time.sleep(0.05)
        return False

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def __enter__(self) -> "VaultWatcher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
