"""Spawned by ``test_mcp_stdio.py``: the real server over real stdio, with the deterministic fake embedder.

Environment: ACADMEM_VAULT, ACADMEM_INDEX_DIR (and optionally STDIO_READ_ONLY=1).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from conftest import FakeEmbedder  # noqa: E402

from acadmem.config import load_config  # noqa: E402
from acadmem.mcp_server import AcademicMemoryService, build_server  # noqa: E402

config = load_config(None, max_chunk_tokens=120, min_chunk_tokens=10, overlap_tokens=10)
if os.environ.get("STDIO_READ_ONLY"):
    config = dataclasses.replace(config, writes_enabled=False)
embedder = FakeEmbedder()
service = AcademicMemoryService(config, embedder=embedder)
if not service.index.exists():
    service.indexer.run(full=True)
print("launcher noise on stderr only", file=sys.stderr)
build_server(service).run("stdio")
