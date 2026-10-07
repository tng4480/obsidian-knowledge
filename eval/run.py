"""Run the retrieval benchmark: ``uv run python eval/run.py [options]``.

A thin wrapper over ``acadmem eval`` so the benchmark has an obvious entry point next to ``benchmark.yaml``.
All options are forwarded (``--modes vector,fts,hybrid``, ``--category pdf``, ``--details``, ``--check-labels`` ...).
"""

from __future__ import annotations

import sys

from acadmem.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
