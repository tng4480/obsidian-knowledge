"""Tune chunking parameters against the benchmark.

PDF extraction dominates indexing time (~6 min for the vault) and does not depend on chunking, so parsed
documents are cached on disk, as are embeddings (keyed by text). Each candidate then costs only the
chunks that actually differ from earlier candidates. This is a tuning aid only: the production indexer is
unchanged and does no caching.

    uv run python eval/tune_chunking.py --grid "min=64;max=384" --grid "min=32;max=384" --grid "min=16;max=256"
    uv run python eval/tune_chunking.py --default-grid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

from acadmem.config import Config, load_config
from acadmem.embeddings import get_embedder
from acadmem.evaluation import copy_vault_with_fixtures, format_report, load_benchmark, run_benchmark
from acadmem.indexer import Indexer
from acadmem.ingest.base import ExtractorRegistry
from acadmem.ingest.registry import build_registry
from acadmem.search import Searcher

ROOT = Path(__file__).resolve().parent
DEFAULT_GRID = ["min=64;max=384;overlap=48", "min=32;max=384;overlap=48", "min=16;max=384;overlap=48", "min=32;max=256;overlap=32"]


class CachedExtractor:
    def __init__(self, inner, cache_dir: Path, vault: Path) -> None:
        self.inner, self.cache_dir, self.vault = inner, cache_dir, vault
        self.extensions, self.file_type = inner.extensions, inner.file_type

    def parse(self, path: Path):
        st = path.stat()
        key = hashlib.sha1(f"{path.relative_to(self.vault).as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        cached = self.cache_dir / f"{key}.pkl"
        if cached.exists():
            return pickle.loads(cached.read_bytes())
        content = self.inner.parse(path)
        cached.write_bytes(pickle.dumps(content))
        return content


def cached_registry(config: Config, cache_dir: Path) -> ExtractorRegistry:
    cache_dir.mkdir(parents=True, exist_ok=True)
    inner = build_registry(config)
    wrapped = ExtractorRegistry()
    seen = set()
    for ext in inner.extensions:
        ex = inner.for_path(f"x{ext}")
        if id(ex) not in seen:
            seen.add(id(ex))
            wrapped.register(CachedExtractor(ex, cache_dir, Path(config.vault_path)))
    return wrapped


class CachedEmbedder:
    """Memoises document embeddings by text (tuning only)."""

    def __init__(self, inner, path: Path) -> None:
        self.inner, self.path = inner, path
        self.model_name = inner.model_name
        self.cache: dict[str, list[float]] = pickle.loads(path.read_bytes()) if path.exists() else {}
        self.hits = self.misses = 0

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed_documents(self, texts):
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        todo = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            for (k, _), vec in zip(todo, self.inner.embed_documents([t for _, t in todo])):
                self.cache[k] = vec
        return [self.cache[k] for k in keys]

    def embed_query(self, text):
        return self.inner.embed_query(text)

    def count_tokens(self, text):
        return self.inner.count_tokens(text)

    def save(self) -> None:
        self.path.write_bytes(pickle.dumps(self.cache))


def parse_grid(spec: str) -> dict[str, int]:
    out = {}
    for part in spec.split(";"):
        k, _, v = part.partition("=")
        out[{"min": "min_chunk_tokens", "max": "max_chunk_tokens", "overlap": "overlap_tokens"}[k.strip()]] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.environ.get("ACADMEM_VAULT"))
    ap.add_argument("--grid", action="append", help='e.g. "min=32;max=384;overlap=48" (repeatable)')
    ap.add_argument("--default-grid", action="store_true")
    ap.add_argument("--modes", default="vector,fts,hybrid")
    ap.add_argument("--work", default=str(Path(os.environ.get("LOCALAPPDATA", ".")) / "acadmem-tune"))
    ap.add_argument("--out", default=str(ROOT / "tuning_results.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    grids = DEFAULT_GRID if args.default_grid or not args.grid else args.grid
    work = Path(args.work)
    base = load_config(vault_path=args.vault)
    vault_copy = copy_vault_with_fixtures(Path(base.vault_path), ROOT / "fixtures" / "notes", work / "vault")
    benchmark = load_benchmark(ROOT / "benchmark.yaml")

    embedder = CachedEmbedder(get_embedder(base), work / "embeddings.pkl")
    summary = []
    for spec in grids:
        params = parse_grid(spec)
        cfg = base.with_overrides(vault_path=vault_copy, index_dir=work / "index", **params)
        if (work / "index").exists():
            shutil.rmtree(work / "index")
        started = time.time()
        indexer = Indexer(cfg, embedder=embedder, registry=cached_registry(cfg, work / "parsed"))
        report = indexer.run(full=True)
        embedder.save()
        built = time.time() - started
        result = run_benchmark(Searcher(cfg, embedder=embedder), benchmark, modes=args.modes.split(","), top_k=5)
        print(f"\n===== {spec}  ({report.chunks_written} chunks, built in {built:.0f}s, embed cache {embedder.hits} hit / {embedder.misses} miss)")
        print(format_report(result))
        summary.append({"spec": spec, "chunks": report.chunks_written, "overall": {m: v["overall"] for m, v in result["modes"].items()},
                        "by_category": {m: {c: a["recall"] for c, a in v["by_category"].items()} for m, v in result["modes"].items()}})
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== summary: Recall@5 / MRR per candidate")
    for s in summary:
        cells = "  ".join(f"{m}: {o['recall'] * 100:5.1f}/{(o['mrr'] or 0) * 100:5.1f}" for m, o in s["overall"].items())
        print(f"  {s['spec']:<34} chunks={s['chunks']:<5} {cells}")


if __name__ == "__main__":
    main()
