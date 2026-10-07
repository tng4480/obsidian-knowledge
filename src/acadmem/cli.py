"""Command line interface: ``acadmem index | rebuild | watch | status | search | eval``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .config import Config, load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vault", help="Path to the Obsidian vault (or set ACADMEM_VAULT)")
    p.add_argument("--index-dir", help="Where the disposable LanceDB index lives (default: %%LOCALAPPDATA%%\\acadmem)")
    p.add_argument("--config", help="TOML config file (or set ACADMEM_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acadmem", description="Semantic memory over an Obsidian vault.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_ in [
        ("index", "Incrementally update the index (new/modified/deleted files only)"),
        ("rebuild", "Drop the index and rebuild it from the vault"),
        ("status", "Show what is indexed, and any files that failed"),
    ]:
        _add_common(sub.add_parser(name, help=help_))

    w = sub.add_parser("watch", help="Keep the index up to date while you work in Obsidian")
    _add_common(w)
    w.add_argument("--no-initial-sync", action="store_true", help="Do not catch up on changes made while stopped")

    s = sub.add_parser("search", help="Query the index")
    _add_common(s)
    s.add_argument("query")
    s.add_argument("-k", "--top-k", type=int)
    s.add_argument("--mode", choices=["vector", "fts", "hybrid"])
    s.add_argument("--module", action="append")
    s.add_argument("--document-type", action="append")
    s.add_argument("--file-type", action="append")
    s.add_argument("--path-prefix")
    s.add_argument("--tag", action="append")
    s.add_argument("--modified-after")
    s.add_argument("--modified-before")
    s.add_argument("--ai-generated", choices=["yes", "no"])
    s.add_argument("--expand", help="Context expansion, e.g. neighbors:1, section, pages:1, parent")
    s.add_argument("--json", action="store_true", help="Machine-readable output")
    s.add_argument("--full-text", action="store_true", help="Print whole chunks instead of snippets")

    e = sub.add_parser("eval", help="Run the retrieval benchmark")
    _add_common(e)
    e.add_argument("--benchmark", default=None, help="Path to benchmark YAML (default: eval/benchmark.yaml)")
    e.add_argument("--modes", default="vector,fts,hybrid")
    e.add_argument("-k", "--top-k", type=int, default=5)
    e.add_argument("--details", action="store_true", help="List every query with its rank")
    e.add_argument("--json", dest="json_out", help="Write the full report to this file")
    e.add_argument("--category", action="append", help="Only run this category (repeatable)")
    e.add_argument("--fixtures", default=None, help="Fixture notes overlaid on a COPY of the vault (default: eval/fixtures/notes)")
    e.add_argument("--no-fixtures", action="store_true", help="Evaluate the real vault/index only; skip queries needing fixture notes")
    e.add_argument("--check-labels", action="store_true", help="Only verify the benchmark labels against the source text, then exit")
    e.add_argument("--keep-overlay", action="store_true", help="Keep the scratch vault/index copy (for debugging)")
    return parser


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, vault_path=args.vault, index_dir=args.index_dir)
    if not cfg.vault_path or not Path(cfg.vault_path).is_dir():
        raise SystemExit(f"Vault not found: {cfg.vault_path!r}. Pass --vault or set ACADMEM_VAULT.")
    return cfg


def _snippet(text: str, width: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _cmd_index(args: argparse.Namespace, full: bool) -> int:
    from .indexer import Indexer

    cfg = _config(args)
    indexer = Indexer(cfg)

    def progress(event: str, path: str) -> None:
        print(f"  {event:9} {path}", flush=True)

    report = indexer.run(full=full, progress=progress)
    print(report.summary())
    for err in report.errors:
        print(f"  ERROR {err['source_path']}: {err['error_type']}: {err['message']}", file=sys.stderr)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .index import VectorIndex

    cfg = _config(args)
    index = VectorIndex(cfg)
    if not index.exists():
        print("No index yet. Run `acadmem index`.")
        return 1
    print(f"vault:     {cfg.vault_path}")
    print(f"index:     {cfg.db_path}")
    meta = index.read_meta() or {}
    print(f"model:     {meta.get('embedding_model')} (dim {meta.get('dimension')}, schema v{meta.get('schema_version')})")
    searchable = len(index.list_documents())
    empty = index.count_documents() - searchable
    note = f"   (+{empty} empty placeholder file(s) tracked, nothing to search)" if empty else ""
    print(f"documents: {searchable}   chunks: {index.count_chunks()}{note}")
    errors = index.errors()
    print(f"failed:    {len(errors)}")
    for e in errors:
        print(f"  {e['source_path']}: {e['error_type']}: {e['message'][:200]}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    from .indexer import Indexer
    from .watch import VaultWatcher

    cfg = _config(args)
    watcher = VaultWatcher(Indexer(cfg), debounce=cfg.watch_debounce_seconds, initial_sync=not args.no_initial_sync)
    print(f"Watching {cfg.vault_path} (Ctrl+C to stop)")
    watcher.run_forever()
    return 0


def _filters_from_args(args: argparse.Namespace) -> dict:
    filters = {
        "module": args.module,
        "document_type": args.document_type,
        "file_type": args.file_type,
        "path_prefix": args.path_prefix,
        "tags": args.tag,
        "modified_after": args.modified_after,
        "modified_before": args.modified_before,
        "ai_generated": None if args.ai_generated is None else args.ai_generated == "yes",
    }
    return {k: v for k, v in filters.items() if v is not None}


def _cmd_search(args: argparse.Namespace) -> int:
    from .search import Searcher

    cfg = _config(args)
    searcher = Searcher(cfg)
    hits = searcher.search(args.query, top_k=args.top_k, filters=_filters_from_args(args), mode=args.mode)

    expanded = None
    if args.expand:
        from .expand import ContextExpander

        expander = ContextExpander(searcher.index)
        expanded = [expander.expand(h, args.expand) for h in hits]

    if args.json:
        payload = [h.to_dict() for h in hits]
        if expanded is not None:
            for item, ctx in zip(payload, expanded):
                item["context"] = ctx.to_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not hits:
        print("No results.")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"[{i}] sim {h.score:.3f}  {h.location}")
        print("    " + (h.text if args.full_text else _snippet(h.text)).replace("\n", "\n    "))
        if expanded is not None:
            for piece in expanded[i - 1].pieces:
                tag = "HIT" if piece.is_hit else "ctx"
                print(f"      ({tag}) {piece.location}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import format_report, load_benchmark, prepare_overlay, run_benchmark, verify_labels
    from .search import Searcher

    cfg = _config(args)
    eval_dir = Path(__file__).resolve().parents[2] / "eval"
    bench_path = Path(args.benchmark) if args.benchmark else eval_dir / "benchmark.yaml"
    fixtures = Path(args.fixtures) if args.fixtures else eval_dir / "fixtures" / "notes"
    benchmark = load_benchmark(bench_path)

    if args.check_labels:
        problems = verify_labels(benchmark, Path(cfg.vault_path), fixtures)
        for p in problems:
            print("  LABEL PROBLEM:", p)
        print(f"{len(benchmark.queries)} queries checked, {len(problems)} problem(s).")
        return 1 if problems else 0

    work = None
    if not args.no_fixtures and fixtures.is_dir():
        print(f"Overlaying fixture notes from {fixtures} onto a scratch copy of the vault + index ...", flush=True)
        cfg, work = prepare_overlay(cfg, fixtures)
    try:
        report = run_benchmark(Searcher(cfg), benchmark, modes=args.modes.split(","), top_k=args.top_k, categories=args.category)
    finally:
        if work is not None and not args.keep_overlay:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
    print(format_report(report, details=args.details))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    handlers = {
        "index": lambda a: _cmd_index(a, full=False),
        "rebuild": lambda a: _cmd_index(a, full=True),
        "status": _cmd_status,
        "watch": _cmd_watch,
        "search": _cmd_search,
        "eval": _cmd_eval,
    }
    return int(handlers[args.command](args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
