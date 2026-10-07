"""How well does a hit's ``score`` (cosine similarity) separate relevant from irrelevant hits?

    uv run python eval/score_calibration.py [--top-k 10] [--json-out FILE]

Uses the benchmark labels: a hit is *relevant* when it matches an ``expected`` item, *irrelevant* otherwise (other
retrieved hits). A handful of off-topic queries (nothing in a 3rd-year engineering vault can answer them) show what the
best hit looks like when there is no good answer. Runs on the same scratch overlay as ``acadmem eval``; the real
vault and index are not modified.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from acadmem.config import load_config
from acadmem.evaluation import item_matches, load_benchmark, prepare_overlay
from acadmem.search import Searcher

OFF_TOPIC = [
    "best sourdough bread recipe with a long cold ferment",
    "who won the football world cup in 2018",
    "how do I renew my passport in the UK",
    "symptoms and treatment of seasonal influenza",
    "history of the Roman empire and the fall of Constantinople",
    "kubernetes pod scheduling and ingress controllers",
    "Shakespeare sonnet analysis of iambic pentameter",
    "how many calories are in a banana",
    "cheap flights from London to Tokyo in spring",
    "Python decorators and asyncio event loop explained",
]

# Engineering-flavoured topics that are NOT in this (signals / information / inference / numerics) vault: harder negatives.
ADJACENT_ABSENT = [
    "Carnot cycle efficiency and entropy generation in a heat engine",
    "bending moment diagram for a reinforced concrete beam",
    "boundary layer turbulence and the Navier-Stokes equations",
    "SN2 reaction mechanism in organic chemistry",
    "price elasticity of demand and consumer surplus",
    "gear ratio and torque in a planetary gearbox",
]


def quantiles(values: list[float], qs=(0, 10, 25, 50, 75, 90, 100)) -> dict[int, float]:
    if not values:
        return {}
    s = sorted(values)
    return {q: s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))] for q in qs}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random relevant hit has a higher score than a random irrelevant hit)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fmt(d: dict[int, float]) -> str:
    return "  ".join(f"p{q}={v:.3f}" for q, v in d.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    eval_dir = Path(__file__).resolve().parent
    cfg, work = prepare_overlay(load_config(), eval_dir / "fixtures" / "notes")
    searcher = Searcher(cfg)
    bench = load_benchmark(eval_dir / "benchmark.yaml")
    available = {d["source_path"] for d in searcher.index.list_documents()}
    queries = [q for q in bench.queries if not q.expect_empty and {e.source for e in q.expected} <= available]

    out: dict = {"n_queries": len(queries), "modes": {}}
    for mode in ("vector", "fts", "hybrid"):
        rel: list[float] = []
        irr: list[float] = []
        top1_rel: list[float] = []  # best-ranked hit of queries whose top-1 hit is relevant
        top1_irr: list[float] = []
        best_rel_per_query: list[float] = []
        for q in queries:
            hits = searcher.search(q.query, top_k=args.top_k, filters=q.filters or None, mode=mode)
            flags = [any(item_matches(e, h.source_path, h.page_start, h.page_end, h.heading_path, h.heading) for e in q.expected) for h in hits]
            for h, ok in zip(hits, flags):
                (rel if ok else irr).append(h.score)
            if hits:
                (top1_rel if flags[0] else top1_irr).append(hits[0].score)
            if any(flags):
                best_rel_per_query.append(max(h.score for h, ok in zip(hits, flags) if ok))
        top1_all = [h[0].score for q in queries if (h := searcher.search(q.query, top_k=1, filters=q.filters or None, mode=mode))]
        adj_top1 = [max(h.score for h in searcher.search(t, top_k=5, mode=mode)) for t in ADJACENT_ABSENT]
        off_top1 = []
        off_top5 = []
        for text in OFF_TOPIC:
            hits = searcher.search(text, top_k=5, mode=mode)
            off_top1.append(max(h.score for h in hits))
            off_top5.extend(h.score for h in hits)
        out["modes"][mode] = {
            "relevant": rel, "irrelevant": irr, "top1_relevant": top1_rel, "top1_irrelevant": top1_irr,
            "best_relevant_per_query": best_rel_per_query, "offtopic_top1": off_top1, "top1_all": top1_all, "adjacent_top1": adj_top1, "offtopic_all": off_top5,
        }

        print(f"\n=== {mode}  (top_k={args.top_k}, {len(queries)} labelled queries, 'score' = cosine similarity)")
        print(f"relevant hits   n={len(rel):3d}  {fmt(quantiles(rel))}")
        print(f"irrelevant hits n={len(irr):3d}  {fmt(quantiles(irr))}")
        print(f"AUC (relevant hit outscores irrelevant hit) = {auc(rel, irr):.3f}")
        print(f"best relevant hit per query     n={len(best_rel_per_query):3d}  {fmt(quantiles(best_rel_per_query))}")
        print(f"off-topic queries: best hit     n={len(off_top1):3d}  {fmt(quantiles(off_top1))}")
        print(f"AUC (answerable query's best relevant hit vs off-topic query's best hit) = {auc(best_rel_per_query, off_top1):.3f}")
        print(f"top-1 hit of every answerable query    n={len(top1_all):3d}  {fmt(quantiles(top1_all))}")
        print(f"adjacent-but-absent topics: best hit    n={len(adj_top1):3d}  {fmt(quantiles(adj_top1))}")
        print("query-level: share of answerable queries whose TOP-1 hit is below t  |  off-topic best hit >= t  |  adjacent-absent best hit >= t")
        for t in (0.60, 0.65, 0.68, 0.70, 0.72):
            print(f"  t={t:.2f}   below: {sum(x < t for x in top1_all) / len(top1_all):.2f}   |  off>=t: {sum(x >= t for x in off_top1) / len(off_top1):.2f}   |  adj>=t: {sum(x >= t for x in adj_top1) / len(adj_top1):.2f}")
        if mode == "hybrid":
            low = sorted(((x, q.id) for x, q in zip(top1_all, queries)))[:5]
            print("lowest top-1 answerable queries:", [(round(x, 3), i) for x, i in low])
        print("threshold sweep over labelled hits (flag 'relevant' if score >= t):")
        print("    t    precision  recall   | share of off-topic queries whose best hit passes t")
        for t in (0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85):
            tp = sum(s >= t for s in rel)
            fp = sum(s >= t for s in irr)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            print(f"  {t:.2f}   {prec:8.2f}   {tp / len(rel):6.2f}   | {sum(s >= t for s in off_top1) / len(off_top1):.2f}")
        print(f"mean relevant {statistics.mean(rel):.3f} (sd {statistics.pstdev(rel):.3f}), irrelevant {statistics.mean(irr):.3f} (sd {statistics.pstdev(irr):.3f})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out), encoding="utf-8")
    import shutil

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
