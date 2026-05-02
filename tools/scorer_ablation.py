#!/usr/bin/env python3.11
"""scorer_ablation.py — drop each scorer term in turn, measure R@5 delta.

Per Tito's brief 2026-05-02 (item A): mazemaker ablation found
BM25/temporal/salience were "dead weight" in their setup. We need our
own ablation to find AE-domain dead weight.

Approach: monkey-patch DEFAULT_WEIGHTS in scoring.py — set each weight
to 0 in turn, run AE-domain bench, capture R@5. Compare against
baseline. Any term whose removal moves R@5 by < 0.01 = dead-weight
candidate (drop after manual review).

Tests removal of:
- 8 weighted features (semantic/sparse/graph/temporal/entity/procedural/locus/rrf)
- baseline (no removal)

Output: JSON to stdout + per-run row showing delta vs baseline.

Usage:
    tools/scorer_ablation.py                      # full sweep, default rerank=True
    tools/scorer_ablation.py --no-rerank          # skip cross-encoder
    tools/scorer_ablation.py --terms semantic,sparse  # only these terms
    tools/scorer_ablation.py --db /tmp/copy.db    # against a DB copy

Cost: ~3-5 min per run × 9 runs = ~30-45 min total.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "benchmarks" / "ae_domain_memory_bench"))


def _hit_at_k(rids, gt_ids, k):
    return 1 if set(rids[:k]) & set(gt_ids) else 0


def _mrr(rids, gt_ids):
    for i, rid in enumerate(rids):
        if rid in gt_ids:
            return 1.0 / (i + 1)
    return 0.0


def run_bench(db_path: str, rerank: bool) -> dict:
    """Returns per-category + global R@5/R@10/MRR."""
    from queries import ALL_QUERIES
    from memory_client import NeuralMemory
    mem = NeuralMemory(
        db_path=db_path,
        embedding_backend="auto",
        use_cpp=False,
        use_hnsw=True,
    )
    by_cat = defaultdict(lambda: {"hit5": 0, "hit10": 0, "mrr_sum": 0.0, "n": 0})
    t_start = time.time()
    for q in ALL_QUERIES:
        if not q.get("ground_truth_ids"):
            continue
        results = mem.hybrid_recall(q["query"], k=10, rerank=rerank)
        rids = [r["id"] for r in results]
        gt = q["ground_truth_ids"]
        cat = q["category"]
        by_cat[cat]["hit5"] += _hit_at_k(rids, gt, 5)
        by_cat[cat]["hit10"] += _hit_at_k(rids, gt, 10)
        by_cat[cat]["mrr_sum"] += _mrr(rids, gt)
        by_cat[cat]["n"] += 1
    elapsed = time.time() - t_start

    summary = {"per_category": {}, "wall_s": round(elapsed, 1)}
    total_hit5 = total_hit10 = total_n = 0
    total_mrr = 0.0
    for cat, agg in sorted(by_cat.items()):
        n = agg["n"]
        summary["per_category"][cat] = {
            "n": n,
            "r@5": round(agg["hit5"] / n, 4),
            "r@10": round(agg["hit10"] / n, 4),
            "mrr": round(agg["mrr_sum"] / n, 4),
        }
        total_hit5 += agg["hit5"]
        total_hit10 += agg["hit10"]
        total_n += n
        total_mrr += agg["mrr_sum"]
    summary["global_r@5"] = round(total_hit5 / total_n, 4) if total_n else 0.0
    summary["global_r@10"] = round(total_hit10 / total_n, 4) if total_n else 0.0
    summary["global_mrr"] = round(total_mrr / total_n, 4) if total_n else 0.0
    summary["n"] = total_n
    mem.close()
    return summary


def ablate_term(term: str, rerank: bool, db_path: str) -> dict:
    """Set DEFAULT_WEIGHTS[term] = 0, run bench, report. Restore after."""
    import scoring
    # Mutate the module-level dict (memory_client imports DEFAULT_WEIGHTS by ref)
    original = scoring.DEFAULT_WEIGHTS.get(term)
    scoring.DEFAULT_WEIGHTS[term] = 0.0
    try:
        result = run_bench(db_path, rerank=rerank)
        result["ablated_term"] = term
        result["original_weight"] = original
        return result
    finally:
        scoring.DEFAULT_WEIGHTS[term] = original


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(Path.home() / ".neural_memory" / "memory.db"))
    p.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument(
        "--terms",
        default="semantic,sparse,graph,temporal,entity,procedural,locus,rrf",
        help="comma-separated terms to ablate (default: all 8)",
    )
    p.add_argument("--out", default=None,
                   help="write JSON to this path (default: stdout only)")
    args = p.parse_args()

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    print(f"[ablation] DB: {args.db}", flush=True)
    print(f"[ablation] rerank: {args.rerank}", flush=True)
    print(f"[ablation] terms: {terms}", flush=True)
    print(f"[ablation] starting baseline...", flush=True)

    baseline = run_bench(args.db, rerank=args.rerank)
    baseline_r5 = baseline["global_r@5"]
    print(f"[baseline] global R@5 = {baseline_r5:.4f} ({baseline['wall_s']}s)", flush=True)

    runs = [{"label": "baseline", **baseline}]
    for term in terms:
        print(f"[ablation] dropping '{term}'...", flush=True)
        result = ablate_term(term, args.rerank, args.db)
        delta = result["global_r@5"] - baseline_r5
        verdict = "DEAD WEIGHT" if abs(delta) < 0.01 else "LOAD-BEARING"
        print(
            f"  drop {term:12s} → R@5={result['global_r@5']:.4f} "
            f"Δ={delta:+.4f} {verdict} ({result['wall_s']}s)",
            flush=True,
        )
        runs.append({"label": f"drop_{term}", **result, "delta_vs_baseline": delta, "verdict": verdict})

    # Final dead-weight summary
    dead = [r for r in runs[1:] if r.get("verdict") == "DEAD WEIGHT"]
    load_bearing = [r for r in runs[1:] if r.get("verdict") == "LOAD-BEARING"]
    print(f"\n[summary] baseline R@5 = {baseline_r5:.4f}", flush=True)
    print(f"[summary] dead-weight terms ({len(dead)}): "
          f"{[r['ablated_term'] for r in dead]}", flush=True)
    print(f"[summary] load-bearing terms ({len(load_bearing)}): "
          f"{[r['ablated_term'] for r in load_bearing]}", flush=True)

    out_payload = {
        "ts": int(time.time()),
        "db_path": args.db,
        "rerank": args.rerank,
        "baseline_r@5": baseline_r5,
        "runs": runs,
        "dead_weight_candidates": [r["ablated_term"] for r in dead],
        "load_bearing": [r["ablated_term"] for r in load_bearing],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out_payload, indent=2))
        print(f"[summary] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
