#!/usr/bin/env python3
"""AE-domain bench runner — Sprint 2 Phase 7.

Runs the 240-query AE-domain bench against a NeuralMemory store. Two modes:

  --mode diagnostic   (default; no ground truth needed)
      For each query, calls recall() and reports which memory IDs surfaced
      via which channels. Useful for inspecting Phase 7 retrieval health
      against existing AE memories before ground-truth labels are filled in.

  --mode scored       (requires ground_truth_ids in queries.py)
      Computes per-category R@5 / R@10 / MRR and compares against the
      addendum thresholds. Fails non-zero if any category misses threshold.

Usage:
    python3 run_ae_domain_bench.py
    python3 run_ae_domain_bench.py --db ~/.neural_memory/memory.db --mode diagnostic
    python3 run_ae_domain_bench.py --category electrical_contracting
    python3 run_ae_domain_bench.py --out report.json

Per addendum lines 580-637 + lines 627-637 (global acceptance thresholds).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from queries import (  # noqa: E402
    ALL_QUERIES, CATEGORY_THRESHOLDS, category_counts, get_queries,
)


# Provenance: env vars whose value affects retrieval ranking. Captured
# verbatim into every scored artifact so future flips are attributable.
_PROVENANCE_ENV_KEYS = (
    "NM_EMBED_MODEL",
    "NM_SPANISH_TRANSLATE",
    "NM_RERANK_MODEL",
    "NM_RERANK_MODEL_ES",
    "NM_RERANK_ES_DISABLE",
    "NM_DISABLE_CONN_HISTORY",
    "NM_NREM_UPDATE_BATCH",
)


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _query_file_hash() -> str:
    qpath = Path(__file__).resolve().parent / "queries.py"
    if not qpath.exists():
        return "unknown"
    return hashlib.md5(qpath.read_bytes()).hexdigest()


def _substrate_counts(db_path: str | None) -> dict:
    """Read memory + active-connection counts via read-only sqlite URI."""
    counts: dict = {"memories": None, "connections_active": None}
    if not db_path or not Path(db_path).exists():
        return counts
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            counts["memories"] = row[0] if row else None
            row = conn.execute(
                "SELECT COUNT(*) FROM connections WHERE valid_to IS NULL"
            ).fetchone()
            counts["connections_active"] = row[0] if row else None
        finally:
            conn.close()
    except Exception:
        pass
    return counts


def _model_provenance(mem) -> dict:
    """Best-effort: pull embedder + reranker identifiers from the live store."""
    out = {
        "embedding_model": None,
        "embedding_dim": None,
        "reranker_en": None,
        "reranker_es": None,
    }
    try:
        emb = getattr(mem, "_embedder", None)
        if emb is not None:
            out["embedding_model"] = (
                getattr(emb, "model_name", None)
                or getattr(emb, "MODEL_NAME", None)
            )
            out["embedding_dim"] = getattr(emb, "dim", None)
    except Exception:
        pass
    try:
        out["reranker_en"] = getattr(mem, "_rerank_model_name", None)
        out["reranker_es"] = getattr(mem, "_rerank_model_es_name", None)
    except Exception:
        pass
    return out


def _env_provenance() -> dict:
    return {k: os.environ.get(k) for k in _PROVENANCE_ENV_KEYS}


def _resolve_db_path(mem, args) -> str | None:
    return (
        getattr(mem, "_db_path", None)
        or getattr(mem, "db_path", None)
        or args.db
    )


def _collect_provenance(mem, args, queries) -> dict:
    """Assemble the provenance block injected into the bench artifact."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    db_path = _resolve_db_path(mem, args)
    db_path_str = db_path if isinstance(db_path, str) else (
        str(db_path) if db_path else "(default)"
    )
    return {
        "git_head": _git_head(repo_root),
        "db_path": db_path_str,
        "substrate_counts": _substrate_counts(
            db_path if isinstance(db_path, str) else None
        ),
        "models": _model_provenance(mem),
        "env": _env_provenance(),
        "query_file_md5": _query_file_hash(),
        "args": {
            "mode": args.mode,
            "category": args.category,
            "k": args.k,
            "embedding_backend": args.embedding_backend,
            "rerank": args.rerank,
            "mmr_lambda": args.mmr_lambda,
            "percentile_floor": args.percentile_floor,
        },
        "queries_loaded": len(queries),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ts_epoch": int(time.time()),
    }


def _category_regression_gate(current: dict, prev_path: str | None,
                              threshold: float = 0.05) -> dict:
    """Compare current per_category R@5 vs a prior artifact; flag drops."""
    if not prev_path:
        return {"enabled": False, "regressions": []}
    try:
        prev = json.loads(Path(prev_path).read_text())
    except Exception as e:
        return {"enabled": True, "error": str(e), "regressions": []}
    regressions = []
    cur_cats = current.get("per_category", {}) or {}
    prev_cats = (prev.get("per_category") if isinstance(prev, dict) else {}) or {}
    for cat, cur_row in cur_cats.items():
        prev_row = prev_cats.get(cat)
        if not prev_row:
            continue
        cur_r5 = cur_row.get("r@5", 0.0)
        prev_r5 = prev_row.get("r@5", 0.0)
        delta = cur_r5 - prev_r5
        if delta < -threshold:
            regressions.append({
                "category": cat,
                "prev_r@5": prev_r5,
                "current_r@5": cur_r5,
                "delta": round(delta, 4),
            })
    return {
        "enabled": True,
        "prev_path": prev_path,
        "threshold": threshold,
        "regressions": regressions,
    }


def _load_neural_memory(db_path: str | None,
                       embedding_backend: str = "auto"):
    """Lazy-load NeuralMemory; supports --db override."""
    from memory_client import NeuralMemory  # noqa: WPS433
    kwargs = {
        "embedding_backend": embedding_backend,
        "use_cpp": False,  # avoid C++ lib dependency in bench env
        "use_hnsw": False,
    }
    if db_path:
        kwargs["db_path"] = db_path
    return NeuralMemory(**kwargs)


def run_diagnostic(mem, queries: list[dict], k: int = 5) -> dict:
    """Run each query in diagnostic mode (no ground truth). Reports channel
    activation + top-k IDs per query. Useful for labeling the corpus."""
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        results = mem.recall(q["query"], k=k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Touch each retrieval channel briefly to record what's available.
        sparse_ids = []
        try:
            sparse_results = mem.sparse_search(q["query"], k=k)
            sparse_ids = [r["id"] for r in sparse_results]
        except Exception:
            pass

        intent = mem._classify_intent(q["query"])
        weights = mem.intent_edge_weights(q["query"])

        rows.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected_channels": q["expected_channels"],
            "intent_classified": intent,
            "edge_weights_for_intent": weights,
            "dense_top_k_ids": [r["id"] for r in results],
            "sparse_top_k_ids": sparse_ids,
            "latency_ms": round(latency_ms, 2),
        })
    return {"mode": "diagnostic", "queries": len(rows), "rows": rows}


def _hit_at_k(retrieved_ids: list[int], gt_ids: list[int], k: int) -> int:
    """1 if any ground-truth id appears in top-k retrieved, else 0."""
    if not gt_ids:
        return 0
    top = retrieved_ids[:k]
    return 1 if any(g in top for g in gt_ids) else 0


def _mrr(retrieved_ids: list[int], gt_ids: list[int]) -> float:
    if not gt_ids:
        return 0.0
    for rank, mid in enumerate(retrieved_ids, start=1):
        if mid in gt_ids:
            return 1.0 / rank
    return 0.0


def run_scored(mem, queries: list[dict], k: int = 10,
              rerank: bool = False,
              mmr_lambda: float = 0.0,
              percentile_floor: float = 0.0) -> dict:
    """Run scored mode against ground_truth_ids. Reports per-category metrics
    + global metrics + threshold pass/fail.

    Uses hybrid_recall (the multi-channel path with all Phase 7.5 wirings:
    entity_score, procedural_score, locus_score, stale_penalty,
    contradiction_penalty, RRF feature). Caught 2026-05-01: previously used
    plain recall() which only exercises the dense channel — my Phase 7.5
    boosts couldn't surface the doctrinal chunks above conversation-turn
    noise without the entity/procedural/locus signal active.

    rerank: pass-through to hybrid_recall's cross-encoder rerank stage.
    Off by default; on for "path to 0.92 R@5" runs. First measurement
    2026-05-01: rerank=off gave global R@5=0.26 (R@10=0.71). Without rerank,
    labeled IDs retrieve in top-10 but rank 6-10 instead of 1-5.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    per_query: list[dict] = []
    for q in queries:
        if not q["ground_truth_ids"]:
            continue
        t0 = time.perf_counter()
        retrieved = mem.hybrid_recall(
            q["query"], k=k, rerank=rerank,
            mmr_lambda=mmr_lambda,
            percentile_floor=percentile_floor,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rids = [r["id"] for r in retrieved]
        gt_set = set(q["ground_truth_ids"])
        first_hit_rank = next(
            (rank for rank, rid in enumerate(rids, start=1) if rid in gt_set),
            None,
        )
        h5 = _hit_at_k(rids, q["ground_truth_ids"], 5)
        h10 = _hit_at_k(rids, q["ground_truth_ids"], 10)
        mrr = _mrr(rids, q["ground_truth_ids"])
        by_cat[q["category"]].append({
            "id": q["id"],
            "hit_at_5": h5,
            "hit_at_10": h10,
            "mrr": mrr,
        })
        per_query.append({
            "id": q["id"],
            "category": q["category"],
            "ground_truth_ids": list(q["ground_truth_ids"]),
            "retrieved_ids": rids,
            "hit_at_5": h5,
            "hit_at_10": h10,
            "mrr": round(mrr, 4),
            "first_hit_rank": first_hit_rank,
            "latency_ms": round(latency_ms, 2),
        })

    summary = {}
    failed = []
    for cat, rows in by_cat.items():
        if not rows:
            continue
        n = len(rows)
        r5 = sum(r["hit_at_5"] for r in rows) / n
        r10 = sum(r["hit_at_10"] for r in rows) / n
        mrr = sum(r["mrr"] for r in rows) / n
        threshold = CATEGORY_THRESHOLDS.get(cat, 0.0)
        passed = r5 >= threshold
        summary[cat] = {
            "n": n, "r@5": round(r5, 4), "r@10": round(r10, 4),
            "mrr": round(mrr, 4), "threshold": threshold, "passed": passed,
        }
        if not passed:
            failed.append(cat)

    total_rows = [r for rs in by_cat.values() for r in rs]
    if total_rows:
        global_r5 = sum(r["hit_at_5"] for r in total_rows) / len(total_rows)
    else:
        global_r5 = 0.0

    return {
        "mode": "scored",
        "queries_evaluated": len(total_rows),
        "queries_skipped_no_ground_truth": sum(1 for q in queries if not q["ground_truth_ids"]),
        "per_category": summary,
        "per_query": per_query,
        "global_r@5": round(global_r5, 4),
        "global_r@5_target": 0.760,
        "global_r@5_passed": global_r5 >= 0.760 if total_rows else None,
        "categories_failed": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="DB path override")
    parser.add_argument("--mode", choices=("diagnostic", "scored"),
                        default="diagnostic")
    parser.add_argument("--category", default=None,
                        help="Run only one category")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedding-backend", default="auto")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable cross-encoder rerank in scored mode "
                             "(per the path-to-0.92 docstring)")
    parser.add_argument("--mmr-lambda", type=float, default=0.0,
                        help="MMR diversity lambda (0=relevance, "
                             "1=novelty); sweet spot 0.5-0.7")
    parser.add_argument("--percentile-floor", type=float, default=0.0,
                        help="Drop bottom (1-floor) fraction by rank "
                             "(0=no-op; 0.1=keep top 90%%)")
    parser.add_argument("--out", default=None,
                        help="JSON output path; stdout if omitted")
    parser.add_argument("--prev-results", default=None,
                        help="Optional path to a prior scored artifact; "
                             "enables per-category regression gate "
                             "(flags drops > 0.05 R@5)")
    args = parser.parse_args()

    queries = get_queries(args.category)
    if not queries:
        print(f"No queries found for category={args.category!r}", file=sys.stderr)
        return 1
    counts = category_counts()
    print(f"Loaded {len(queries)} queries "
          f"(corpus has {sum(counts.values())} total across "
          f"{len(counts)} categories)", file=sys.stderr)

    mem = _load_neural_memory(args.db, embedding_backend=args.embedding_backend)
    try:
        if args.mode == "diagnostic":
            result = run_diagnostic(mem, queries, k=args.k)
        else:
            result = run_scored(mem, queries, k=args.k,
                                rerank=args.rerank,
                                mmr_lambda=args.mmr_lambda,
                                percentile_floor=args.percentile_floor)
        result["provenance"] = _collect_provenance(mem, args, queries)
        if args.mode == "scored":
            result["category_regression_gate"] = _category_regression_gate(
                result, args.prev_results,
            )
    finally:
        try:
            mem.close()
        except Exception:
            pass

    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(output)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(output)

    if args.mode == "scored" and result.get("categories_failed"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
