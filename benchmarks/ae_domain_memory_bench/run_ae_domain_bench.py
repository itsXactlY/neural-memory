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


# ---------------------------------------------------------------------------
# Stable label-set anchors — derived from git history audits.
#
# PRESERVED_33_QUERY_IDS: IDs labeled at commit 03f4785
#   "fix(bench): +11 GT labels (22→33) + 2 bugs caught by Sonnet hostile-reviewer"
#   Derived by running:
#     git show 03f4785:benchmarks/ae_domain_memory_bench/queries.py |
#       python3 -c "... regex for non-empty ground_truth_ids ..."
#   These 33 IDs are the first cohort with validated ground-truth labels.
#
# SUBSET_38_QUERY_IDS: IDs present in artifact ae-domain-2026-05-03-062619.json
#   (per_query block, 38 rows). Five IDs added after commit 03f4785:
#   ELC-009, ELC-040, LOT-010, LOT-027, MAT-004.
# ---------------------------------------------------------------------------

PRESERVED_33_QUERY_IDS: tuple[str, ...] = (
    "ELC-001", "ELC-005", "ELC-011", "ELC-013", "ELC-014", "ELC-016", "ELC-027",
    "FIN-002", "FIN-018", "FIN-026", "FIN-039",
    "LOT-001", "LOT-002", "LOT-003", "LOT-005", "LOT-006", "LOT-015",
    "LOT-023", "LOT-035", "LOT-039",
    "MAT-006", "MAT-015", "MAT-016", "MAT-021", "MAT-023", "MAT-025",
    "MAT-026", "MAT-030", "MAT-037",
    "SPA-003", "SPA-010", "SPA-028", "SPA-035",
)

SUBSET_38_QUERY_IDS: tuple[str, ...] = (
    "ELC-001", "ELC-005", "ELC-009", "ELC-011", "ELC-013", "ELC-014",
    "ELC-016", "ELC-027", "ELC-040",
    "FIN-002", "FIN-018", "FIN-026", "FIN-039",
    "LOT-001", "LOT-002", "LOT-003", "LOT-005", "LOT-006", "LOT-010",
    "LOT-015", "LOT-023", "LOT-027", "LOT-035", "LOT-039",
    "MAT-004", "MAT-006", "MAT-015", "MAT-016", "MAT-021", "MAT-023",
    "MAT-025", "MAT-026", "MAT-030", "MAT-037",
    "SPA-003", "SPA-010", "SPA-028", "SPA-035",
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
    """Best-effort: pull embedder + reranker identifiers from the live store.

    Reads PUBLIC attributes per memory_client.py: ``mem.embedder`` (set at
    line 592) and ``mem.dim`` (set at line 604). Falls back to private
    ``_embedder`` only if the public name is absent — earlier code read
    ``_embedder`` first, which was always None on the live NeuralMemory
    instance, so embedding_model/embedding_dim were silently null.

    On EmbeddingProvider the model name lives on ``backend.MODEL_NAME``
    (see embed_provider.py:34) — try the provider directly first, then
    fall through to ``provider.backend``.
    """
    out = {
        "embedding_model": None,
        "embedding_dim": None,
        "reranker_en": None,
        "reranker_es": None,
    }
    try:
        emb = getattr(mem, "embedder", None)
        if emb is None:
            # Defensive fallback: pre-2026-05 code may have used the private name.
            emb = getattr(mem, "_embedder", None)
        if emb is not None:
            model_name = (
                getattr(emb, "model_name", None)
                or getattr(emb, "MODEL_NAME", None)
            )
            if model_name is None:
                backend = getattr(emb, "backend", None)
                if backend is not None:
                    model_name = (
                        getattr(backend, "model_name", None)
                        or getattr(backend, "MODEL_NAME", None)
                    )
            out["embedding_model"] = model_name
            # Prefer the top-level mem.dim (public on NeuralMemory), fall
            # back to the embedder's own dim.
            out["embedding_dim"] = (
                getattr(mem, "dim", None)
                or getattr(emb, "dim", None)
            )
        else:
            # No embedder at all — still surface mem.dim if NeuralMemory has one.
            out["embedding_dim"] = getattr(mem, "dim", None)
    except Exception:
        pass
    try:
        # Reranker model names live on the NeuralMemory instance directly
        # (memory_client.py:633 sets self._rerank_model_name). Verified by
        # grep — no public alias, so the underscore-name read stays.
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
    """Compare current per_category R@5 vs a prior artifact on the comparable
    label intersect only.

    Label-expansion categories (current has more queries than prev due to new
    GT labels being added) are reported separately and never trigger a
    regression failure. Only categories whose R@5 dropped beyond `threshold`
    on the comparable intersect fire ``regression_detected``.
    """
    if not prev_path:
        return {"enabled": False, "regressions": [], "regression_detected": False}
    try:
        prev = json.loads(Path(prev_path).read_text())
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e),
            "regressions": [],
            "regression_detected": False,
        }

    # Build query_id → per_query row maps for intersect computation.
    cur_pq: dict[str, dict] = {
        r["id"]: r for r in (current.get("per_query") or [])
    }
    prev_pq: dict[str, dict] = {
        r["id"]: r for r in (prev.get("per_query") or [])
    }
    comparable_ids = set(cur_pq.keys()) & set(prev_pq.keys())

    # Per-category comparable-intersect R@5 for current and previous.
    def _cat_r5_on_ids(pq_map: dict[str, dict], ids: set[str]) -> dict[str, dict]:
        by_cat: dict[str, list[float]] = defaultdict(list)
        for qid in ids:
            row = pq_map.get(qid)
            if row:
                by_cat[row["category"]].append(float(row.get("hit_at_5", 0)))
        return {
            cat: {"r@5": round(sum(vs) / len(vs), 4), "n": len(vs)}
            for cat, vs in by_cat.items() if vs
        }

    cur_intersect_cats = _cat_r5_on_ids(cur_pq, comparable_ids)
    prev_intersect_cats = _cat_r5_on_ids(prev_pq, comparable_ids)

    # Identify categories that gained queries (label expansion).
    cur_all_cats: dict[str, int] = defaultdict(int)
    prev_all_cats: dict[str, int] = defaultdict(int)
    for qid, row in cur_pq.items():
        cur_all_cats[row["category"]] += 1
    for qid, row in prev_pq.items():
        prev_all_cats[row["category"]] += 1
    label_expansion_categories = sorted(
        cat for cat in cur_all_cats
        if cur_all_cats[cat] > prev_all_cats.get(cat, 0)
    )

    regressions = []
    for cat, cur_row in cur_intersect_cats.items():
        prev_row = prev_intersect_cats.get(cat)
        if not prev_row:
            continue
        # Skip regression check for label-expansion categories.
        if cat in label_expansion_categories:
            continue
        delta = cur_row["r@5"] - prev_row["r@5"]
        if delta < -threshold:
            regressions.append({
                "category": cat,
                "prev_r@5": prev_row["r@5"],
                "current_r@5": cur_row["r@5"],
                "delta": round(delta, 4),
                "comparable_n": cur_row["n"],
            })

    return {
        "enabled": True,
        "prev_path": prev_path,
        "threshold": threshold,
        "comparable_query_count": len(comparable_ids),
        "label_expansion_categories": label_expansion_categories,
        "regressions": regressions,
        "regression_detected": len(regressions) > 0,
    }


def _build_subset(
    label: str,
    anchor_ids: tuple[str, ...] | list[str],
    scored_result: dict,
    provenance: dict,
) -> dict:
    """Slice a scored result to the queries in anchor_ids.

    Returns a subset block with query_md5, label_count, per_category R@5,
    per_query rows, and dropped_ids (anchor IDs not found in current run).
    """
    cur_pq_map = {r["id"]: r for r in (scored_result.get("per_query") or [])}
    present = [qid for qid in anchor_ids if qid in cur_pq_map]
    dropped = [qid for qid in anchor_ids if qid not in cur_pq_map]

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for qid in present:
        row = cur_pq_map[qid]
        by_cat[row["category"]].append(row)

    per_category: dict[str, dict] = {}
    for cat, rows in by_cat.items():
        n = len(rows)
        r5 = sum(r["hit_at_5"] for r in rows) / n
        r10 = sum(r["hit_at_10"] for r in rows) / n
        mrr = sum(r["mrr"] for r in rows) / n
        per_category[cat] = {
            "n": n,
            "r@5": round(r5, 4),
            "r@10": round(r10, 4),
            "mrr": round(mrr, 4),
        }

    total_rows = [cur_pq_map[qid] for qid in present]
    global_r5 = (
        sum(r["hit_at_5"] for r in total_rows) / len(total_rows)
        if total_rows else 0.0
    )

    return {
        "label": label,
        "label_count": len(present),
        "anchor_count": len(anchor_ids),
        "dropped_ids": dropped,
        "query_md5": provenance.get("query_file_md5"),
        "git_head": provenance.get("git_head"),
        "db_path": provenance.get("db_path"),
        "substrate_counts": provenance.get("substrate_counts"),
        "global_r@5": round(global_r5, 4),
        "per_category_r@5": per_category,
        "per_query": total_rows,
    }


def _build_subsets(scored_result: dict, provenance: dict) -> dict:
    """Build the top-level ``subsets`` block for a scored artifact.

    Produces up to four slices: preserved_33, subset_38, new_label_only
    (queries in current run that are NOT in the 33-label anchor), and
    full_57 (all currently-scored queries). Only slices with at least one
    present query are included.
    """
    cur_pq_map = {r["id"]: r for r in (scored_result.get("per_query") or [])}
    all_current_ids = list(cur_pq_map.keys())

    subsets: dict[str, dict] = {}

    s33 = _build_subset("preserved_33", PRESERVED_33_QUERY_IDS, scored_result, provenance)
    if s33["label_count"] > 0:
        subsets["preserved_33"] = s33

    s38 = _build_subset("subset_38", SUBSET_38_QUERY_IDS, scored_result, provenance)
    if s38["label_count"] > 0:
        subsets["subset_38"] = s38

    # new_label_only: queries in current run NOT in the 33-label anchor.
    preserved_set = set(PRESERVED_33_QUERY_IDS)
    new_ids = [qid for qid in all_current_ids if qid not in preserved_set]
    if new_ids:
        subsets["new_label_only"] = _build_subset(
            "new_label_only", new_ids, scored_result, provenance
        )

    # full_57: all currently-scored queries (mirrors top-level but nested).
    if all_current_ids:
        subsets["full_57"] = _build_subset(
            "full_57", all_current_ids, scored_result, provenance
        )

    return subsets


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
                             "(flags drops > 0.05 R@5 on comparable intersect)")
    parser.add_argument("--enforce-regression", action="store_true",
                        default=False,
                        help="Exit non-zero (rc=3) when the comparable-label "
                             "category regression gate fires. Off by default; "
                             "set AE_BENCH_ENFORCE_REGRESSION=1 to enable via env.")
    args = parser.parse_args()

    # Env-var override for --enforce-regression (lets AOR plist control it).
    enforce_regression = (
        args.enforce_regression
        or os.environ.get("AE_BENCH_ENFORCE_REGRESSION", "").strip() == "1"
    )

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
            result["subsets"] = _build_subsets(result, result["provenance"])
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
    if args.mode == "scored" and enforce_regression:
        gate = result.get("category_regression_gate", {})
        if gate.get("regression_detected"):
            print(
                f"REGRESSION DETECTED on comparable intersect: "
                f"{gate.get('regressions')}",
                file=sys.stderr,
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
