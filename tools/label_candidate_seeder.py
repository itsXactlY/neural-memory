#!/usr/bin/env python3
"""label_candidate_seeder.py — Step 2 of labeling methodology.

Per codex labeling-strategy finding 2026-05-02 (~/.neural_memory/codex-subagent-runs/
nm-builder/20260502-035227-research-labeling-strategy_.md):

  Step 2: For each unlabeled query, run three read-only retrieval views:
          hybrid_recall(k=10, rerank=on/off) + sparse_search(k=10).
          Reason: scored mode uses hybrid_recall, not plain recall().

  Step 3: Auto-seed candidate GT sets from:
          - overlap with top-k IDs across retrieval modes
          - overlap with neighboring labeled queries in same family
          - lexical/entity anchor matches

This script does Steps 2 + 3. Output is per-query candidate JSONL ready
for human/Sonnet family-level review (Step 5) before writeback (Step 6).

Inputs:
  - ~/.neural_memory/label-families/<TS>-buckets.jsonl  (Step 1 output)
  - ~/.neural_memory/memory.db                          (substrate)

Outputs:
  - ~/.neural_memory/label-families/<TS>-candidates.jsonl
    Per unlabeled query: candidate_ids ranked by retrieval-mode-overlap
    + seed_ids + memory_snippets for downstream review

Usage:
    python3 tools/label_candidate_seeder.py             # priority families only
    python3 tools/label_candidate_seeder.py --all       # all families
    python3 tools/label_candidate_seeder.py --family sarah_contact

Defaults exclude unfamilied bucket (197 queries — too noisy without anchor signal).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

LABEL_DIR = Path.home() / ".neural_memory" / "label-families"
DB_PATH = str(Path.home() / ".neural_memory" / "memory.db")

# Families codex flagged as highest-priority for first labeling pass.
PRIORITY_FAMILIES = {
    "sarah_contact",         # 5 unlabeled, seeds 264/268/280/282
    "panel_labels",          # 2 unlabeled, seeds 274/286
    "qbo_reauth",            # 2 unlabeled, no seeds
    "permit_paperwork",      # 5 unlabeled, no seeds
    "gfci",                  # 3 unlabeled, seeds 277/288/4666
    "ev_charger",            # 1 unlabeled, seed 158
    "bonding_grounding",     # 3 unlabeled, seed 5961
}


def load_buckets(jsonl_path: Path) -> List[dict]:
    return [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]


def latest_buckets() -> Path:
    files = sorted(LABEL_DIR.glob("*-buckets.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No buckets file in {LABEL_DIR}. Run tools/label_family_bucketing.py first.")
    return files[-1]


def make_neural_memory():
    """Lazy-load NeuralMemory with bench-friendly settings."""
    from memory_client import NeuralMemory  # noqa: WPS433
    return NeuralMemory(
        db_path=DB_PATH,
        embedding_backend="auto",
        use_cpp=False,
        use_hnsw=False,
    )


def fetch_snippet(mem, memory_id: int) -> str:
    """Get first ~120 chars of a memory's content for review JSONL.
    NeuralMemory.get_memory() returns dict with 'content'/'label' (not 'text').
    """
    try:
        m = mem.get_memory(memory_id)
        if not m:
            return ""
        for k in ("content", "label"):
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                return v[:120].replace("\n", " ")
    except Exception:
        pass
    return ""


def seed_candidates_for_query(mem, query: str, family: dict, k: int = 10) -> dict:
    """Run 3 retrieval views, aggregate candidate IDs with mode overlap.

    Returns: {
        "candidate_ids": [(id, modes_count, modes_list), ...],
        "seed_ids_from_family": [...],
        "snippets": {id: first_120_chars},
    }
    """
    mode_hits: Dict[int, List[str]] = {}

    # Mode 1: hybrid_recall with rerank ON (the scored bench's actual surface)
    try:
        for r in mem.hybrid_recall(query, k=k, rerank=True):
            mid = r["id"]
            mode_hits.setdefault(mid, []).append("hybrid_rerank")
    except Exception as e:
        print(f"    hybrid_recall(rerank=on) error: {e}", file=sys.stderr)

    # Mode 2: hybrid_recall with rerank OFF (raw scored channels)
    try:
        for r in mem.hybrid_recall(query, k=k, rerank=False):
            mid = r["id"]
            mode_hits.setdefault(mid, []).append("hybrid_no_rerank")
    except Exception as e:
        print(f"    hybrid_recall(rerank=off) error: {e}", file=sys.stderr)

    # Mode 3: sparse_search (lexical-only signal)
    try:
        for r in mem.sparse_search(query, k=k):
            mid = r["id"]
            mode_hits.setdefault(mid, []).append("sparse")
    except Exception as e:
        print(f"    sparse_search error: {e}", file=sys.stderr)

    # Add family seeds (treated as bonus candidates per codex methodology)
    seed_ids = family.get("seed_ground_truth_ids", [])
    for sid in seed_ids:
        mode_hits.setdefault(sid, []).append("family_seed")

    # Rank candidates: more modes = higher confidence
    candidates = sorted(
        mode_hits.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )

    # Fetch snippets for top candidates only (keeps output tight)
    snippets = {mid: fetch_snippet(mem, mid) for mid, _ in candidates[:15]}

    return {
        "candidate_ids": [
            {"id": mid, "modes_count": len(modes), "modes": sorted(set(modes))}
            for mid, modes in candidates[:15]  # top 15 per query
        ],
        "seed_ids_from_family": seed_ids,
        "snippets": snippets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets-file", help="Override buckets JSONL path")
    ap.add_argument("--all", action="store_true",
                    help="Process all families (default: priority + ev_charger + bonding_grounding)")
    ap.add_argument("--family", help="Process only this family name")
    ap.add_argument("--k", type=int, default=10, help="Top-k per retrieval mode")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max queries to process (default: all matching)")
    args = ap.parse_args()

    buckets_path = Path(args.buckets_file) if args.buckets_file else latest_buckets()
    print(f"Reading buckets: {buckets_path}", file=sys.stderr)
    buckets = load_buckets(buckets_path)

    # Filter families
    if args.family:
        buckets = [b for b in buckets if b["family"] == args.family]
    elif not args.all:
        buckets = [b for b in buckets if b["family"] in PRIORITY_FAMILIES]
    # Skip unfamilied unconditionally (too noisy without anchor)
    buckets = [b for b in buckets if b["family"] != "unfamilied"]

    queries_to_process = []
    for fam in buckets:
        for q in fam["unlabeled_queries"]:
            queries_to_process.append((fam, q))
    if args.limit:
        queries_to_process = queries_to_process[:args.limit]

    print(f"Will process {len(queries_to_process)} unlabeled queries across "
          f"{len(buckets)} families", file=sys.stderr)

    if not queries_to_process:
        print("No queries to process.", file=sys.stderr)
        return 0

    print("Loading NeuralMemory (this takes ~10s for cold sentence-transformers)...",
          file=sys.stderr)
    mem = make_neural_memory()
    # Substrate size — try a few common surfaces, skip if none work
    try:
        import sqlite3 as _sql
        n = _sql.connect(DB_PATH).execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print(f"  loaded; substrate has {n} memories", file=sys.stderr)
    except Exception:
        print(f"  loaded (substrate count unavailable)", file=sys.stderr)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = LABEL_DIR / f"{ts}-candidates.jsonl"

    with out_path.open("w") as out:
        for i, (fam, q) in enumerate(queries_to_process, 1):
            print(f"[{i}/{len(queries_to_process)}] {q['id']} ({fam['family']}): "
                  f"{q['query'][:60]}...", file=sys.stderr)
            t0 = time.time()
            seed_result = seed_candidates_for_query(mem, q["query"], fam, k=args.k)
            elapsed = time.time() - t0
            record = {
                "query_id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "family": fam["family"],
                "family_description": fam["description"],
                "elapsed_sec": round(elapsed, 2),
                **seed_result,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    print(f"\n→ wrote {out_path}", file=sys.stderr)
    print(f"\nNext step: human/Sonnet review of the candidate JSONL — approve "
          f"per-family GT writebacks, then re-run scored bench to measure lift.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
