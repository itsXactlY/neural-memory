#!/usr/bin/env python3
"""post_ingest_sanity.py — verify the AE canonical corpus is retrievable.

Runs 10-15 known-good retrieval contracts against the live DB. Each
contract has a query + an expected substring/source-label that should
appear in the top-K results. Exits non-zero if any contract fails —
suitable for use as a daily health check after the ingest cron.

These are SANITY checks, not benchmark scoring. Designed to catch:
- FTS5 index drift (e.g., entity rows polluting again)
- Embedding backend regression (different vectors → wrong rankings)
- Stopword over-aggression (real terms filtered out)
- Schema migration issues that break retrieval paths

Run:
    python3 tools/post_ingest_sanity.py
    python3 tools/post_ingest_sanity.py --db ~/.neural_memory/memory.db
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))


# Each contract: (description, query, channel, expected_substring_in_any_top_result)
# expected_substring is matched case-insensitively against
# top-K result content OR label OR metadata.source_path.
_CONTRACTS = [
    # Operator skill content
    ("operator skill — Tito identity",
     "Who is Tito?",
     "sparse",
     "tito"),
    ("operator skill — canonical 7-item workflow",
     "canonical 7-item workflow",
     "sparse",
     "ae_operator_skill"),
    ("operator skill — materials catalog reference",
     "Amperage catalog supplier",
     "sparse",
     "ae_operator_skill"),
    # LangGraph kernel content
    ("kernel — executive autonomy doctrine",
     "executive autonomy doctrine",
     "sparse",
     "executive_autonomy"),
    ("kernel — connector policy",
     "connector policy",
     "sparse",
     "connector"),
    ("kernel — data authority map",
     "data authority map",
     "sparse",
     "data_authority"),
    # Phase 7 / neural-memory state
    ("session wrap — Phase 7 commits",
     "Phase 7 unified-graph donor-organ integration",
     "sparse",
     "phase7"),
    # Sprint research
    ("sprint research — Pulse verdict",
     "Pulse 2-tier CDP Exa",
     "sparse",
     "pulse"),
    # Entity expansion
    ("entity — Hermes appears as named entity",
     "Hermes",
     "graph",
     None),  # graph search; check top-3 ids exist
    ("entity — Lennar appears as named entity",
     "Lennar",
     "graph",
     None),
    # Procedural retrieval
    ("procedural — kind filter returns operator-skill chunks",
     "How does Tito want estimates handled",
     "recall_kind_procedural",
     "ae_operator_skill"),
    # Recent valiendo handoff
    ("valiendo handoff — V7 convergence content",
     "V7 convergence canon sync",
     "sparse",
     "valiendo"),
]


def _load(db_path: str | None):
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        from memory_client import NeuralMemory
        # Prefer sentence-transformers if installed (semantic > lexical for
        # natural-language queries). Falls through to auto-detect if not.
        # Caught 2026-05-01: TF-IDF backend top-25 doesn't carry enough
        # procedural-kind memories for kind-filter queries to work; sentence-
        # transformers does.
        backend = "sentence-transformers"
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            backend = "auto"
        kwargs = {"embedding_backend": backend,
                  "use_cpp": False, "use_hnsw": False}
        if db_path:
            kwargs["db_path"] = db_path
        mem = NeuralMemory(**kwargs)
    print(captured.getvalue(), file=sys.stderr, end="")
    return mem


def _result_contains(results: list[dict], expected: str | None) -> bool:
    """Check if any top-K result has expected substring in content / label /
    metadata.source_path. None means just check len > 0."""
    if not results:
        return False
    if expected is None:
        return True  # contract just checks "got results"
    needle = expected.lower()
    for r in results:
        if needle in (r.get("content") or "").lower():
            return True
        if needle in (r.get("label") or "").lower():
            return True
        meta = r.get("metadata_json") or "{}"
        try:
            meta_obj = json.loads(meta) if isinstance(meta, str) else meta
            if needle in (meta_obj.get("source_path") or "").lower():
                return True
            if needle in (meta_obj.get("source_label") or "").lower():
                return True
        except Exception:
            pass
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=None, help="DB path override")
    p.add_argument("--k", type=int, default=5, help="top-K to inspect")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mem = _load(args.db)

    passed = 0
    failed: list[tuple[str, str, str]] = []

    for desc, query, channel, expected in _CONTRACTS:
        if channel == "sparse":
            results = mem.sparse_search(query, k=args.k)
        elif channel == "graph":
            results = mem.graph_search(query, k=args.k, hops=1)
        elif channel == "recall_kind_procedural":
            results = mem.recall(query, k=args.k, kind="procedural")
        elif channel == "hybrid":
            results = mem.hybrid_recall(query, k=args.k)
        else:
            results = mem.recall(query, k=args.k)

        ok = _result_contains(results, expected)
        if ok:
            passed += 1
            if args.verbose:
                print(f"  PASS  {desc}")
        else:
            failed.append((desc, query, channel))
            print(f"  FAIL  {desc}")
            print(f"        query: {query}")
            print(f"        channel: {channel}, expected: {expected!r}")
            print(f"        got: {[r.get('id') for r in results[:5]]}")

    print(f"\n{passed}/{len(_CONTRACTS)} contracts passed")

    try:
        mem.close()
    except Exception:
        pass

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
