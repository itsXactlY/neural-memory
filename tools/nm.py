#!/usr/bin/env python3
"""nm — neural-memory CLI.

Direct terminal handle on Phase 7 features. Bypasses hermes for cases where
tito wants to inspect / record / recall / forget / audit memories from the
shell.

Usage:
    nm remember "Sarah called about lot 27" --kind=experience --source=phone
    nm recall "Lennar inspection" --k=5
    nm recall "current Lennar contact" --as-of=now
    nm sparse "GFCI exterior receptacles"
    nm graph "Who handles Lennar?" --hops=2
    nm explain "How do we estimate panels?" --kind=procedural
    nm audit
    nm count
    nm entities --top=10
    nm forget 123 --mode=background
    nm forget 123 --mode=redact
    nm bench --category=lennar_lots

All commands accept --db PATH to override the default ~/.neural_memory/memory.db.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))


def _parse_as_of(s: str | None) -> float | None:
    if s is None or s.lower() == "none":
        return None
    if s.lower() == "now":
        return time.time()
    # Accept ISO-ish or unix epoch
    try:
        return float(s)
    except ValueError:
        pass
    import datetime as _dt
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise SystemExit(f"can't parse --as-of value: {s!r}")


def _load(db_path: str | None):
    from memory_client import NeuralMemory  # noqa
    kwargs = {"embedding_backend": "auto", "use_cpp": False, "use_hnsw": False}
    if db_path:
        kwargs["db_path"] = db_path
    return NeuralMemory(**kwargs)


def _print_results(results: list[dict], format_: str = "compact") -> None:
    if format_ == "json":
        print(json.dumps(results, indent=2, default=str))
        return
    if not results:
        print("(no results)")
        return
    for r in results:
        sim = r.get("similarity", r.get("activation", 0))
        sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else str(sim)
        label = (r.get("label") or "")[:30]
        content = (r.get("content") or "")[:80]
        print(f"  id={r['id']:5d}  sim={sim_s:8s}  [{label:30s}]  {content}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_remember(args) -> int:
    mem = _load(args.db)
    metadata = json.loads(args.metadata) if args.metadata else None
    valid_from = _parse_as_of(args.valid_from) if args.valid_from else None
    mid = mem.remember(
        args.text,
        label=args.label or "",
        detect_conflicts=not args.no_conflict_detect,
        kind=args.kind,
        source=args.source,
        origin_system=args.origin or "cli",
        valid_from=valid_from,
        metadata=metadata,
    )
    print(f"stored memory id={mid}")
    return 0


def cmd_recall(args) -> int:
    mem = _load(args.db)
    as_of = _parse_as_of(args.as_of) if args.as_of else None
    results = mem.recall(args.query, k=args.k, kind=args.kind, as_of=as_of)
    _print_results(results, args.format)
    return 0


def cmd_sparse(args) -> int:
    mem = _load(args.db)
    results = mem.sparse_search(args.query, k=args.k)
    _print_results(results, args.format)
    return 0


def cmd_graph(args) -> int:
    mem = _load(args.db)
    results = mem.graph_search(args.query, k=args.k, hops=args.hops)
    _print_results(results, args.format)
    return 0


def cmd_explain(args) -> int:
    mem = _load(args.db)
    as_of = _parse_as_of(args.as_of) if args.as_of else None
    results = mem.explain_recall(args.query, k=args.k, kind=args.kind, as_of=as_of)
    if args.format == "json":
        print(json.dumps(results, indent=2, default=str))
        return 0
    if not results:
        print("(no results)")
        return 0
    for r in results:
        e = r.get("explanation", {})
        print(f"id={r['id']}  score={e.get('final_score', 0):.3f}  intent={e.get('intent','?')}")
        print(f"  channels: {e.get('channels')}")
        print(f"  features: {e.get('features')}")
        print(f"  content: {(r.get('content') or '')[:120]}")
        print()
    return 0


def cmd_audit(args) -> int:
    sys.path.insert(0, str(_ROOT / "tools"))
    from phase7_audit import audit, render_text  # noqa
    db_path = args.db or str(Path.home() / ".neural_memory" / "memory.db")
    report = audit(db_path)
    print(render_text(report))
    return 0


def cmd_count(args) -> int:
    mem = _load(args.db)
    stats = mem.store.get_stats()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_entities(args) -> int:
    mem = _load(args.db)
    if not mem.entities:
        print("(entity registry not available)")
        return 1
    rows = mem.store.conn.execute(
        "SELECT id, label, metadata_json FROM memories WHERE kind='entity'"
    ).fetchall()
    parsed = []
    for r in rows:
        meta = json.loads(r[2]) if r[2] else {}
        parsed.append((r[0], r[1], meta.get("frequency", 0)))
    parsed.sort(key=lambda x: -x[2])
    if args.format == "json":
        print(json.dumps([
            {"id": p[0], "label": p[1], "frequency": p[2]}
            for p in parsed[:args.top]
        ], indent=2))
        return 0
    print(f"{'id':>6}  {'freq':>5}  label")
    for p in parsed[:args.top]:
        print(f"{p[0]:>6}  {p[2]:>5}  {p[1]}")
    return 0


def cmd_forget(args) -> int:
    mem = _load(args.db)
    mem.forget(args.id, mode=args.mode)
    print(f"forgot id={args.id} mode={args.mode}")
    return 0


def cmd_bench(args) -> int:
    sys.path.insert(0, str(_ROOT / "benchmarks" / "ae_domain_memory_bench"))
    from run_ae_domain_bench import (  # noqa
        run_diagnostic, run_scored, _load_neural_memory,
    )
    from queries import get_queries  # noqa
    queries = get_queries(args.category)
    mem = _load_neural_memory(args.db)
    if args.mode == "scored":
        result = run_scored(mem, queries, k=args.k)
    else:
        result = run_diagnostic(mem, queries, k=args.k)
    print(json.dumps(result, indent=2))
    return 0


def cmd_memify(args) -> int:
    mem = _load(args.db)
    stats = mem.run_memify_once(decay_factor=args.decay)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_contradiction(args) -> int:
    mem = _load(args.db)
    stats = mem.run_contradiction_detection_once(jaccard_threshold=args.threshold)
    print(json.dumps(stats, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(prog="nm", description="neural-memory CLI")
    p.add_argument("--db", default=None, help="DB path override")
    sub = p.add_subparsers(dest="cmd", required=True)

    # remember
    r = sub.add_parser("remember", help="Store a new memory")
    r.add_argument("text")
    r.add_argument("--label", default="")
    r.add_argument("--kind", default=None)
    r.add_argument("--source", default=None)
    r.add_argument("--origin", default=None)
    r.add_argument("--valid-from", dest="valid_from", default=None)
    r.add_argument("--metadata", default=None, help="JSON dict")
    r.add_argument("--no-conflict-detect", action="store_true")
    r.set_defaults(fn=cmd_remember)

    # recall
    rc = sub.add_parser("recall", help="Semantic recall (Phase 7 typed)")
    rc.add_argument("query")
    rc.add_argument("--k", type=int, default=5)
    rc.add_argument("--kind", default=None)
    rc.add_argument("--as-of", dest="as_of", default=None,
                    help="ISO date, unix epoch, or 'now'")
    rc.add_argument("--format", choices=("compact", "json"), default="compact")
    rc.set_defaults(fn=cmd_recall)

    # sparse
    sp = sub.add_parser("sparse", help="FTS5 sparse retrieval")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--format", choices=("compact", "json"), default="compact")
    sp.set_defaults(fn=cmd_sparse)

    # graph
    g = sub.add_parser("graph", help="PPR graph search with intent weights")
    g.add_argument("query")
    g.add_argument("--k", type=int, default=5)
    g.add_argument("--hops", type=int, default=2)
    g.add_argument("--format", choices=("compact", "json"), default="compact")
    g.set_defaults(fn=cmd_graph)

    # explain
    e = sub.add_parser("explain", help="Recall with explanation paths")
    e.add_argument("query")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--kind", default=None)
    e.add_argument("--as-of", dest="as_of", default=None)
    e.add_argument("--format", choices=("compact", "json"), default="compact")
    e.set_defaults(fn=cmd_explain)

    # audit
    a = sub.add_parser("audit", help="Phase 7 health audit")
    a.set_defaults(fn=cmd_audit)

    # count
    c = sub.add_parser("count", help="Memory + connection counts")
    c.set_defaults(fn=cmd_count)

    # entities
    en = sub.add_parser("entities", help="Top entities by mention frequency")
    en.add_argument("--top", type=int, default=10)
    en.add_argument("--format", choices=("compact", "json"), default="compact")
    en.set_defaults(fn=cmd_entities)

    # forget
    f = sub.add_parser("forget", help="Soft-forget / redact / delete a memory")
    f.add_argument("id", type=int)
    f.add_argument("--mode", choices=("background", "redact", "delete"),
                   default="background")
    f.set_defaults(fn=cmd_forget)

    # bench
    b = sub.add_parser("bench", help="Run AE-domain bench")
    b.add_argument("--mode", choices=("diagnostic", "scored"), default="diagnostic")
    b.add_argument("--category", default=None)
    b.add_argument("--k", type=int, default=10)
    b.set_defaults(fn=cmd_bench)

    # memify
    m = sub.add_parser("memify", help="Run dream Memify hygiene pass")
    m.add_argument("--decay", type=float, default=0.5)
    m.set_defaults(fn=cmd_memify)

    # contradiction
    cd = sub.add_parser("contradiction", help="Run contradiction detection")
    cd.add_argument("--threshold", type=float, default=0.4)
    cd.set_defaults(fn=cmd_contradiction)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
