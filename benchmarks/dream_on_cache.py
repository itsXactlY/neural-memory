#!/usr/bin/env python3
"""dream_on_cache.py — run a full dream cycle directly on the cache schema.

Binds Mazemaker + DreamEngine to `mm10m_bench.longmemeval_<variant>_bgem3_1024`
and runs N cycles until quiescent (NREM strengthened+pruned, REM bridges,
and Insight clusters all drop to ~zero for 2 consecutive cycles).

Writes every dream artifact INTO the cache schema so the eventual pg_dump
snapshot is fully self-contained. After this script + snapshot, bench
runs just pipe the snapshot and recall against fully-dreamt state.

USAGE
    python benchmarks/dream_on_cache.py                # --variant s
    python benchmarks/dream_on_cache.py --variant oracle
    python benchmarks/dream_on_cache.py --max-cycles 8
    python benchmarks/dream_on_cache.py --max-cycles 1   # smoke

Tunable via env (defaults set in dream_engine.py):
    MM_REM_SIM_LOW          (default 0.3, dense AFE corpora often want 0.3)
    MM_REM_SIM_HIGH         (default 0.95, AFE chunks may want 0.99)
    MM_INSIGHT_MIN_CLUSTER  (default 10, bench corpora may want 4)
    MM_INSIGHT_MAX_CLUSTERS (default 50)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    p.add_argument("--schema", default=None,
                   help="Override schema (default: longmemeval_<variant>_bgem3_1024)")
    p.add_argument("--max-cycles", type=int, default=10,
                   help="Hard upper bound on cycles (default 10)")
    p.add_argument("--min-cycles", type=int, default=3,
                   help="Always run at least this many cycles even if quiescent (default 3)")
    p.add_argument("--max-memories", type=int, default=4000,
                   help="NREM batch size (default 4000)")
    p.add_argument("--max-isolated", type=int, default=2000,
                   help="REM batch size (default 2000)")
    p.add_argument("--no-afe", action="store_true",
                   help="Skip AFE phase (already baked separately via bake_afe_facts.py)")
    args = p.parse_args()

    schema = args.schema or f"longmemeval_{args.variant}_bgem3_1024"
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = CACHE_DB
    os.environ["MM_POSTGRES_SCHEMA"] = schema
    # Bench-corpus tuning baked in unless operator pre-set them.
    os.environ.setdefault("MM_INSIGHT_MIN_CLUSTER", "4")
    os.environ.setdefault("MM_REM_SIM_HIGH", "0.99")
    # AFE re-runs are expensive — skip unless operator opts in.
    if args.no_afe:
        os.environ["MAZEMAKER_AFE_MIN_LEN"] = "10000000"  # effectively disable

    print(f"[dream-on-cache] DB={CACHE_DB} schema={schema}")
    print(f"[dream-on-cache] max_cycles={args.max_cycles} max_memories={args.max_memories} max_isolated={args.max_isolated}")
    print(f"[dream-on-cache] MM_INSIGHT_MIN_CLUSTER={os.environ['MM_INSIGHT_MIN_CLUSTER']} MM_REM_SIM_HIGH={os.environ['MM_REM_SIM_HIGH']}")

    from memory_client import Mazemaker
    from dream_engine import DreamEngine
    from dream_postgres_store import DreamPostgresStore

    print(f"[dream-on-cache] Building Mazemaker on {schema}...")
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",
        rerank=False,
    )
    nm.store._ensure_embedding_column(1024)

    backend = DreamPostgresStore()
    de = DreamEngine(
        backend,
        neural_memory=nm,
        max_memories_per_cycle=args.max_memories,
        max_isolated_per_cycle=args.max_isolated,
    )

    t0 = time.perf_counter()
    quiescent_streak = 0
    for cycle in range(1, args.max_cycles + 1):
        t_cyc = time.perf_counter()
        stats = de._run_dream_cycle()
        cyc_s = time.perf_counter() - t_cyc

        nrem = stats.get("nrem", {}) or {}
        rem = stats.get("rem", {}) or {}
        sup = stats.get("supersedes", {}) or {}
        ins = stats.get("insights", {}) or {}
        afe = stats.get("afe", {}) or {}
        dae = stats.get("dae", {}) or {}

        bridges = int(rem.get("bridges", 0) or 0)
        insights = int(ins.get("insights", 0) or 0)
        strengthened = int(nrem.get("strengthened", 0) or 0)

        print(
            f"  cycle {cycle:>2}/{args.max_cycles}  "
            f"NREM: proc={nrem.get('processed', 0)} +{strengthened}/-{nrem.get('weakened', 0)}/p{nrem.get('pruned', 0)}  "
            f"SUP: {sup.get('supersedes_found', 0)}  "
            f"REM: bridges={bridges}/rej={rem.get('rejected', 0)}  "
            f"INS: communities={ins.get('communities', 0)} insights={insights} derived={ins.get('derived_facts', 0)}  "
            f"AFE: written={afe.get('written', 0)}  "
            f"DAE: written={dae.get('written', 0) if isinstance(dae, dict) else 0}  "
            f"{cyc_s:.1f}s",
            flush=True,
        )

        # Quiescence: REM produces no new bridges AND Insight produces no
        # new clusters AND NREM strengthens almost nothing.
        is_quiescent = (
            bridges < 10 and insights == 0 and strengthened < 10
        )
        if is_quiescent and cycle >= args.min_cycles:
            quiescent_streak += 1
            if quiescent_streak >= 2:
                print(f"[dream-on-cache] Quiescent for 2 cycles — done.")
                break
        else:
            quiescent_streak = 0

    elapsed = time.perf_counter() - t0
    print(f"\n[dream-on-cache] DONE  {cycle} cycles in {elapsed/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
