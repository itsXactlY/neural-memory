#!/usr/bin/env python3
"""
bake_afe_facts.py — run AFE once over the pre-baked session cache.

Reads `mm10m_bench.longmemeval_s_bgem3_1024.memories` (populated by
`bake_longmemeval_s_cache.py`) and feeds every long session through the
DreamEngine's AFE phase.  The extracted atomic facts land in the SAME
schema with labels of the form `session:<sid>::afe::<stage><idx>` — the
bench scorer's `label.split(':')` still surfaces the `<sid>` segment,
so AFE facts inherit gold-matchability for free.

Why this exists
---------------
LongMemEval-S sessions are 2-18 KB of multi-turn dialogue.  Dense
embeddings of the whole session are dominated by surface topic;
answer-bearing sentences get diluted.  AFE breaks each session into
atomic facts (one sentence per memory) and re-embeds — recall now
finds the exact-fact rows, which still carry the source sid in their
label.

Modes
-----
Default — regex + spaCy NER.  ~1 h for 23,882 sessions on this host.
  python benchmarks/bake_afe_facts.py

LLM fallback — DeepHermes-3-3B locally, much slower but higher recall.
  MAZEMAKER_AFE_LLM_FALLBACK=1 python benchmarks/bake_afe_facts.py

USAGE
    python benchmarks/bake_afe_facts.py                   # full pass
    python benchmarks/bake_afe_facts.py --max-sources 100 # smoke test
    python benchmarks/bake_afe_facts.py --rebuild         # wipe AFE rows first

After completion, re-dump the snapshot — the dump now includes facts:
    pg_dump -U mazemaker -d mm10m_bench \\
        --schema=longmemeval_s_bgem3_1024 \\
        -Fc -f benchmarks/snapshots/longmemeval_s_bgem3_1024.dump
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_SRC = REPO / "python"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

CACHE_DB = "mm10m_bench"
CACHE_SCHEMA = "longmemeval_s_bgem3_1024"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sources", type=int, default=0,
                        help="Cap on sources processed per cycle (0 = all). "
                             "Overrides MAZEMAKER_AFE_MAX_PER_CYCLE.")
    parser.add_argument("--min-len", type=int, default=500,
                        help="Skip memories shorter than this (default 500 "
                             "chars; matches the AFE default).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing AFE facts (label LIKE "
                             "'%%::afe::%%') before running.")
    parser.add_argument("--variant", default="s",
                        choices=["s", "m", "oracle"],
                        help="LongMemEval variant: s (default), m, or oracle")
    args = parser.parse_args()
    global CACHE_SCHEMA
    CACHE_SCHEMA = f"longmemeval_{args.variant}_bgem3_1024"

    # PG env — point Mazemaker at the cache schema directly.
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = CACHE_DB
    os.environ["MM_POSTGRES_SCHEMA"] = CACHE_SCHEMA
    # Defer HNSW index updates — AFE writes thousands of facts; rebuild
    # the embedding HNSW once at the end (already idempotent via
    # `CREATE INDEX IF NOT EXISTS` in PostgresStore.create_bulk_indexes).
    os.environ["MM_DEFER_HNSW"] = "1"

    # AFE config — let env override `--max-sources` so this script
    # doesn't fight the operator's env if they've already set it.
    if args.max_sources:
        os.environ["MAZEMAKER_AFE_MAX_PER_CYCLE"] = str(args.max_sources)
    os.environ["MAZEMAKER_AFE_MIN_LEN"] = str(args.min_len)

    print(f"[afe-bake] DB:     {CACHE_DB}.{CACHE_SCHEMA}")
    print(f"[afe-bake] min_len={args.min_len}  "
          f"max_per_cycle={os.environ.get('MAZEMAKER_AFE_MAX_PER_CYCLE', 'unbounded')}")
    print(f"[afe-bake] LLM fallback: "
          f"{os.environ.get('MAZEMAKER_AFE_LLM_FALLBACK', '0')}")

    if args.rebuild:
        from postgres_store import _build_dsn
        import psycopg
        import urllib.parse as _u
        p = _u.urlparse(_build_dsn())
        dsn = _u.urlunparse(p._replace(path=f"/{CACHE_DB}"))
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DELETE FROM "{CACHE_SCHEMA}".memories '
                    f'WHERE label LIKE \'%::afe::%\''
                )
                cur.execute(
                    f'DELETE FROM "{CACHE_SCHEMA}".memories '
                    f'WHERE label LIKE \'%::afe::%\''
                )
                # Wipe the AFE done-set so the phase reprocesses
                # everything next run.
                cur.execute(
                    f'DELETE FROM "{CACHE_SCHEMA}".meta '
                    f'WHERE key=\'afe_processed_ids\''
                )
                print("[afe-bake] --rebuild: AFE rows + done-set cleared")

    from memory_client import Mazemaker
    from dream_engine import DreamEngine
    from dream_postgres_store import DreamPostgresStore

    print("[afe-bake] Building Mazemaker on cache schema...")
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",  # AFE doesn't recall; semantic is fine
        rerank=False,
    )
    nm.store._ensure_embedding_column(1024)

    # DreamEngine takes a DreamBackend; the convenience path (passing a
    # Mazemaker as first arg) hardcodes SQLite via nm._db_path.  On PG we
    # bind a DreamPostgresStore explicitly so the dream-side tables land
    # in the same schema as the memory store.
    backend = DreamPostgresStore()
    de = DreamEngine(backend, neural_memory=nm)

    # Pre-flight: how many sources qualify?
    with nm.store._cursor() as (_c, cur):
        cur.execute(
            f'SELECT count(*) FROM memories '
            f'WHERE length(content) >= %s '
            f'  AND label NOT LIKE %s',
            (args.min_len, '%::afe::%'),
        )
        n_eligible = int(cur.fetchone()[0])
    print(f"[afe-bake] Eligible sources: {n_eligible:,}")

    # Cycle until no more work.  `_phase_afe()` processes up to
    # MAZEMAKER_AFE_MAX_PER_CYCLE per call and persists its done-set,
    # so re-calling resumes cleanly.
    t_start = time.perf_counter()
    cycle = 0
    total_written = 0
    while True:
        cycle += 1
        t_cycle = time.perf_counter()
        stats = de._phase_afe()
        cycle_s = time.perf_counter() - t_cycle
        processed = stats.get("processed", 0)
        written = stats.get("written", 0)
        facts = stats.get("facts_extracted", 0)
        by_stage = stats.get("by_stage", {})
        err = stats.get("error")
        total_written += written
        print(
            f"  cycle {cycle:>3}  sources={processed:>4}  "
            f"facts={facts:>5}  written={written:>5}  "
            f"by_stage={by_stage}  {cycle_s:.1f}s"
            + (f"  ERROR={err}" if err else "")
        )
        if err and "no_store" in err or "afe_import_failed" in (err or ""):
            print(f"[afe-bake] FATAL: {err}")
            return 2
        if processed == 0:
            print("[afe-bake] No more sources to process — done.")
            break

    elapsed = time.perf_counter() - t_start
    print(f"\n[afe-bake] DONE  {total_written:,} fact memories written "
          f"in {elapsed/60:.1f}m")
    print(f"[afe-bake] Next: pg_dump -U mazemaker -d {CACHE_DB} "
          f"--schema={CACHE_SCHEMA} -Fc "
          f"-f benchmarks/snapshots/{CACHE_SCHEMA}.dump")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
