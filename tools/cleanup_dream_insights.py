#!/usr/bin/env python3.11
"""cleanup_dream_insights.py — one-shot dedup for the dream_insights table.

Caught 2026-05-01: dream_engine._backend.add_insight() does unconditional
INSERT, so every dream cycle re-emits the same bridge/cluster insights.
Live DB had 4.3M rows where only 1,879 were unique (insight_type,
source_memory_id, content) tuples — 99.95% duplication, ~494 MB of disk.

This tool deduplicates by keeping MAX(id) per unique tuple. Wrapped in a
transaction with explicit confirmation: defaults to --dry-run; pass --execute
to actually mutate.

Note: this fixes the data layer. Preventing future bloat requires adding a
UNIQUE INDEX + INSERT OR IGNORE in dream_engine.py — that's a code change
in the dream-engine lane (Valiendo's D5 work area) and not done here.

Usage:
    tools/cleanup_dream_insights.py            # dry-run; reports what would happen
    tools/cleanup_dream_insights.py --execute  # actually delete + VACUUM
    tools/cleanup_dream_insights.py --execute --no-vacuum  # skip VACUUM (faster)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

_DEFAULT_DB = Path.home() / ".neural_memory" / "memory.db"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(_DEFAULT_DB), help="DB path")
    p.add_argument("--execute", action="store_true",
                   help="Actually mutate (default is dry-run)")
    p.add_argument("--no-vacuum", action="store_true",
                   help="Skip VACUUM (faster but no disk reclaim)")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db))

    # --- before ---------------------------------------------------------
    before = conn.execute("SELECT COUNT(*) FROM dream_insights").fetchone()[0]
    unique = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT 1 FROM dream_insights "
        "  GROUP BY insight_type, source_memory_id, COALESCE(content, '')"
        ")"
    ).fetchone()[0]
    will_delete = before - unique
    pct = 100.0 * will_delete / before if before else 0.0

    print(f"BEFORE:")
    print(f"  total rows:    {before}")
    print(f"  unique combos: {unique}")
    print(f"  to delete:     {will_delete} ({pct:.2f}%)")
    print()

    if not args.execute:
        print(f"DRY-RUN. Pass --execute to actually delete.")
        return 0

    # --- mutate ---------------------------------------------------------
    print(f"EXECUTE: deleting {will_delete} rows in a single transaction...")
    t0 = time.time()
    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "DELETE FROM dream_insights "
            "WHERE id NOT IN ("
            "  SELECT MAX(id) FROM dream_insights "
            "  GROUP BY insight_type, source_memory_id, COALESCE(content, '')"
            ")"
        )
        deleted = cur.rowcount
        after = conn.execute("SELECT COUNT(*) FROM dream_insights").fetchone()[0]
        if after != unique:
            print(f"SAFETY ABORT: post-delete count {after} != expected unique {unique}")
            print("Rolling back. No changes committed.")
            conn.rollback()
            return 2
        conn.commit()
        elapsed = time.time() - t0
        print(f"  deleted: {deleted} rows in {elapsed:.1f}s")
        print(f"  after:   {after} rows remain")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}; rolled back. No changes committed.", file=sys.stderr)
        return 3

    # --- vacuum ---------------------------------------------------------
    if args.no_vacuum:
        print(f"\nSkipping VACUUM (--no-vacuum). Disk will reclaim on next auto-VACUUM.")
        return 0

    print(f"\nRunning VACUUM to reclaim disk...")
    t0 = time.time()
    size_before = db.stat().st_size
    conn.execute("VACUUM")
    elapsed = time.time() - t0
    size_after = db.stat().st_size
    reclaimed = size_before - size_after
    print(f"  VACUUM done in {elapsed:.1f}s")
    print(f"  reclaimed: {reclaimed / (1024 * 1024):.1f} MB "
          f"({size_before / (1024 ** 3):.2f} GB → {size_after / (1024 ** 3):.2f} GB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
