#!/usr/bin/env python3
"""neural_memory_production_upgrade.py — PoC → Production Upgrade

Fixes database growth problems in Mazemaker SQLite without data loss.

What it does (in order):
  1. Backup database before any changes
  2. Diagnose current state (sizes, duplicates, orphans, freelist)
  3. Clean connection_history (logging bloat)
  4. Remove orphan connections (pointing to deleted memories)
  5. Deduplicate edges + add UNIQUE constraint (prevent future dupes)
  6. VACUUM (reclaim freelist pages)
  7. Add retention indexes for fast cleanup
  8. Verify integrity after all changes

Usage:
  python3 neural_memory_production_upgrade.py [--db PATH] [--dry-run] [--history-days 7] [--skip-backup]

  --db PATH          Path to SQLite database (default: ~/.mazemaker/data/memory.db)
  --dry-run          Show what would change without modifying anything
  --history-days N   Keep connection_history entries from last N days (default: 7)
  --skip-backup      Skip backup creation (NOT recommended)
  --force            Skip confirmation prompts

Requirements:
  - Python 3.10+
  - sqlite3 (stdlib)
  - Enough disk space for a DB copy (237 MB as of 2026-04-16)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def db_size(db_path: str) -> int:
    return os.path.getsize(db_path)


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [r[0] for r in cur.fetchall()]
    counts = {}
    for t in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        counts[t] = cnt
    return counts


def freelist_info(conn: sqlite3.Connection) -> tuple[int, int, int]:
    pages = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return pages, page_size, freelist


# ---------------------------------------------------------------------------
# Actions (each returns bytes saved or rows affected)
# ---------------------------------------------------------------------------

def backup_db(db_path: str) -> str:
    """Create timestamped backup using sqlite3's online backup API.

    File-copying a WAL-mode SQLite database while writes are in flight
    yields an inconsistent snapshot — the -wal file may be mid-write.
    sqlite3.Connection.backup() is the supported way to clone a live DB.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.bak.{ts}"
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(backup_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return backup_path


def diagnose(conn: sqlite3.Connection, db_path: str) -> dict:
    """Full diagnostic snapshot."""
    counts = table_counts(conn)
    pages, page_size, freelist = freelist_info(conn)
    file_size = db_size(db_path)

    # Duplicate edges
    dupes = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, edge_type, COUNT(*) as cnt
            FROM connections
            GROUP BY source_id, target_id, edge_type
            HAVING cnt > 1
        )
    """).fetchone()[0]

    # Orphan connections
    orphans = conn.execute("""
        SELECT COUNT(*) FROM connections
        WHERE source_id NOT IN (SELECT id FROM memories)
           OR target_id NOT IN (SELECT id FROM memories)
    """).fetchone()[0]

    # Orphan history
    orphan_hist = conn.execute("""
        SELECT COUNT(*) FROM connection_history
        WHERE source_id NOT IN (SELECT id FROM memories)
           OR target_id NOT IN (SELECT id FROM memories)
    """).fetchone()[0]

    # Weight distribution
    weight_dist = {}
    for bucket, lo, hi in [
        ("dead (<0.05)", 0, 0.05),
        ("weak (0.05-0.1)", 0.05, 0.1),
        ("low (0.1-0.3)", 0.1, 0.3),
        ("medium (0.3-0.5)", 0.3, 0.5),
        ("strong (0.5-0.7)", 0.5, 0.7),
        ("very_strong (0.7+)", 0.7, 999),
    ]:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE weight >= ? AND weight < ?",
            (lo, hi)
        ).fetchone()[0]
        weight_dist[bucket] = cnt

    # History age
    hist_oldest = conn.execute(
        "SELECT MIN(changed_at) FROM connection_history"
    ).fetchone()[0]
    hist_newest = conn.execute(
        "SELECT MAX(changed_at) FROM connection_history"
    ).fetchone()[0]

    # History by reason
    hist_reasons = conn.execute(
        "SELECT reason, COUNT(*) FROM connection_history GROUP BY reason ORDER BY 2 DESC"
    ).fetchall()

    # Most connected nodes
    hub_nodes = conn.execute("""
        SELECT source_id, COUNT(*) as cnt FROM connections
        GROUP BY source_id ORDER BY cnt DESC LIMIT 5
    """).fetchall()

    return {
        "file_size": file_size,
        "tables": counts,
        "pages": pages,
        "page_size": page_size,
        "freelist": freelist,
        "freelist_mb": (freelist * page_size) / (1024 * 1024),
        "duplicate_edge_groups": dupes,
        "orphan_connections": orphans,
        "orphan_history": orphan_hist,
        "weight_distribution": weight_dist,
        "history_oldest": hist_oldest,
        "history_newest": hist_newest,
        "history_reasons": dict(hist_reasons) if hist_reasons else {},
        "hub_nodes": hub_nodes,
    }


def clean_history(conn: sqlite3.Connection, keep_days: int, dry_run: bool) -> int:
    """Delete connection_history entries older than keep_days."""
    cutoff = time.time() - (keep_days * 86400)
    count = conn.execute(
        "SELECT COUNT(*) FROM connection_history WHERE changed_at < ?",
        (cutoff,)
    ).fetchone()[0]

    if count == 0:
        return 0

    if not dry_run:
        conn.execute("DELETE FROM connection_history WHERE changed_at < ?", (cutoff,))
        conn.commit()

    return count


def clean_orphan_connections(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete connections pointing to non-existent memories."""
    count = conn.execute("""
        SELECT COUNT(*) FROM connections
        WHERE source_id NOT IN (SELECT id FROM memories)
           OR target_id NOT IN (SELECT id FROM memories)
    """).fetchone()[0]

    if count == 0:
        return 0

    if not dry_run:
        conn.execute("""
            DELETE FROM connections
            WHERE source_id NOT IN (SELECT id FROM memories)
               OR target_id NOT IN (SELECT id FROM memories)
        """)
        conn.commit()

    return count


def clean_orphan_history(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete history entries for deleted memories."""
    count = conn.execute("""
        SELECT COUNT(*) FROM connection_history
        WHERE source_id NOT IN (SELECT id FROM memories)
           OR target_id NOT IN (SELECT id FROM memories)
    """).fetchone()[0]

    if count == 0:
        return 0

    if not dry_run:
        conn.execute("""
            DELETE FROM connection_history
            WHERE source_id NOT IN (SELECT id FROM memories)
               OR target_id NOT IN (SELECT id FROM memories)
        """)
        conn.commit()

    return count


def clean_old_dream_sessions(conn: sqlite3.Connection, keep_days: int, dry_run: bool) -> int:
    """Delete dream sessions older than keep_days."""
    cutoff = time.time() - (keep_days * 86400)
    count = conn.execute(
        "SELECT COUNT(*) FROM dream_sessions WHERE started_at < ?",
        (cutoff,)
    ).fetchone()[0]

    if count == 0:
        return 0

    if not dry_run:
        conn.execute("DELETE FROM dream_sessions WHERE started_at < ?", (cutoff,))
        conn.commit()

    return count


def deduplicate_and_constrain(conn: sqlite3.Connection, dry_run: bool) -> tuple[int, bool]:
    """Deduplicate connections and add UNIQUE constraint.

    Returns (duplicate_rows_removed, constraint_added).
    """
    # Count duplicates first
    dupe_count = conn.execute("""
        SELECT SUM(cnt - 1) FROM (
            SELECT source_id, target_id, edge_type, COUNT(*) as cnt
            FROM connections
            GROUP BY source_id, target_id, edge_type
            HAVING cnt > 1
        )
    """).fetchone()[0] or 0

    # Check if UNIQUE index already exists
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_connections_unique'"
    ).fetchone()

    if existing and dupe_count == 0:
        # Constraint exists and no dupes — nothing to do
        return 0, True

    if dry_run:
        return dupe_count, True

    if dupe_count > 0:
        # Full rebuild with constraint (dedup + create index in one transaction)
        conn.execute("BEGIN TRANSACTION")

        try:
            # Create deduplicated table
            conn.execute("""
                CREATE TABLE connections_dedup AS
                SELECT MIN(id) as id, source_id, target_id,
                       MAX(weight) as weight,
                       edge_type,
                       MIN(created_at) as created_at
                FROM connections
                GROUP BY source_id, target_id, edge_type
            """)

            # Drop old table
            conn.execute("DROP TABLE connections")

            # Rename
            conn.execute("ALTER TABLE connections_dedup RENAME TO connections")

            # Recreate indexes
            conn.execute("""
                CREATE UNIQUE INDEX idx_connections_unique
                ON connections(source_id, target_id, edge_type)
            """)
            conn.execute("CREATE INDEX idx_connections_source ON connections(source_id)")
            conn.execute("CREATE INDEX idx_connections_target ON connections(target_id)")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    elif not existing:
        # No dupes but constraint doesn't exist yet — just add it
        conn.execute("""
            CREATE UNIQUE INDEX idx_connections_unique
            ON connections(source_id, target_id, edge_type)
        """)
        conn.commit()

    return dupe_count, True


def add_retention_indexes(conn: sqlite3.Connection, dry_run: bool) -> list[str]:
    """Add indexes that speed up retention cleanup queries."""
    created = []

    # Index on connection_history.changed_at for fast time-based cleanup
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conn_history_time'"
    ).fetchone()
    if not exists:
        if not dry_run:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conn_history_time ON connection_history(changed_at)"
            )
            conn.commit()
        created.append("idx_conn_history_time")

    # Index on dream_sessions.started_at for fast cleanup
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_dream_sessions_time'"
    ).fetchone()
    if not exists:
        if not dry_run:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dream_sessions_time ON dream_sessions(started_at)"
            )
            conn.commit()
        created.append("idx_dream_sessions_time")

    return created


def vacuum_db(conn: sqlite3.Connection, dry_run: bool) -> int:
    """VACUUM to reclaim freelist pages. Returns bytes freed estimate."""
    _, _, freelist = freelist_info(conn)
    _, page_size, _ = freelist_info(conn)
    estimated = freelist * page_size

    if not dry_run and freelist > 0:
        conn.execute("VACUUM")

    return estimated


def verify_integrity(conn: sqlite3.Connection) -> tuple[bool, str]:
    """Run PRAGMA integrity_check and verify referential integrity."""
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        return False, f"SQLite integrity_check failed: {result}"

    # Check no orphans remain
    orphans = conn.execute("""
        SELECT COUNT(*) FROM connections
        WHERE source_id NOT IN (SELECT id FROM memories)
           OR target_id NOT IN (SELECT id FROM memories)
    """).fetchone()[0]

    if orphans > 0:
        return False, f"{orphans} orphan connections still exist"

    # Check unique constraint works
    try:
        conn.execute("""
            INSERT INTO connections (source_id, target_id, weight, edge_type, created_at)
            SELECT source_id, target_id, weight, edge_type, created_at
            FROM connections LIMIT 1
        """)
        conn.execute("ROLLBACK")
        # If we get here without error, constraint might not be active
        has_constraint = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_connections_unique'"
        ).fetchone()
        if has_constraint:
            return False, "UNIQUE constraint exists but INSERT didn't fail — check constraint"
    except sqlite3.IntegrityError:
        pass  # Good — constraint is working

    return True, "OK"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mazemaker SQLite Production Upgrade — PoC → Production"
    )
    parser.add_argument(
        "--db", default=os.path.expanduser("~/.mazemaker/data/memory.db"),
        help="Path to SQLite database (default: ~/.hermes/hermes-agent/plugins/memory/neural/neural_memory.db)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without modifying anything"
    )
    parser.add_argument(
        "--history-days", type=int, default=7,
        help="Keep connection_history entries from last N days (default: 7)"
    )
    parser.add_argument(
        "--session-days", type=int, default=30,
        help="Keep dream_sessions from last N days (default: 30)"
    )
    parser.add_argument(
        "--skip-backup", action="store_true",
        help="Skip backup creation (NOT recommended)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip confirmation prompts"
    )

    args = parser.parse_args()
    db_path = os.path.abspath(args.db)

    if not os.path.exists(db_path):
        print(f"FATAL: Database not found: {db_path}")
        sys.exit(1)

    print("=" * 70)
    print("NEURAL MEMORY — PRODUCTION UPGRADE")
    print("=" * 70)
    print(f"Database:  {db_path}")
    print(f"Size:      {human_size(db_size(db_path))}")
    print(f"Mode:      {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will modify!)'}")
    print(f"History:   keep {args.history_days} days")
    print(f"Sessions:  keep {args.session_days} days")
    print()

    # --- Phase 0: Diagnose ---
    print("[0/8] Diagnostic scan...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    diag = diagnose(conn, db_path)

    print(f"  Tables:")
    for table, count in diag["tables"].items():
        print(f"    {table}: {count:,} rows")
    print(f"  Duplicate edge groups: {diag['duplicate_edge_groups']:,}")
    print(f"  Orphan connections:    {diag['orphan_connections']:,}")
    print(f"  Orphan history:        {diag['orphan_history']:,}")
    print(f"  Freelist pages:        {diag['freelist']:,} ({diag['freelist_mb']:.1f} MB reclaimable)")
    print(f"  Weight distribution:")
    for bucket, cnt in diag["weight_distribution"].items():
        if cnt > 0:
            print(f"    {bucket}: {cnt:,}")

    if diag["history_oldest"]:
        oldest = datetime.fromtimestamp(diag["history_oldest"])
        newest = datetime.fromtimestamp(diag["history_newest"])
        age_days = (time.time() - diag["history_oldest"]) / 86400
        print(f"  History age: {oldest} → {newest} ({age_days:.1f} days)")
    print(f"  History by reason: {diag['history_reasons']}")

    conn.close()

    # Confirm if not dry-run and not forced
    if not args.dry_run and not args.force:
        total_cleanable = (
            diag["orphan_connections"]
            + diag["orphan_history"]
            + diag["duplicate_edge_groups"]
        )
        if total_cleanable > 0:
            print(f"\nWill remove ~{total_cleanable:,} rows and reclaim ~{diag['freelist_mb']:.0f} MB.")
            resp = input("Continue? [y/N] ").strip().lower()
            if resp != "y":
                print("Aborted.")
                sys.exit(0)

    # --- Phase 1: Backup ---
    if not args.skip_backup:
        print(f"\n[1/8] Creating backup...")
        if not args.dry_run:
            backup_path = backup_db(db_path)
            print(f"  Backup: {backup_path} ({human_size(db_size(backup_path))})")
        else:
            print(f"  Would create: {db_path}.bak.<timestamp>")
    else:
        print(f"\n[1/8] Backup SKIPPED (--skip-backup)")

    # Reconnect for modifications
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = {}

    # --- Phase 2: Clean connection_history ---
    print(f"\n[2/8] Cleaning connection_history (keeping {args.history_days} days)...")
    removed = clean_history(conn, args.history_days, args.dry_run)
    results["history_cleaned"] = removed
    print(f"  {'Would remove' if args.dry_run else 'Removed'}: {removed:,} rows")

    # --- Phase 3: Clean orphan connections ---
    print(f"\n[3/8] Cleaning orphan connections...")
    removed = clean_orphan_connections(conn, args.dry_run)
    results["orphan_connections"] = removed
    print(f"  {'Would remove' if args.dry_run else 'Removed'}: {removed:,} rows")

    # --- Phase 4: Clean orphan history ---
    print(f"\n[4/8] Cleaning orphan history entries...")
    removed = clean_orphan_history(conn, args.dry_run)
    results["orphan_history"] = removed
    print(f"  {'Would remove' if args.dry_run else 'Removed'}: {removed:,} rows")

    # --- Phase 5: Clean old dream sessions ---
    print(f"\n[5/8] Cleaning old dream sessions (keeping {args.session_days} days)...")
    removed = clean_old_dream_sessions(conn, args.session_days, args.dry_run)
    results["sessions_cleaned"] = removed
    print(f"  {'Would remove' if args.dry_run else 'Removed'}: {removed:,} rows")

    # --- Phase 6: Deduplicate + UNIQUE constraint ---
    print(f"\n[6/8] Deduplicating edges + UNIQUE constraint...")
    dupes, had_dupes = deduplicate_and_constrain(conn, args.dry_run)
    results["dupes_removed"] = dupes
    print(f"  {'Would remove' if args.dry_run else 'Removed'}: {dupes:,} duplicate rows")
    if had_dupes:
        print(f"  UNIQUE constraint: {'would add' if args.dry_run else 'added'} (idx_connections_unique)")

    # --- Phase 7: Retention indexes + VACUUM ---
    print(f"\n[7/8] Adding retention indexes + VACUUM...")
    new_indexes = add_retention_indexes(conn, args.dry_run)
    if new_indexes:
        print(f"  {'Would create' if args.dry_run else 'Created'} indexes: {', '.join(new_indexes)}")
    else:
        print(f"  Retention indexes already exist")

    if not args.dry_run:
        before_size = db_size(db_path)
        vacuum_db(conn, False)
        after_size = db_size(db_path)
        freed = before_size - after_size
        print(f"  VACUUM: {human_size(before_size)} → {human_size(after_size)} (freed {human_size(freed)})")
    else:
        _, _, freelist = freelist_info(conn)
        _, page_size, _ = freelist_info(conn)
        estimated = freelist * page_size
        print(f"  Would VACUUM: reclaim ~{human_size(estimated)}")

    # --- Phase 8: Verify ---
    print(f"\n[8/8] Verifying integrity...")
    if not args.dry_run:
        ok, msg = verify_integrity(conn)
        print(f"  Integrity: {'PASS' if ok else 'FAIL — ' + msg}")
        if not ok:
            print(f"\n  WARNING: Integrity check failed!")
            print(f"  Your backup is at: {backup_path if not args.skip_backup else 'N/A'}")
    else:
        print(f"  Skipped (dry run)")

    conn.close()

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'=' * 70}")

    if not args.dry_run:
        final_size = db_size(db_path)
        saved = diag["file_size"] - final_size
        print(f"  Before:  {human_size(diag['file_size'])}")
        print(f"  After:   {human_size(final_size)}")
        print(f"  Saved:   {human_size(saved)}")
    else:
        print(f"  Current: {human_size(diag['file_size'])}")
        print(f"  Would save: ~{human_size(diag['freelist_mb'] * 1024 * 1024)} (freelist) + cleanup rows")

    print(f"\n  Rows affected:")
    for key, val in results.items():
        if val > 0:
            print(f"    {key}: {val:,}")

    # Final stats
    if not args.dry_run:
        conn = sqlite3.connect(db_path)
        final_counts = table_counts(conn)
        _, _, final_freelist = freelist_info(conn)
        conn.close()

        print(f"\n  Final table counts:")
        for table, count in final_counts.items():
            print(f"    {table}: {count:,} rows")
        print(f"  Freelist: {final_freelist:,} pages")

    print(f"\n{'DONE' if not args.dry_run else 'DRY RUN COMPLETE — run without --dry-run to apply'}")


if __name__ == "__main__":
    main()
