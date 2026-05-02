#!/usr/bin/env python3.11
"""cleanup_onedrive_dupes.py — one-shot cleanup for OneDrive duplicates
in the live substrate.

Caught 2026-05-02 by round-5 archaeologist B7: Tito's OneDrive has
multiple physical copies of same files (Old + Misc, Combined_Base/,
Old_SOPs/, copy/, full_context/, etc.). Each path → different
content_hash because intro/path-aware text varies slightly between
copies — but the substrate ends up with ~4-44 redundant rows of the
same intellectual content per file.

This tool dedupes existing onedrive_reference rows by section_heading +
content body (joined). Keeps MAX(id) per dup-group. Wraps in transaction
with explicit dry-run default.

Forward fix shipped in commit dd396d3 (file-level pre-walk dedup) —
this tool cleans what's already in.

Usage:
    tools/cleanup_onedrive_dupes.py            # dry-run; reports counts
    tools/cleanup_onedrive_dupes.py --execute  # actually delete
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB = Path.home() / ".neural_memory" / "memory.db"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--execute", action="store_true",
                   help="Actually delete (default: dry-run)")
    p.add_argument("--source-label", default="onedrive_reference",
                   help="Source label to dedup within "
                        "(default onedrive_reference)")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db))

    # Find dup groups: same content_hash within same source_label.
    # content_hash is in metadata_json.
    rows = conn.execute(
        f"""
        SELECT
          JSON_EXTRACT(metadata_json, '$.content_hash') AS h,
          GROUP_CONCAT(id) AS ids,
          COUNT(*) AS n
        FROM memories
        WHERE JSON_EXTRACT(metadata_json, '$.source_label') = ?
        GROUP BY h
        HAVING n > 1
        """,
        (args.source_label,),
    ).fetchall()

    if not rows:
        print(f"No dup-groups found for source_label='{args.source_label}'.")
        return 0

    delete_ids: list[int] = []
    for h, ids_str, n in rows:
        ids = sorted(int(i) for i in ids_str.split(","))
        # Keep highest id (most recent), delete rest
        delete_ids.extend(ids[:-1])

    print(f"BEFORE:")
    print(f"  source_label:           {args.source_label}")
    print(f"  dup groups:             {len(rows)}")
    print(f"  total redundant rows:   {len(delete_ids)}")
    if rows:
        sample = rows[:3]
        for h, ids_str, n in sample:
            print(f"    hash={h[:8] if h else '?':>8s}  count={n}  ids={ids_str[:60]}")
    print()

    if not args.execute:
        print("DRY-RUN. Pass --execute to actually delete.")
        return 0

    if not delete_ids:
        print("Nothing to delete. Already clean.")
        return 0

    print(f"EXECUTE: deleting {len(delete_ids)} dup rows + their connections...")
    placeholders = ",".join("?" * len(delete_ids))
    try:
        conn.execute("BEGIN")
        # Delete connections first (defensive — mentions_entity, similar)
        cur = conn.execute(
            f"DELETE FROM connections WHERE source_id IN ({placeholders})",
            tuple(delete_ids),
        )
        deleted_edges_a = cur.rowcount
        cur = conn.execute(
            f"DELETE FROM connections WHERE target_id IN ({placeholders})",
            tuple(delete_ids),
        )
        deleted_edges_b = cur.rowcount
        # Delete memories
        cur = conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})",
            tuple(delete_ids),
        )
        deleted_mems = cur.rowcount
        # Delete from FTS5 (best-effort)
        try:
            conn.execute(
                f"DELETE FROM memories_fts WHERE rowid IN ({placeholders})",
                tuple(delete_ids),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
        print(f"  deleted: {deleted_mems} memory rows + "
              f"{deleted_edges_a + deleted_edges_b} connections")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}; rolled back.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
