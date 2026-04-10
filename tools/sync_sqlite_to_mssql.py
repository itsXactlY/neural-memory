#!/usr/bin/env python3
"""
sync_sqlite_to_mssql.py - Sync Neural Memory from SQLite to MSSQL.

Preserves original SQLite IDs via IDENTITY_INSERT.
Handles FK constraints, incremental sync, and batch inserts.

Usage:
    python sync_sqlite_to_mssql.py                         # Full sync
    python sync_sqlite_to_mssql.py --incremental           # Only missing rows
    python sync_sqlite_to_mssql.py --skip-connections      # Memories only
    python sync_sqlite_to_mssql.py --db /path/to/memory.db # Custom SQLite path
"""
import argparse
import os
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone

EMBEDDING_DIM = 384
DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")


def get_mssql(password=None):
    """Return a pyodbc connection to local MSSQL NeuralMemory database."""
    try:
        import pyodbc
    except ImportError:
        print("pyodbc required: pip install pyodbc")
        sys.exit(1)

    pw = password or os.environ.get("MSSQL_PASSWORD", "q?}33YIToo:H%xue$Kr*")
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=localhost;DATABASE=NeuralMemory;"
        f"UID=SA;PWD={pw};TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, autocommit=True)


def unix_to_dt(ts):
    if not ts:
        return "1970-01-01 00:00:00.0000000"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def sync_memories(sqlite_db, mssql_conn, incremental=False):
    """Sync memories table. Returns (synced, errors)."""
    sconn = sqlite3.connect(sqlite_db)
    sc = sconn.cursor()
    mc = mssql_conn.cursor()

    if incremental:
        mc.execute("SELECT ISNULL(MAX(id), 0) FROM memories")
        max_id = mc.fetchone()[0]
        sc.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count "
            "FROM memories WHERE id > ? ORDER BY id", (max_id,)
        )
    else:
        sc.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count "
            "FROM memories ORDER BY id"
        )

    rows = sc.fetchall()
    sconn.close()
    if not rows:
        return 0, 0

    print(f"  Memories to sync: {len(rows)}")

    if not incremental:
        mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
        mc.execute("DELETE FROM connections")
        mc.execute("DELETE FROM memories")
        mssql_conn.commit()

    mc.execute("SET IDENTITY_INSERT memories ON")
    mssql_conn.commit()

    synced, errors = 0, 0
    t0 = time.time()

    for i, row in enumerate(rows):
        id_, label, content, blob, salience, created, accessed, acc = row
        emb = list(struct.unpack(f"{EMBEDDING_DIM}f", blob)) if blob else None
        emb_blob = struct.pack(f"{EMBEDDING_DIM}f", *emb) if emb else None

        try:
            mc.execute(
                "INSERT INTO memories (id, label, content, embedding, vector_dim, "
                "salience, created_at, last_accessed, access_count) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                id_, label, content, emb_blob, EMBEDDING_DIM,
                salience or 1.0, unix_to_dt(created), unix_to_dt(accessed), acc or 0,
            )
            synced += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error memory {id_}: {e}")

        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            mssql_conn.commit()
            rate = (i + 1) / (time.time() - t0) if time.time() - t0 > 0 else 0
            print(f"    {i+1}/{len(rows)} ({rate:.0f}/s)")

    mc.execute("SET IDENTITY_INSERT memories OFF")
    mssql_conn.commit()

    if not incremental:
        mc.execute("ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL")
        mssql_conn.commit()

    return synced, errors


def sync_connections(sqlite_db, mssql_conn, incremental=False):
    """Sync connections table. Returns (synced, errors)."""
    sconn = sqlite3.connect(sqlite_db)
    sc = sconn.cursor()
    mc = mssql_conn.cursor()

    if incremental:
        # Find missing (source_id, target_id) pairs
        mc.execute("SELECT source_id, target_id FROM connections")
        mssql_set = set((r.source_id, r.target_id) for r in mc.fetchall())
        sc.execute("SELECT source_id, target_id, weight, edge_type FROM connections ORDER BY id")
        missing = [(s, t, w, e) for s, t, w, e in sc.fetchall() if (s, t) not in mssql_set]
        sconn.close()
        if not missing:
            return 0, 0
        rows_to_insert = missing
        mc.execute("SELECT ISNULL(MAX(id), 0) FROM connections")
        next_id = mc.fetchone()[0] + 1
    else:
        sc.execute("SELECT source_id, target_id, weight, edge_type FROM connections ORDER BY id")
        rows_to_insert = sc.fetchall()
        sconn.close()
        next_id = 1

    print(f"  Connections to sync: {len(rows_to_insert)}")

    mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
    if not incremental:
        mc.execute("DELETE FROM connections")
    mc.execute("SET IDENTITY_INSERT connections ON")
    mssql_conn.commit()

    synced, errors = 0, 0
    t0 = time.time()
    batch = 1000

    for i in range(0, len(rows_to_insert), batch):
        chunk = rows_to_insert[i:i + batch]
        data = [
            (next_id + j, s, t, round(w, 6), (e or "similar"))
            for j, (s, t, w, e) in enumerate(chunk)
        ]
        try:
            mc.fast_executemany = True
            mc.executemany(
                "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                "VALUES (?,?,?,?,?)",
                data,
            )
            mssql_conn.commit()
            synced += len(chunk)
            next_id += len(chunk)
        except Exception as ex:
            print(f"    Batch error at {i}: {ex}")
            mssql_conn.rollback()
            for row in data:
                try:
                    mc.execute(
                        "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                        "VALUES (?,?,?,?,?)", *row
                    )
                    synced += 1
                    next_id += 1
                except:
                    errors += 1
            mssql_conn.commit()

        done = min(i + batch, len(rows_to_insert))
        rate = done / (time.time() - t0) if time.time() - t0 > 0 else 0
        if done % 5000 == 0 or done >= len(rows_to_insert):
            print(f"    {done}/{len(rows_to_insert)} ({rate:.0f}/s)")

    mc.execute("SET IDENTITY_INSERT connections OFF")
    mc.execute("ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL")
    mssql_conn.commit()

    return synced, errors


def verify(sqlite_db, mssql_conn):
    """Print comparison of SQLite vs MSSQL counts."""
    sconn = sqlite3.connect(sqlite_db)
    sc = sconn.cursor()
    sc.execute("SELECT COUNT(*) FROM memories"); sm = sc.fetchone()[0]
    sc.execute("SELECT COUNT(*) FROM connections"); sc2 = sc.fetchone()[0]
    sconn.close()

    mc = mssql_conn.cursor()
    mc.execute("SELECT COUNT(*) FROM memories"); mm = mc.fetchone()[0]
    mc.execute("SELECT COUNT(*) FROM connections"); mc2 = mc.fetchone()[0]

    print(f"\n{'='*45}")
    print(f"  SQLite:   {sm:>5} memories, {sc2:>6} connections")
    print(f"  MSSQL:    {mm:>5} memories, {mc2:>6} connections")
    status = "SYNCED" if sm == mm and sc2 == mc2 else "PARTIAL"
    print(f"  Status:   {status}")
    print(f"{'='*45}")
    return status == "SYNCED"


def main():
    parser = argparse.ArgumentParser(description="Sync Neural Memory SQLite -> MSSQL")
    parser.add_argument("--db", default=DEFAULT_SQLITE, help="SQLite database path")
    parser.add_argument("--incremental", action="store_true", help="Only sync missing rows")
    parser.add_argument("--skip-connections", action="store_true", help="Skip connections table")
    parser.add_argument("--password", default=None, help="MSSQL SA password")
    args = parser.parse_args()

    print("=== Neural Memory SQLite -> MSSQL Sync ===\n")

    print(f"[1/3] SQLite: {args.db}")
    mconn = get_mssql(args.password)

    print("[2/3] Syncing memories...")
    t0 = time.time()
    ms, me = sync_memories(args.db, mconn, args.incremental)
    print(f"  Done: {ms} synced, {me} errors ({time.time()-t0:.1f}s)\n")

    if not args.skip_connections:
        print("[3/3] Syncing connections...")
        t1 = time.time()
        cs, ce = sync_connections(args.db, mconn, args.incremental)
        print(f"  Done: {cs} synced, {ce} errors ({time.time()-t1:.1f}s)")
    else:
        print("[3/3] Skipped connections")

    verify(args.db, mconn)
    mconn.close()


if __name__ == "__main__":
    main()
