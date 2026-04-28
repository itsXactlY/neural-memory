#!/usr/bin/env python3
"""
sync_bridge.py - Hot/Cold Sync Bridge for Neural Memory

Architecture:
  SQLite (hot store)  ──one-way sync──→  MSSQL (cold store)
  Agent reads/writes                       Backup/analytics
  here in real-time                        Disaster recovery

Sync modes:
  - continuous: Background thread, syncs every N seconds
  - batch:      One-shot sync of everything
  - incremental: Only new/changed records since last sync

State tracked in ~/.neural_memory/sync_state.json

Usage:
    from sync_bridge import SyncBridge

    bridge = SyncBridge()
    bridge.start()             # Start continuous sync (background thread)
    bridge.sync_now()          # Force one-shot batch sync
    bridge.sync_incremental()  # Sync only new records
    bridge.stop()              # Stop continuous sync
    bridge.status()            # Get sync stats

    # Or as CLI:
    python sync_bridge.py --mode continuous --interval 300
    python sync_bridge.py --mode batch
    python sync_bridge.py --mode incremental
    python sync_bridge.py --status
"""
import json
import logging
import os
import sqlite3
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("neural.sync_bridge")

DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")
SYNC_STATE_FILE = os.path.expanduser("~/.neural_memory/sync_state.json")
DEFAULT_INTERVAL = 300  # 5 minutes


class SyncState:
    """Persistent sync state tracker."""

    def __init__(self, path: str = SYNC_STATE_FILE):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "last_sync_time": None,
                "last_memory_id": 0,
                "last_connection_id": 0,
                "synced_memories": 0,
                "synced_connections": 0,
                "sync_errors": 0,
                "total_synced_memories": 0,
                "total_synced_connections": 0,
                "consecutive_failures": 0,
            }

    def save(self):
        """Persist sync state atomically.

        Writes to a sibling .tmp then os.replace()'s onto the final path so a
        Ctrl+C / OOM-kill mid-write can never leave a half-written JSON file.
        Without this, _load() catches JSONDecodeError on the next start and
        falls back to defaults, silently zeroing out cumulative counters
        (total_synced_*, last_memory_id, etc.) — so a single interruption
        wipes weeks of running totals.
        """
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_sync_time"] = datetime.now(timezone.utc).isoformat()
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError:
            # Best-effort cleanup; surface the error.
            try: tmp.unlink()
            except OSError: pass
            raise

    def record_success(self, memories: int, connections: int):
        self.data["synced_memories"] = memories
        self.data["synced_connections"] = connections
        self.data["total_synced_memories"] = self.data.get("total_synced_memories", 0) + memories
        self.data["total_synced_connections"] = self.data.get("total_synced_connections", 0) + connections
        self.data["consecutive_failures"] = 0
        self.data["sync_errors"] = 0
        self.save()

    def record_failure(self, error: str):
        self.data["sync_errors"] = self.data.get("sync_errors", 0) + 1
        self.data["consecutive_failures"] = self.data.get("consecutive_failures", 0) + 1
        self.data["last_error"] = error
        self.save()

    @property
    def last_memory_id(self) -> int:
        return self.data.get("last_memory_id", 0)

    @last_memory_id.setter
    def last_memory_id(self, val: int):
        self.data["last_memory_id"] = val

    @property
    def consecutive_failures(self) -> int:
        return self.data.get("consecutive_failures", 0)


class SyncBridge:
    """
    One-way sync bridge: SQLite (hot) → MSSQL (cold)

    SQLite is always the source of truth. MSSQL serves as backup/analytics.
    """

    def __init__(self, sqlite_path: str = DEFAULT_SQLITE, mssql_password: str = None):
        self.sqlite_path = sqlite_path
        self.mssql_password = mssql_password
        self.state = SyncState()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._interval = DEFAULT_INTERVAL
        self._mssql_conn = None
        # Stop event for the continuous loop. Replaces the prior
        # "for _ in range(N): time.sleep(1)" + "time.sleep(N*3)" pair that
        # made stop() block for up to (N*3) seconds during a backoff window
        # — at the default 5-minute interval, a 15-minute uncancellable
        # sleep. With Event.wait(timeout), stop() wakes the loop instantly.
        self._stop_event = threading.Event()

    def _get_mssql_conn(self):
        """Get or create MSSQL connection. Returns None if unavailable."""
        if self._mssql_conn:
            try:
                self._mssql_conn.execute("SELECT 1")
                return self._mssql_conn
            except Exception:
                self._mssql_conn = None

        try:
            import pyodbc
        except ImportError:
            logger.debug("pyodbc not installed, MSSQL sync disabled")
            return None

        pw = self.mssql_password or os.environ.get("MSSQL_PASSWORD", "")
        if not pw:
            env_file = os.path.expanduser("~/.hermes/.env")
            if os.path.exists(env_file):
                for line in open(env_file):
                    if line.startswith("MSSQL_PASSWORD="):
                        pw = line.split("=", 1)[1].strip().strip("\"'")
                        break

        if not pw:
            logger.debug("MSSQL_PASSWORD not configured")
            return None

        try:
            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                "SERVER=localhost;DATABASE=NeuralMemory;"
                f"UID=SA;PWD={pw};TrustServerCertificate=yes;"
            )
            self._mssql_conn = pyodbc.connect(conn_str, autocommit=True)
            return self._mssql_conn
        except Exception as e:
            logger.debug(f"MSSQL connection failed: {e}")
            return None

    def sync_incremental(self, filter_garbage: bool = False) -> dict:
        """
        Sync only new records since last sync.

        Returns dict with 'memories', 'connections', 'errors', 'skipped'.
        """
        result = {"memories": 0, "connections": 0, "errors": 0, "skipped": 0}

        mssql = self._get_mssql_conn()
        if not mssql:
            logger.warning("MSSQL unavailable, skipping sync")
            self.state.record_failure("MSSQL unavailable")
            return result

        try:
            # Sync new memories
            mem_result = self._sync_new_memories(mssql, filter_garbage)
            result["memories"] = mem_result[0]
            result["errors"] += mem_result[1]
            result["skipped"] += mem_result[2]

            # Sync new connections
            conn_result = self._sync_new_connections(mssql)
            result["connections"] = conn_result[0]
            result["errors"] += conn_result[1]

            # Update state
            sconn = sqlite3.connect(self.sqlite_path)
            last_id = sconn.execute("SELECT MAX(id) FROM memories").fetchone()[0] or 0
            sconn.close()
            self.state.last_memory_id = last_id
            self.state.record_success(result["memories"], result["connections"])

            if result["memories"] or result["connections"]:
                logger.info(
                    f"Sync complete: {result['memories']} memories, "
                    f"{result['connections']} connections"
                )

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.state.record_failure(str(e))
            result["errors"] += 1

        return result

    def _sync_new_memories(self, mssql, filter_garbage: bool = False):
        """Sync memories newer than last_memory_id. Returns (synced, errors, skipped)."""
        import re
        garbage_pats = [re.compile(r'^turn-\d+'), re.compile(r'^DD\d+%')]

        sconn = sqlite3.connect(self.sqlite_path)
        sc = sconn.cursor()
        mc = mssql.cursor()

        max_id = self.state.last_memory_id
        mc.execute("SELECT ISNULL(MAX(id), 0) FROM memories")
        mssql_max = mc.fetchone()[0]
        max_id = max(max_id, mssql_max)

        sc.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count "
            "FROM memories WHERE id > ? ORDER BY id", (max_id,)
        )
        rows = sc.fetchall()
        sconn.close()

        if not rows:
            return 0, 0, 0

        skipped = 0
        if filter_garbage:
            filtered = []
            for row in rows:
                label = row[1] or ""
                if any(p.match(label) for p in garbage_pats):
                    skipped += 1
                else:
                    filtered.append(row)
            rows = filtered

        if not rows:
            return 0, 0, skipped

        mc.execute("SET IDENTITY_INSERT memories ON")
        mssql.commit()

        synced, errors = 0, 0
        # IDENTITY_INSERT is per-session; if the per-row INSERT loop raises
        # (e.g. MSSQL connection drops mid-batch), the OFF must still run or
        # this connection is left in a state where every future autoincrement
        # INSERT fails. Wrap in try/finally — and best-effort the OFF since
        # the caller might already be torn down by a connection error.
        try:
            for row in rows:
                id_, label, content, blob, salience, created, accessed, acc = row
                if blob:
                    elem_count = len(blob) // 4
                    emb_blob = struct.pack(f"{elem_count}f", *struct.unpack(f"{elem_count}f", blob))
                    emb_dim = elem_count
                else:
                    emb_blob = None
                    emb_dim = 1024

                def _ts(ts):
                    if not ts:
                        return "1970-01-01 00:00:00.0000000"
                    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

                try:
                    mc.execute(
                        "INSERT INTO memories (id, label, content, embedding, vector_dim, "
                        "salience, created_at, last_accessed, access_count) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        id_, label, content, emb_blob, emb_dim,
                        salience or 1.0, _ts(created), _ts(accessed), acc or 0,
                    )
                    synced += 1
                except Exception as e:
                    errors += 1
                    if errors <= 2:
                        logger.warning(f"Error syncing memory {id_}: {e}")
            mssql.commit()
        finally:
            try:
                mc.execute("SET IDENTITY_INSERT memories OFF")
                mssql.commit()
            except Exception:
                pass

        return synced, errors, skipped

    def _sync_new_connections(self, mssql):
        """Sync connections not yet in MSSQL. Returns (synced, errors)."""
        sconn = sqlite3.connect(self.sqlite_path)
        sc = sconn.cursor()
        mc = mssql.cursor()

        mc.execute("SELECT source_id, target_id FROM connections")
        mssql_set = set((r.source_id, r.target_id) for r in mc.fetchall())
        sc.execute("SELECT source_id, target_id, weight FROM connections ORDER BY id")
        missing = [(s, t, w) for s, t, w in sc.fetchall() if (s, t) not in mssql_set]
        sconn.close()

        if not missing:
            return 0, 0

        mc.execute("SELECT ISNULL(MAX(id), 0) FROM connections")
        next_id = mc.fetchone()[0] + 1

        mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
        mc.execute("SET IDENTITY_INSERT connections ON")
        mssql.commit()

        synced, errors = 0, 0
        # As with _sync_new_memories: the IDENTITY_INSERT toggle and the
        # NOCHECK constraint disable MUST be paired with their counterparts
        # in finally, otherwise a mid-loop MSSQL disconnect leaves the
        # connection wedged with IDENTITY_INSERT ON and FK checks disabled.
        # Both are session-scoped, but a session that survives the failure
        # (e.g. a connection-pooler) carries the broken state to the next
        # caller and breaks unrelated work.
        try:
            batch = 500
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                data = [
                    (next_id + j, s, t, round(w, 6), "similar")
                    for j, (s, t, w) in enumerate(chunk)
                ]
                try:
                    mc.fast_executemany = True
                    mc.executemany(
                        "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                        "VALUES (?,?,?,?,?)", data,
                    )
                    mssql.commit()
                    synced += len(chunk)
                    next_id += len(chunk)
                except Exception:
                    for row in data:
                        try:
                            mc.execute(
                                "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                                "VALUES (?,?,?,?,?)", *row,
                            )
                            synced += 1
                            next_id += 1
                        except Exception:
                            errors += 1
                    mssql.commit()
        finally:
            for cleanup_sql in (
                "SET IDENTITY_INSERT connections OFF",
                "ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL",
            ):
                try:
                    mc.execute(cleanup_sql)
                    mssql.commit()
                except Exception:
                    pass

        return synced, errors

    def sync_batch(self, filter_garbage: bool = False) -> dict:
        """Full batch sync — clears MSSQL and re-syncs everything."""
        result = {"memories": 0, "connections": 0, "errors": 0, "skipped": 0}

        mssql = self._get_mssql_conn()
        if not mssql:
            logger.warning("MSSQL unavailable")
            return result

        try:
            sconn = sqlite3.connect(self.sqlite_path)

            # Clear MSSQL
            mc = mssql.cursor()
            mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
            mc.execute("DELETE FROM connections")
            mc.execute("DELETE FROM memories")
            mssql.commit()

            # Sync memories
            sc = sconn.cursor()
            sc.execute(
                "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count "
                "FROM memories ORDER BY id"
            )
            rows = sc.fetchall()

            import re
            garbage_pats = [re.compile(r'^turn-\d+'), re.compile(r'^DD\d+%')]
            skipped = 0
            if filter_garbage:
                filtered = []
                for row in rows:
                    if not any(p.match(row[1] or "") for p in garbage_pats):
                        filtered.append(row)
                    else:
                        skipped += 1
                rows = filtered

            mc.execute("SET IDENTITY_INSERT memories ON")
            mssql.commit()

            try:
                for row in rows:
                    id_, label, content, blob, salience, created, accessed, acc = row
                    if blob:
                        elem_count = len(blob) // 4
                        emb_blob = struct.pack(f"{elem_count}f", *struct.unpack(f"{elem_count}f", blob))
                        emb_dim = elem_count
                    else:
                        emb_blob = None
                        emb_dim = 1024

                    def _ts(ts):
                        if not ts:
                            return "1970-01-01 00:00:00.0000000"
                        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

                    try:
                        mc.execute(
                            "INSERT INTO memories (id, label, content, embedding, vector_dim, "
                            "salience, created_at, last_accessed, access_count) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            id_, label, content, emb_blob, emb_dim,
                            salience or 1.0, _ts(created), _ts(accessed), acc or 0,
                        )
                        result["memories"] += 1
                    except Exception:
                        result["errors"] += 1
                mssql.commit()
            finally:
                # Always clear IDENTITY_INSERT so a mid-batch failure doesn't
                # leave the connection in a state that breaks every future
                # autoincrement INSERT (see _sync_new_memories for the same fix).
                try:
                    mc.execute("SET IDENTITY_INSERT memories OFF")
                    mssql.commit()
                except Exception:
                    pass

            # Sync connections
            sc.execute("SELECT source_id, target_id, weight FROM connections ORDER BY id")
            conn_rows = sc.fetchall()
            sconn.close()

            mc.execute("SET IDENTITY_INSERT connections ON")
            mssql.commit()

            try:
                next_id = 1
                batch = 1000
                for i in range(0, len(conn_rows), batch):
                    chunk = conn_rows[i:i + batch]
                    data = [
                        (next_id + j, s, t, round(w, 6), "similar")
                        for j, (s, t, w) in enumerate(chunk)
                    ]
                    try:
                        mc.fast_executemany = True
                        mc.executemany(
                            "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                            "VALUES (?,?,?,?,?)", data,
                        )
                        mssql.commit()
                        result["connections"] += len(chunk)
                        next_id += len(chunk)
                    except Exception:
                        for row in data:
                            try:
                                mc.execute(
                                    "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                                    "VALUES (?,?,?,?,?)", *row,
                                )
                                result["connections"] += 1
                                next_id += 1
                            except Exception:
                                result["errors"] += 1
                        mssql.commit()
            finally:
                for cleanup_sql in (
                    "SET IDENTITY_INSERT connections OFF",
                    "ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL",
                ):
                    try:
                        mc.execute(cleanup_sql)
                        mssql.commit()
                    except Exception:
                        pass

            result["skipped"] = skipped
            self.state.record_success(result["memories"], result["connections"])
            logger.info(f"Batch sync: {result['memories']} memories, {result['connections']} connections")

        except Exception as e:
            logger.error(f"Batch sync failed: {e}")
            self.state.record_failure(str(e))
            result["errors"] += 1

        return result

    def _continuous_loop(self):
        """Background sync loop."""
        logger.info(f"Continuous sync started (interval={self._interval}s)")
        while not self._stop_event.is_set():
            try:
                result = self.sync_incremental(filter_garbage=True)
                if result["errors"] and self.state.consecutive_failures > 5:
                    logger.warning(
                        f"{self.state.consecutive_failures} consecutive failures, "
                        "backing off..."
                    )
                    # Cancellable backoff. Event.wait returns True if set,
                    # which means stop() was called — exit the loop instead
                    # of starting another sync after the backoff window.
                    if self._stop_event.wait(self._interval * 3):
                        break
                    continue
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            # Cancellable inter-cycle wait — stop() unblocks immediately.
            if self._stop_event.wait(self._interval):
                break

        logger.info("Continuous sync stopped")

    def start(self, interval: int = DEFAULT_INTERVAL):
        """Start continuous background sync."""
        if self._running:
            logger.warning("Sync bridge already running")
            return

        self._interval = interval
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop continuous sync. Wakes the loop immediately via stop_event."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def status(self) -> dict:
        """Get current sync status."""
        return {
            "running": self._running,
            "interval": self._interval,
            **self.state.data,
        }

    def close(self):
        """Clean shutdown."""
        self.stop()
        if self._mssql_conn:
            try:
                self._mssql_conn.close()
            except Exception:
                pass
            self._mssql_conn = None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Memory Sync Bridge (SQLite → MSSQL)")
    parser.add_argument("--mode", choices=["continuous", "batch", "incremental"],
                        default="incremental", help="Sync mode")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Sync interval in seconds (continuous mode)")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument("--db", default=DEFAULT_SQLITE, help="SQLite database path")
    parser.add_argument("--filter-garbage", action="store_true", help="Skip turn-DD labels")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    bridge = SyncBridge(sqlite_path=args.db)

    if args.status:
        s = bridge.status()
        print(json.dumps(s, indent=2))
        return

    if args.mode == "continuous":
        print(f"Starting continuous sync (interval={args.interval}s)...")
        print("Press Ctrl+C to stop.")
        bridge.start(interval=args.interval)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            bridge.stop()
    elif args.mode == "batch":
        print("Running batch sync...")
        result = bridge.sync_batch(filter_garbage=args.filter_garbage)
        print(f"Done: {result['memories']} memories, {result['connections']} connections, "
              f"{result['errors']} errors, {result['skipped']} skipped")
    elif args.mode == "incremental":
        print("Running incremental sync...")
        result = bridge.sync_incremental(filter_garbage=args.filter_garbage)
        print(f"Done: {result['memories']} memories, {result['connections']} connections, "
              f"{result['errors']} errors, {result['skipped']} skipped")

    bridge.close()


if __name__ == "__main__":
    main()
