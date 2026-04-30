"""Dream Engine — autonomous background memory consolidation.

Implements three phases inspired by biological sleep:
  1. NREM — Replay & consolidation (strengthen, prune)
  2. REM  — Exploration & bridge discovery
  3. Insight — Abstraction & community detection

Runs as a background daemon during idle periods. Stores results
in the same SQLite DB as the main mazemaker, extended with
dream-specific tables.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema extensions for dream tables (SQLite)
# ---------------------------------------------------------------------------

_DREAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    phase TEXT NOT NULL,
    memories_processed INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned INTEGER DEFAULT 0,
    bridges_found INTEGER DEFAULT 0,
    insights_created INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dream_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    insight_type TEXT NOT NULL,
    source_memory_id INTEGER,
    content TEXT,
    confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS connection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    changed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dream_insights_type
    ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session
    ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes
    ON connection_history(source_id, target_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_changed_at
    ON connection_history(changed_at);
CREATE INDEX IF NOT EXISTS idx_dream_sessions_started_at
    ON dream_sessions(started_at);
"""


# ---------------------------------------------------------------------------
# Abstract Dream Backend
# ---------------------------------------------------------------------------

class DreamBackend:
    """Interface for dream storage backends (SQLite or Postgres)."""

    def start_session(self, phase: str) -> int:
        raise NotImplementedError

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_connections(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        raise NotImplementedError

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        raise NotImplementedError

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05) -> int:
        """Bulk strengthen. Returns count updated."""
        count = 0
        for src, tgt in edges:
            self.strengthen_connection(src, tgt, delta)
            count += 1
        return count

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01) -> int:
        """Bulk weaken all connections above threshold. Returns count updated."""
        raise NotImplementedError

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> bool:
        """Insert a bridge edge. Returns True if newly inserted, False if skipped
        (e.g. edge already exists or self-loop). Backends must canonicalise
        source<target before INSERT so connection_history and the connections
        table stay aligned. Callers should only log_connection_change on True.
        """
        raise NotImplementedError

    def prune_weak(self, threshold: float = 0.05) -> int:
        raise NotImplementedError

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        raise NotImplementedError

    def get_memory_vectors(self, memory_ids: List[int]) -> Dict[int, List[float]]:
        """Return embeddings for memory IDs. Optional backend capability."""
        return {}

    def set_connection_weight(self, source_id: int, target_id: int,
                              weight: float, reason: str = "semantic_reweight") -> bool:
        raise NotImplementedError

    def add_typed_connection(self, source_id: int, target_id: int,
                             weight: float = 0.5,
                             edge_type: str = "similar") -> bool:
        if edge_type == "bridge":
            self.add_bridge(source_id, target_id, weight)
            return True
        raise NotImplementedError

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Delete history entries older than keep_days. Returns count deleted."""
        raise NotImplementedError

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Delete dream sessions older than keep_days. Returns count deleted."""
        raise NotImplementedError

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories."""
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQLite Dream Backend
# ---------------------------------------------------------------------------

class SQLiteDreamBackend(DreamBackend):
    """Dream backend using the existing mazemaker SQLite DB."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._persistent_conn = None
        self._ensure_tables()

    @property
    def conn(self):
        """Persistent connection for DreamWorker compatibility."""
        if self._persistent_conn is None:
            self._persistent_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        return self._persistent_conn

    def close(self):
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_DREAM_SCHEMA)
            # Existing mazemaker DBs may predate typed/bi-temporal edges.
            has_connections = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='connections'"
            ).fetchone()
            if has_connections:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)").fetchall()}
                if "edge_type" not in cols:
                    conn.execute("ALTER TABLE connections ADD COLUMN edge_type TEXT DEFAULT 'similar'")
                for col, sql in {
                    "event_time": "ALTER TABLE connections ADD COLUMN event_time REAL",
                    "ingestion_time": "ALTER TABLE connections ADD COLUMN ingestion_time REAL DEFAULT (unixepoch())",
                    "valid_from": "ALTER TABLE connections ADD COLUMN valid_from REAL",
                    "valid_to": "ALTER TABLE connections ADD COLUMN valid_to REAL",
                }.items():
                    if col not in cols:
                        try:
                            conn.execute(sql)
                        except sqlite3.OperationalError:
                            pass
                conn.execute("UPDATE connections SET edge_type = 'similar' WHERE edge_type IS NULL OR edge_type = ''")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_connections_edge_type_weight "
                    "ON connections(edge_type, weight)"
                )
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def start_session(self, phase: str) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO dream_sessions (started_at, phase) VALUES (?, ?)",
                (time.time(), phase)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE dream_sessions SET "
                "finished_at = ?, memories_processed = ?, "
                "connections_strengthened = ?, connections_pruned = ?, "
                "bridges_found = ?, insights_created = ? WHERE id = ?",
                (
                    time.time(),
                    stats.get("processed", stats.get("explored", 0)),
                    stats.get("strengthened", 0),
                    stats.get("pruned", 0),
                    stats.get("bridges", 0),
                    stats.get("insights", 0),
                    session_id,
                )
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, content FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": r["id"], "content": r["content"] or ""} for r in rows]
        finally:
            conn.close()

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT m.id, m.content,
                       (SELECT COUNT(*) FROM connections
                        WHERE source_id = m.id OR target_id = m.id) as cnt
                FROM memories m
                WHERE (SELECT COUNT(*) FROM connections
                       WHERE source_id = m.id OR target_id = m.id) < ?
                ORDER BY m.created_at DESC LIMIT ?
            """, (max_connections, limit)).fetchall()
            return [
                {"id": r["id"], "content": r["content"] or "", "connection_count": r["cnt"]}
                for r in rows
            ]
        finally:
            conn.close()

    def get_connections(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, target_id, weight, COALESCE(edge_type, 'similar') AS edge_type "
                "FROM connections WHERE weight >= 0.05"
            ).fetchall()
            return [
                {
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "weight": r["weight"],
                    "edge_type": r["edge_type"],
                    "type": r["edge_type"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_memory_vectors(self, memory_ids: List[int]) -> Dict[int, List[float]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT id, embedding FROM memories WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                tuple(memory_ids),
            ).fetchall()
            out: Dict[int, List[float]] = {}
            for r in rows:
                blob = r["embedding"]
                dim = len(blob) // 4 if blob else 0
                if dim:
                    out[r["id"]] = list(struct.unpack(f"{dim}f", blob))
            return out
        finally:
            conn.close()

    def set_connection_weight(self, source_id: int, target_id: int,
                              weight: float, reason: str = "semantic_reweight") -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, source_id, target_id, weight FROM connections "
                "WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                (source_id, target_id, target_id, source_id),
            ).fetchone()
            if not row:
                return False
            new_weight = max(0.0, min(1.0, float(weight)))
            conn.execute("UPDATE connections SET weight = ? WHERE id = ?", (new_weight, row["id"]))
            conn.execute(
                "INSERT INTO connection_history (source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (row["source_id"], row["target_id"], row["weight"], new_weight, reason, time.time()),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def add_typed_connection(self, source_id: int, target_id: int,
                             weight: float = 0.5,
                             edge_type: str = "similar") -> bool:
        if source_id == target_id:
            return False
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id, weight FROM connections WHERE source_id = ? AND target_id = ? AND COALESCE(edge_type, 'similar') = ?",
                (source_id, target_id, edge_type),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE connections SET weight = MAX(weight, ?) WHERE id = ?",
                    (weight, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at, event_time, ingestion_time, valid_from) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (source_id, target_id, weight, edge_type, time.time(), time.time(), time.time(), time.time()),
                )
            conn.commit()
            return True
        finally:
            conn.close()

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        """Bump an edge's weight by delta, capped at 1.0.

        Canonicalises (source < target) before the UPDATE so this works
        on the canonical rows produced by the iter-62 migration. Without
        the swap, a caller passing (max, min) would silently match no
        rows and the update would be a no-op — observed in practice
        when NREM activated_edges happened to be derived from a
        non-canonical source.
        """
        if source_id == target_id:
            return
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05) -> int:
        """Bulk strengthen via executemany. Returns count updated.

        Canonicalises every (src, tgt) pair before the UPDATE — same
        rationale as strengthen_connection above. NREM's activated_edges
        set already stores (min, max), so this is a no-op for the
        primary caller; the canonicalisation is here so any future
        caller that forgets to swap still hits the right rows.
        """
        if not edges:
            return 0
        canon = []
        for src, tgt in edges:
            src, tgt = int(src), int(tgt)
            if src == tgt:
                continue
            if src > tgt:
                src, tgt = tgt, src
            canon.append((delta, src, tgt))
        if not canon:
            return 0
        conn = self._connect()
        try:
            conn.executemany(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                canon,
            )
            conn.commit()
            return len(canon)
        finally:
            conn.close()

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01) -> int:
        """Bulk weaken all connections above threshold in one UPDATE."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE weight > ?",
                (delta, threshold)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> bool:
        """Insert a REM-discovered bridge edge. Returns True if newly inserted.

        Canonicalises source<target to match add_connection's invariant.
        Without this, the connections table held mixed-orientation rows:
        an edge added by remember()'s auto_connect path was always (min,max),
        but a bridge from REM could be (max,min). Any downstream code that
        assumed canonical form (\"WHERE source=? AND target=?\") would miss
        the bridge edge half the time.

        Returns False on self-loop or pre-existing edge so the caller can
        avoid writing a misleading connection_history row claiming the
        weight changed from 0.0.
        """
        if source_id == target_id:
            return False
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM connections WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at) "
                "VALUES (?, ?, ?, 'bridge', ?)",
                (source_id, target_id, weight, time.time())
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def prune_weak(self, threshold: float = 0.05) -> int:
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM connections WHERE weight < ?",
                (threshold,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        # Defensive canonicalisation: every connections row is canonical
        # (source<target), so log rows must match or any join on
        # (source_id, target_id) loses half the history. Callers fixed in
        # iter-M already canonicalise; this guard makes the contract
        # function-level so a future caller can't silently regress.
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO connection_history (source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, target_id, old_weight, new_weight, reason, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, insight_type, source_memory_id, content, confidence, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Delete history entries older than keep_days."""
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM connection_history WHERE changed_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Delete dream sessions older than keep_days and their associated insights.

        Uses correlated DELETE+subquery (no per-id parameter binding) so first-run
        cleanup of a large backlog can't trip SQLITE_MAX_VARIABLE_NUMBER. The
        previous implementation built `IN (?,?,?,...)` with one placeholder per
        old session — fine for the steady-state (3/cycle), broken if the user
        enabled cleanup after months of accumulation.
        """
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            # Insights first — SQLite has no FK cascade in default schemas.
            conn.execute(
                "DELETE FROM dream_insights "
                "WHERE session_id IN (SELECT id FROM dream_sessions WHERE started_at < ?)",
                (cutoff,)
            )
            count = conn.execute(
                "DELETE FROM dream_sessions WHERE started_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories."""
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM connections "
                "WHERE source_id NOT IN (SELECT id FROM memories) "
                "OR target_id NOT IN (SELECT id FROM memories)"
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(memories_processed), 0), "
                "COALESCE(SUM(connections_strengthened), 0), "
                "COALESCE(SUM(connections_pruned), 0), "
                "COALESCE(SUM(bridges_found), 0), "
                "COALESCE(SUM(insights_created), 0) "
                "FROM dream_sessions"
            ).fetchone()

            insights = conn.execute(
                "SELECT insight_type, COUNT(*) FROM dream_insights GROUP BY insight_type"
            ).fetchall()

            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "insight_types": {r[0]: r[1] for r in insights},
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

class DreamEngine:
    """Autonomous background consolidation for mazemaker.

    Three phases:
      NREM  — Replay recent memories, strengthen active, prune dead
      REM   — Explore isolated memories, discover bridges via embedding
      Insight — Community detection, bridge identification, abstraction
    """

    def __init__(
        self,
        backend: DreamBackend,
        neural_memory: Optional[Any] = None,
        idle_threshold: float = 300.0,     # 5 min idle
        memory_threshold: int = 50,         # dream every N new memories
        max_memories_per_cycle: int = 100,
    ):
        self._backend = backend
        self._memory = neural_memory        # Mazemaker instance for think/recall
        self._idle_threshold = idle_threshold
        self._memory_threshold = memory_threshold
        self._max_memories = max_memories_per_cycle

        self._stop_event = threading.Event()  # set = stop requested
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._memory_count_at_last_dream = 0
        self._dream_count = 0

    @classmethod
    def sqlite(cls, db_path: str, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with SQLite backend."""
        backend = SQLiteDreamBackend(db_path)
        return cls(backend, neural_memory, **kwargs)

    @classmethod
    def postgres(cls, dsn: Optional[str] = None,
                 neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with Postgres+pgvector backend."""
        from dream_postgres_store import DreamPostgresStore
        backend = DreamPostgresStore(dsn=dsn)
        return cls(backend, neural_memory, **kwargs)

    def start(self) -> None:
        """Start the background dream daemon."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dream_loop, daemon=True, name="dream-engine"
        )
        self._thread.start()
        logger.info(
            "Dream engine started: idle=%ss, threshold=%d",
            self._idle_threshold, self._memory_threshold,
        )

    def stop(self) -> None:
        """Stop the dream daemon."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("Dream engine stopped after %d cycles", self._dream_count)

    def touch(self) -> None:
        """Signal activity — resets idle timer."""
        self._last_activity = time.time()

    def dream_now(self) -> Dict[str, Any]:
        """Force an immediate dream cycle. Returns stats."""
        return self._run_dream_cycle()

    # -- Main loop -----------------------------------------------------------

    def _dream_loop(self) -> None:
        """Background daemon: dream when idle or threshold reached."""
        while not self._stop_event.is_set():
            try:
                # Wait 30s but wake immediately if stop() is called
                if self._stop_event.wait(timeout=30.0):
                    break

                idle = time.time() - self._last_activity
                try:
                    stats = self._memory.stats() if self._memory else {"memories": 0}
                    total = stats.get("memories", 0)
                except Exception:
                    total = 0
                new_since_last = total - self._memory_count_at_last_dream

                should_dream = (
                    idle >= self._idle_threshold
                    or new_since_last >= self._memory_threshold
                )

                if should_dream:
                    logger.info(
                        "Dream cycle triggered: idle=%.0fs, new=%d",
                        idle, new_since_last,
                    )
                    self._run_dream_cycle()
                    # Reset idle timer so the next cycle doesn't re-trigger
                    # on the very next 30s wake.  Without this, every poll
                    # after the threshold is crossed sees idle >= threshold
                    # and fires another cycle, pegging CPU on spreading
                    # activation indefinitely.
                    self._last_activity = time.time()

            except Exception as e:
                logger.debug("Dream loop error: %s", e)
                if self._stop_event.wait(timeout=60):
                    break

    # -- Dream Cycle ---------------------------------------------------------

    def _run_dream_cycle(self) -> Dict[str, Any]:
        """Execute a full NREM → REM → Insight cycle."""
        with self._lock:
            start = time.time()
            total_stats: Dict[str, Any] = {"nrem": {}, "rem": {}, "insights": {}}

            try:
                total_stats["nrem"] = self._phase_nrem()
                total_stats["rem"] = self._phase_rem()
                total_stats["insights"] = self._phase_insights()

                self._dream_count += 1
                if self._memory:
                    try:
                        s = self._memory.stats()
                        self._memory_count_at_last_dream = s.get("memories", 0)
                    except Exception:
                        pass

                total_stats["duration"] = time.time() - start
                total_stats["dream_id"] = self._dream_count

                logger.info(
                    "Dream #%d complete: %.1fs | NREM: %d+/ %d- / %d pruned | REM: %d bridges | Insights: %d",
                    self._dream_count, total_stats["duration"],
                    total_stats["nrem"].get("strengthened", 0),
                    total_stats["nrem"].get("weakened", 0),
                    total_stats["nrem"].get("pruned", 0),
                    total_stats["rem"].get("bridges", 0),
                    total_stats["insights"].get("insights", 0),
                )

            except Exception as e:
                logger.error("Dream cycle failed: %s", e)
                total_stats["error"] = str(e)
            finally:
                # Reset the idle timer after every cycle, regardless of how
                # _run_dream_cycle was invoked (loop poll OR explicit dream_now()).
                # Without this in the finally, dream_now() leaves _last_activity
                # untouched, so the very next 30s loop wake sees idle >>
                # idle_threshold and immediately fires another cycle — pegging
                # CPU on spreading activation. The duplicate reset in the loop
                # caller is now redundant but harmless.
                self._last_activity = time.time()

            return total_stats

    # -- Phase 1: NREM -------------------------------------------------------

    def _phase_nrem(self) -> Dict[str, Any]:
        """NREM: Replay, strengthen active, weaken inactive, prune dead.

        For each recent memory:
          1. Fire spreading activation
          2. Activated edges: batch strengthen
          3. Non-activated edges: bulk weaken (single SQL UPDATE)
          4. Edges below threshold: prune
        """
        stats = {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}
        session_id = self._backend.start_session("nrem")

        try:
            memories = self._backend.get_recent_memories(self._max_memories)
            if not memories:
                return stats

            activated_edges: Set[Tuple[int, int]] = set()

            for mem in memories:
                mid = mem["id"]
                if self._memory:
                    try:
                        activated = self._memory.think(mid, depth=2)
                        for a in activated:
                            aid = a.get("id")
                            if aid and aid != mid:
                                activated_edges.add((min(mid, aid), max(mid, aid)))
                    except Exception:
                        pass
                stats["processed"] += 1

            # Batch strengthen activated edges (usually small set)
            if activated_edges:
                stats["strengthened"] = self._backend.batch_strengthen_connections(
                    list(activated_edges), 0.05
                )

            # Bulk weaken ALL non-activated connections above threshold
            # Single SQL UPDATE instead of per-row loop
            stats["weakened"] = self._backend.batch_weaken_connections(
                threshold=0.05, delta=0.01
            )

            # Prune dead connections
            stats["pruned"] = self._backend.prune_weak(0.05)

            # Every NREM cycle: prune old history + orphans
            try:
                pruned_hist = self._backend.prune_connection_history(keep_days=7)
                if pruned_hist:
                    logger.info("Pruned %d old connection_history entries", pruned_hist)
                pruned_sessions = self._backend.prune_old_dream_sessions(keep_days=30)
                if pruned_sessions:
                    logger.info("Pruned %d old dream sessions", pruned_sessions)
                pruned_orphans = self._backend.prune_orphans()
                if pruned_orphans:
                    logger.info("Pruned %d orphan connections", pruned_orphans)
            except Exception as e:
                logger.debug("Maintenance cleanup error: %s", e)

        except Exception as e:
            logger.debug("NREM phase error: %s", e)
        finally:
            try:
                self._backend.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("NREM finish_session failed: %s", e)

        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def _phase_rem(self) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Search via embedding similarity for unconnected but similar
        3. Create tentative bridge connections (weight 0.1-0.3)
        """
        stats = {"explored": 0, "bridges": 0, "rejected": 0}
        session_id = self._backend.start_session("rem")

        try:
            isolated = self._backend.get_isolated_memories(max_connections=3, limit=50)
            if not isolated:
                return stats

            for mem in isolated:
                mid = mem["id"]
                content = mem["content"]

                if not self._memory or not content:
                    continue

                try:
                    similar = self._memory.recall(content[:200], k=10)
                    stats["explored"] += 1

                    for sim in similar:
                        sim_id = sim.get("id")
                        sim_score = sim.get("similarity", 0)

                        if not sim_id or sim_id == mid:
                            continue
                        if sim_score < 0.3 or sim_score > 0.95:
                            continue

                        bridge_weight = round(sim_score * 0.3, 3)
                        # Only count + log when add_bridge actually inserted.
                        # Pre-existing edges return False; logging a "0.0 → w"
                        # change on those would falsify connection_history.
                        # Canonicalise the (src, tgt) for the history row to
                        # match the connections row's canonical orientation.
                        if self._backend.add_bridge(mid, sim_id, bridge_weight):
                            src, tgt = (mid, sim_id) if mid < sim_id else (sim_id, mid)
                            self._backend.log_connection_change(
                                src, tgt, 0.0, bridge_weight, "rem_bridge"
                            )
                            stats["bridges"] += 1

                except Exception:
                    pass

        except Exception as e:
            logger.debug("REM phase error: %s", e)
        finally:
            try:
                self._backend.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("REM finish_session failed: %s", e)

        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def _phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction.

        1. Find connected components (communities)
        2. Identify bridge nodes connecting communities
        3. Create insight memories for dense clusters
        """
        stats = {"communities": 0, "bridges": 0, "insights": 0, "derived_facts": 0}
        session_id = self._backend.start_session("insight")

        try:
            edges = self._backend.get_connections()
            if not edges:
                return stats

            # Build adjacency
            adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            nodes = set()
            for e in edges:
                s, t, w = e["source_id"], e["target_id"], e["weight"]
                adj[s].append((t, w))
                adj[t].append((s, w))
                nodes.add(s)
                nodes.add(t)

            communities = self._detect_communities(edges, nodes, adj)
            stats["communities"] = len(communities)

            # Map nodes to communities
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i

            # Find bridge nodes
            bridge_nodes = set()
            for e in edges:
                s_comm = node_to_comm.get(e["source_id"], -1)
                t_comm = node_to_comm.get(e["target_id"], -1)
                if s_comm != t_comm and s_comm >= 0 and t_comm >= 0:
                    bridge_nodes.add(e["source_id"])
                    bridge_nodes.add(e["target_id"])
            stats["bridges"] = len(bridge_nodes)

            # Create cluster insights
            for i, comm in enumerate(communities):
                if len(comm) < 3:
                    continue
                theme = self._extract_theme(comm)
                confidence = min(len(comm) / 10.0, 1.0)
                content = f"Cluster of {len(comm)} related memories: {theme}"
                self._backend.add_insight(session_id, "cluster", comm[0], content, confidence)
                stats["insights"] += 1
                if self._write_derived_cluster_memory(comm, content, confidence) is not None:
                    stats["derived_facts"] += 1

            # Create bridge insights — use the adjacency map we already built
            # (one O(E) construction up front) instead of rescanning the
            # entire edge list per bridge node. The previous loop was
            # O(|bridges| * |edges|), which on a 10K-edge / 500-bridge graph
            # meant 5M iterations every Insight phase.
            for bnode in bridge_nodes:
                bridging_communities = set()
                for neighbor, _w in adj.get(bnode, ()):
                    bridging_communities.add(node_to_comm.get(neighbor, -1))
                bridging_communities.discard(-1)

                if len(bridging_communities) >= 2:
                    content = (
                        f"Bridge connecting {len(bridging_communities)} communities, "
                        f"memory #{bnode}"
                    )
                    self._backend.add_insight(session_id, "bridge", bnode, content, 0.8)
                    stats["insights"] += 1

        except Exception as e:
            logger.debug("Insight phase error: %s", e)
        finally:
            try:
                self._backend.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("Insight finish_session failed: %s", e)

        return stats

    def _detect_communities(self, edges: List[Dict[str, Any]], nodes: set,
                            adj: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
        """Louvain community detection with deterministic BFS fallback."""
        if HAS_NETWORKX:
            try:
                graph = nx.Graph()
                graph.add_nodes_from(nodes)
                for e in edges:
                    graph.add_edge(e["source_id"], e["target_id"], weight=float(e.get("weight", 0.0) or 0.0))
                if hasattr(nx.algorithms.community, "louvain_communities"):
                    comms = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=42)
                    out = [sorted(int(n) for n in comm) for comm in comms if comm]
                    if out:
                        out.sort(key=lambda c: (-len(c), c[0]))
                        return out
            except Exception:
                pass

        visited = set()
        communities: List[List[int]] = []
        for node in nodes:
            if node in visited:
                continue
            component = []
            queue = [node]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.append(curr)
                for neighbor, _ in adj.get(curr, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            communities.append(component)
        return communities

    def _write_derived_cluster_memory(self, comm: List[int], content: str, confidence: float) -> int | None:
        """Materialize a dream insight as a first-class derived memory.

        Dedup strategy:
        1) Reuse existing identical derived:cluster content when present.
        2) Otherwise create a new memory entry (with conflict detection enabled).
        """
        if not self._memory:
            return None

        store = getattr(self._memory, "store", None)
        if store is None and hasattr(self._memory, "_sqlite_memory"):
            store = self._memory._sqlite_memory.store

        derived_id: Optional[int] = None

        # Reuse exact duplicate first to prevent unbounded growth.
        if store is not None:
            try:
                lock = getattr(store, "_lock", None)
                if lock is not None:
                    with lock:
                        row = store.conn.execute(
                            "SELECT id FROM memories WHERE label = ? AND content = ? ORDER BY id DESC LIMIT 1",
                            ("derived:cluster", content),
                        ).fetchone()
                else:
                    row = store.conn.execute(
                        "SELECT id FROM memories WHERE label = ? AND content = ? ORDER BY id DESC LIMIT 1",
                        ("derived:cluster", content),
                    ).fetchone()

                if row is not None:
                    derived_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
                    try:
                        store.touch(derived_id)
                    except Exception:
                        pass
            except Exception:
                pass

        # No exact duplicate found: create one.
        if derived_id is None:
            try:
                created = self._memory.remember(
                    content,
                    label="derived:cluster",
                    auto_connect=False,
                    detect_conflicts=True,
                )
                if isinstance(created, list):
                    created = created[0]
                derived_id = int(created)
            except Exception:
                return None

        # Ensure derived_from links to source memories exist.
        try:
            if store is None:
                store = getattr(self._memory, "store", None)
                if store is None and hasattr(self._memory, "_sqlite_memory"):
                    store = self._memory._sqlite_memory.store
            if store is not None:
                link_weight = max(0.35, min(0.95, confidence))
                for source_id in comm:
                    sid = int(source_id)
                    if sid != int(derived_id):
                        store.add_connection(int(derived_id), sid, link_weight, edge_type="derived_from")
        except Exception:
            pass

        return int(derived_id)

    # -- Helpers -------------------------------------------------------------

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Extract common themes from node IDs (simple keyword frequency)."""
        # If we have memory access, get contents via store (thread-safe)
        contents = []
        if self._memory and hasattr(self._memory, 'store'):
            try:
                placeholders = ",".join("?" * len(node_ids))
                with self._memory.store._lock:
                    rows = self._memory.store.conn.execute(
                        f"SELECT content FROM memories WHERE id IN ({placeholders})",
                        tuple(node_ids)
                    ).fetchall()
                    contents = [r[0] for r in rows if r[0]]
            except Exception:
                pass

        if not contents:
            return f"{len(node_ids)} memories"

        word_freq: Dict[str, int] = defaultdict(int)
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "it", "its", "this", "that", "user", "assistant", "and", "or",
            "but", "not", "if", "then", "so", "just", "also", "very",
            "really", "like", "get", "got", "want", "need", "think",
            "know", "see", "look", "make", "let", "use", "still",
        }
        for c in contents:
            for w in c.lower().split():
                w = w.strip(".,!?;:'\"()[]{}#@")
                if len(w) > 3 and w not in stopwords:
                    word_freq[w] += 1

        top = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
        return ", ".join(w for w, _ in top) if top else "mixed topics"

    def get_stats(self) -> Dict[str, Any]:
        """Get dream engine statistics."""
        base = self._backend.get_dream_stats()
        base["engine_running"] = not self._stop_event.is_set() and (self._thread is not None and self._thread.is_alive())
        base["dream_cycles"] = self._dream_count
        return base
