"""
CppDreamBackend — Dream backend that uses C++ bridge for MSSQL operations.

All graph operations (strengthen, weaken, prune, spreading activation)
go through the C++ libneural_memory.so → ODBC → MSSQL.

SQLite fallback is handled by SQLiteDreamBackend in dream_engine.py.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the C++ bridge
sys.path.insert(0, str(Path(__file__).parent))
from cpp_bridge import NeuralMemoryCpp

from dream_engine import DreamBackend

logger = logging.getLogger(__name__)


class CppDreamBackend(DreamBackend):
    """Dream backend backed by C++ bridge → MSSQL.

    Graph operations (strengthen, weaken, prune) go through the C++ bridge.
    Memory queries go DIRECTLY to MSSQL via pyodbc (the C++ bridge's in-memory
    index may not be populated with existing data).
    """

    def __init__(self, dim: int = 1024):
        import sqlite3

        self._cpp = NeuralMemoryCpp()
        self._cpp.initialize(dim=dim)
        self._dim = dim
        logger.info("CppDreamBackend initialized (dim=%d)", dim)

        # Direct MSSQL connection for queries
        self._mssql_conn = None
        import os
        mssql_server = os.environ.get("MSSQL_SERVER", "")
        mssql_password = os.environ.get("MSSQL_PASSWORD", "")
        if mssql_server and mssql_password:
            try:
                import pyodbc
                self._mssql_conn = pyodbc.connect(
                    f"DRIVER={os.environ.get('MSSQL_DRIVER', '{ODBC Driver 18 for SQL Server}')};"
                    f"SERVER={mssql_server};"
                    f"DATABASE={os.environ.get('MSSQL_DATABASE', 'NeuralMemory')};"
                    f"UID={os.environ.get('MSSQL_USERNAME', 'SA')};"
                    f"PWD={mssql_password};"
                    f"TrustServerCertificate=yes;",
                    autocommit=True
                )
                logger.info("CppDreamBackend: direct MSSQL connection established")
            except Exception as e:
                logger.warning("CppDreamBackend: MSSQL direct connection failed: %s", e)

        # Lightweight SQLite for dream session tracking only
        db_path = str(Path.home() / ".neural_memory" / "dream_sessions.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._session_db = db_path

        conn = sqlite3.connect(db_path)
        conn.executescript("""
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
        """)
        conn.commit()
        conn.close()

    def close(self):
        """Shutdown C++ bridge."""
        if self._cpp:
            self._cpp.shutdown()
            self._cpp = None

    # -- Dream Sessions (lightweight SQLite for tracking) --

    def start_session(self, phase: str) -> int:
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        cur = conn.execute(
            "INSERT INTO dream_sessions (started_at, phase) VALUES (?, ?)",
            (time.time(), phase)
        )
        conn.commit()
        sid = cur.lastrowid
        conn.close()
        return sid

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        conn.execute(
            "UPDATE dream_sessions SET finished_at=?, memories_processed=?, "
            "connections_strengthened=?, connections_pruned=?, "
            "bridges_found=?, insights_created=? WHERE id=?",
            (time.time(),
             stats.get("processed", stats.get("explored", 0)),
             stats.get("strengthened", 0),
             stats.get("pruned", 0),
             stats.get("bridges", 0),
             stats.get("insights", 0),
             session_id)
        )
        conn.commit()
        conn.close()

    # -- Graph Operations (ALL via C++ → MSSQL) --

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memories from MSSQL. Returns [{id, label, content}]."""
        if not self._mssql_conn:
            return []
        try:
            rows = self._mssql_conn.execute(
                "SELECT TOP (?) id, label, content FROM memories ORDER BY created_at DESC",
                (limit,)
            ).fetchall()
            return [{"id": r[0], "label": r[1] or "", "content": r[2] or ""} for r in rows]
        except Exception as e:
            logger.debug("get_recent_memories failed: %s", e)
            return []

    def get_random_memories(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get a random sample of memories directly from MSSQL."""
        if not self._mssql_conn:
            return []
        try:
            rows = self._mssql_conn.execute(
                "SELECT TOP (?) id FROM memories ORDER BY NEWID()",
                (limit,)
            ).fetchall()
            return [{"id": r[0]} for r in rows]
        except Exception as e:
            logger.debug("get_random_memories failed: %s", e)
            return []

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50,
                               oldest_first: bool = False) -> List[Dict[str, Any]]:
        """Find memories with few edges directly from MSSQL."""
        if not self._mssql_conn:
            return []
        try:
            order = "ASC" if oldest_first else "DESC"
            rows = self._mssql_conn.execute(f"""
                SELECT TOP (?) m.id, m.content,
                    (SELECT COUNT(*) FROM connections
                     WHERE source_id = m.id OR target_id = m.id) as cnt
                FROM memories m
                WHERE (SELECT COUNT(*) FROM connections
                       WHERE source_id = m.id OR target_id = m.id) < ?
                ORDER BY m.created_at {order}
            """, (limit, max_connections)).fetchall()
            return [{"id": r[0], "content": r[1] or "", "connection_count": r[2]} for r in rows]
        except Exception as e:
            logger.debug("get_isolated_memories failed: %s", e)
            return []

    def get_connections(self) -> List[Dict[str, Any]]:
        """Get all edges from MSSQL (capped for perf)."""
        if not self._mssql_conn:
            return []
        try:
            rows = self._mssql_conn.execute("""
                SELECT TOP 50000 source_id, target_id, weight
                FROM connections
                WHERE weight >= 0.05
            """).fetchall()
            return [{"source_id": r[0], "target_id": r[1], "weight": r[2]} for r in rows]
        except Exception as e:
            logger.debug("get_connections failed: %s", e)
            return []

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        """Strengthen edge in connections table via MSSQL."""
        if not self._mssql_conn:
            return
        try:
            self._mssql_conn.execute(
                "UPDATE connections SET weight = CASE WHEN weight + ? > 1.0 THEN 1.0 ELSE weight + ? END "
                "WHERE source_id = ? AND target_id = ?",
                delta, delta, source_id, target_id
            )
        except Exception as e:
            logger.debug("strengthen_connection failed: %s", e)

    def batch_strengthen_connections(self, edges: list, delta: float = 0.05) -> int:
        """Batch strengthen activated edges.

        Accepts 2-tuples (source_id, target_id) — fetches current weight and adds delta.
        Also accepts 3-tuples (new_weight, source_id, target_id) for direct weight set.
        """
        if not edges or not self._mssql_conn:
            return 0
        try:
            cursor = self._mssql_conn.cursor()
            count = 0
            for item in edges:
                try:
                    new_w, src, tgt = item
                except ValueError:
                    # 2-tuple: (source_id, target_id) — fetch weight then add delta
                    src, tgt = item
                    try:
                        cursor.execute(
                            "SELECT weight FROM connections WHERE source_id = ? AND target_id = ?",
                            (src, tgt)
                        )
                        row = cursor.fetchone()
                        old_w = row[0] if row else 0.5
                    except Exception:
                        old_w = 0.5
                    new_w = min(old_w + delta, 1.0)
                cursor.execute(
                    "UPDATE connections SET weight = ? "
                    "WHERE source_id = ? AND target_id = ?",
                    (new_w, src, tgt)
                )
                count += 1
            return count
        except Exception as e:
            logger.debug("batch_strengthen failed: %s", e)
            return 0

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        """Weaken edge in connections table via MSSQL."""
        if not self._mssql_conn:
            return
        try:
            self._mssql_conn.execute(
                "UPDATE connections SET weight = CASE WHEN weight - ? < 0.0 THEN 0.0 ELSE weight - ? END "
                "WHERE source_id = ? AND target_id = ?",
                delta, delta, source_id, target_id
            )
        except Exception as e:
            logger.debug("weaken_connection failed: %s", e)

    def batch_weaken_connections(self, updates: list = None, **kwargs) -> int:
        """Batch weaken connections.
        
        Two modes:
        1. Explicit: updates = [(new_weight, source_id, target_id), ...]
        2. Bulk: threshold=0.05, delta=0.01 — single SQL UPDATE for all edges above threshold
        """
        if not self._mssql_conn:
            return 0
        
        # Bulk mode: threshold + delta
        threshold = kwargs.get('threshold')
        delta = kwargs.get('delta', 0.01)
        if threshold is not None:
            try:
                cursor = self._mssql_conn.cursor()
                cursor.execute(
                    "UPDATE connections SET weight = CASE WHEN weight - ? < 0.0 THEN 0.0 ELSE weight - ? END "
                    "WHERE weight > ?",
                    delta, delta, threshold
                )
                return cursor.rowcount
            except Exception as e:
                logger.debug("batch_weaken (bulk) failed: %s", e)
                return 0
        
        # Explicit mode: list of tuples
        if not updates:
            return 0
        try:
            cursor = self._mssql_conn.cursor()
            for new_w, src, tgt in updates:
                cursor.execute(
                    "UPDATE connections SET weight = ? "
                    "WHERE source_id = ? AND target_id = ?",
                    new_w, src, tgt
                )
            return len(updates)
        except Exception as e:
            logger.debug("batch_weaken failed: %s", e)
            return 0

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        """Add bridge edge to connections table via MSSQL."""
        if not self._mssql_conn:
            return
        try:
            # Check if connection already exists (either direction)
            existing = self._mssql_conn.execute(
                "SELECT id FROM connections "
                "WHERE (source_id = ? AND target_id = ?) "
                "OR (source_id = ? AND target_id = ?)",
                source_id, target_id, target_id, source_id
            ).fetchone()
            if not existing:
                self._mssql_conn.execute(
                    "INSERT INTO connections (source_id, target_id, weight, edge_type) "
                    "VALUES (?, ?, ?, ?)",
                    source_id, target_id, weight, "bridge"
                )
        except Exception as e:
            logger.debug("add_bridge failed: %s", e)

    def prune_weak(self, threshold: float = 0.05) -> int:
        """Prune weak connections from connections table via MSSQL."""
        if not self._mssql_conn:
            return 0
        try:
            cursor = self._mssql_conn.execute(
                "DELETE FROM connections WHERE weight < ?",
                threshold
            )
            return cursor.rowcount if hasattr(cursor, 'rowcount') else 0
        except Exception as e:
            logger.debug("prune_weak failed: %s", e)
            return 0

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        """Log connection change to connection_history table (UPSERT)."""
        if not self._mssql_conn:
            return
        try:
            self._mssql_conn.execute(
                "MERGE connection_history AS target "
                "USING (VALUES (?, ?)) AS src(source_id, target_id) "
                "ON target.source_id = src.source_id AND target.target_id = src.target_id "
                "WHEN MATCHED THEN UPDATE SET old_weight=?, new_weight=?, reason=?, changed_at=SYSUTCDATETIME() "
                "WHEN NOT MATCHED THEN INSERT (source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (?, ?, ?, ?, ?, SYSUTCDATETIME());",
                source_id, target_id,   # USING
                old_weight, new_weight, reason,  # UPDATE SET
                source_id, target_id, old_weight, new_weight, reason  # INSERT
            )
        except Exception as e:
            logger.debug("log_connection_change failed: %s", e)
    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Skip — C++/MSSQL handles history internally."""
        return 0

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Prune old dream sessions from SQLite tracking DB."""
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM dream_sessions WHERE started_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_orphans(self) -> int:
        """Skip — C++/MSSQL handles referential integrity."""
        return 0



    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        """Store insight in lightweight SQLite."""
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        conn.execute(
            "INSERT INTO dream_insights "
            "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, insight_type, source_memory_id, content, confidence, time.time())
        )
        conn.commit()
        conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        """Get dream stats from SQLite tracking + C++ graph stats."""
        import sqlite3
        conn = sqlite3.connect(self._session_db)
        try:
            s = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(memories_processed),0), "
                "COALESCE(SUM(connections_strengthened),0), "
                "COALESCE(SUM(connections_pruned),0), "
                "COALESCE(SUM(bridges_found),0), "
                "COALESCE(SUM(insights_created),0) FROM dream_sessions"
            ).fetchone()
            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "graph_stats": self._cpp.get_stats(),
            }
        finally:
            conn.close()
