"""
Dream Postgres Store — dream-specific tables on Postgres backend.

Reuses the connection-pool pattern from postgres_store.py and shares the
same DSN env vars (MM_POSTGRES_DSN or MM_POSTGRES_HOST/PORT/DB/USER/PASSWORD).

Canonical edge ordering and bridge-merge semantics match the SQLite
dream store so anything toggling between backends sees consistent
graph state.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env loader (lightweight, mirrors postgres_store.py)
# ---------------------------------------------------------------------------

def _load_dotenv(paths: list[str]) -> dict:
    env: dict = {}
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip("\"'")
                        if key and val:
                            env.setdefault(key, val)
            except Exception:
                pass
    return env


_dotenv = _load_dotenv([
    ".env",
    str(Path.home() / ".hermes" / ".env"),
    str(Path(__file__).parent / ".env"),
])


def _env(key: str, fallback: str = "") -> str:
    return os.environ.get(key) or _dotenv.get(key, fallback)


def _build_dsn() -> str:
    dsn = _env("MM_POSTGRES_DSN", "")
    if dsn:
        return dsn
    host = _env("MM_POSTGRES_HOST", "127.0.0.1")
    port = _env("MM_POSTGRES_PORT", "5432")
    db = _env("MM_POSTGRES_DB", "mazemaker")
    user = _env("MM_POSTGRES_USER", "mazemaker")
    password = _env("MM_POSTGRES_PASSWORD", "")
    if not password:
        logger.warning(
            "MM_POSTGRES_PASSWORD not set — add it to ~/.hermes/.env or "
            "the MM_POSTGRES_DSN env var"
        )
    return (
        f"host={host} port={port} dbname={db} user={user} password={password}"
    )


_DREAM_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_sessions (
    id                       BIGSERIAL PRIMARY KEY,
    started_at               DOUBLE PRECISION NOT NULL,
    finished_at              DOUBLE PRECISION,
    phase                    TEXT NOT NULL,
    memories_processed       INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned       INTEGER DEFAULT 0,
    bridges_found            INTEGER DEFAULT 0,
    insights_created         INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dream_insights (
    id               BIGSERIAL PRIMARY KEY,
    session_id       BIGINT,
    insight_type     TEXT NOT NULL,
    source_memory_id BIGINT,
    content          TEXT,
    confidence       DOUBLE PRECISION DEFAULT 0.0,
    created_at       DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS connection_history (
    id          BIGSERIAL PRIMARY KEY,
    source_id   BIGINT NOT NULL,
    target_id   BIGINT NOT NULL,
    old_weight  DOUBLE PRECISION,
    new_weight  DOUBLE PRECISION,
    reason      TEXT,
    changed_at  DOUBLE PRECISION NOT NULL,
    UNIQUE (source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_dream_insights_type
    ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session
    ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_dream_conn_history
    ON connection_history(source_id, target_id);
CREATE INDEX IF NOT EXISTS idx_dream_conn_history_changed_at
    ON connection_history(changed_at);
CREATE INDEX IF NOT EXISTS idx_dream_sessions_started_at
    ON dream_sessions(started_at);
"""


class DreamPostgresStore:
    """Postgres-backed dream store. Creates dream tables on first use.

    Provides the dream-store API surface used by DreamWorker.
    """

    def __init__(self, dsn: str | None = None, min_size: int = 1, max_size: int = 4):
        try:
            import psycopg  # noqa: F401
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise ImportError(
                "psycopg[binary]>=3 and psycopg_pool are required for the "
                "Postgres dream backend"
            ) from exc

        self._dsn = dsn or _build_dsn()
        self.pool = ConnectionPool(
            self._dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            open=True,
        )
        self._ensure_schema()

    @classmethod
    def from_config(cls, config: dict) -> "DreamPostgresStore":
        """Create from config dict (postgres section). Env vars override."""
        dsn = config.get("dsn", "") or None
        return cls(dsn=dsn)

    @contextmanager
    def _cursor(self):
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                yield conn, cur

    def _ensure_schema(self) -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(_DREAM_PG_SCHEMA)

    # -- Dream Sessions ------------------------------------------------------

    def start_session(self, phase: str) -> int:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO dream_sessions (started_at, phase) "
                "VALUES (%s, %s) RETURNING id",
                (time.time(), phase),
            )
            row = cur.fetchone()
            return int(row[0]) if row else -1

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE dream_sessions SET "
                "finished_at = %s, "
                "memories_processed = %s, "
                "connections_strengthened = %s, "
                "connections_pruned = %s, "
                "bridges_found = %s, "
                "insights_created = %s "
                "WHERE id = %s",
                (
                    time.time(),
                    stats.get("processed", stats.get("explored", 0)),
                    stats.get("strengthened", 0),
                    stats.get("pruned", 0),
                    stats.get("bridges", 0),
                    stats.get("insights", 0),
                    session_id,
                ),
            )

    # -- Connections ---------------------------------------------------------

    def get_connections(self) -> List[Dict[str, Any]]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT source_id, target_id, weight FROM connections "
                "WHERE weight >= 0.05 ORDER BY weight DESC"
            )
            return [
                {"source_id": int(r[0]), "target_id": int(r[1]), "weight": r[2]}
                for r in cur.fetchall()
            ]

    def get_isolated_memories(self, max_connections: int = 3,
                              limit: int = 50) -> List[Dict[str, Any]]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                """
                SELECT m.id, m.content, COALESCE(conn.cnt, 0) AS conn_count
                FROM memories m
                LEFT JOIN (
                    SELECT source_id AS mid, COUNT(*) AS cnt
                    FROM connections GROUP BY source_id
                ) conn ON m.id = conn.mid
                WHERE COALESCE(conn.cnt, 0) < %s
                ORDER BY m.id DESC
                LIMIT %s
                """,
                (max_connections, limit),
            )
            return [
                {
                    "id": int(r[0]),
                    "content": r[1] or "",
                    "connection_count": int(r[2]),
                }
                for r in cur.fetchall()
            ]

    @staticmethod
    def _canon_pair(source_id: int, target_id: int):
        """Canonicalise (source<target) and reject self-edges.

        All connection rows satisfy source<target (postgres_store.add_connection,
        add_bridge, the dream-side migration). Strict-WHERE updates without
        canonicalisation silently match nothing on (max, min) input.
        """
        if source_id == target_id:
            return None
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        return source_id, target_id

    def strengthen_connection(self, source_id: int, target_id: int,
                              delta: float = 0.05) -> None:
        pair = self._canon_pair(source_id, target_id)
        if pair is None:
            return
        source_id, target_id = pair
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE connections SET weight = LEAST(weight + %s, 1.0) "
                "WHERE source_id = %s AND target_id = %s",
                (delta, source_id, target_id),
            )

    def weaken_connection(self, source_id: int, target_id: int,
                          delta: float = 0.01) -> None:
        pair = self._canon_pair(source_id, target_id)
        if pair is None:
            return
        source_id, target_id = pair
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE connections SET weight = GREATEST(weight - %s, 0.0) "
                "WHERE source_id = %s AND target_id = %s",
                (delta, source_id, target_id),
            )

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                     delta: float = 0.05) -> int:
        if not edges:
            return 0
        canon: list[tuple] = []
        for src, tgt in edges:
            pair = self._canon_pair(int(src), int(tgt))
            if pair is None:
                continue
            canon.append((delta, pair[0], pair[1]))
        if not canon:
            return 0
        with self._cursor() as (_conn, cur):
            cur.executemany(
                "UPDATE connections SET weight = LEAST(weight + %s, 1.0) "
                "WHERE source_id = %s AND target_id = %s",
                canon,
            )
            return len(canon)

    def batch_weaken_connections(self, threshold: float = 0.05,
                                 delta: float = 0.01) -> int:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE connections SET weight = GREATEST(weight - %s, 0.0) "
                "WHERE weight > %s",
                (delta, threshold),
            )
            return cur.rowcount or 0

    def add_bridge(self, source_id: int, target_id: int,
                   weight: float = 0.3) -> bool:
        """Add a new bridge connection. Returns True if newly inserted.

        Canonicalises source<target so strict-WHERE updates match
        regardless of caller orientation. Returns False on self-loop or
        pre-existing edge so
        the REM caller can skip a misleading "0.0 → w" history row.
        """
        if source_id == target_id:
            return False
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id FROM connections WHERE source_id = %s AND target_id = %s",
                (source_id, target_id),
            )
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO connections (source_id, target_id, weight) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (source_id, target_id) DO UPDATE SET "
                "  weight = GREATEST(connections.weight, EXCLUDED.weight)",
                (source_id, target_id, weight),
            )
            return True

    def prune_weak(self, threshold: float = 0.05) -> int:
        with self._cursor() as (_conn, cur):
            cur.execute("DELETE FROM connections WHERE weight < %s", (threshold,))
            return cur.rowcount or 0

    # -- Connection History ---------------------------------------------------

    def log_connection_change(self, source_id: int, target_id: int,
                              old_weight: float, new_weight: float,
                              reason: str) -> None:
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (source_id, target_id) DO UPDATE SET "
                "  old_weight = EXCLUDED.old_weight, "
                "  new_weight = EXCLUDED.new_weight, "
                "  reason = EXCLUDED.reason, "
                "  changed_at = EXCLUDED.changed_at",
                (source_id, target_id, old_weight, new_weight, reason, time.time()),
            )

    # -- Insights -------------------------------------------------------------

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, "
                " confidence, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, insight_type, source_memory_id,
                 content, confidence, time.time()),
            )

    def get_insights(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, session_id, insight_type, source_memory_id, "
                "content, confidence, created_at "
                "FROM dream_insights ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [
                {
                    "id": int(r[0]),
                    "session_id": int(r[1]) if r[1] is not None else None,
                    "type": r[2],
                    "memory_id": r[3],
                    "content": r[4],
                    "confidence": r[5],
                    "created_at": r[6],
                }
                for r in cur.fetchall()
            ]

    # -- Memory access (for NREM replay) -------------------------------------

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, content FROM memories ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [{"id": int(r[0]), "content": r[1] or ""} for r in cur.fetchall()]

    # -- Stats ---------------------------------------------------------------

    def get_dream_stats(self) -> Dict[str, Any]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(memories_processed), 0),
                       COALESCE(SUM(connections_strengthened), 0),
                       COALESCE(SUM(connections_pruned), 0),
                       COALESCE(SUM(bridges_found), 0),
                       COALESCE(SUM(insights_created), 0)
                FROM dream_sessions
                """
            )
            row = cur.fetchone()
            if not row:
                return {"sessions": 0}
            cur.execute(
                "SELECT insight_type, COUNT(*) FROM dream_insights GROUP BY insight_type"
            )
            insight_types = {r[0]: int(r[1]) for r in cur.fetchall()}
            return {
                "sessions": int(row[0]),
                "total_processed": int(row[1]),
                "total_strengthened": int(row[2]),
                "total_pruned": int(row[3]),
                "total_bridges": int(row[4]),
                "total_insights": int(row[5]),
                "insight_types": insight_types,
            }

    def prune_connection_history(self, keep_days: int = 7) -> int:
        cutoff = time.time() - keep_days * 86400
        with self._cursor() as (_conn, cur):
            cur.execute(
                "DELETE FROM connection_history WHERE changed_at < %s",
                (cutoff,),
            )
            return cur.rowcount or 0

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        cutoff = time.time() - keep_days * 86400
        with self._cursor() as (_conn, cur):
            cur.execute(
                "DELETE FROM dream_insights WHERE session_id IN "
                "(SELECT id FROM dream_sessions WHERE started_at < %s)",
                (cutoff,),
            )
            cur.execute(
                "DELETE FROM dream_sessions WHERE started_at < %s",
                (cutoff,),
            )
            return cur.rowcount or 0

    def prune_orphans(self) -> int:
        with self._cursor() as (_conn, cur):
            cur.execute(
                """
                DELETE FROM connections
                WHERE source_id NOT IN (SELECT id FROM memories)
                   OR target_id NOT IN (SELECT id FROM memories)
                """
            )
            return cur.rowcount or 0

    def close(self) -> None:
        try:
            self.pool.close()
        except Exception:
            pass
