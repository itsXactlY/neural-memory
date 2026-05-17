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


_PW_WARN_EMITTED = False


def _build_dsn() -> str:
    """URI-form DSN with percent-encoded user/password. The previous
    keyword-DSN form broke on passwords containing spaces, '#', or
    backslashes — see postgres_store._build_dsn for the same fix.
    """
    global _PW_WARN_EMITTED
    dsn = _env("MM_POSTGRES_DSN", "")
    if dsn:
        return dsn
    host = _env("MM_POSTGRES_HOST", "127.0.0.1")
    port = _env("MM_POSTGRES_PORT", "5432")
    db = _env("MM_POSTGRES_DB", "mazemaker")
    user = _env("MM_POSTGRES_USER", "mazemaker")
    password = _env("MM_POSTGRES_PASSWORD", "")
    if not password and not _PW_WARN_EMITTED:
        _PW_WARN_EMITTED = True
        logger.info(
            "MM_POSTGRES_PASSWORD empty — relying on local trust/peer auth"
        )
    import urllib.parse
    user_q = urllib.parse.quote(user, safe="")
    pw_q = urllib.parse.quote(password, safe="")
    return f"postgresql://{user_q}:{pw_q}@{host}:{port}/{db}"


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
    dream_session_id BIGINT
);

-- Drop the legacy UNIQUE (source_id, target_id) on connection_history if
-- a previous bootstrap created it. The table is an append-only audit log
-- of every weight change — a single edge can have many rows over its
-- lifetime (one per dream cycle that touched it). Older schemas modelled
-- it as latest-event-per-edge which silently dropped 99 % of the history.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'connection_history_source_id_target_id_key'
    ) THEN
        ALTER TABLE connection_history
          DROP CONSTRAINT connection_history_source_id_target_id_key;
    END IF;
END$$;

-- Ensure the dream_session_id column exists on legacy bootstraps.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'connection_history' AND column_name = 'dream_session_id'
    ) THEN
        ALTER TABLE connection_history ADD COLUMN dream_session_id BIGINT;
    END IF;
END$$;

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


from dream_engine import DreamBackend  # noqa: E402


class DreamPostgresStore(DreamBackend):
    """Postgres-backed dream store. Creates dream tables on first use.

    Provides the dream-store API surface used by DreamWorker. Inherits
    from DreamBackend so DreamEngine's isinstance check accepts it.
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

        # Honour the same schema env var that PostgresStore uses, so a
        # bench run with MM_POSTGRES_SCHEMA=conv_N targets that schema's
        # memories table for the dream cycle's reads as well.
        import re
        raw_schema = os.environ.get("MM_POSTGRES_SCHEMA", "public").strip() or "public"
        if not re.fullmatch(r"[A-Za-z0-9_]{1,32}", raw_schema):
            raise ValueError(f"Invalid MM_POSTGRES_SCHEMA: {raw_schema!r}")
        self._schema = raw_schema

        self._dsn = dsn or _build_dsn()
        self.pool = ConnectionPool(
            self._dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            configure=self._configure_conn,
            open=True,
        )
        self._ensure_schema()

    def _configure_conn(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path = {self._schema}, public")

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

    def sample_isolated_for_dream(
        self,
        limit: int,
        max_connections: int = 3,
        recent_pct: float = 0.5,
        random_old_pct: float = 0.3,
        low_salience_pct: float = 0.2,
    ) -> List[Dict[str, Any]]:
        recent_n = max(1, int(limit * recent_pct))
        random_n = max(0, int(limit * random_old_pct))
        low_n = max(0, limit - recent_n - random_n)

        seen: set = set()
        out: List[Dict[str, Any]] = []

        def _push(rows):
            for r in rows:
                rid = int(r[0])
                if rid in seen:
                    continue
                seen.add(rid)
                out.append({
                    "id": rid,
                    "content": r[1] or "",
                    "connection_count": int(r[2]),
                })
                if len(out) >= limit:
                    return True
            return False

        base = (
            "SELECT m.id, m.content, COALESCE(conn.cnt, 0) AS conn_count "
            "FROM memories m "
            "LEFT JOIN ( "
            "    SELECT source_id AS mid, COUNT(*) AS cnt "
            "    FROM connections GROUP BY source_id "
            ") conn ON m.id = conn.mid "
            "WHERE COALESCE(conn.cnt, 0) < %s "
        )

        with self._cursor() as (_conn, cur):
            cur.execute(base + "ORDER BY m.created_at DESC LIMIT %s",
                        (max_connections, recent_n))
            _push(cur.fetchall())

            if random_n > 0 and len(out) < limit:
                cur.execute(base + "ORDER BY random() LIMIT %s",
                            (max_connections, random_n + max(0, recent_n // 4)))
                _push(cur.fetchall())

            if low_n > 0 and len(out) < limit:
                try:
                    cur.execute(
                        base + "ORDER BY m.salience ASC, m.last_accessed ASC LIMIT %s",
                        (max_connections, low_n + max(0, (recent_n + random_n) // 4)),
                    )
                    _push(cur.fetchall())
                except Exception:
                    cur.execute(base + "ORDER BY m.created_at ASC LIMIT %s",
                                (max_connections, low_n))
                    _push(cur.fetchall())

        return out[:limit]

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
                                     delta: float = 0.05,
                                     dream_session_id: Optional[int] = None) -> int:
        """Bulk-strengthen activated edges and capture per-edge history.

        Mirrors SQLiteDreamBackend semantics: NREM logs one row per
        affected edge so the dashboard can replay the cycle's strengthen
        burst. dream_session_id ties them to the session row.
        """
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
        now = time.time()
        # Bulk path: COPY pairs into a temp table, then ONE UPDATE FROM
        # joining temp → connections. Replaces an executemany N-round-trip
        # loop. Same pattern for the audit-row insert: derive history rows
        # from the post-update weights via RETURNING + COPY.
        with self._cursor() as (_conn, cur):
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _nrem_pairs "
                "(source_id BIGINT, target_id BIGINT) ON COMMIT PRESERVE ROWS"
            )
            cur.execute("TRUNCATE _nrem_pairs")
            with cur.copy(
                "COPY _nrem_pairs (source_id, target_id) FROM STDIN"
            ) as cp:
                for _d, s, t in canon:
                    cp.write_row((s, t))
            if dream_session_id is not None:
                cur.execute(
                    "UPDATE connections c "
                    "SET weight = LEAST(c.weight + %s, 1.0) "
                    "FROM _nrem_pairs p "
                    "WHERE c.source_id = p.source_id "
                    "  AND c.target_id = p.target_id "
                    "RETURNING c.source_id, c.target_id, c.weight",
                    (delta,),
                )
                rows = cur.fetchall()
                if rows:
                    cur.execute(
                        "CREATE TEMP TABLE IF NOT EXISTS _nrem_hist "
                        "(source_id BIGINT, target_id BIGINT, "
                        " old_weight DOUBLE PRECISION, new_weight DOUBLE PRECISION, "
                        " reason TEXT, changed_at DOUBLE PRECISION, "
                        " dream_session_id BIGINT) ON COMMIT PRESERVE ROWS"
                    )
                    cur.execute("TRUNCATE _nrem_hist")
                    with cur.copy(
                        "COPY _nrem_hist (source_id, target_id, old_weight, "
                        "new_weight, reason, changed_at, dream_session_id) "
                        "FROM STDIN"
                    ) as cp:
                        dsid = int(dream_session_id)
                        for r in rows:
                            new_w = float(r[2] or 0.0)
                            cp.write_row((int(r[0]), int(r[1]),
                                          max(0.0, new_w - delta),
                                          new_w, "nrem_strengthen",
                                          now, dsid))
                    cur.execute(
                        "INSERT INTO connection_history "
                        "(source_id, target_id, old_weight, new_weight, "
                        " reason, changed_at, dream_session_id) "
                        "SELECT source_id, target_id, old_weight, new_weight, "
                        "       reason, changed_at, dream_session_id "
                        "FROM _nrem_hist"
                    )
            else:
                cur.execute(
                    "UPDATE connections c "
                    "SET weight = LEAST(c.weight + %s, 1.0) "
                    "FROM _nrem_pairs p "
                    "WHERE c.source_id = p.source_id "
                    "  AND c.target_id = p.target_id",
                    (delta,),
                )
        return len(canon)

    def batch_weaken_connections(self, threshold: float = 0.05,
                                 delta: float = 0.01,
                                 dream_session_id: Optional[int] = None) -> int:
        """Bulk-weaken every connection above the threshold.

        Writes ONE summary connection_history row per cycle when
        dream_session_id is given — batch_weaken can touch tens of
        thousands of edges per cycle and per-edge audit rows would
        inflate connection_history without adding signal.
        """
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE connections SET weight = GREATEST(weight - %s, 0.0) "
                "WHERE weight > %s",
                (delta, threshold),
            )
            n = cur.rowcount or 0
            if dream_session_id is not None and n > 0:
                cur.execute(
                    "INSERT INTO connection_history "
                    "(source_id, target_id, old_weight, new_weight, "
                    " reason, changed_at, dream_session_id) "
                    "VALUES (-1, -1, %s, %s, %s, %s, %s)",
                    (delta, 0.0, f"nrem_bulk_weaken:{n}",
                     time.time(), int(dream_session_id)),
                )
            return n

    def add_bridges_batch(
        self,
        bridges,
        dream_session_id=None,
        reason: str = "rem_bridge",
    ) -> int:
        """Override of DreamBackend default to give PG a true bulk path.

        Default fallback loops `add_bridge` — each call does a SELECT
        existence check + INSERT — so a 6000-bridge REM phase burns
        12 000 round-trips. This override is the same as the SQLite
        bulk impl: COPY into a temp staging table, EXISTS-anti-join
        against `connections` to find new rows, single bulk INSERT.
        One commit per cycle instead of one per bridge.
        """
        if not bridges:
            return 0
        canon = []
        seen = set()
        for s, t, w in bridges:
            si, ti = int(s), int(t)
            if si == ti:
                continue
            if si > ti:
                si, ti = ti, si
            if (si, ti) in seen:
                continue
            seen.add((si, ti))
            canon.append((si, ti, max(0.0, min(1.0, float(w)))))
        if not canon:
            return 0
        now = time.time()
        with self._cursor() as (_conn, cur):
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _bridge_stage "
                "(source_id BIGINT, target_id BIGINT, "
                " weight DOUBLE PRECISION) ON COMMIT PRESERVE ROWS"
            )
            cur.execute("TRUNCATE _bridge_stage")
            with cur.copy(
                "COPY _bridge_stage (source_id, target_id, weight) FROM STDIN"
            ) as cp:
                for s, t, w in canon:
                    cp.write_row((s, t, w))
            cur.execute(
                "INSERT INTO connections "
                "(source_id, target_id, weight, edge_type, created_at) "
                "SELECT s.source_id, s.target_id, s.weight, 'bridge', NOW() "
                "FROM _bridge_stage s "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM connections c "
                "  WHERE c.source_id = s.source_id "
                "    AND c.target_id = s.target_id"
                ") "
                "RETURNING source_id, target_id, weight"
            )
            new_rows = cur.fetchall()
            if dream_session_id is not None and new_rows:
                cur.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS _bridge_hist "
                    "(source_id BIGINT, target_id BIGINT, "
                    " old_weight DOUBLE PRECISION, new_weight DOUBLE PRECISION, "
                    " reason TEXT, changed_at DOUBLE PRECISION, "
                    " dream_session_id BIGINT) ON COMMIT PRESERVE ROWS"
                )
                cur.execute("TRUNCATE _bridge_hist")
                with cur.copy(
                    "COPY _bridge_hist (source_id, target_id, old_weight, "
                    "new_weight, reason, changed_at, dream_session_id) "
                    "FROM STDIN"
                ) as cp:
                    dsid = int(dream_session_id)
                    for r in new_rows:
                        cp.write_row((int(r[0]), int(r[1]), 0.0,
                                      float(r[2] or 0.0), reason, now, dsid))
                cur.execute(
                    "INSERT INTO connection_history "
                    "(source_id, target_id, old_weight, new_weight, "
                    " reason, changed_at, dream_session_id) "
                    "SELECT source_id, target_id, old_weight, new_weight, "
                    "       reason, changed_at, dream_session_id "
                    "FROM _bridge_hist"
                )
        return len(new_rows)

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
                              reason: str,
                              dream_session_id: Optional[int] = None) -> None:
        """Append a connection_history row (audit log).

        Plain INSERT — connection_history is intentionally append-only;
        a single edge has many rows over its lifetime. The legacy
        UNIQUE (source_id, target_id) constraint that an earlier
        bootstrap created collapsed this into a latest-event-only
        table; that constraint was dropped as part of the schema
        migration in _DREAM_PG_SCHEMA.
        """
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, "
                " reason, changed_at, dream_session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (source_id, target_id, old_weight, new_weight,
                 reason, time.time(),
                 int(dream_session_id) if dream_session_id is not None else None),
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

    def recent_cluster_anchors(self, window_seconds: float) -> set:
        cutoff = time.time() - float(window_seconds)
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT DISTINCT source_memory_id FROM dream_insights "
                "WHERE insight_type = 'cluster' AND created_at >= %s",
                (cutoff,),
            )
            return {int(r[0]) for r in cur.fetchall() if r[0] is not None}

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

    def get_memory_vectors(self, memory_ids: List[int]) -> Dict[int, List[float]]:
        """Return {id: vector_floats} for the requested ids.

        Needed by the supersedes phase. Without this, the default-empty
        DreamBackend implementation kicked in and _phase_supersedes
        silently no-op'd whenever the dream worker ran against PG.
        """
        if not memory_ids:
            return {}
        ids = [int(i) for i in memory_ids if i is not None]
        try:
            with self._cursor() as (_conn, cur):
                cur.execute(
                    "SELECT id, embedding FROM memories "
                    "WHERE id = ANY(%s) AND embedding IS NOT NULL",
                    (ids,),
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning("PG get_memory_vectors failed: %s", exc)
            return {}
        # psycopg's pgvector adapter returns numpy arrays / lists already.
        out: Dict[int, List[float]] = {}
        for mid, vec in rows:
            if vec is None:
                continue
            try:
                out[int(mid)] = [float(x) for x in vec]
            except Exception:
                continue
        return out

    def get_memory_metadata(self, memory_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """PG counterpart for the SQLite supersedes-phase metadata fetch."""
        if not memory_ids:
            return {}
        ids = [int(i) for i in memory_ids if i is not None]
        try:
            with self._cursor() as (_conn, cur):
                cur.execute(
                    "SELECT id, EXTRACT(EPOCH FROM created_at) AS ts, content "
                    "FROM memories WHERE id = ANY(%s)",
                    (ids,),
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning("PG get_memory_metadata failed: %s", exc)
            return {}
        return {
            int(r[0]): {
                "created_at": float(r[1] or 0.0),
                "content": r[2] or "",
            }
            for r in rows
        }

    def sample_for_dream(
        self,
        limit: int,
        recent_pct: float = 0.5,
        random_old_pct: float = 0.3,
        low_salience_pct: float = 0.2,
    ) -> List[Dict[str, Any]]:
        recent_n = max(1, int(limit * recent_pct))
        random_n = max(0, int(limit * random_old_pct))
        low_n = max(0, limit - recent_n - random_n)

        seen: set = set()
        out: List[Dict[str, Any]] = []

        def _push(rows):
            for r in rows:
                rid = int(r[0])
                if rid in seen:
                    continue
                seen.add(rid)
                out.append({"id": rid, "content": r[1] or ""})
                if len(out) >= limit:
                    return True
            return False

        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, content FROM memories ORDER BY created_at DESC LIMIT %s",
                (recent_n,),
            )
            _push(cur.fetchall())

            if random_n > 0 and len(out) < limit:
                cur.execute(
                    "SELECT id, content FROM memories ORDER BY random() LIMIT %s",
                    (random_n + max(0, recent_n // 4),),
                )
                _push(cur.fetchall())

            # Postgres memories table may or may not have salience/last_accessed
            # columns depending on which schema migration the deployment ran.
            # Try the salience-aware sort and fall back to created_at ASC if
            # the columns aren't there — the goal (reach old memories) is
            # still served by the random pool above.
            if low_n > 0 and len(out) < limit:
                try:
                    cur.execute(
                        "SELECT id, content FROM memories "
                        "ORDER BY salience ASC, last_accessed ASC LIMIT %s",
                        (low_n + max(0, (recent_n + random_n) // 4),),
                    )
                    _push(cur.fetchall())
                except Exception:
                    cur.execute(
                        "SELECT id, content FROM memories "
                        "ORDER BY created_at ASC LIMIT %s",
                        (low_n,),
                    )
                    _push(cur.fetchall())

        return out[:limit]

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
        # FK constraint REFERENCES memories(id) already prevents orphan
        # connections at INSERT time. The legacy NOT IN double-scan was
        # O(connections × memories) — 30+ min on 300k-row cache schemas.
        # Use NOT EXISTS (anti-join, hash-joined in PG) and gate behind a
        # cheap fast-path: if the FK is enforced, this can only return 0.
        with self._cursor() as (_conn, cur):
            cur.execute(
                "DELETE FROM connections c "
                "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.source_id) "
                "   OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.target_id)"
            )
            return cur.rowcount or 0

    def close(self) -> None:
        try:
            self.pool.close()
        except Exception:
            pass
