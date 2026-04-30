#!/usr/bin/env python3
"""
postgres_store.py — PostgreSQL + pgvector storage backend for Mazemaker.

Acts as a graph/cold-storage companion to the SQLite source-of-truth.
ID alignment, canonical edge ordering, and MERGE-with-max upsert
semantics let memory_client / mazemaker treat this store as a drop-in
mirror of the SQLite primary.

Credentials resolution order:
  1. Environment variable: MM_POSTGRES_DSN  (preferred)
  2. Discrete env vars: MM_POSTGRES_HOST/PORT/DB/USER/PASSWORD
  3. .env file (CWD, ~/.hermes/.env, plugin dir)
  4. Defaults (localhost:5432, mazemaker DB, mazemaker user)

Requires: psycopg[binary]>=3, pgvector, psycopg_pool.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env loader (lightweight)
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
    """Resolve a DSN from env, .env, or discrete fields."""
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
    # Use keyword DSN (libpq syntax) — psycopg accepts it directly.
    return (
        f"host={host} port={port} dbname={db} user={user} password={password}"
    )


# ---------------------------------------------------------------------------
# Schema — pgvector + HNSW indexes
# ---------------------------------------------------------------------------
#
# Why vector(MAX_DIM)? pgvector requires a fixed dim per column. We cannot
# know the embedding dim until the first write, so we defer column creation
# until then via _ensure_embedding_column(). The memories table itself is
# created up-front without the embedding column; the column (and HNSW
# index) is added on first write with the observed dim.
#
# Canonical edge invariant (source<target) is enforced in add_connection()
# and the migration block. Without that the graph traversal in
# recall_multihop would miss half the bridges.

_BASE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id            BIGSERIAL PRIMARY KEY,
    label         TEXT,
    content       TEXT,
    vector_dim    INTEGER NOT NULL,
    salience      DOUBLE PRECISION DEFAULT 1.0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS connections (
    id          BIGSERIAL PRIMARY KEY,
    source_id   BIGINT REFERENCES memories(id),
    target_id   BIGINT REFERENCES memories(id),
    weight      DOUBLE PRECISION DEFAULT 0.5,
    edge_type   TEXT DEFAULT 'similar',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_conn_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_conn_target ON connections(target_id);
CREATE INDEX IF NOT EXISTS idx_conn_weight ON connections(weight);
CREATE INDEX IF NOT EXISTS idx_conn_edge_type_weight ON connections(edge_type, weight);
CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON memories
    USING gin (to_tsvector('simple', coalesce(content, '')));
"""


class PostgresStore:
    """Postgres + pgvector store; graph/cold-storage mirror of SQLite.

    Connection pooling via psycopg_pool.ConnectionPool keeps short-lived
    cursors fast. The pool is lazily opened on construction and closed
    by close().
    """

    def __init__(self, dsn: str | None = None, min_size: int = 1, max_size: int = 8):
        try:
            import psycopg  # noqa: F401
            from psycopg.rows import tuple_row  # noqa: F401
            from psycopg_pool import ConnectionPool
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise ImportError(
                "psycopg[binary]>=3, psycopg_pool and pgvector are required "
                "for the Postgres backend"
            ) from exc

        self._dsn = dsn or _build_dsn()
        self._register_vector = register_vector
        self._lock = threading.Lock()
        # The pool's `configure` hook registers the vector adapter on each
        # new connection so list[float]<->vector conversion works without
        # per-call casts.
        self.pool = ConnectionPool(
            self._dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            configure=self._configure_conn,
            open=True,
        )
        self._embedding_dim: Optional[int] = None
        self._ensure_schema()

    # -- pool / connection helpers ------------------------------------------

    def _configure_conn(self, conn) -> None:
        """Register pgvector adapter on a freshly-opened connection."""
        # CREATE EXTENSION must already have run for register_vector to find
        # the vector OID; _ensure_schema() runs that on the first borrow.
        try:
            self._register_vector(conn)
        except Exception:
            # On the very first connect (before _ensure_schema has CREATE
            # EXTENSION'd) the type lookup fails. We re-register from
            # _ensure_schema once the extension is present.
            pass

    @contextmanager
    def _cursor(self):
        """Borrow a connection + cursor from the pool."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                yield conn, cur

    # -- schema -------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create base tables, extension, and re-register vector adapter."""
        with self._cursor() as (conn, cur):
            cur.execute(_BASE_SCHEMA)
            # Re-register vector now that the extension definitely exists,
            # in case _configure_conn ran before CREATE EXTENSION.
            try:
                self._register_vector(conn)
            except Exception:
                pass
            # Cache embedding dim if the column already exists (existing DB).
            cur.execute(
                "SELECT atttypmod FROM pg_attribute "
                " JOIN pg_class ON attrelid = pg_class.oid "
                " WHERE relname = 'memories' AND attname = 'embedding'"
            )
            row = cur.fetchone()
            if row is not None and row[0] is not None and row[0] > 0:
                self._embedding_dim = int(row[0])

    def _ensure_embedding_column(self, dim: int) -> None:
        """Add the vector(dim) column + HNSW index on first write.

        Idempotent. If the column exists with a different dim, raise — same
        invariant as the SQLite-side dim-lock (one model per DB). Mirrors
        an explicit dim-lock migration.
        """
        with self._lock:
            if self._embedding_dim == dim:
                return
            with self._cursor() as (conn, cur):
                cur.execute(
                    "SELECT atttypmod FROM pg_attribute "
                    " JOIN pg_class ON attrelid = pg_class.oid "
                    " WHERE relname = 'memories' AND attname = 'embedding'"
                )
                row = cur.fetchone()
                if row is not None and row[0] is not None and row[0] > 0:
                    existing = int(row[0])
                    if existing == dim:
                        self._embedding_dim = dim
                        return
                    raise RuntimeError(
                        f"Postgres embeddings column has dim={existing}, "
                        f"but current backend produces dim={dim}. Drop or "
                        f"migrate the DB before switching models."
                    )
                # Add the column + HNSW cosine index.
                cur.execute(f"ALTER TABLE memories ADD COLUMN embedding vector({dim})")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
                    "ON memories USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m=16, ef_construction=64)"
                )
                self._embedding_dim = dim

    # -- writes -------------------------------------------------------------

    def store(self, label: str, content: str, embedding: list[float],
              id_: Optional[int] = None) -> int:
        """Insert or upsert a memory.

        When `id_` is provided, the row is written with that exact id and
        the BIGSERIAL sequence is bumped past it on collision-free insert.
        Keeps Postgres mirror
        IDs aligned with SQLite's source-of-truth IDs so recall_multihop
        and graph traversal don't silently miss memories.
        """
        dim = len(embedding)
        self._ensure_embedding_column(dim)
        with self._cursor() as (conn, cur):
            if id_ is None:
                cur.execute(
                    "INSERT INTO memories (label, content, embedding, vector_dim) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (label, content, embedding, dim),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
            # Explicit-id path: ON CONFLICT DO UPDATE — idempotent if the
            # mirror sees the same id twice (e.g. from sync_bridge).
            cur.execute(
                "INSERT INTO memories (id, label, content, embedding, vector_dim) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "  label = EXCLUDED.label, content = EXCLUDED.content, "
                "  embedding = EXCLUDED.embedding, vector_dim = EXCLUDED.vector_dim",
                (int(id_), label, content, embedding, dim),
            )
            # Bump the sequence past the inserted id so future autoinc
            # inserts don't collide. setval(..., is_called=true) means the
            # next nextval() returns id+1.
            cur.execute(
                "SELECT setval(pg_get_serial_sequence('memories','id'), "
                "GREATEST(%s, (SELECT COALESCE(MAX(id),0) FROM memories)))",
                (int(id_),),
            )
            return int(id_)

    def touch(self, id_: int) -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE memories SET last_accessed = NOW(), "
                "access_count = access_count + 1 WHERE id = %s",
                (id_,),
            )

    def add_connection(self, source: int, target: int, weight: float,
                       edge_type: str = "similar") -> None:
        """Upsert an edge with canonical ordering (source<target).

        ON CONFLICT keeps the larger weight (MERGE-with-max semantics).
        and the SQLite-side iter 23 invariant. Mixed-orientation rows would
        break cross-backend graph traversal.
        """
        if source == target:
            return
        if source > target:
            source, target = target, source
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO connections (source_id, target_id, weight, edge_type) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (source_id, target_id) DO UPDATE SET "
                "  weight = GREATEST(connections.weight, EXCLUDED.weight), "
                "  edge_type = EXCLUDED.edge_type",
                (source, target, weight, edge_type),
            )

    # -- reads --------------------------------------------------------------

    @staticmethod
    def _epoch(dt) -> float | None:
        """Convert TIMESTAMPTZ to epoch seconds (cross-backend parity)."""
        if dt is None:
            return None
        try:
            return dt.timestamp()
        except Exception:
            return None

    def _row_to_dict(self, row: tuple, with_embedding: bool = True) -> dict:
        """Project a memories row into the standard cross-backend dict.

        Matches SQLiteStore.get exactly so any
        consumer reading via either backend sees the same shape.
        """
        if with_embedding:
            id_, label, content, emb, dim, salience, created_at, last_accessed, access = row
            embedding = list(emb) if emb is not None else []
        else:
            id_, label, content, salience, created_at, last_accessed, access = row
            embedding = []
        return {
            "id": int(id_),
            "label": label,
            "content": content,
            "embedding": embedding,
            "salience": salience,
            "created_at": self._epoch(created_at),
            "last_accessed": self._epoch(last_accessed),
            "access_count": access,
        }

    def get_all(self) -> list[dict]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, label, content, embedding, vector_dim, salience, "
                "created_at, last_accessed, access_count "
                "FROM memories ORDER BY id"
            )
            return [self._row_to_dict(r, with_embedding=True) for r in cur.fetchall()]

    def get(self, id_: int, include_embedding: bool = True) -> Optional[dict]:
        """Fetch one memory.

        Embedding-skip path matches SQLiteStore.get(include_embedding=False)
        and SQLiteStore.get — pulling the vector blob for label-only
        neighbour lookups during recall_multihop is wasteful (4KB+ per
        memory at 1024-d).
        """
        with self._cursor() as (_conn, cur):
            if include_embedding:
                cur.execute(
                    "SELECT id, label, content, embedding, vector_dim, salience, "
                    "created_at, last_accessed, access_count "
                    "FROM memories WHERE id = %s",
                    (id_,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_dict(row, with_embedding=True)
            cur.execute(
                "SELECT id, label, content, salience, "
                "created_at, last_accessed, access_count "
                "FROM memories WHERE id = %s",
                (id_,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_dict(row, with_embedding=False)

    def exists(self, id_: int) -> bool:
        with self._cursor() as (_conn, cur):
            cur.execute("SELECT 1 FROM memories WHERE id = %s", (id_,))
            return cur.fetchone() is not None

    def exists_many(self, ids: "list[int] | set[int] | Iterable[int]") -> set[int]:
        """Batch existence check — single round-trip via ANY()."""
        if not ids:
            return set()
        ids_list = [int(i) for i in ids]
        present: set[int] = set()
        with self._cursor() as (_conn, cur):
            # Postgres has no rigid parameter cap like SQL Server (default
            # 2100), but we still chunk to keep individual queries fast.
            for i in range(0, len(ids_list), 5000):
                chunk = ids_list[i:i + 5000]
                cur.execute(
                    "SELECT id FROM memories WHERE id = ANY(%s)",
                    (chunk,),
                )
                for row in cur.fetchall():
                    present.add(int(row[0]))
        return present

    def get_connections(self, node_id: int) -> list[dict]:
        """Return edges incident to node_id with both 'type' and 'edge_type' keys.

        Matches SQLiteStore.get_connections's dict shape exactly (the
        get_connections also exposes both legacy `type` and modern
        `edge_type` so cross-backend dashboards/recall don't lose the
        edge classification).
        """
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT source_id, target_id, weight, edge_type "
                "FROM connections WHERE source_id = %s OR target_id = %s "
                "ORDER BY weight DESC",
                (node_id, node_id),
            )
            out = []
            for r in cur.fetchall():
                etype = r[3] or "similar"
                out.append({
                    "source": int(r[0]),
                    "target": int(r[1]),
                    "weight": r[2],
                    "type": etype,
                    "edge_type": etype,
                })
            return out

    def stats(self) -> dict:
        with self._cursor() as (_conn, cur):
            cur.execute("SELECT COUNT(*) FROM memories")
            mc = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM connections")
            cc = cur.fetchone()[0]
            return {"memories": int(mc), "connections": int(cc)}

    def close(self) -> None:
        try:
            self.pool.close()
        except Exception:
            pass


# Smoke test
if __name__ == "__main__":
    try:
        store = PostgresStore()
        mid = store.store("test", "Hello Postgres", [0.1] * 1024)
        print(f"Stored: {mid}")
        m = store.get(mid)
        print(f"Retrieved: {m['label']}")
        s = store.stats()
        print(f"Stats: {s}")
        store.close()
        print("Postgres: OK")
    except Exception as e:
        print(f"Postgres error: {e}")
