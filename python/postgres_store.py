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
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
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


_PW_WARN_EMITTED = False


def _build_dsn() -> str:
    """Resolve a DSN from env, .env, or discrete fields.

    Returns a URI-form DSN (`postgresql://user:pw@host:port/db`) with
    user and password percent-encoded. The previous keyword-DSN form
    interpolated the raw password into an f-string, which broke for
    passwords containing spaces, single quotes, '#', or '\\' — libpq
    keyword values need shell-style quoting that we were not doing.
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
    if not password:
        # Empty password is fine when pg_hba.conf maps the local user to
        # `trust`/`peer` auth (the default for a single-operator host). Warn
        # only once per process so multi-connection pipelines don't spam the
        # log, and demote to INFO so it doesn't look like an error.
        if not _PW_WARN_EMITTED:
            _PW_WARN_EMITTED = True
            logger.info(
                "MM_POSTGRES_PASSWORD empty — relying on local trust/peer "
                "auth. If PG rejects the connection, add the password to "
                "~/.hermes/.env or MM_POSTGRES_DSN."
            )
    import urllib.parse
    user_q = urllib.parse.quote(user, safe="")
    pw_q = urllib.parse.quote(password, safe="")
    return f"postgresql://{user_q}:{pw_q}@{host}:{port}/{db}"


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
    access_count  INTEGER DEFAULT 0,
    -- ColBERT-style late-interaction token cache. Optional; populated
    -- by remember() only when MM_COLBERT_ENABLED=1. ~64 KB/row at 32
    -- tokens × 1024 dims fp16; mirrored from the SQLite primary's
    -- `colbert_tokens` BLOB column. See python/colbert_helper.py for
    -- the byte layout (CB1 magic header) and the design rationale.
    colbert_tokens BYTEA
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

-- Mirrors SQLiteStore.add_revision: every supersession or conflict-fusion
-- write captures the prior content here so audits can reconstruct the
-- pre-fusion state.
CREATE TABLE IF NOT EXISTS memory_revisions (
    id          BIGSERIAL PRIMARY KEY,
    memory_id   BIGINT REFERENCES memories(id) ON DELETE CASCADE,
    old_content TEXT,
    new_content TEXT,
    reason      TEXT DEFAULT 'conflict_fusion',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_revisions_memory_id
    ON memory_revisions(memory_id);

-- Key-value config: embed-fingerprint, dim-lock, schema version, etc.
-- One-row-per-key, last-writer-wins semantics matching SQLiteStore.set_meta.
CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- License/tier audit log. Append-only; license-client adds a row on every
-- detected tier change. Used by re-upgrade flows to recognise a returning
-- account on a previously-bound volume rather than greeting them as a
-- fresh install.
CREATE TABLE IF NOT EXISTS tier_history (
    id           BIGSERIAL PRIMARY KEY,
    tier         TEXT NOT NULL,
    started_at   TIMESTAMPTZ DEFAULT NOW(),
    ended_at     TIMESTAMPTZ,
    last_jti     TEXT,
    fp_hash      TEXT
);

CREATE INDEX IF NOT EXISTS idx_tier_history_started_at
    ON tier_history(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_conn_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_conn_target ON connections(target_id);
CREATE INDEX IF NOT EXISTS idx_conn_weight ON connections(weight);
CREATE INDEX IF NOT EXISTS idx_conn_edge_type_weight ON connections(edge_type, weight);
CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON memories
    USING gin (to_tsvector('english', coalesce(content, '')));
CREATE INDEX IF NOT EXISTS idx_memories_label ON memories(label);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
"""


class PostgresStore:
    """Postgres + pgvector store; graph/cold-storage mirror of SQLite.

    Connection pooling via psycopg_pool.ConnectionPool keeps short-lived
    cursors fast. The pool is lazily opened on construction and closed
    by close().
    """

    _SCHEMA_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")

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
        schema = _env("MM_POSTGRES_SCHEMA", "public") or "public"
        if not self._SCHEMA_RE.match(schema):
            raise ValueError(
                f"MM_POSTGRES_SCHEMA={schema!r} rejected: must match "
                f"[A-Za-z0-9_]{{1,32}}"
            )
        self._schema = schema
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
        # search_path is set BEFORE register_vector so any DDL/queries the
        # adapter issues during type lookup land in the operator-scoped
        # schema. Identifier is regex-validated in __init__ — safe to inline.
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path = {self._schema}, public")
        except Exception:
            pass
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
            # CREATE SCHEMA before search_path can land us anywhere useful.
            # _configure_conn already set search_path on this borrowed conn,
            # so re-issue SET after the CREATE in case the schema is brand
            # new (search_path silently no-ops on missing schemas in PG14+).
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
            cur.execute(f"SET search_path = {self._schema}, public")
            cur.execute(_BASE_SCHEMA)
            # Idempotent migration for pre-existing DBs that predate the
            # colbert_tokens column. Postgres's IF NOT EXISTS clause makes
            # this no-op when already applied.
            cur.execute(
                "ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_tokens BYTEA"
            )
            # DAE table mirrors the SQLite layout: vector is the
            # float32-packed BYTEA blob (same wire format as SQLite's
            # BLOB column so the dream engine's compute path stays
            # backend-agnostic). engine_config.py used to disable DAE
            # on PG because this table was missing — now both backends
            # carry it and Mazemaker._dae_score_candidates dispatches
            # via store.fetch_dae_vectors().
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_dae_embeddings (
                    memory_id        BIGINT PRIMARY KEY,
                    vector           BYTEA NOT NULL,
                    self_weight      DOUBLE PRECISION NOT NULL,
                    neighbour_k      INTEGER NOT NULL,
                    schema_version   INTEGER NOT NULL,
                    computed_at      DOUBLE PRECISION NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dae_computed_at "
                "ON memory_dae_embeddings(computed_at)"
            )
            # Re-register vector now that the extension definitely exists,
            # in case _configure_conn ran before CREATE EXTENSION.
            try:
                self._register_vector(conn)
            except Exception:
                pass
            # Cache embedding dim if the column already exists (existing DB).
            # Schema-scoped: pg_attribute has no namespace filter by default,
            # so unqualified `relname = 'memories'` would surface ANY schema's
            # memories.embedding and break MM_POSTGRES_SCHEMA isolation.
            cur.execute(
                "SELECT atttypmod FROM pg_attribute a "
                " JOIN pg_class c ON a.attrelid = c.oid "
                " JOIN pg_namespace n ON c.relnamespace = n.oid "
                " WHERE c.relname = 'memories' AND a.attname = 'embedding' "
                "   AND n.nspname = %s",
                (self._schema,),
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
                    "SELECT atttypmod FROM pg_attribute a "
                    " JOIN pg_class c ON a.attrelid = c.oid "
                    " JOIN pg_namespace n ON c.relnamespace = n.oid "
                    " WHERE c.relname = 'memories' AND a.attname = 'embedding' "
                    "   AND n.nspname = %s",
                    (self._schema,),
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
                # Add the column. HNSW index creation is deferred when
                # MM_DEFER_HNSW=1 so bulk-ingest paths can drop the
                # O(M log N) per-insert overhead and rebuild the index
                # once at the end of the batch. Without that env the
                # index is created eagerly (correctness for ad-hoc writers).
                cur.execute(f"ALTER TABLE memories ADD COLUMN embedding vector({dim})")
                if os.environ.get("MM_DEFER_HNSW", "").strip() not in ("1", "true", "True"):
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

    # -- bulk-ingest helpers (godbench / large-corpus loaders) --------------
    #
    # Three ingest fast-paths cooperate:
    #   1. `drop_bulk_indexes()` removes the HNSW vector index, the GIN FTS
    #      index, and the cheap btrees that PG would otherwise maintain on
    #      every INSERT.  CREATE INDEX on a populated table is parallel and
    #      orders of magnitude cheaper than incremental maintenance.
    #   2. `remember_batch_copy()` (engaged automatically when MM_PG_COPY=1
    #      or rows >= MM_PG_COPY_THRESHOLD, default 1000) bypasses INSERT …
    #      VALUES and streams via `COPY memories FROM STDIN` after claiming
    #      a contiguous id range from the BIGSERIAL sequence.
    #   3. `create_bulk_indexes(dim)` rebuilds the indexes after ingest.
    #      pg_hint here uses parallel-workers and increased
    #      maintenance_work_mem for the HNSW build.

    # HNSW intentionally excluded — pgvector's parallel CREATE INDEX on a
    # populated 333k×1024-d table takes ~25 min with 3 workers fighting
    # over the shared graph layers, vs ~5 min of incremental maintenance
    # during INSERT-from-SELECT.  Leave the HNSW index live during ingest
    # and let the bulk INSERT update it row-by-row.
    _BULK_INDEXES = (
        "idx_memories_content_fts",
        "idx_memories_label",
        "idx_memories_created_at",
    )

    def drop_bulk_indexes(self) -> list[str]:
        """Drop all ingest-cost indexes on `memories`.

        Returns the list of index names actually dropped.  Idempotent — a
        missing index is silently skipped.  Caller must invoke
        `create_bulk_indexes(dim)` after bulk ingest completes.
        """
        dropped: list[str] = []
        with self._cursor() as (_conn, cur):
            for idx in self._BULK_INDEXES:
                cur.execute(f"DROP INDEX IF EXISTS {idx}")
                dropped.append(idx)
        return dropped

    def create_bulk_indexes(self, dim: Optional[int] = None) -> None:
        """Recreate the ingest-cost indexes on `memories`.

        Run after a `drop_bulk_indexes()` + bulk-ingest cycle.  The HNSW
        index is only created when `dim` is provided and the column exists.
        """
        with self._cursor() as (_conn, cur):
            # Bump maintenance memory + parallelism for this session only
            # so the HNSW + GIN builds use the full machine.
            try:
                cur.execute("SET LOCAL maintenance_work_mem = '2GB'")
                cur.execute("SET LOCAL max_parallel_maintenance_workers = 8")
            except Exception:
                pass
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_label "
                "ON memories(label)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at "
                "ON memories(created_at DESC)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_content_fts "
                "ON memories USING gin (to_tsvector('english', "
                "coalesce(content, '')))"
            )
            if dim is not None and (self._embedding_dim or dim) > 0:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
                    "ON memories USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m=16, ef_construction=64)"
                )

    def remember_batch_copy(self, rows: list[dict]) -> list[int]:
        """COPY-based bulk ingest path.

        Pre-claims a contiguous id range from `memories_id_seq`, then
        streams the rows through `COPY memories (...) FROM STDIN` using
        psycopg3's binary copy with the pgvector adapter.  Roughly 4–8×
        faster than multi-row INSERT for >1k rows on a populated table,
        and even more when paired with `drop_bulk_indexes()`.

        Returns the assigned ids in input order.
        """
        if not rows:
            return []
        dim = len(rows[0].get("embedding") or [])
        if dim == 0:
            raise ValueError("remember_batch_copy: first row has empty embedding")
        for r in rows:
            emb = r.get("embedding") or []
            if len(emb) != dim:
                raise ValueError(
                    f"remember_batch_copy: mixed embedding dims "
                    f"({len(emb)} != {dim})"
                )
        self._ensure_embedding_column(dim)

        n = len(rows)
        with self._cursor() as (_conn, cur):
            # Claim n contiguous ids in one round-trip.  Returns them in
            # ascending order so we can zip back with `rows`.
            cur.execute(
                "SELECT nextval(pg_get_serial_sequence('memories','id')) "
                "FROM generate_series(1, %s)",
                (n,),
            )
            ids = [int(r[0]) for r in cur.fetchall()]

            # pgvector accepts the text format `[v1,v2,…]` in COPY but
            # NOT psycopg's default array form `{v1,v2,…}`.  psycopg3 also
            # doesn't register a binary send adapter for the `vector` type
            # (binary COPY raises ProtocolViolation), so we serialise the
            # embedding by hand and stream as text COPY.
            with cur.copy(
                "COPY memories (id, label, content, embedding, vector_dim, salience) "
                "FROM STDIN"
            ) as cp:
                for mid, r in zip(ids, rows):
                    sal = r.get("salience")
                    emb = r.get("embedding") or []
                    emb_text = "[" + ",".join(repr(float(v)) for v in emb) + "]"
                    cp.write_row((
                        mid,
                        r.get("label"),
                        r.get("content", ""),
                        emb_text,
                        dim,
                        float(sal) if sal is not None else 1.0,
                    ))

            # Bump the sequence past the claimed range.  setval with
            # is_called=true means the next nextval() returns max(ids)+1.
            cur.execute(
                "SELECT setval(pg_get_serial_sequence('memories','id'), %s)",
                (ids[-1],),
            )
        return ids

    def remember_batch(self, rows: list[dict]) -> list[int]:
        if not rows:
            return []
        # Bulk-ingest fast path.  Engaged automatically when the caller
        # opts in via MM_PG_COPY=1 (always-on) or the row count crosses
        # MM_PG_COPY_THRESHOLD (default 1000 — below that, the multi-row
        # INSERT below already round-trips once and isn't the bottleneck).
        copy_flag = (os.environ.get("MM_PG_COPY", "").strip() in ("1", "true", "True"))
        try:
            threshold = int(os.environ.get("MM_PG_COPY_THRESHOLD", "1000"))
        except ValueError:
            threshold = 1000
        if copy_flag or len(rows) >= threshold:
            return self.remember_batch_copy(rows)

        dim = len(rows[0].get("embedding") or [])
        if dim == 0:
            raise ValueError("remember_batch: first row has empty embedding")
        for r in rows:
            emb = r.get("embedding") or []
            if len(emb) != dim:
                raise ValueError(
                    f"remember_batch: mixed embedding dims "
                    f"({len(emb)} != {dim})"
                )
        self._ensure_embedding_column(dim)

        params: list[tuple] = []
        for r in rows:
            salience = r.get("salience")
            params.append((
                r.get("label"),
                r.get("content", ""),
                r.get("embedding") or [],
                dim,
                float(salience) if salience is not None else 1.0,
            ))

        # Single multi-row INSERT … VALUES (..), (..), … RETURNING id.
        # Avoids psycopg3's executemany(returning=True) which opens an
        # implicit pipeline that wasn't reliably closing across multiple
        # PostgresStore instances in one process (pipeline=ON leaked,
        # subsequent INSERTs hit pipeline-aborted state).
        #
        # PG's bind-parameter ceiling is 65535. With 5 params/row the
        # call breaks above ~13k rows, so chunk before issuing the
        # INSERT. Chunk of 10k = 50k params, well under the ceiling
        # and one round-trip per chunk — still O(rows / 10k) calls,
        # not O(rows) like executemany.
        PARAM_CEIL = 65000
        params_per_row = 5
        chunk_size = max(1, PARAM_CEIL // params_per_row)
        ids: list[int] = []
        with self._cursor() as (conn, cur):
            for start in range(0, len(params), chunk_size):
                slab = params[start:start + chunk_size]
                values_clause = ", ".join(["(%s, %s, %s, %s, %s)"] * len(slab))
                flat: list = []
                for p in slab:
                    flat.extend(p)
                cur.execute(
                    f"INSERT INTO memories (label, content, embedding, vector_dim, salience) "
                    f"VALUES {values_clause} RETURNING id",
                    flat,
                )
                ids.extend(int(r[0]) for r in cur.fetchall())
        if len(ids) != len(rows):
            raise RuntimeError(
                f"remember_batch: expected {len(rows)} ids, got {len(ids)}"
            )
        return ids

    def touch(self, id_: int) -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE memories SET last_accessed = NOW(), "
                "access_count = access_count + 1 WHERE id = %s",
                (id_,),
            )

    @staticmethod
    def _sanitize_weight(w) -> float | None:
        """Normalise a weight to a finite float in [0, 1].

        Rejects NaN/Inf — those poison ``ORDER BY weight DESC`` because
        Postgres treats NaN as greater than every real number, and
        ``GREATEST(real, NaN) = NaN`` propagates the poison through every
        subsequent UPSERT. SQLite parity: SQLite's ``REAL`` sort already
        treats NaN as null-equivalent, so the SQLite path never surfaced
        this; we mirror that semantic by refusing to store the value at
        all. Returns None on rejection so callers can skip the row.
        """
        try:
            wf = float(w)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(wf):
            return None
        if wf < 0.0:
            return 0.0
        if wf > 1.0:
            return 1.0
        return wf

    def add_connection(self, source: int, target: int, weight: float,
                       edge_type: str = "similar") -> None:
        """Upsert an edge with canonical ordering (source<target).

        ON CONFLICT keeps the larger weight (MERGE-with-max semantics).
        and the SQLite-side iter 23 invariant. Mixed-orientation rows would
        break cross-backend graph traversal.

        NaN/Inf weights are silently dropped — see ``_sanitize_weight``.
        """
        if source == target:
            return
        if source > target:
            source, target = target, source
        wf = self._sanitize_weight(weight)
        if wf is None:
            return
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO connections (source_id, target_id, weight, edge_type) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (source_id, target_id) DO UPDATE SET "
                "  weight = GREATEST(connections.weight, EXCLUDED.weight), "
                "  edge_type = EXCLUDED.edge_type",
                (source, target, wf, edge_type),
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

    def search_semantic(self, query_vec: "list[float]",
                        limit: int = 50) -> list[dict]:
        """pgvector HNSW cosine search. Sub-50 ms on a 200 k-row corpus
        because the `idx_memories_embedding_hnsw` index does the kNN.

        Mazemaker._semantic_candidates falls through to ``get_all()`` +
        Python cosine when none of {GPU recall, HNSW (Python), C++} are
        armed — which on the PG backend was burning ~75 s/query scanning
        186 k rows in Python. The store knows how to ask Postgres for the
        same answer in milliseconds; this method exposes it.

        Returns list of dicts with id, score (cosine similarity), similarity,
        channel — same shape Mazemaker._semantic_candidates expects so it
        can drop the result straight into the RRF fusion.
        """
        import psycopg
        try:
            from pgvector.psycopg import register_vector  # type: ignore
        except Exception:
            register_vector = None

        # pgvector wants its own type adapter — register it on the borrowed
        # connection. The pool may return a conn that hasn't been
        # registered yet for vector input.
        emb_str = "[" + ",".join(f"{v:.7g}" for v in query_vec) + "]"
        with self._cursor() as (_conn, cur):
            try:
                cur.execute(
                    "SELECT id, 1 - (embedding <=> %s::vector) AS sim "
                    "FROM memories "
                    "WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> %s::vector "
                    "LIMIT %s",
                    (emb_str, emb_str, int(limit)),
                )
                rows = cur.fetchall()
            except psycopg.errors.UndefinedColumn:
                return []
        return [
            {
                "id": int(r[0]),
                "score": float(r[1]) if r[1] is not None else 0.0,
                "similarity": float(r[1]) if r[1] is not None else 0.0,
                "channel": "semantic",
            }
            for r in rows
        ]

    def get_all(self) -> list[dict]:
        # Fresh schema has no `embedding` column yet (added lazily on first
        # write once the dim is known). Treat that as an empty table — the
        # column will appear after the first remember_batch().
        import psycopg
        with self._cursor() as (_conn, cur):
            try:
                cur.execute(
                    "SELECT id, label, content, embedding, vector_dim, salience, "
                    "created_at, last_accessed, access_count "
                    "FROM memories ORDER BY id"
                )
            except psycopg.errors.UndefinedColumn:
                return []
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

    def get_connections(self, node_id: int,
                        at_time: Optional[float] = None) -> list[dict]:
        """Return edges incident to node_id with both 'type' and 'edge_type'.

        `at_time` is accepted for SQLiteStore parity but ignored — the
        Postgres connections table doesn't carry valid_from/valid_to
        bitemporal columns yet. Recall code that passes at_time gets a
        single union of all edges instead of a time-window slice.
        """
        with self._cursor() as (_conn, cur):
            # ``weight = weight`` strips NaN rows (see top_weighted_edges)
            # so spreading-activation never seeds an edge with a poison
            # weight. Falls through to the same DESC sort otherwise.
            cur.execute(
                "SELECT source_id, target_id, weight, edge_type "
                "FROM connections WHERE (source_id = %s OR target_id = %s) "
                "AND weight = weight "
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

    # -- Phase B parity methods --------------------------------------------
    # Below are the methods callers expect from SQLiteStore but that
    # PostgresStore lacked while it was a partial mirror. Now that
    # PostgresStore is the primary on Pro/Enterprise tiers, every read/
    # write path memory_client and dream_engine call must be available
    # here too.

    def get_many(self, ids: list[int], include_embedding: bool = False) -> dict[int, dict]:
        """Batch fetch memories by id. Returns {id: dict}, missing ids omitted."""
        ids_list = [int(i) for i in ids if i is not None]
        if not ids_list:
            return {}
        out: dict[int, dict] = {}
        if include_embedding:
            sql = (
                "SELECT id, label, content, embedding, vector_dim, salience, "
                "created_at, last_accessed, access_count "
                "FROM memories WHERE id = ANY(%s)"
            )
        else:
            sql = (
                "SELECT id, label, content, salience, "
                "created_at, last_accessed, access_count "
                "FROM memories WHERE id = ANY(%s)"
            )
        with self._cursor() as (_conn, cur):
            for i in range(0, len(ids_list), 5000):
                cur.execute(sql, (ids_list[i:i + 5000],))
                for row in cur.fetchall():
                    item = self._row_to_dict(row, with_embedding=include_embedding)
                    out[item["id"]] = item
        return out

    def find_by_label(self, label: str) -> list[dict]:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, label, content, embedding, vector_dim, salience, "
                "created_at, last_accessed, access_count "
                "FROM memories WHERE label = %s ORDER BY id",
                (label,),
            )
            return [self._row_to_dict(r, with_embedding=True) for r in cur.fetchall()]

    def add_revision(self, memory_id: int, old_content: str, new_content: str,
                     reason: str = "conflict_fusion") -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO memory_revisions (memory_id, old_content, new_content, reason) "
                "VALUES (%s, %s, %s, %s)",
                (int(memory_id), old_content, new_content, reason),
            )

    def update_memory(self, memory_id: int, content: str, embedding: list[float],
                      label: Optional[str] = None) -> None:
        """Rewrite content/embedding without bumping access_count or last_accessed.

        Same invariant as SQLiteStore.update_memory: writes don't affect the
        access timeline; that belongs to touch().
        """
        dim = len(embedding)
        self._ensure_embedding_column(dim)
        with self._cursor() as (_conn, cur):
            if label is None:
                cur.execute(
                    "UPDATE memories SET content = %s, embedding = %s, "
                    "vector_dim = %s WHERE id = %s",
                    (content, embedding, dim, int(memory_id)),
                )
            else:
                cur.execute(
                    "UPDATE memories SET label = %s, content = %s, "
                    "embedding = %s, vector_dim = %s WHERE id = %s",
                    (label, content, embedding, dim, int(memory_id)),
                )

    # ---- ColBERT token cache (mirror of SQLiteStore methods) ---------------
    def set_colbert_tokens(self, memory_id: int, blob: Optional[bytes]) -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "UPDATE memories SET colbert_tokens = %s WHERE id = %s",
                (blob, int(memory_id)),
            )

    def set_colbert_tokens_many(self, rows) -> int:
        """Bulk-write ColBERT blobs via COPY into a temp table + single
        UPDATE FROM. ~50–100× faster than looping set_colbert_tokens for
        large batches: one COPY + one UPDATE per call instead of N
        round-trips. rows is an iterable of (memory_id, blob) pairs."""
        rows = [(int(mid), bytes(blob)) for (mid, blob) in rows if blob is not None]
        if not rows:
            return 0
        with self._cursor() as (_conn, cur):
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _cb_stage "
                "(id BIGINT PRIMARY KEY, blob BYTEA) ON COMMIT PRESERVE ROWS"
            )
            cur.execute("TRUNCATE _cb_stage")
            with cur.copy("COPY _cb_stage (id, blob) FROM STDIN") as cp:
                for mid, blob in rows:
                    cp.write_row((mid, blob))
            cur.execute(
                "UPDATE memories SET colbert_tokens = s.blob "
                "FROM _cb_stage s WHERE memories.id = s.id"
            )
            return int(cur.rowcount or 0)

    def get_colbert_tokens_many(self, ids: "list[int]") -> "dict[int, bytes]":
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return {}
        out: dict[int, bytes] = {}
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, colbert_tokens FROM memories "
                "WHERE id = ANY(%s) AND colbert_tokens IS NOT NULL",
                (ids,),
            )
            for r in cur.fetchall():
                blob = r[1]
                if blob:
                    out[int(r[0])] = bytes(blob)
        return out

    def ensure_dae_schema(self) -> bool:
        """The PG `memory_dae_embeddings` table is created by
        `_ensure_schema()` on first connect (see _BASE_SCHEMA + the
        inline DAE block). Nothing to do here beyond confirming the
        license gate — the table either exists or the operator isn't
        on a Pro tier in which case the read path returns {}.
        """
        try:
            from dae import is_enabled
            return bool(is_enabled())
        except Exception:
            return False

    def upsert_dae_vectors(
        self,
        rows: "list[tuple[int, bytes, float, int, int, float]]",
    ) -> int:
        """PG counterpart of SQLiteStore.upsert_dae_vectors.

        Uses ON CONFLICT (memory_id) DO UPDATE so the operation has the
        same upsert semantics as SQLite's INSERT OR REPLACE.
        """
        if not rows:
            return 0
        # Convert blobs to memoryview / bytes so psycopg uploads them
        # as BYTEA without a redundant copy.
        payload = [
            (
                int(mid),
                bytes(blob) if not isinstance(blob, (bytes, memoryview)) else blob,
                float(self_w),
                int(nk),
                int(sv),
                float(ts),
            )
            for (mid, blob, self_w, nk, sv, ts) in rows
        ]
        try:
            with self._cursor() as (_conn, cur):
                cur.executemany(
                    "INSERT INTO memory_dae_embeddings "
                    "(memory_id, vector, self_weight, neighbour_k, "
                    " schema_version, computed_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (memory_id) DO UPDATE SET "
                    "  vector         = EXCLUDED.vector, "
                    "  self_weight    = EXCLUDED.self_weight, "
                    "  neighbour_k    = EXCLUDED.neighbour_k, "
                    "  schema_version = EXCLUDED.schema_version, "
                    "  computed_at    = EXCLUDED.computed_at",
                    payload,
                )
        except Exception as exc:
            logger.warning("PG upsert_dae_vectors failed: %s", exc)
            return 0
        return len(payload)

    def prune_memories_by_label_prefix(self, prefix: str, older_than_ts: float) -> int:
        """PG counterpart to SQLiteStore.prune_memories_by_label_prefix.

        `connections.source_id` / `target_id` reference `memories.id` without
        ON DELETE CASCADE, so a naive DELETE FROM memories trips the FK on
        any row that still has edges (derived:cluster outputs always do —
        they're emitted with derived_from connections in the same cycle).
        Clear the connections first, then the memories. Single transaction
        so a crash mid-prune doesn't leak orphan edges.
        """
        from datetime import datetime, timezone
        cutoff = datetime.fromtimestamp(float(older_than_ts), tz=timezone.utc)
        try:
            with self._cursor() as (conn, cur):
                cur.execute(
                    "DELETE FROM connections "
                    "WHERE source_id IN (SELECT id FROM memories "
                    "                    WHERE label LIKE %s AND created_at < %s) "
                    "   OR target_id IN (SELECT id FROM memories "
                    "                    WHERE label LIKE %s AND created_at < %s)",
                    (prefix + "%", cutoff, prefix + "%", cutoff),
                )
                cur.execute(
                    "DELETE FROM memories WHERE label LIKE %s AND created_at < %s",
                    (prefix + "%", cutoff),
                )
                return int(cur.rowcount or 0)
        except Exception as exc:
            logger.warning("PG prune_memories_by_label_prefix failed: %s", exc)
            return 0

    def fetch_dae_vectors(self, ids: "list[int]") -> "dict[int, list[float]]":
        """Backend-agnostic DAE vector fetch — Postgres flavour.

        Mirrors dae.fetch_dae_vectors but uses PG syntax (`ANY(%s)`,
        BYTEA → bytes) so Mazemaker._dae_score_candidates can dispatch
        the same call against either backend. The blob wire format is
        identical to SQLite's: float32-packed little-endian, decoded
        with struct.unpack.
        """
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return {}
        import struct
        try:
            with self._cursor() as (_conn, cur):
                cur.execute(
                    "SELECT memory_id, vector FROM memory_dae_embeddings "
                    "WHERE memory_id = ANY(%s)",
                    (ids,),
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning("PG fetch_dae_vectors failed: %s", exc)
            return {}
        out: dict[int, list[float]] = {}
        for r in rows:
            mid, blob = int(r[0]), r[1]
            if not blob:
                continue
            try:
                data = bytes(blob)
                n = len(data) // 4
                out[mid] = list(struct.unpack(f"{n}f", data))
            except Exception:
                continue
        return out

    def stream_missing_colbert(self, batch_size: int = 1000, start_after_id: int = 0):
        last_id = int(start_after_id or 0)
        while True:
            with self._cursor() as (_conn, cur):
                cur.execute(
                    "SELECT id, content FROM memories "
                    "WHERE id > %s AND colbert_tokens IS NULL "
                    "ORDER BY id ASC LIMIT %s",
                    (last_id, int(batch_size)),
                )
                rows = cur.fetchall()
            if not rows:
                return
            for r in rows:
                yield int(r[0]), (r[1] or "")
            last_id = int(rows[-1][0])

    def stream_long_memories_for_afe(self, *, min_len: int = 500,
                                     limit: int = 1000,
                                     exclude_label_pattern: str = "%::afe::%",
                                     exclude_ids: "set[int] | None" = None):
        """Backend-agnostic enumeration of long memories that need AFE
        extraction. Skips rows whose label already carries the AFE
        marker, plus any explicitly-excluded ids from the processed-set.

        Yields ``(id, label, content)`` ordered by content length DESC.
        Caller handles the in-Python exclude_ids filter so we don't have
        to materialise a giant temp table for the NOT-IN list.
        """
        exclude_ids = exclude_ids or set()
        # Push exclude_ids into the SQL when non-trivial. The earlier
        # design fetched `limit*2` rows and filtered in Python, which
        # silently capped progress once `len(exclude_ids) > limit*2`: a
        # full conv looked drained even with thousands of sources left.
        # `id != ALL(%s::int[])` handles 10k+ ids in PG without a temp
        # table or expression-blowup.
        with self._cursor() as (_conn, cur):
            if exclude_ids:
                cur.execute(
                    "SELECT id, label, content FROM memories "
                    "WHERE length(content) >= %s "
                    "  AND (label IS NULL OR label NOT LIKE %s) "
                    "  AND id <> ALL(%s::int[]) "
                    "ORDER BY length(content) DESC LIMIT %s",
                    (int(min_len), exclude_label_pattern,
                     list(exclude_ids), int(limit)),
                )
            else:
                cur.execute(
                    "SELECT id, label, content FROM memories "
                    "WHERE length(content) >= %s "
                    "  AND (label IS NULL OR label NOT LIKE %s) "
                    "ORDER BY length(content) DESC LIMIT %s",
                    (int(min_len), exclude_label_pattern, int(limit)),
                )
            rows = cur.fetchall()
        for r in rows:
            yield int(r[0]), (r[1] or ""), (r[2] or "")

    def add_connections_batch(self, pairs, edge_type: str = "similar") -> int:
        """Bulk-upsert undirected weighted edges in a single transaction.

        ``pairs`` is an iterable of ``(source, target, weight)``. Self-loops
        dropped, endpoints canonicalised to (min, max) order, weights clamped
        to [0, 1]. ON CONFLICT keeps the larger weight (MERGE-with-max).

        Returns the number of rows the batch UPSERT touched. With
        ON CONFLICT DO UPDATE this counts both new inserts AND existing
        rows whose weight got bumped, which differs from SQLiteStore's
        "newly inserted only" semantic — callers using the count for
        progress reporting should treat it as an upper bound.
        """
        normalised: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for s, t, w in pairs:
            si, ti = int(s), int(t)
            if si == ti:
                continue
            if si > ti:
                si, ti = ti, si
            if (si, ti) in seen:
                continue
            wf = self._sanitize_weight(w)
            # NaN/Inf get rejected upstream — the previous
            # ``max(0, min(1, float(w)))`` pattern silently turned NaN
            # into 1.0 on the Python side, which then sorted to the top
            # of ``ORDER BY weight DESC`` because Postgres treats NaN
            # itself as greater than every real number too.
            if wf is None:
                continue
            seen.add((si, ti))
            normalised.append((si, ti, wf))

        if not normalised:
            return 0

        # COPY into a temp staging table, then a SINGLE upsert from staging.
        # Replaces an executemany loop (N round-trips) with two statements
        # for any batch size. ~50–100× faster on batches of 1k+ edges.
        with self._cursor() as (_conn, cur):
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _conn_stage "
                "(source_id BIGINT, target_id BIGINT, "
                " weight DOUBLE PRECISION, edge_type TEXT) "
                "ON COMMIT PRESERVE ROWS"
            )
            cur.execute("TRUNCATE _conn_stage")
            with cur.copy(
                "COPY _conn_stage (source_id, target_id, weight, edge_type) "
                "FROM STDIN"
            ) as cp:
                for s, t, w in normalised:
                    cp.write_row((s, t, w, edge_type))
            cur.execute(
                "INSERT INTO connections (source_id, target_id, weight, edge_type) "
                "SELECT source_id, target_id, weight, edge_type FROM _conn_stage "
                "ON CONFLICT (source_id, target_id) DO UPDATE SET "
                "  weight = GREATEST(connections.weight, EXCLUDED.weight), "
                "  edge_type = EXCLUDED.edge_type"
            )
        return len(normalised)

    def get_all_connections(self) -> list[dict]:
        """Return every edge as a list — used for bulk graph hydration.

        NaN weights are filtered out so downstream graph builders don't
        compute traversal scores against poison values. See
        ``top_weighted_edges`` for the underlying NaN problem.
        """
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT source_id, target_id, weight, edge_type FROM connections "
                "WHERE weight = weight"
            )
            return [
                {
                    "source": int(r[0]),
                    "target": int(r[1]),
                    "weight": float(r[2] or 0.0),
                    "type": r[3] or "similar",
                }
                for r in cur.fetchall()
            ]

    @staticmethod
    def _sanitize_tsquery_terms(query: str, max_tokens: int = 12) -> list[str]:
        """Extract searchable tokens from a free-text query.

        Mirrors SQLiteStore._sanitize_fts_query token rules: word
        characters + hyphen, dropped quotes, capped at max_tokens.
        Returned as a list so callers can build either AND (' & ') or
        OR (' | ') tsquery strings.
        """
        tokens = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_\-]{1,}", query or "")
        return [t.replace('"', '') for t in tokens[:max_tokens]]

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """Same heuristic as SQLiteStore.extract_entities — duplicated to
        avoid a circular import (memory_client imports postgres_store)."""
        text = text or ""
        entities: list[str] = []
        for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", text):
            if tok[0].isupper() or any(ch.isdigit() for ch in tok) or re.search(r"[a-z][A-Z]", tok):
                entities.append(tok)
        for phrase in re.findall(r"['\"]([^'\"]{3,80})['\"]", text):
            entities.append(phrase.strip())
        seen = set()
        out = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    def search_bm25(self, query: str, limit: int = 50) -> list[dict]:
        """tsvector-based lexical search. Equivalent of SQLiteStore.search_bm25.

        Postgres uses ts_rank_cd over to_tsvector('english', content) — the
        idx_memories_content_fts GIN index from _BASE_SCHEMA serves the
        match. Score field mirrors SQLiteStore: reciprocal-rank
        (1/(i+1)) so RRF fusion in memory_client._parallel_retrieve keeps
        working across backends.
        """
        terms = self._sanitize_tsquery_terms(query)
        if not terms:
            return []
        tsq = " & ".join(terms)
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id, "
                "  ts_rank_cd(to_tsvector('english', coalesce(content, '')), "
                "             to_tsquery('english', %s)) AS rank "
                "FROM memories "
                "WHERE to_tsvector('english', coalesce(content, '')) @@ to_tsquery('english', %s) "
                "ORDER BY rank DESC LIMIT %s",
                (tsq, tsq, int(limit)),
            )
            rows = cur.fetchall()
        return [
            {
                "id": int(r[0]),
                "score": 1.0 / (i + 1),
                "rank": float(r[1] or 0.0),
                "channel": "bm25",
            }
            for i, r in enumerate(rows)
        ]

    def search_entity(self, query: str, limit: int = 50) -> list[dict]:
        """Entity-token tsvector search. OR over extracted entities."""
        entities = self.extract_entities(query)
        if not entities:
            entities = [t for t in re.findall(r"\b[A-Za-z0-9_\-]{6,}\b", query or "")[:5]]
        if not entities:
            return []
        cleaned = [e.replace('"', '').replace("'", "") for e in entities[:8] if e.strip()]
        if not cleaned:
            return []
        tsq = " | ".join(cleaned)
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT id FROM memories "
                "WHERE to_tsvector('english', coalesce(content, '')) @@ to_tsquery('english', %s) "
                "LIMIT %s",
                (tsq, int(limit)),
            )
            rows = cur.fetchall()
        return [
            {
                "id": int(r[0]),
                "score": 1.0 / (i + 1),
                "matched_entities": entities,
                "channel": "entity",
            }
            for i, r in enumerate(rows)
        ]

    # Mirrors SQLiteStore._TEMPORAL_CUES. Kept in sync with the SQLite
    # side; if you add a phrase here, add it there. The PG variant of
    # search_temporal used to short-circuit with only "today/yesterday/
    # last week" and return "most recent N" when nothing matched, which
    # polluted the RRF fusion with recency bias on every query and is
    # half the reason update_tracking sat at 0.508 in the v5 matrix.
    _TEMPORAL_CUES = (
        ("today", 86400),
        ("heute", 86400),
        ("yesterday", 2 * 86400),
        ("gestern", 2 * 86400),
        ("last week", 7 * 86400),
        ("letzte woche", 7 * 86400),
        ("this week", 7 * 86400),
        ("diese woche", 7 * 86400),
        ("last month", 30 * 86400),
        ("this month", 30 * 86400),
        ("latest", 30 * 86400),
        ("newest", 30 * 86400),
        ("most recent", 30 * 86400),
        ("currently", 30 * 86400),
        ("current value", 30 * 86400),
        ("now what", 30 * 86400),
        ("recently", 14 * 86400),
        ("updated", 14 * 86400),
    )

    def search_temporal(self, query: str, limit: int = 50,
                        now: Optional[float] = None) -> list[dict]:
        """Time-window scan keyed off temporal phrases in the query.

        Returns [] when no temporal cue is present. The previous form
        returned an unfiltered most-recent-N pool regardless of intent;
        on the RRF fusion that meant every query — including pure
        semantic ones — got a recency bias dragging the wrong answers
        up the ranking. See [[bug-search-temporal-recency-bias]] in
        the SQLite side (memory_client.py); this is the PG parity fix.
        """
        import re as _re
        now = now or time.time()
        q = (query or "").lower()

        # ISO date anchor: YYYY-MM-DD pins a ±1d window.
        iso_match = _re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
        anchor_ts: Optional[float] = None
        if iso_match:
            try:
                import datetime as _dt
                y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
                anchor_ts = _dt.datetime(y, m, d).timestamp()
            except (ValueError, OverflowError):
                anchor_ts = None

        where = ""
        params: list[Any] = []
        if anchor_ts is None:
            for cue, win in self._TEMPORAL_CUES:
                if cue in q:
                    if cue in ("yesterday", "gestern"):
                        where = "WHERE created_at BETWEEN to_timestamp(%s) AND to_timestamp(%s)"
                        params = [now - 2 * 86400, now - 86400]
                    else:
                        where = "WHERE created_at >= to_timestamp(%s)"
                        params = [now - float(win)]
                    break
            else:
                return []
        else:
            where = "WHERE created_at BETWEEN to_timestamp(%s) AND to_timestamp(%s)"
            params = [anchor_ts - 86400, anchor_ts + 86400]

        with self._cursor() as (_conn, cur):
            cur.execute(
                f"SELECT id, created_at, last_accessed FROM memories "
                f"{where} ORDER BY created_at DESC LIMIT %s",
                tuple(params + [int(limit)]),
            )
            rows = cur.fetchall()
        return [
            {
                "id": int(r[0]),
                "score": 1.0 / (i + 1),
                "created_at": self._epoch(r[1]),
                "channel": "temporal",
            }
            for i, r in enumerate(rows)
        ]

    def get_meta(self, key: str) -> Optional[str]:
        with self._cursor() as (_conn, cur):
            cur.execute("SELECT value FROM meta WHERE key = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "INSERT INTO meta (key, value, updated_at) VALUES (%s, %s, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET "
                "  value = EXCLUDED.value, updated_at = NOW()",
                (key, value),
            )

    def get_stats(self) -> dict:
        """Cross-backend parity name; returns SQLiteStore.get_stats shape
        (memories, connections, revisions, fts) so consumers don't branch."""
        with self._cursor() as (_conn, cur):
            cur.execute("SELECT COUNT(*) FROM memories")
            mc = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM connections")
            cc = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM memory_revisions")
            rc = cur.fetchone()[0]
        return {
            "memories": int(mc),
            "connections": int(cc),
            "revisions": int(rc),
            "fts": True,
        }

    # -- NeuralMemory hot-path helpers (parity with SQLiteStore) ----------

    def get_max_id(self) -> int:
        with self._cursor() as (_conn, cur):
            cur.execute("SELECT MAX(id) FROM memories")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def recent_semantic_pool(self, limit: int,
                             exclude_label_prefix: Optional[str] = None) -> list[dict]:
        """Top-N newest memories with non-null embedding, optionally
        excluding labels with a prefix. Returns dicts with id/content/
        embedding (already adapted to list[float] by pgvector)."""
        if exclude_label_prefix:
            sql = (
                "SELECT id, content, embedding FROM memories "
                "WHERE embedding IS NOT NULL "
                "  AND (label IS NULL OR label NOT LIKE %s) "
                "ORDER BY created_at DESC LIMIT %s"
            )
            params: tuple = (exclude_label_prefix + "%", int(limit))
        else:
            sql = (
                "SELECT id, content, embedding FROM memories "
                "WHERE embedding IS NOT NULL "
                "ORDER BY created_at DESC LIMIT %s"
            )
            params = (int(limit),)
        with self._cursor() as (_conn, cur):
            cur.execute(sql, params)
            out = []
            for r in cur.fetchall():
                emb = r[2]
                # pgvector adapter returns numpy array or list; normalise.
                if emb is None:
                    continue
                emb_list = list(emb) if not isinstance(emb, list) else emb
                out.append({
                    "id": int(r[0]),
                    "content": r[1],
                    "embedding": emb_list,
                })
            return out

    def weighted_edges(self) -> list[tuple[int, int, float]]:
        # ``weight = weight`` excludes NaN — see top_weighted_edges note.
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT source_id, target_id, weight FROM connections "
                "WHERE weight > 0 AND weight = weight"
            )
            return [
                (int(r[0]), int(r[1]), float(r[2] or 0.0))
                for r in cur.fetchall()
            ]

    def top_weighted_edges(self, limit: int = 500) -> list[dict]:
        # Postgres sorts NaN as greater than every real number, so a
        # naked ``ORDER BY weight DESC`` surfaces NaN-weight rows first
        # if any slipped past the writer (or were written before the
        # writer was hardened). The ``weight = weight`` predicate
        # evaluates to false for NaN, which both filters them out and
        # lets the planner use idx_conn_weight for the sort.
        with self._cursor() as (_conn, cur):
            cur.execute(
                "SELECT source_id, target_id, weight, edge_type FROM connections "
                "WHERE weight = weight "
                "ORDER BY weight DESC LIMIT %s",
                (int(limit),),
            )
            return [
                {
                    "source": int(r[0]),
                    "target": int(r[1]),
                    "weight": float(r[2] or 0.0),
                    "type": r[3] or "similar",
                    "edge_type": r[3] or "similar",
                }
                for r in cur.fetchall()
            ]

    def prune_connections_below(self, threshold: float) -> int:
        with self._cursor() as (_conn, cur):
            cur.execute(
                "DELETE FROM connections WHERE weight < %s",
                (float(threshold),),
            )
            return int(cur.rowcount or 0)

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
