#!/usr/bin/env python3
"""
bake_longmemeval_s_cache.py — embed LongMemEval-S ONCE, snapshot, forever.

Reads raw turns from `mm_bench_raw.longmemeval_s.sessions`, groups them
by (question_id, session_idx), feeds the resulting session text through
the shared bge-m3 embed-server, and writes the result into
`mm10m_bench.longmemeval_s_bgem3_1024.memories`.  Two databases because
the `vector` extension is only installed in `mm10m_bench` on this host
(mazemaker user lacks superuser to CREATE EXTENSION in `mm_bench_raw`).
The cache schema in mm10m_bench is the canonical pre-embedded snapshot
— pg_dump it once, restore it forever.

The script is RESUMABLE: on every run it checks which labels already
exist in the cache and skips them.  Kill at any point, re-run, it picks
up where it left off.

USAGE
    # Default — embed everything that isn't already cached
    python benchmarks/bake_longmemeval_s_cache.py

    # Smoke-test 100 sessions then exit
    python benchmarks/bake_longmemeval_s_cache.py --limit 100

    # Wipe and rebuild from scratch
    python benchmarks/bake_longmemeval_s_cache.py --rebuild

SNAPSHOT
    # Dump (after the bake completes, ~24k rows, ~100 MB compressed)
    pg_dump -U mazemaker -d mm10m_bench \\
        --schema=longmemeval_s_bgem3_1024 \\
        -Fc -f benchmarks/snapshots/longmemeval_s_bgem3_1024.dump

    # Restore (idempotent — drops the schema first via --clean)
    pg_restore -U mazemaker -d mm10m_bench \\
        --clean --if-exists \\
        benchmarks/snapshots/longmemeval_s_bgem3_1024.dump

WORKFLOW
    restore  →  godbench (writes to godbench_<ts>)  →  drop godbench_<ts>
        ↑                                                       │
        └───────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_SRC = REPO / "python"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from postgres_store import _build_dsn  # noqa: E402
from embed_provider import EmbeddingProvider  # noqa: E402

SOURCE_DB = "mm_bench_raw"        # raw turns live here
CACHE_DB = "mm10m_bench"          # vector extension is installed here
EMBED_DIM = 1024
# SOURCE_SCHEMA / CACHE_SCHEMA are resolved from --variant at runtime.
SOURCE_SCHEMA = "longmemeval_s"
CACHE_SCHEMA = "longmemeval_s_bgem3_1024"


def _dsn_for(database: str) -> str:
    """Resolve the DSN, overriding the default database name.

    `_build_dsn` returns a URI form `postgresql://user:pw@host:port/db`.
    Swap the path component for the requested database so this script can
    target mm_bench_raw regardless of the operator's default MM_POSTGRES_DB.
    """
    import urllib.parse
    base = _build_dsn()
    parsed = urllib.parse.urlparse(base)
    new = parsed._replace(path=f"/{database}")
    return urllib.parse.urlunparse(new)


def ensure_cache_schema(conn) -> None:
    """Create the cache schema + memories table if they don't exist.

    The vector extension is assumed to already exist in this database —
    mazemaker user isn't superuser, so CREATE EXTENSION is a no-go here.

    LongMemEval-S reuses session ids across questions with subtly-different
    content per appearance (23,867 unique (sid, content) combos vs 18,770
    unique sids), so the cache stores one row per (question_id, sid) pair
    — labels are NOT unique, matching the original ingest_haystack
    semantics where every haystack-session a question references gets its
    own memory.
    """
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{CACHE_SCHEMA}"')
        cur.execute(
            f'CREATE TABLE IF NOT EXISTS "{CACHE_SCHEMA}".memories ('
            f'  id BIGSERIAL PRIMARY KEY,'
            f'  label TEXT NOT NULL,'
            f'  content TEXT NOT NULL,'
            f'  embedding vector({EMBED_DIM}) NOT NULL,'
            f'  vector_dim INTEGER NOT NULL DEFAULT {EMBED_DIM},'
            f'  salience DOUBLE PRECISION DEFAULT 1.0,'
            f'  colbert_tokens BYTEA,'
            f'  question_id TEXT,'
            f'  session_idx INTEGER,'
            f'  created_at TIMESTAMPTZ DEFAULT NOW(),'
            f'  last_accessed TIMESTAMPTZ DEFAULT NOW(),'
            f'  access_count INTEGER DEFAULT 0'
            f')'
        )
        # Idempotent: add audit columns on older cache schemas. PostgresStore's
        # get_all() SELECTs last_accessed + access_count, so missing them makes
        # GPU recall's load_from_store return 0 rows → silent CPU fallback.
        cur.execute(
            f'ALTER TABLE "{CACHE_SCHEMA}".memories '
            f'ADD COLUMN IF NOT EXISTS colbert_tokens BYTEA, '
            f'ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ DEFAULT NOW(), '
            f'ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0'
        )
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{CACHE_SCHEMA}_label '
            f'ON "{CACHE_SCHEMA}".memories(label)'
        )
        # NOTE: no unique constraint on (question_id, session_idx).
        # Chunked-session bake adds many rows per (qid, sidx) tuple, and
        # AFE adds facts whose (qid, sidx) is inherited from the source.
        # The id column (BIGSERIAL) is the only true natural key.


def existing_keys(conn) -> set[tuple[str, int]]:
    """Set of (question_id, session_idx) tuples already in the cache."""
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT question_id, session_idx FROM "{CACHE_SCHEMA}".memories'
        )
        return {(r[0], int(r[1])) for r in cur.fetchall()}


def iter_sessions(conn, limit: int | None):
    """Yield (question_id, label, content) for every (qid, sidx) pair.

    Labels are formatted as `session:<sid>` where sid comes from the
    question's `haystack_session_ids` array at position session_idx —
    matches the original `ingest_haystack` label format so the bench's
    `rank_of_gold` scorer (which colon-splits labels and checks each
    segment against `answer_session_ids`) accepts cache-ingested results.

    The full result set is ~25 MB of text for 24k sessions; materialised
    once via fetchall() rather than a named cursor because autocommit
    connections can't DECLARE CURSOR.
    """
    # haystack_session_ids is jsonb; jsonb arrays are 0-indexed, so the
    # session_idx (which is 0-based) maps directly without offset.  An
    # earlier +1 here produced an off-by-one across the entire cache.
    sql = (
        f'SELECT q.question_id, s.session_idx, '
        f'       q.haystack_session_ids ->> s.session_idx AS sid, '
        f'       string_agg(s.role || \': \' || s.content, E\'\\n\' '
        f'                  ORDER BY s.msg_idx) AS content '
        f'FROM "{SOURCE_SCHEMA}".questions q '
        f'JOIN "{SOURCE_SCHEMA}".sessions s ON s.question_id = q.question_id '
        f'GROUP BY q.question_id, s.session_idx, '
        f'         q.haystack_session_ids ->> s.session_idx '
        f'ORDER BY q.question_id, s.session_idx'
    )
    if limit:
        sql += f' LIMIT {int(limit)}'
    with conn.cursor() as cur:
        cur.execute(sql)
        for question_id, session_idx, sid, content in cur.fetchall():
            label = f"session:{sid}"
            yield question_id, int(session_idx), label, content


def chunked(it, n):
    chunk: list = []
    for item in it:
        chunk.append(item)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_chunk(conn, rows: list[tuple[str, int, str, str, list[float]]]) -> None:
    """COPY (qid, session_idx, label, content, embedding) tuples into the cache."""
    with conn.cursor() as cur:
        with cur.copy(
            f'COPY "{CACHE_SCHEMA}".memories '
            f'(question_id, session_idx, label, content, embedding, vector_dim, salience) '
            f'FROM STDIN'
        ) as cp:
            for qid, sidx, label, content, emb in rows:
                emb_text = "[" + ",".join(repr(float(v)) for v in emb) + "]"
                cp.write_row((qid, sidx, label, content, emb_text, EMBED_DIM, 1.0))


def main():
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N sessions (0 = all)")
    parser.add_argument("--rebuild", action="store_true",
                        help="DROP the cache schema first, then re-bake")
    parser.add_argument("--chunk", type=int, default=256,
                        help="Sessions per embed-batch + COPY chunk")
    parser.add_argument("--variant", default="s", choices=["s", "m", "oracle"],
                        help="LongMemEval corpus variant: s (default), m, or oracle")
    args = parser.parse_args()

    # Resolve source + cache schema names from the variant.  Globals stay
    # named SOURCE_SCHEMA / CACHE_SCHEMA so the rest of the script (which
    # references them as module-level constants) keeps working unchanged.
    global SOURCE_SCHEMA, CACHE_SCHEMA
    SOURCE_SCHEMA = f"longmemeval_{args.variant}"
    CACHE_SCHEMA = f"longmemeval_{args.variant}_bgem3_1024"

    src_dsn = _dsn_for(SOURCE_DB)
    cache_dsn = _dsn_for(CACHE_DB)
    print(f"[bake] Source: {SOURCE_DB}.{SOURCE_SCHEMA}.sessions")
    print(f"[bake] Cache:  {CACHE_DB}.{CACHE_SCHEMA}.memories  "
          f"chunk={args.chunk}  limit={args.limit or 'all'}")

    with psycopg.connect(src_dsn, autocommit=True) as src_conn, \
         psycopg.connect(cache_dsn, autocommit=True) as cache_conn:
        if args.rebuild:
            print(f"[bake] --rebuild: DROP SCHEMA {CACHE_SCHEMA} CASCADE")
            with cache_conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{CACHE_SCHEMA}" CASCADE')

        ensure_cache_schema(cache_conn)
        existing = existing_keys(cache_conn)
        if existing:
            print(f"[bake] Resuming — {len(existing):,} sessions already cached")

        # Count total work so progress is meaningful
        with src_conn.cursor() as cur:
            cur.execute(
                f'SELECT COUNT(*) FROM ('
                f'  SELECT 1 FROM "{SOURCE_SCHEMA}".questions q '
                f'  JOIN "{SOURCE_SCHEMA}".sessions s ON s.question_id = q.question_id '
                f'  GROUP BY q.question_id, s.session_idx) AS t'
            )
            total_sessions = cur.fetchone()[0]
        print(f"[bake] Source has {total_sessions:,} (qid, sidx) pairs")
        remaining = total_sessions - len(existing)
        if args.limit:
            remaining = min(remaining, args.limit)
        print(f"[bake] To embed this run: {remaining:,}")

        if remaining <= 0:
            print("[bake] Nothing to do.  Snapshot is up to date.")
            return 0

        # Embed via the shared CUDA server (same as godbench uses).
        # Initialising the EmbeddingProvider auto-starts the server.
        print("[bake] Initialising embed-server...")
        embedder = EmbeddingProvider(backend="auto")
        if embedder.dim != EMBED_DIM:
            raise RuntimeError(
                f"Embedder produces dim={embedder.dim}, expected "
                f"{EMBED_DIM} (bge-m3).  Refusing to mix dims in cache."
            )
        embed_batch = getattr(embedder, "embed_batch", None)
        if embed_batch is None:
            raise RuntimeError(
                "Embedder backend lacks embed_batch — cannot bake at scale"
            )

        t_start = time.perf_counter()
        done = 0
        skipped = 0

        def session_filter():
            nonlocal skipped
            for qid, sidx, label, content in iter_sessions(src_conn, args.limit or None):
                if (qid, sidx) in existing:
                    skipped += 1
                    continue
                yield qid, sidx, label, content

        for chunk in chunked(session_filter(), args.chunk):
            texts = [c[3] for c in chunk]
            t_embed = time.perf_counter()
            embeddings = embed_batch(texts)
            embed_s = time.perf_counter() - t_embed

            t_copy = time.perf_counter()
            rows = [(qid, sidx, lbl, content, emb)
                    for (qid, sidx, lbl, content), emb in zip(chunk, embeddings)]
            write_chunk(cache_conn, rows)
            copy_s = time.perf_counter() - t_copy

            done += len(chunk)
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta_s = (remaining - done) / rate if rate > 0 else float("inf")
            print(
                f"  +{len(chunk):>4}  total {done:>6,}/{remaining:,}  "
                f"embed={embed_s:.2f}s copy={copy_s*1000:.0f}ms  "
                f"rate={rate:.1f} sess/s  eta={eta_s/60:.1f}m",
                flush=True,
            )

        elapsed = time.perf_counter() - t_start
        print(f"\n[bake] DONE  {done:,} embedded in {elapsed/60:.1f}m "
              f"({done/elapsed:.1f} sessions/s)  skipped={skipped:,}")
        print(f"[bake] Next: pg_dump -U mazemaker -d {CACHE_DB} "
              f"--schema={CACHE_SCHEMA} -Fc "
              f"-f benchmarks/snapshots/{CACHE_SCHEMA}.dump")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
