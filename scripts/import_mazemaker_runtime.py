#!/usr/bin/env python3
"""mm_bench_raw → mm10m_bench (Mazemaker runtime) bulk-load.

Pure pgvector bulk-load pattern: COPY into memories table WITHOUT any
indexes, then CREATE INDEX once at the end. Avoids the O(M log N)
HNSW per-row insert death-spiral that brings Mazemaker's
remember_batch to a crawl past ~10 k rows.

After this runs, the conv_<N> schema in mm10m_bench is ready for
Mazemaker queries + dream cycle. No Mazemaker engine constructed in
this script — embeddings come straight from the shared embed-server,
rows go straight via COPY.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import psycopg
from pgvector.psycopg import register_vector

from embed_provider import SharedEmbedClient
from postgres_store import _build_dsn


def _schema_ddl(schema: str, dim: int) -> str:
    return f"""
DROP SCHEMA IF EXISTS {schema} CASCADE;
CREATE SCHEMA {schema};

CREATE TABLE {schema}.memories (
    id             BIGSERIAL PRIMARY KEY,
    label          TEXT,
    content        TEXT,
    embedding      vector({dim}),
    vector_dim     INTEGER NOT NULL,
    salience       DOUBLE PRECISION DEFAULT 1.0,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    last_accessed  TIMESTAMPTZ DEFAULT NOW(),
    access_count   INTEGER DEFAULT 0,
    colbert_tokens BYTEA
);

CREATE TABLE {schema}.connections (
    source_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    weight    DOUBLE PRECISION DEFAULT 0.0,
    edge_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE {schema}.memory_revisions (
    id BIGSERIAL PRIMARY KEY,
    memory_id BIGINT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE {schema}.meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _build_indexes(conn, schema: str, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_memories_label "
            f"ON {schema}.memories(label)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_memories_created_at "
            f"ON {schema}.memories(created_at DESC)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_memories_content_fts "
            f"ON {schema}.memories USING gin "
            f"(to_tsvector('simple', COALESCE(content, '')))"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
            f"ON {schema}.memories USING hnsw "
            f"(embedding vector_cosine_ops) "
            f"WITH (m=16, ef_construction=64)"
        )
    conn.commit()


def import_conv(conv: int, scale: str, chunk: int,
                client: SharedEmbedClient,
                raw_dsn: str, runtime_dsn: str) -> tuple[int, float]:
    schema = f"conv_{conv}"
    src_schema = f"beam_{scale.lower()}"

    # 1) Pull all turns from mm_bench_raw into memory (small per-conv:
    #    ~20K rows × ~2KB each ≈ 40 MB).
    with psycopg.connect(raw_dsn) as src:
        with src.cursor() as cur:
            cur.execute(
                f"SELECT seq, plan_id, batch_id, group_id, tidx, role, "
                f"content, time_anchor "
                f"FROM {src_schema}.turns WHERE conv_id = %s ORDER BY seq",
                (conv,),
            )
            turns = cur.fetchall()
    n_total = len(turns)
    print(f"[{schema}] loaded {n_total} turns from {src_schema}", flush=True)

    # 2) Recreate runtime schema, no indexes yet.
    rt = psycopg.connect(runtime_dsn)
    register_vector(rt)
    with rt.cursor() as cur:
        cur.execute(_schema_ddl(schema, client.dim))
    rt.commit()

    # 3) Bulk insert. Embed via shared server, COPY into table.
    t0 = time.time()
    n_written = 0
    last_log = t0

    # Truncate at 4000 chars (~1000 tokens). BGE-M3 hard-truncates at
    # 8192 tokens anyway, and attention is O(seq_len^2): a single 247K
    # outlier turn pads the entire batch to 8K tokens and triggers OOM
    # on the embed-server. The CONTENT column in PG keeps the full
    # text — only the embedded text is truncated.
    EMBED_TRUNCATE = 4000

    for start in range(0, n_total, chunk):
        batch = turns[start:start + chunk]
        texts: list[str] = []
        labels: list[str] = []
        full_contents: list[str] = []
        for (seq, plan_id, batch_id, group_id, tidx, role, content, anchor) in batch:
            prefix = f"[{anchor}] " if anchor else ""
            full_text = f"{prefix}{role}: {content}"
            full_contents.append(full_text)
            texts.append(full_text[:EMBED_TRUNCATE])
            plan_part = f".p{plan_id}" if plan_id is not None else ""
            labels.append(
                f"seq{seq}{plan_part}.b{batch_id}.g{group_id}.t{tidx}"
            )

        embs = client.embed_batch(texts)

        # pgvector's COPY text format wants "[v1,v2,...]" (brackets);
        # Python lists serialise as "{v1,v2,...}" (PG array braces),
        # which pgvector rejects. Format it ourselves for COPY.
        with rt.cursor() as cur:
            with cur.copy(
                f"COPY {schema}.memories "
                "(label, content, embedding, vector_dim) FROM STDIN"
            ) as cp:
                for lbl, full, emb in zip(labels, full_contents, embs):
                    emb_str = "[" + ",".join(f"{v:.7g}" for v in emb) + "]"
                    cp.write_row((lbl, full, emb_str, client.dim))
        rt.commit()
        n_written += len(batch)

        now = time.time()
        if now - last_log >= 10.0 or n_written >= n_total:
            rate = n_written / (now - t0)
            eta_min = (n_total - n_written) / max(rate, 0.1) / 60
            print(f"[{schema}] {n_written}/{n_total}  "
                  f"{rate:.1f}/s  ETA {eta_min:.1f} min", flush=True)
            last_log = now

    ingest_t = time.time() - t0
    print(f"[{schema}] ingest done: {n_written} in {ingest_t:.1f}s "
          f"({n_written/ingest_t:.0f}/s)", flush=True)

    # 4) Build indexes once.
    idx_t0 = time.time()
    print(f"[{schema}] building indexes (HNSW + FTS) ...", flush=True)
    _build_indexes(rt, schema, client.dim)
    print(f"[{schema}] indexes built in {time.time() - idx_t0:.1f}s",
          flush=True)
    rt.close()

    return n_written, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--conv", type=int, help="single conv id")
    g.add_argument("--all", action="store_true",
                   help="all convs at the given scale")
    ap.add_argument("--scale", default="10M",
                    choices=["10M", "1M", "500K", "100K"])
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    raw_dsn = _build_dsn().replace(
        "dbname=mazemaker", "dbname=mm_bench_raw"
    ).replace("dbname=mm10m_bench", "dbname=mm_bench_raw")
    runtime_dsn = _build_dsn().replace(
        "dbname=mazemaker", "dbname=mm10m_bench"
    )

    client = SharedEmbedClient(timeout=float(
        os.environ.get("MM_EMBED_TIMEOUT", "120.0")
    ))
    print(f"[embed] dim={client.dim}", flush=True)

    if args.all:
        with psycopg.connect(raw_dsn) as src:
            with src.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT conv_id FROM beam_{args.scale.lower()}.turns "
                    f"ORDER BY conv_id"
                )
                convs = [r[0] for r in cur.fetchall()]
        print(f"importing {len(convs)} convs at scale {args.scale}", flush=True)
    else:
        convs = [args.conv]

    overall_t0 = time.time()
    for c in convs:
        print(f"\n=== conv {c} START ({time.strftime('%H:%M:%S')}) ===", flush=True)
        n, dt = import_conv(c, args.scale, args.chunk, client, raw_dsn, runtime_dsn)
        print(f"=== conv {c} DONE: {n} in {dt/60:.2f} min ===", flush=True)

    print(f"\nALL DONE in {(time.time()-overall_t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
