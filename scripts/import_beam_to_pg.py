#!/usr/bin/env python3
"""One-shot BEAM → PG import.

Pure data pipeline: BEAM JSON → embed_batch via shared embed-server →
COPY/INSERT into mm10m_bench.conv_<n>.memories.

No Mazemaker engine stack. No retrieval mode. No HNSW lazy state. No
license gates. Just psycopg + SharedEmbedClient. Idempotent per conv
(drops and recreates the schema's memories table on each run).

After every conv finishes, snapshot via:
    pg_dump --schema=conv_N -f conv_N_snapshot.sql

Usage:
    python import_beam_to_pg.py --conv 1
    python import_beam_to_pg.py --all
    python import_beam_to_pg.py --conv 5 --chunk 128 --limit 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks/neural_memory_benchmark/mm_10m_eval/corpus"))

import psycopg
from pgvector.psycopg import register_vector

from embed_provider import SharedEmbedClient
from corpus_helpers import load_flat_turns
from postgres_store import _build_dsn


SCHEMA_DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};
DROP TABLE IF EXISTS {schema}.memories CASCADE;
CREATE TABLE {schema}.memories (
    id            BIGSERIAL PRIMARY KEY,
    label         TEXT,
    content       TEXT,
    embedding     vector({dim}),
    vector_dim    INTEGER NOT NULL,
    salience      DOUBLE PRECISION DEFAULT 1.0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count  INTEGER DEFAULT 0,
    colbert_tokens BYTEA
);
CREATE INDEX {schema}_mem_hnsw
    ON {schema}.memories USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
CREATE INDEX {schema}_mem_fts
    ON {schema}.memories USING gin (
        to_tsvector('simple', COALESCE(content, ''))
    );
"""


def import_conv(conv: int, chunk: int, limit: int,
                client: SharedEmbedClient, conn: psycopg.Connection) -> tuple[int, float]:
    schema = f"conv_{conv}"
    turns = load_flat_turns(conv)
    if limit > 0:
        turns = turns[:limit]
    n_total = len(turns)

    with conn.cursor() as cur:
        cur.execute(SCHEMA_DDL.format(schema=schema, dim=client.dim))
    conn.commit()

    t0 = time.time()
    n_written = 0
    last_log = t0

    for start in range(0, n_total, chunk):
        batch = turns[start:start + chunk]
        texts = [
            f"[{t['time_anchor']}] {t['role']}: {t['content']}"
            if t.get('time_anchor') else f"{t['role']}: {t['content']}"
            for t in batch
        ]
        labels = [
            f"seq{t['seq']}.p{t['plan']}.b{t['batch']}.g{t['group']}.t{t['tidx']}"
            for t in batch
        ]

        embs = client.embed_batch(texts)

        with conn.cursor() as cur:
            params = []
            for lbl, txt, emb in zip(labels, texts, embs):
                params.append((lbl, txt, emb, client.dim))
            values_clause = ", ".join(["(%s, %s, %s, %s)"] * len(params))
            flat: list = []
            for p in params:
                flat.extend(p)
            cur.execute(
                f"INSERT INTO {schema}.memories "
                f"(label, content, embedding, vector_dim) VALUES {values_clause}",
                flat,
            )
        conn.commit()
        n_written += len(batch)

        now = time.time()
        if now - last_log >= 10.0 or n_written >= n_total:
            rate = n_written / (now - t0)
            eta_min = (n_total - n_written) / max(rate, 0.1) / 60
            print(f"[{schema}] {n_written}/{n_total}  "
                  f"{rate:.1f}/s  ETA {eta_min:.1f} min", flush=True)
            last_log = now

    dt = time.time() - t0
    return n_written, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--conv", type=int, help="single conv 1..10")
    g.add_argument("--all", action="store_true", help="import all 10 convs sequentially")
    ap.add_argument("--chunk", type=int, default=256,
                    help="embed/insert batch size (default 256)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap turns per conv (0 = no cap, default)")
    ap.add_argument("--from", dest="start_at", type=int, default=1,
                    help="with --all: start at conv N (default 1)")
    args = ap.parse_args()

    client = SharedEmbedClient(timeout=float(
        os.environ.get("MM_EMBED_TIMEOUT", "120.0")
    ))
    print(f"[embed] connected, dim={client.dim}", flush=True)

    conn = psycopg.connect(_build_dsn())
    register_vector(conn)

    convs = [args.conv] if args.conv else list(range(args.start_at, 11))
    overall_t0 = time.time()
    for c in convs:
        print(f"\n=== conv_{c} START ({time.strftime('%H:%M:%S')}) ===", flush=True)
        n, dt = import_conv(c, args.chunk, args.limit, client, conn)
        print(f"=== conv_{c} DONE: {n} rows in {dt/60:.2f} min "
              f"({n/dt:.1f}/s) ===", flush=True)

    conn.close()
    print(f"\nALL DONE in {(time.time()-overall_t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
