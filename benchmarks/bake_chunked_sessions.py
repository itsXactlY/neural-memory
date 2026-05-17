#!/usr/bin/env python3
"""
bake_chunked_sessions.py — slide a ColBERT-fit window over every session.

LongMemEval-S sessions run 2-18 KB.  bge-m3 + ColBERT both truncate at
512 tokens (~2000 chars) — answers buried past char 2000 are literally
invisible to the late-interaction channel.  This bake splits every
session into ~500-token chunks with overlap, embeds each chunk, and
computes ColBERT tokens on each chunk individually.  Every sentence
ends up inside SOME chunk's encodable window.

Each chunk becomes its own row in the cache schema:
    label = "<source_label>::chunk::<N>"

Because the bench scorer colon-splits labels and tests each segment
against `answer_session_ids`, the source sid is preserved — a high-
ranking chunk surfaces the correct gold via the unchanged sid segment.

USAGE
    python benchmarks/bake_chunked_sessions.py                 # full pass
    python benchmarks/bake_chunked_sessions.py --limit 100     # smoke
    python benchmarks/bake_chunked_sessions.py --rebuild       # drop chunks first

SNAPSHOT
    pg_dump -U mazemaker -d mm10m_bench \\
        --schema=longmemeval_s_bgem3_1024 \\
        -Fc -f benchmarks/snapshots/longmemeval_s_bgem3_1024.dump
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

CACHE_DB = "mm10m_bench"
CACHE_SCHEMA = "longmemeval_s_bgem3_1024"
EMBED_DIM = 1024
CHUNK_CHARS = 2000      # ~500 bge-m3 tokens — fits ColBERT max_length=512
OVERLAP_CHARS = 200     # ~50-token overlap so sentence boundaries don't split mid-claim


def _dsn() -> str:
    from postgres_store import _build_dsn
    import urllib.parse
    base = _build_dsn()
    parsed = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(parsed._replace(path=f"/{CACHE_DB}"))


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS,
               overlap: int = OVERLAP_CHARS) -> list[str]:
    """Slide a fixed-width window over text with overlap.

    Returns a list of overlapping substrings.  Shorter than chunk_chars
    inputs return as a single chunk so we don't lose tiny rows.
    """
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def fetch_long_sessions(conn, min_chars: int, limit: int | None):
    """Yield (id, label, question_id, session_idx, content) for every
    SESSION row (not AFE facts, not already-chunked rows) longer than
    min_chars.  Skips rows already chunked (presence of any row with
    label LIKE '<source_label>::chunk::%' in the same schema).
    """
    sql = (
        f'SELECT m.id, m.label, m.question_id, m.session_idx, m.content '
        f'FROM "{CACHE_SCHEMA}".memories m '
        f'WHERE length(m.content) >= %s '
        f"  AND m.label LIKE 'session:%%'"
        f"  AND m.label NOT LIKE '%%::afe::%%'"
        f"  AND m.label NOT LIKE '%%::chunk::%%'"
        f'  AND NOT EXISTS ('
        f'    SELECT 1 FROM "{CACHE_SCHEMA}".memories c '
        f"    WHERE c.label = m.label || '::chunk::0' "
        f'      AND c.question_id IS NOT DISTINCT FROM m.question_id '
        f'      AND c.session_idx IS NOT DISTINCT FROM m.session_idx) '
        f'ORDER BY m.id'
    )
    if limit:
        sql += f' LIMIT {int(limit)}'
    with conn.cursor() as cur:
        cur.execute(sql, (min_chars,))
        yield from cur.fetchall()


def write_chunks(conn, rows: list[tuple]) -> None:
    """COPY (qid, sidx, label, content, embedding, colbert) → memories."""
    with conn.cursor() as cur:
        with cur.copy(
            f'COPY "{CACHE_SCHEMA}".memories '
            f'(question_id, session_idx, label, content, embedding, '
            f' vector_dim, salience, colbert_tokens) '
            f'FROM STDIN'
        ) as cp:
            for qid, sidx, label, content, emb, cb_bytes in rows:
                emb_text = "[" + ",".join(repr(float(v)) for v in emb) + "]"
                cp.write_row((qid, sidx, label, content, emb_text,
                              EMBED_DIM, 1.0, cb_bytes))


def main():
    import psycopg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N source sessions (0 = all)")
    parser.add_argument("--rebuild", action="store_true",
                        help="DELETE existing chunk rows before baking")
    parser.add_argument("--chunk-batch", type=int, default=64,
                        help="Sessions per embed/colbert batch (default 64)")
    parser.add_argument("--min-chars", type=int, default=CHUNK_CHARS + 1,
                        help=f"Skip sessions shorter than this "
                             f"(default {CHUNK_CHARS+1} — those already fit "
                             f"in one ColBERT window)")
    parser.add_argument("--variant", default="s",
                        choices=["s", "m", "oracle"],
                        help="LongMemEval variant: s (default), m, or oracle")
    args = parser.parse_args()
    global CACHE_SCHEMA
    CACHE_SCHEMA = f"longmemeval_{args.variant}_bgem3_1024"

    print(f"[chunk-bake] DB: {CACHE_DB}.{CACHE_SCHEMA}")
    print(f"[chunk-bake] chunk_chars={CHUNK_CHARS}  overlap={OVERLAP_CHARS}  "
          f"min_chars={args.min_chars}")

    if args.rebuild:
        with psycopg.connect(_dsn(), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DELETE FROM "{CACHE_SCHEMA}".memories '
                    f"WHERE label LIKE '%::chunk::%'"
                )
                print(f"[chunk-bake] --rebuild: {cur.rowcount} chunk rows removed")

    print("[chunk-bake] Initialising embed-server + colbert encoder...")
    from embed_provider import EmbeddingProvider
    from colbert_helper import (
        colbert_available, encode_tokens_batch, pack_tokens,
    )
    embedder = EmbeddingProvider(backend="auto")
    if embedder.dim != EMBED_DIM:
        raise RuntimeError(f"embedder dim={embedder.dim} != {EMBED_DIM}")
    embed_batch = getattr(embedder, "embed_batch")
    if not colbert_available():
        raise RuntimeError("ColBERT unavailable — check colbert_helper")

    with psycopg.connect(_dsn(), autocommit=True) as conn:
        # Count work
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT count(*) FROM "{CACHE_SCHEMA}".memories '
                f'WHERE length(content) >= %s '
                f"  AND label LIKE 'session:%%' "
                f"  AND label NOT LIKE '%%::afe::%%' "
                f"  AND label NOT LIKE '%%::chunk::%%'",
                (args.min_chars,),
            )
            total_sources = int(cur.fetchone()[0])
        print(f"[chunk-bake] Source long sessions (≥{args.min_chars} chars): "
              f"{total_sources:,}")
        if args.limit:
            total_sources = min(total_sources, args.limit)
        if total_sources == 0:
            print("[chunk-bake] Nothing to do.")
            return 0

        t_start = time.perf_counter()
        done_sources = 0
        total_chunks = 0
        batch_buf: list[tuple] = []  # buffered chunk tuples awaiting embed

        def flush(buf: list[tuple]) -> int:
            if not buf:
                return 0
            texts = [b[3] for b in buf]
            t_e = time.perf_counter()
            embs = embed_batch(texts)
            t_c = time.perf_counter()
            cb_arrs = encode_tokens_batch(texts, top_k=32, batch_size=32)
            t_w = time.perf_counter()
            rows = []
            for (qid, sidx, label, content), emb, cb in zip(
                ((b[0], b[1], b[2], b[3]) for b in buf), embs, cb_arrs
            ):
                cb_bytes = pack_tokens(cb) if cb is not None else None
                rows.append((qid, sidx, label, content, emb, cb_bytes))
            write_chunks(conn, rows)
            print(
                f"    flushed {len(buf):>4} chunks  "
                f"embed={(t_c-t_e):.2f}s colbert={(t_w-t_c):.2f}s "
                f"copy={(time.perf_counter()-t_w):.2f}s",
                flush=True,
            )
            return len(buf)

        for src_id, src_label, qid, sidx, content in fetch_long_sessions(
            conn, args.min_chars, args.limit or None
        ):
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_label = f"{src_label}::chunk::{i}"
                batch_buf.append((qid, sidx, chunk_label, chunk))
            done_sources += 1

            # Flush when buffer reaches the batch size
            if len(batch_buf) >= args.chunk_batch:
                total_chunks += flush(batch_buf)
                batch_buf.clear()
                elapsed = time.perf_counter() - t_start
                rate = done_sources / elapsed if elapsed else 0
                eta = (total_sources - done_sources) / rate / 60 if rate else float("inf")
                print(
                    f"  sources={done_sources:>5}/{total_sources}  "
                    f"chunks={total_chunks:>6}  "
                    f"rate={rate:.1f} src/s  eta={eta:.1f}m",
                    flush=True,
                )

        # Final flush
        if batch_buf:
            total_chunks += flush(batch_buf)
            batch_buf.clear()

        elapsed = time.perf_counter() - t_start
        print(f"\n[chunk-bake] DONE  sources={done_sources:,}  "
              f"chunks={total_chunks:,}  {elapsed/60:.1f}m  "
              f"({done_sources/elapsed:.1f} sources/s)")
        print(f"[chunk-bake] Next: pg_dump -U mazemaker -d {CACHE_DB} "
              f"--schema={CACHE_SCHEMA} -Fc "
              f"-f benchmarks/snapshots/{CACHE_SCHEMA}.dump")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
