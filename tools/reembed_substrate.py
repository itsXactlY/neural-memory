#!/usr/bin/env python3
"""reembed_substrate.py — re-embed all memories in a SQLite substrate using
a different sentence-transformer model. Used for embedder ablation A/B tests.

Per Sonnet upgrade-scout 2026-05-02: switching MiniLM-L6 to bge-small-en-v1.5
should lift R@5 by 0.02-0.04 absolute on AE-domain bench. This script
performs the substrate-side re-embed required to run the bench against
the new model.

Usage:
    NM_EMBED_MODEL=BAAI/bge-small-en-v1.5 \\
        python3 tools/reembed_substrate.py \\
            --db /Users/tito/.neural_memory/memory.bge-small-test.db

WARNING: this updates the embedding column in-place. Always run on a COPY
of the production DB. The script refuses to run on the canonical path
unless --i-know-what-im-doing is set.
"""

from __future__ import annotations
import argparse
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

CANONICAL_DB = str(Path.home() / ".neural_memory" / "memory.db")


def reembed(db_path: str, batch_size: int = 256) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    from embed_provider import DIMENSION, EmbeddingProvider
    provider = EmbeddingProvider(backend="sentence-transformers")
    print(f"[reembed] backend dim = {provider.dim}")
    print(f"[reembed] model = {provider.backend.__class__.__name__}")
    if hasattr(provider.backend, "MODEL_NAME"):
        print(f"[reembed] MODEL_NAME = {provider.backend.MODEL_NAME}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"[reembed] {total} memories to re-embed")

    t0 = time.time()
    done = 0
    skipped_empty = 0

    cur = conn.execute("SELECT id, content FROM memories ORDER BY id")
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        ids, texts = [], []
        for mid, content in rows:
            if not content or not content.strip():
                skipped_empty += 1
                continue
            ids.append(mid)
            texts.append(content if len(content) <= 8000 else content[:8000])

        if not texts:
            continue

        embeddings = provider.backend.embed_batch(texts)

        # Pack into binary format (matches existing storage)
        update_rows = []
        for mid, emb in zip(ids, embeddings):
            if len(emb) != DIMENSION:
                raise ValueError(
                    f"Refusing to write memory {mid}: embedding dim {len(emb)} "
                    f"!= DIMENSION ({DIMENSION}). Substrate is fixed-width."
                )
            packed = struct.pack(f"{len(emb)}f", *emb)
            update_rows.append((packed, mid))

        conn.executemany("UPDATE memories SET embedding = ? WHERE id = ?", update_rows)
        conn.commit()

        done += len(ids)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"[reembed] [{done}/{total}] {rate:.1f} mem/s ETA {eta:.0f}s",
              flush=True)

    conn.close()
    return {
        "total": total,
        "reembedded": done,
        "skipped_empty": skipped_empty,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="SQLite DB path to re-embed")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--i-know-what-im-doing", action="store_true",
                        help="Allow running on the canonical production DB")
    args = parser.parse_args()

    if args.db == CANONICAL_DB and not args.i_know_what_im_doing:
        print(f"REFUSING to re-embed canonical DB at {args.db}", file=sys.stderr)
        print("Pass --i-know-what-im-doing to override.", file=sys.stderr)
        return 2

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    stats = reembed(args.db, batch_size=args.batch_size)
    print(f"\n=== Re-embed complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
