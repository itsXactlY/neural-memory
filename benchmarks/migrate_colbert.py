#!/usr/bin/env python3
"""migrate_colbert.py — populate colbert_tokens on rows that lack them.

Walks the cache schema's memories table, finds any row with
content ≥ MIN_LEN chars and a NULL colbert_tokens, encodes top-K tokens
via ColBERT, packs and writes back via UPDATE.

USAGE
    python benchmarks/migrate_colbert.py                # --variant s
    python benchmarks/migrate_colbert.py --variant oracle
    python benchmarks/migrate_colbert.py --batch 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"


def _dsn() -> str:
    from postgres_store import _build_dsn
    import urllib.parse
    base = _build_dsn()
    p = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(p._replace(path=f"/{CACHE_DB}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    parser.add_argument("--schema", default=None)
    parser.add_argument("--batch", type=int, default=64,
                        help="Rows per encode/UPDATE batch (default 64)")
    parser.add_argument("--min-len", type=int, default=50,
                        help="Skip content shorter than this (default 50)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap on rows processed this run (0 = all)")
    parser.add_argument("--label-like", default=None,
                        help="Optional SQL LIKE pattern to restrict (e.g. '%%::afe::%%')")
    parser.add_argument("--label-not-like", default=None,
                        help="Optional SQL NOT LIKE pattern to exclude (e.g. 'session:%%')")
    args = parser.parse_args()
    schema = args.schema or f"longmemeval_{args.variant}_bgem3_1024"

    from colbert_helper import colbert_available, encode_tokens_batch, pack_tokens
    if not colbert_available():
        print("[colbert-migrate] FATAL: ColBERT unavailable", file=sys.stderr)
        return 2

    import psycopg
    print(f"[colbert-migrate] DB={CACHE_DB} schema={schema} batch={args.batch}")

    where = "colbert_tokens IS NULL AND length(content) >= %s"
    params = [args.min_len]
    if args.label_like:
        where += " AND label LIKE %s"
        params.append(args.label_like)
    if args.label_not_like:
        where += " AND label NOT LIKE %s"
        params.append(args.label_not_like)
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT count(*) FROM "{schema}".memories WHERE ' + where,
                tuple(params),
            )
            total = int(cur.fetchone()[0])
        print(f"[colbert-migrate] {total:,} rows need ColBERT tokens")
        if total == 0:
            return 0
        if args.limit:
            total = min(total, args.limit)

        t_start = time.perf_counter()
        done = 0
        while done < total:
            chunk = args.batch if not args.limit else min(args.batch, args.limit - done)
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT id, content FROM "{schema}".memories WHERE ' + where +
                    ' ORDER BY id LIMIT %s',
                    tuple(params) + (chunk,),
                )
                rows = cur.fetchall()
            if not rows:
                break
            ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            t_e = time.perf_counter()
            cb_arrs = encode_tokens_batch(texts, top_k=32, batch_size=32)
            t_w = time.perf_counter()
            updates = []
            for mid, arr in zip(ids, cb_arrs):
                if arr is None:
                    continue
                updates.append((pack_tokens(arr), mid))
            if updates:
                with conn.cursor() as cur:
                    cur.executemany(
                        f'UPDATE "{schema}".memories SET colbert_tokens = %s WHERE id = %s',
                        updates,
                    )
            done += len(rows)
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate / 60 if rate else float("inf")
            print(
                f"  +{len(rows):>3}  done={done:>6,}/{total}  "
                f"encode={(t_w-t_e):.2f}s update={(time.perf_counter()-t_w):.2f}s "
                f"rate={rate:.1f}/s  eta={eta:.1f}m",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start
    print(f"\n[colbert-migrate] DONE  {done:,} rows in {elapsed/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
