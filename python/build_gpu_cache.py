#!/usr/bin/env python3
"""Build the ~/.mazemaker/engine/gpu_cache/ files that gpu_recall.GpuRecallEngine
loads. Until Mazemaker.remember() learns to append to the GPU tensor on the
fly, run this after large bulk-imports to refresh the cache.

Usage:
    python3 python/scripts/build_gpu_cache.py
    python3 python/scripts/build_gpu_cache.py --db /path/to/memory.db
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np


def build(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, label, content, embedding FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        # Fresh customer pods have an empty memory.db at install time —
        # the GPU recall cache simply can't be built yet. Return without
        # creating any files; the next call (after first remember()) will
        # have rows and succeed. Was raise SystemExit which propagated
        # past `except Exception` blocks and killed engine startup.
        print(f"  build_gpu_cache: no rows with embedding in {db_path} — skipping")
        return

    sample = len(rows[0]["embedding"])
    if sample % 4 == 0:
        dtype, dim = np.float32, sample // 4
    elif sample % 2 == 0:
        dtype, dim = np.float16, sample // 2
    else:
        raise SystemExit(f"can't infer dtype from blob length {sample}")

    print(f"  rows = {len(rows)}, dim = {dim}, dtype = {np.dtype(dtype).name}")

    emb = np.empty((len(rows), dim), dtype=dtype)
    ids, labels, contents = [], [], []
    for i, r in enumerate(rows):
        emb[i] = np.frombuffer(r["embedding"], dtype=dtype)
        ids.append(r["id"])
        labels.append(r["label"])
        contents.append(r["content"])

    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)

    t0 = time.time()
    np.save(str(out_dir / "embeddings.npy"), emb)
    with open(str(out_dir / "metadata.pkl"), "wb") as f:
        pickle.dump({"ids": ids, "labels": labels, "contents": contents}, f)
    print(f"  wrote {emb.nbytes/1e6:.1f} MB embeddings.npy + metadata.pkl in {time.time()-t0:.2f}s")
    print(f"  → {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",  default=str(Path.home() / ".mazemaker" / "engine" / "memory.db"))
    ap.add_argument("--out", default=str(Path.home() / ".mazemaker" / "engine" / "gpu_cache"))
    args = ap.parse_args()
    build(Path(args.db), Path(args.out))


if __name__ == "__main__":
    main()
