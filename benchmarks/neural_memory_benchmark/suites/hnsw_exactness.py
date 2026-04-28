"""
HNSW Exactness Benchmark
========================
HNSW is an approximate nearest-neighbor index — it trades recall for
speed. The neural-memory-adapter activates HNSW automatically above
some threshold ("auto"). This suite quantifies the trade:

  At memory tiers 1k / 10k / 50k:
    * exact (use_hnsw=False) — semantic recall via full cosine over
      the memory matrix.  Ground truth for ANN recall.
    * HNSW (use_hnsw=True)   — approximate recall.

For each tier we measure:
  * recall_at_k_overlap — for each query, fraction of HNSW's top-k
    that are also in exact's top-k. 1.0 = HNSW is lossless. 0.0 = HNSW
    returned a completely different set.
  * latency speedup.

This is the only suite that tells you whether HNSW is hurting recall
quality at the cost of latency on YOUR data, or earning the latency
honestly.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def _build_and_query(
    use_hnsw: bool,
    memories: List[Dict[str, Any]],
    queries: List[str],
    k: int,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        nm = NeuralMemory(
            db_path=db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
            use_hnsw=use_hnsw,
        )
        t0 = time.perf_counter()
        for m in memories:
            nm.remember(m["text"], label=m["label"], auto_connect=False)
        setup_s = time.perf_counter() - t0

        latencies = []
        ids_per_query: List[List[int]] = []
        for q in queries:
            t0 = time.perf_counter()
            results = nm.recall(q, k=k)
            latencies.append(time.perf_counter() - t0)
            ids_per_query.append([int(r.get("id", -1)) for r in results])

        return {
            "setup_s": round(setup_s, 2),
            "p50_ms": round(sorted(latencies)[len(latencies) // 2] * 1000, 3),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 3),
            "ids_per_query": ids_per_query,
            "use_hnsw": use_hnsw,
        }
    finally:
        for ext in ("", "-wal", "-shm"):
            try:
                os.unlink(db_path + ext)
            except OSError:
                pass


class HNSWExactnessBenchmark:
    def __init__(
        self,
        memories: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        tiers: Optional[List[int]] = None,
        k: int = 10,
    ):
        self.memories = memories
        self.queries = queries
        self.tiers = tiers or [1_000, 10_000]
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.k = k
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== HNSW Exactness Benchmark ===")
        results: Dict[str, Any] = {"tiers": {}, "k": self.k}
        # Pad memories by cycling — the original corpus may not reach 10k.
        pool = self.memories
        if len(pool) < max(self.tiers):
            repeats = (max(self.tiers) // max(len(pool), 1)) + 1
            scaled: List[Dict[str, Any]] = []
            for r in range(repeats):
                for i, m in enumerate(pool):
                    md = dict(m)
                    md["id"] = f"{md.get('id','m')}-rep{r}-{i}"
                    md["text"] = md["text"] + f" [rep{r}-{i}]"
                    scaled.append(md)
            pool = scaled[: max(self.tiers)]

        query_strings = [q["query"] for q in self.queries]

        for tier in sorted(self.tiers):
            print(f"\n  --- tier: {tier:,} memories ---")
            mems_tier = pool[:tier]

            print("    [exact] building...")
            exact = _build_and_query(False, mems_tier, query_strings, self.k)
            print("    [hnsw]  building...")
            ann = _build_and_query(True, mems_tier, query_strings, self.k)

            # Per-query overlap of HNSW's top-k with exact's top-k.
            overlaps: List[float] = []
            for ex_ids, ann_ids in zip(exact["ids_per_query"], ann["ids_per_query"]):
                ex_set = set(i for i in ex_ids if i >= 0)
                ann_set = set(i for i in ann_ids if i >= 0)
                if not ex_set:
                    continue
                overlaps.append(len(ex_set & ann_set) / len(ex_set))

            speedup_p50 = (
                exact["p50_ms"] / ann["p50_ms"]
                if ann["p50_ms"] > 0 else None
            )
            tier_result = {
                "exact": {
                    "setup_s": exact["setup_s"],
                    "p50_ms": exact["p50_ms"],
                    "p95_ms": exact["p95_ms"],
                },
                "hnsw": {
                    "setup_s": ann["setup_s"],
                    "p50_ms": ann["p50_ms"],
                    "p95_ms": ann["p95_ms"],
                },
                "ann_recall_overlap": round(statistics.mean(overlaps), 4) if overlaps else 0.0,
                "ann_recall_floor":   round(min(overlaps), 4) if overlaps else 0.0,
                "speedup_p50":        round(speedup_p50, 2) if speedup_p50 else None,
            }
            results["tiers"][f"{tier}"] = tier_result
            print(f"    overlap={tier_result['ann_recall_overlap']}  "
                  f"speedup_p50={tier_result['speedup_p50']}x  "
                  f"exact={exact['p50_ms']}ms hnsw={ann['p50_ms']}ms")

        out = self.output_dir / "hnsw_exactness_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  [saved] {out}")
        return results
