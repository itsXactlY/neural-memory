"""
Baseline Comparison Suite — naive cosine vs neural-memory-adapter
==================================================================
Runs the SAME queries against the SAME paraphrase corpus through:

  1. Raw cosine (numpy):  embed every memory + every query, take
     argmax_k of dot product. No graph, no MMR, no LSTM, no PPR.
     This is the "vector-DB-with-extra-steps" line — anything we
     can't beat here, the fancy machinery isn't earning its keep.

  2. neural-memory-adapter (semantic mode): the closest like-for-like.

  3. neural-memory-adapter (skynet mode): full pipeline.

Reports recall@k, MRR, and per-query latency for each. The user gets
to see whether the multi-channel fusion actually outperforms a 50-line
numpy implementation on the same data.

The baseline uses the SAME embedding model neural-memory uses (via
EmbeddingProvider) so the comparison isolates RETRIEVAL quality, not
embedding quality — otherwise we'd be measuring two confounded things.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory
from embed_provider import EmbeddingProvider


class _RawCosineIndex:
    """Tiny in-process baseline. No graph, no rerank, no temporal."""

    def __init__(self, embedder: EmbeddingProvider):
        self._embedder = embedder
        self._ids: List[str] = []
        self._labels: List[str] = []
        self._texts: List[str] = []
        self._mat: Optional[np.ndarray] = None

    def add_batch(self, memories: List[Dict[str, Any]]) -> float:
        t0 = time.perf_counter()
        embs = []
        for m in memories:
            self._ids.append(m["id"])
            self._labels.append(m["label"])
            self._texts.append(m["text"])
            embs.append(self._embedder.embed(m["text"]))
        self._mat = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(self._mat, axis=1, keepdims=True).clip(min=1e-12)
        self._mat = self._mat / norms
        return time.perf_counter() - t0

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._mat is None or len(self._ids) == 0:
            return []
        q = np.asarray(self._embedder.embed(query), dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        q = q / q_norm
        sims = self._mat @ q
        topk = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        topk = topk[np.argsort(-sims[topk])]
        return [
            {
                "id": self._ids[int(i)],
                "label": self._labels[int(i)],
                "content": self._texts[int(i)],
                "similarity": float(sims[int(i)]),
            }
            for i in topk
        ]


def _measure(recall_fn, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    hits = 0
    rrs: List[float] = []
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = recall_fn(q["query"], k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        anchor = q.get("anchor", "")
        rank = 0
        for i, r in enumerate(results, 1):
            if anchor and anchor in (r.get("content") or ""):
                rank = i
                break
        hits += 1 if rank > 0 else 0
        rrs.append(1.0 / rank if rank > 0 else 0.0)
    n = len(queries) or 1
    latencies_ms.sort()
    return {
        "recall_at_k": round(hits / n, 4),
        "mrr": round(statistics.mean(rrs) if rrs else 0.0, 4),
        "p50_ms": round(latencies_ms[len(latencies_ms) // 2], 3) if latencies_ms else 0.0,
        "p95_ms": round(latencies_ms[int(len(latencies_ms) * 0.95)], 3) if latencies_ms else 0.0,
        "n": len(queries),
    }


class BaselineComparisonBenchmark:
    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        k: int = 5,
    ):
        self.db_path = db_path
        self.memories = memories
        self.queries = queries
        self.k = k
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Baseline Comparison Benchmark ===")
        # Use the SAME embedder for both sides so the comparison is about
        # retrieval, not encoding.
        embedder = EmbeddingProvider(backend="auto")

        print("  [baseline] building raw cosine index...")
        raw = _RawCosineIndex(embedder)
        raw_setup_s = raw.add_batch(self.memories)
        raw_perf = _measure(raw.recall, self.queries, self.k)
        raw_perf["setup_s"] = round(raw_setup_s, 2)

        print("  [neural] building neural-memory-adapter (semantic mode)...")
        nm = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )
        nm_setup_t0 = time.perf_counter()
        for m in self.memories:
            nm.remember(m["text"], label=m["label"], auto_connect=True)
        nm_setup_s = time.perf_counter() - nm_setup_t0
        nm_perf = _measure(lambda q, k=5: nm.recall(q, k=k), self.queries, self.k)
        nm_perf["setup_s"] = round(nm_setup_s, 2)

        # Skynet mode reuses the same DB + index.
        nm._retrieval_mode = "skynet"
        nm_skynet = _measure(lambda q, k=5: nm.recall(q, k=k), self.queries, self.k)

        results = {
            "raw_cosine_baseline": raw_perf,
            "neural_memory_semantic": nm_perf,
            "neural_memory_skynet": nm_skynet,
            "deltas_vs_baseline": {
                "semantic_recall_lift": round(
                    nm_perf["recall_at_k"] - raw_perf["recall_at_k"], 4
                ),
                "semantic_mrr_lift": round(
                    nm_perf["mrr"] - raw_perf["mrr"], 4
                ),
                "skynet_recall_lift": round(
                    nm_skynet["recall_at_k"] - raw_perf["recall_at_k"], 4
                ),
                "skynet_mrr_lift": round(
                    nm_skynet["mrr"] - raw_perf["mrr"], 4
                ),
                "skynet_latency_overhead_p50_ms": round(
                    nm_skynet["p50_ms"] - raw_perf["p50_ms"], 3
                ),
            },
            "k": self.k,
            "n_queries": len(self.queries),
            "n_memories": len(self.memories),
        }
        print(f"  raw      : R@k={raw_perf['recall_at_k']}  MRR={raw_perf['mrr']}  "
              f"p50={raw_perf['p50_ms']}ms")
        print(f"  semantic : R@k={nm_perf['recall_at_k']}  MRR={nm_perf['mrr']}  "
              f"p50={nm_perf['p50_ms']}ms")
        print(f"  skynet   : R@k={nm_skynet['recall_at_k']}  MRR={nm_skynet['mrr']}  "
              f"p50={nm_skynet['p50_ms']}ms")

        out = self.output_dir / "baseline_comparison_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
