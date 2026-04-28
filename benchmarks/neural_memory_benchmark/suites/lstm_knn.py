"""
LSTM+kNN Auto-Enhancement Ablation
===================================
NeuralMemory wraps base recall results with an LSTMPredictor +
multi-signal kNN re-ranker (`_enhance_recall`) when libneural_memory.so
is available. The auto-enhancement is invisible in the public API: it
either fires or it doesn't depending on whether the C++ library loads.

This suite measures the lift it produces — vs the same query set with
the enhancement disabled — on the paraphrase ground-truth set.

Two configurations:

  * baseline  — _lstm_knn_ready forced False, no LSTM, no kNN re-rank
  * enhanced — left at its natural state (active iff .so loaded)

Each is compared on:
  - recall@k
  - MRR
  - per-query latency (the enhancement is on the hot path, the cost is
    real and worth quantifying)

If the C++ library is not available, the suite reports that explicitly
rather than silently skipping — a "lift = 0" because the library
didn't load is very different from "lift = 0" because the algorithm
didn't help.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def _measure(nm: NeuralMemory, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    hits = 0
    rrs: List[float] = []
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = nm.recall(q["query"], k=k)
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


class LSTMKnnBenchmark:
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
        print("\n=== LSTM+kNN Ablation Benchmark ===")
        nm = NeuralMemory(db_path=self.db_path, embedding_backend="auto")
        for m in self.memories:
            nm.remember(m["text"], label=m["label"], auto_connect=True)

        cpp_loaded = bool(getattr(nm, "_lstm_knn_ready", False))
        results: Dict[str, Any] = {"cpp_loaded": cpp_loaded}

        if not cpp_loaded:
            print("  [warning] libneural_memory.so not loaded — only baseline path runs.")
            results["baseline"] = _measure(nm, self.queries, self.k)
            results["enhanced"] = None
            results["delta"] = None
            self._save(results)
            return results

        # Warm up so JIT/lazy init doesn't bias the first batch.
        for q in self.queries[:3]:
            nm.recall(q["query"], k=self.k)

        print("  [enhanced] running with LSTM+kNN active...")
        results["enhanced"] = _measure(nm, self.queries, self.k)

        # Toggle off and re-measure on the same NeuralMemory instance and DB.
        prior = nm._lstm_knn_ready
        try:
            nm._lstm_knn_ready = False
            for q in self.queries[:3]:
                nm.recall(q["query"], k=self.k)
            print("  [baseline] running with LSTM+kNN forced OFF...")
            results["baseline"] = _measure(nm, self.queries, self.k)
        finally:
            nm._lstm_knn_ready = prior

        results["delta"] = {
            "recall_at_k": round(
                results["enhanced"]["recall_at_k"] - results["baseline"]["recall_at_k"], 4
            ),
            "mrr": round(
                results["enhanced"]["mrr"] - results["baseline"]["mrr"], 4
            ),
            "p50_ms_overhead": round(
                results["enhanced"]["p50_ms"] - results["baseline"]["p50_ms"], 3
            ),
            "p95_ms_overhead": round(
                results["enhanced"]["p95_ms"] - results["baseline"]["p95_ms"], 3
            ),
        }
        print(f"  [delta] {results['delta']}")
        self._save(results)
        return results

    def _save(self, results: Dict[str, Any]) -> None:
        out = self.output_dir / "lstm_knn_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
