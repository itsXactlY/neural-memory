"""
LSTM+kNN Auto-Enhancement Ablation
===================================
NeuralMemory (`memory_client.NeuralMemory`) is the SQL/graph store.
The LSTM + multi-signal kNN re-ranker (`_enhance_recall`) lives on the
PUBLIC wrapper `neural_memory.Memory`, not on `NeuralMemory`. Codex
audit 2026-04-28 caught the prior version of this suite imported the
wrong class and never actually toggled the feature — it always
reported "not loaded".

Two configurations:

  * baseline  — Memory()._lstm_knn_ready forced False (skips both the
    AccessLogger write AND the kNN re-rank in Memory._enhance_recall)
  * enhanced  — left at its natural state (active iff libneural_memory.so
    + lstm_knn_bridge import successfully)

We also seed the AccessLogger with a sequence of related queries before
the measurement, so the LSTM has temporal context to predict from. The
prior version fired the queries in a single pass with no warmup — even
if the C++ library had loaded, the LSTM had a 1-element history and
predicted noise.

Reports recall@k, MRR, p50/p95 latency for each, and the per-knob
delta. Cleanly skips with a warning if the C++ library isn't built —
"lift = 0 because the library didn't load" is very different from
"lift = 0 because the algorithm didn't help" and the report makes that
distinction explicit.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from neural_memory import Memory


def _measure(mem: Memory, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    hits = 0
    rrs: List[float] = []
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = mem.recall(q["query"], k=k)
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
        warmup_passes: int = 3,
    ):
        self.db_path = db_path
        self.memories = memories
        self.queries = queries
        self.k = k
        self.warmup_passes = warmup_passes
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== LSTM+kNN Ablation Benchmark ===")
        # Use the PUBLIC Memory wrapper — that's where _lstm_knn_ready and
        # _enhance_recall actually live.  The prior suite imported
        # NeuralMemory and looked for _lstm_knn_ready, which always returned
        # False because that attribute doesn't exist there.
        mem = Memory(db_path=self.db_path, embedding_backend="auto")
        for m in self.memories:
            mem.remember(m["text"], label=m["label"], auto_connect=True)

        cpp_loaded = bool(getattr(mem, "_lstm_knn_ready", False))
        results: Dict[str, Any] = {
            "cpp_loaded": cpp_loaded,
            "warmup_passes": self.warmup_passes,
        }

        if not cpp_loaded:
            print("  [warning] libneural_memory.so not loaded — LSTM+kNN inactive.")
            print("            Reporting baseline only; lift cannot be measured.")
            results["baseline"] = _measure(mem, self.queries, self.k)
            results["enhanced"] = None
            results["delta"] = None
            results["note"] = (
                "lift==0 here means the C++ extension is not built, NOT that "
                "the algorithm has no effect. Build with `cd build && cmake "
                "--build . -j` and re-run."
            )
            self._save(results)
            return results

        # Seed the AccessLogger with prior-turn context so the LSTM has a
        # non-trivial sequence to predict from. Repeats every query
        # `warmup_passes` times to build a non-IID access pattern.
        print(f"  [warmup] {self.warmup_passes}x pass to seed AccessLogger...")
        for _ in range(self.warmup_passes):
            for q in self.queries:
                mem.recall(q["query"], k=self.k)

        # Toggle off and re-measure on the same Memory instance + DB so the
        # only changing variable is _enhance_recall firing or not.
        prior = mem._lstm_knn_ready
        try:
            mem._lstm_knn_ready = False
            print("  [baseline] LSTM+kNN forced OFF...")
            results["baseline"] = _measure(mem, self.queries, self.k)
        finally:
            mem._lstm_knn_ready = prior

        print("  [enhanced] LSTM+kNN active...")
        results["enhanced"] = _measure(mem, self.queries, self.k)

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
