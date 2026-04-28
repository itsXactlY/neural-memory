"""
Retrieval Benchmark Suite
==========================
Measures recall accuracy, latency, and throughput across all retrieval modes.

Metrics:
  - Recall@k       : Does the correct answer appear in top-k?
  - MRR            : Mean Reciprocal Rank (1 / rank_of_first_hit)
  - Latency p50/p95/p99 : Per-query latency percentiles
  - Throughput     : Queries per second
  - Hit rate       : Fraction of queries with at least one hit
"""
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int((len(s) - 1) * p / 100)
    return float(s[min(idx, len(s) - 1)])


def recall_at_k(hit_rank: int, k: int) -> float:
    """1.0 if hit_rank <= k, else 0.0."""
    return 1.0 if 0 < hit_rank <= k else 0.0


def reciprocal_rank(hit_rank: int) -> float:
    """1 / rank if hit, else 0."""
    return 1.0 / hit_rank if hit_rank > 0 else 0.0


class RetrievalBenchmark:
    """
    Comprehensive retrieval benchmark for Neural Memory.

    Loads a dataset, stores it in Neural Memory, then queries it
    across all configured retrieval modes and measures quality + speed.
    """

    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        modes: List[str] = None,
        top_ks: List[int] = None,
        latency_runs: int = 5,
        output_dir: Optional[Path] = None,
    ):
        self.db_path = db_path
        self.memories = memories
        self.queries = queries
        self.modes = modes or ["semantic", "hybrid", "advanced", "skynet"]
        self.top_ks = top_ks or [1, 3, 5, 10, 20, 50]
        self.latency_runs = latency_runs
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.neural_mem = None
        self.results = {}
        # Map synthetic dataset id (e.g. "episodic-000000-...") → integer
        # rowid returned by NeuralMemory.remember(). Built during setup() so
        # ground-truth comparison can translate dataset ids to DB ids.
        self._id_map: Dict[str, int] = {}

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self) -> Dict[str, Any]:
        """Store all memories in Neural Memory. Returns store stats."""
        print(f"  [setup] Storing {len(self.memories)} memories in {self.db_path}")
        start = time.perf_counter()

        # Initialize Neural Memory
        self.neural_mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )

        stored = 0
        for m in self.memories:
            db_id = self.neural_mem.remember(
                text=m["text"],
                label=m["label"],
                auto_connect=True,
            )
            if db_id is not None and m.get("id"):
                self._id_map[m["id"]] = int(db_id)
            stored += 1
            if stored % 500 == 0:
                print(f"    stored {stored}/{len(self.memories)}...")

        elapsed = time.perf_counter() - start
        insert_rate = stored / elapsed if elapsed > 0 else 0

        stats = self.neural_mem.stats()
        print(f"  [setup] Done: {stored} memories in {elapsed:.1f}s ({insert_rate:.0f}/s)")
        print(f"  [setup] DB stats: {stats}")
        return {
            "stored_count": stored,
            "insert_elapsed_s": round(elapsed, 2),
            "insert_rate_per_s": round(insert_rate, 1),
            "db_stats": stats,
        }

    # ── Core benchmark ───────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run all retrieval benchmarks. Returns full results dict."""
        results = {
            "setup": self.setup(),
            "modes": {},
            "summary": {},
        }

        for mode in self.modes:
            print(f"\n  === Mode: {mode} ===")
            mode_results = self._benchmark_mode(mode)
            results["modes"][mode] = mode_results

        # Compute cross-mode summary
        results["summary"] = self._summarize(results["modes"])
        return results

    def _benchmark_mode(self, mode: str) -> Dict[str, Any]:
        """Benchmark a single retrieval mode."""
        # Switch mode (NeuralMemory stores it as the private _retrieval_mode;
        # there is no public property setter, so we set the underlying attribute
        # directly to avoid creating a stray no-op attribute).
        self.neural_mem._retrieval_mode = mode
        mode_stats = {"queries": {}, "latency": {}, "throughput": {}}

        # ── Recall quality ────────────────────────────────────────────────────
        for k in self.top_ks:
            key = f"recall@{k}"
            recalls = []
            mrrs = []
            hits = []

            for qi, q in enumerate(self.queries):
                result = self._evaluate_query(q, k)
                recalls.append(recall_at_k(result["hit_rank"], k))
                mrrs.append(reciprocal_rank(result["hit_rank"]))
                hits.append(1.0 if result["hit_rank"] > 0 else 0.0)

                if qi < 3:  # Print first few for inspection
                    print(
                        f"    {key}: query={q['query'][:40]!r}, "
                        f"hit_rank={result['hit_rank']}, "
                        f"top_labels={[r.get('label','') for r in result['results'][:3]]}"
                    )

            recall_score = statistics.mean(recalls) if recalls else 0.0
            mrr_score = statistics.mean(mrrs) if mrrs else 0.0
            hit_rate = statistics.mean(hits) if hits else 0.0

            mode_stats["queries"][key] = {
                "recall": round(recall_score, 4),
                "mrr": round(mrr_score, 4),
                "hit_rate": round(hit_rate, 4),
                "total_queries": len(recalls),
            }
            print(f"    {key}: R={recall_score:.4f}, MRR={mrr_score:.4f}, Hit={hit_rate:.4f}")

        # ── Latency ─────────────────────────────────────────────────────────
        latencies = []
        for _ in range(self.latency_runs):
            t0 = time.perf_counter()
            for q in self.queries[:20]:  # Sample for speed
                self.neural_mem.recall(q["query"], k=10)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed / min(20, len(self.queries)))

        mode_stats["latency"] = {
            "p50_ms": round(percentile(latencies, 50) * 1000, 2),
            "p95_ms": round(percentile(latencies, 95) * 1000, 2),
            "p99_ms": round(percentile(latencies, 99) * 1000, 2),
            "mean_ms": round(statistics.mean(latencies) * 1000, 2),
        }
        print(f"    Latency: p50={mode_stats['latency']['p50_ms']}ms, "
              f"p95={mode_stats['latency']['p95_ms']}ms, "
              f"p99={mode_stats['latency']['p99_ms']}ms")

        # ── Throughput ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        count = 0
        for q in self.queries:
            self.neural_mem.recall(q["query"], k=10)
            count += 1
        elapsed = time.perf_counter() - t0
        qps = count / elapsed if elapsed > 0 else 0
        mode_stats["throughput"] = {
            "queries_per_second": round(qps, 2),
            "total_queries": count,
            "elapsed_s": round(elapsed, 2),
        }
        print(f"    Throughput: {qps:.2f} queries/second")

        return mode_stats

    def _evaluate_query(
        self, query: Dict[str, Any], k: int
    ) -> Dict[str, Any]:
        """Run a single query and evaluate against ground truth.

        NeuralMemory returns integer rowids; the synthetic dataset uses
        string ids (e.g. "episodic-000000-..."). Translate ground-truth
        through the id_map built in setup(). If the map is empty (e.g. a
        suite reused setup), fall back to label-based matching so the
        recall metric is at least informative.
        """
        results = self.neural_mem.recall(query["query"], k=k)
        gt_synthetic = [
            gid for gid in query.get(
                "ground_truth_ids", [query.get("ground_truth_id")]
            ) if gid is not None
        ]
        gt_db_ids = {self._id_map[g] for g in gt_synthetic if g in self._id_map}
        gt_label = query.get("label")

        hit_rank = 0
        for i, r in enumerate(results, 1):
            rid = r.get("id", r.get("memory_id"))
            try:
                rid_int = int(rid)
            except (TypeError, ValueError):
                rid_int = None
            if rid_int is not None and rid_int in gt_db_ids:
                hit_rank = i
                break
            if not gt_db_ids and gt_label and r.get("label") == gt_label:
                hit_rank = i
                break

        return {
            "hit_rank": hit_rank,
            "results": results,
            "ground_truth": list(gt_db_ids) or gt_synthetic,
        }

    def _summarize(self, mode_results: Dict) -> Dict[str, Any]:
        """Cross-mode comparison summary."""
        summary = {"best_mode": None, "modes": {}}
        best_mrr = 0.0

        for mode, data in mode_results.items():
            qdata = data.get("queries", {})
            r5 = qdata.get("recall@5", {}).get("recall", 0.0)
            mrr = qdata.get("recall@5", {}).get("mrr", 0.0)
            lat = data.get("latency", {}).get("p50_ms", 999)
            qps = data.get("throughput", {}).get("queries_per_second", 0.0)

            summary["modes"][mode] = {
                "recall@5": r5,
                "mrr@5": mrr,
                "latency_p50_ms": lat,
                "qps": qps,
            }
            if mrr > best_mrr:
                best_mrr = mrr
                summary["best_mode"] = mode

        return summary

    def save(self, results: Dict) -> Path:
        out_path = self.output_dir / "retrieval_results.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  [saved] {out_path}")
        return out_path


# ── CLI entry point ─────────────────────────────────────────────────────────

def run_retrieval_benchmark(
    db_path: str,
    memories: List[Dict],
    queries: List[Dict],
    output_dir: Optional[Path] = None,
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bm = RetrievalBenchmark(
        db_path=db_path,
        memories=memories,
        queries=queries,
        modes=modes,
        output_dir=output_dir,
    )
    results = bm.run()
    bm.save(results)
    return results


if __name__ == "__main__":
    # Smoke test
    import tempfile
    from ..dataset import load_or_generate_dataset

    print("Retrieval Benchmark — smoke test")
    memories, queries = load_or_generate_dataset(seed=42)

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test_memory.db")
        results = run_retrieval_benchmark(
            db_path=db,
            memories=memories[:500],
            queries=queries[:20],
            modes=["semantic", "hybrid"],
        )
        print("\n=== Results ===")
        for mode, data in results["modes"].items():
            print(f"  {mode}:")
            for metric, val in data.get("queries", {}).items():
                print(f"    {metric}: {val}")
