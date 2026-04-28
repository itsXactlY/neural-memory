"""
GPU Recall Benchmark Suite
============================
Measures GPU-accelerated embedding recall performance.

Tests:
  1. Embedding load time (cold vs warm cache)
  2. Batch matmul throughput (embeddings/second)
  3. Per-query latency vs batch size
  4. GPU vs CPU comparison
  5. Memory scaling: 10K → 100K → 500K embeddings
"""
import sys
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))


def measure_warm_vs_cold_load(
    embed_provider_path: str,
) -> Dict[str, float]:
    """Measure embedding model load time cold vs warm."""
    # Cold load (first time)
    t0 = time.perf_counter()
    # import — forces model load
    from embed_provider import EmbeddingProvider
    ep = EmbeddingProvider()
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm load (second time — should hit cache/singleton)
    t0 = time.perf_counter()
    ep2 = EmbeddingProvider()
    warm_ms = (time.perf_counter() - t0) * 1000

    return {"cold_load_ms": round(cold_ms, 2), "warm_load_ms": round(warm_ms, 2)}


def batch_matmul_throughput(
    nm_module,
    memories: List[Dict],
    batch_sizes: List[int],
    backend: str = "auto",
) -> Dict[str, Any]:
    """Measure embedding + matmul throughput at different batch sizes."""
    import tempfile
    from memory_client import NeuralMemory

    # Use a fresh tempfile DB per call so concurrent runs don't collide and the
    # real hot-path DB is never touched.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        bench_db = f.name
    nm = NeuralMemory(db_path=bench_db, embedding_backend=backend)

    # Pre-load all embeddings
    print("    Pre-loading embeddings...")
    t0 = time.perf_counter()
    for m in memories:
        nm.remember(m["text"], label=m["label"])
    load_time = time.perf_counter() - t0
    rate = len(memories) / load_time if load_time > 0 else 0

    results = {"preload": {"total_memories": len(memories), "load_time_s": round(load_time, 2), "rate_per_s": round(rate, 1)}}

    # Batch recall benchmark
    for bs in batch_sizes:
        queries = [m["text"][:100] for m in memories[:min(bs * 2, len(memories))]]

        t0 = time.perf_counter()
        for q in queries:
            nm.recall(q, k=5)
        elapsed = time.perf_counter() - t0
        qps = len(queries) / elapsed if elapsed > 0 else 0

        results[f"batch_{bs}"] = {
            "queries": len(queries),
            "elapsed_s": round(elapsed, 2),
            "qps": round(qps, 2),
            "ms_per_query": round(elapsed / len(queries) * 1000, 3),
        }
        print(f"    batch={bs}: {qps:.1f} qps, {elapsed/len(queries)*1000:.2f}ms/q")

    return results


def gpu_vs_cpu_comparison(
    db_path: str,
    memories: List[Dict],
) -> Dict[str, Any]:
    """Compare GPU vs CPU recall performance."""
    from memory_client import NeuralMemory

    results = {}

    for backend in ["cpu", "cuda"]:
        print(f"\n  [GPU vs CPU] Testing {backend}...")
        try:
            nm = NeuralMemory(db_path=db_path, embedding_backend=backend)
        except RuntimeError as e:
            print(f"    Skipping {backend}: {e}")
            results[backend] = {"error": str(e)}
            continue

        # Store memories
        t0 = time.perf_counter()
        for m in memories[:1000]:  # Limit for speed
            nm.remember(m["text"], label=m["label"])
        store_time = time.perf_counter() - t0

        # Recall benchmark
        queries = [m["text"][:80] for m in memories[:500]]
        t0 = time.perf_counter()
        for q in queries:
            nm.recall(q, k=10)
        recall_time = time.perf_counter() - t0

        results[backend] = {
            "store_time_s": round(store_time, 2),
            "recall_time_s": round(recall_time, 2),
            "qps": round(len(queries) / recall_time, 1) if recall_time > 0 else 0,
            "ms_per_query": round(recall_time / len(queries) * 1000, 3),
        }
        print(f"    {backend}: {results[backend]['qps']} qps, {results[backend]['ms_per_query']:.2f}ms/q")

    return results


class GPUBenchmark:
    def __init__(
        self,
        memories: List[Dict],
        output_dir: Optional[Path] = None,
        batch_sizes: List[int] = None,
    ):
        self.memories = memories
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.batch_sizes = batch_sizes or [1, 8, 32, 64, 128, 256]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        import json

        print("\n=== GPU Recall Benchmark ===")
        results = {}

        # 1. Embedding load time
        print("\n[1] Embedding load time")
        try:
            results["load_time"] = measure_warm_vs_cold_load("embed_provider")
            print(f"  Cold: {results['load_time']['cold_load_ms']}ms, "
                  f"Warm: {results['load_time']['warm_load_ms']}ms")
        except Exception as e:
            results["load_time"] = {"error": str(e)}

        # 2. Throughput at different scales
        print("\n[2] Throughput vs memory count")
        results["throughput"] = {}
        for count in [1000, 5000, 10000, 50000]:
            if count > len(self.memories):
                continue
            subset = self.memories[:count]
            print(f"\n  Scale: {count} memories")
            results["throughput"][f"{count}"] = batch_matmul_throughput(
                None, subset, self.batch_sizes[:4]
            )

        # 3. GPU vs CPU
        print("\n[3] GPU vs CPU comparison")
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db = f.name
            results["gpu_vs_cpu"] = gpu_vs_cpu_comparison(db, self.memories[:1000])
        except Exception as e:
            results["gpu_vs_cpu"] = {"error": str(e)}

        # Save
        out_path = self.output_dir / "gpu_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n[saved] {out_path}")
        return results
