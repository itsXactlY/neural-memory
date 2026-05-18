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
    """Measure embedding model load time cold vs warm.

    F68 fix (audit 2026-05-13): the previous implementation conflated
    Python import time (cold) with no-op object construction (warm) —
    the warm call still pointed at the in-memory singleton, so it
    measured ~0ms regardless of model size. To get a real cold load we
    have to fork a subprocess that doesn't share the parent's loaded
    module. We keep the cheap "construct second instance" timing too
    and label it as `construct_instance_ms` so its meaning is clear.
    """
    # Cold load — first instantiation IN THIS PROCESS. Will include
    # Python import + model weight load if no SharedEmbedClient is
    # already alive.
    import subprocess
    import sys as _sys
    t0 = time.perf_counter()
    from embed_provider import EmbeddingProvider
    ep = EmbeddingProvider()  # noqa: F841
    cold_ms = (time.perf_counter() - t0) * 1000

    # Construct-only cost — second instance reuses the loaded module.
    t0 = time.perf_counter()
    ep2 = EmbeddingProvider()  # noqa: F841
    construct_ms = (time.perf_counter() - t0) * 1000

    # True "cold from scratch" via subprocess: timing INSIDE a fresh
    # interpreter where nothing is pre-imported. This is the metric
    # operators care about for first-call latency on a new host.
    code = (
        "import time as _t, sys as _s\n"
        "_t0 = _t.perf_counter()\n"
        f"_s.path.insert(0, {_sys.path[0]!r})\n"
        "from embed_provider import EmbeddingProvider\n"
        "_ep = EmbeddingProvider()\n"
        "print(round((_t.perf_counter() - _t0) * 1000, 2))\n"
    )
    cold_subprocess_ms = None
    try:
        out = subprocess.check_output(
            [_sys.executable, "-c", code], stderr=subprocess.STDOUT, timeout=180
        ).decode().strip().splitlines()[-1]
        cold_subprocess_ms = float(out)
    except Exception as e:
        # Don't fail the whole suite on a subprocess hiccup; just note it.
        cold_subprocess_ms = -1.0
        cold_subprocess_err = f"{type(e).__name__}: {e}"  # noqa: F841

    return {
        "cold_load_ms": round(cold_ms, 2),
        "construct_instance_ms": round(construct_ms, 2),
        "cold_subprocess_ms": round(cold_subprocess_ms, 2),
    }


def batch_matmul_throughput(
    nm_module,
    memories: List[Dict],
    batch_sizes: List[int],
    backend: str = "auto",
) -> Dict[str, Any]:
    """Measure embedding + matmul throughput at different batch sizes."""
    import tempfile
    from memory_client import Mazemaker

    # Use a fresh tempfile DB per call so concurrent runs don't collide and the
    # real hot-path DB is never touched.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        bench_db = f.name
    nm = Mazemaker(db_path=bench_db, embedding_backend=backend)

    # Pre-load all embeddings. Setup phase — not the measurement target,
    # so skip auto_connect / conflict detection to keep ingest lean.
    print("    Pre-loading embeddings...")
    t0 = time.perf_counter()
    for m in memories:
        nm.remember(m["text"], label=m["label"],
                    auto_connect=False, detect_conflicts=False)
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
    """Compare GPU vs CPU recall performance.

    F30 fix (audit 2026-05-13): each backend now uses its own temp DB
    and is properly closed before the next one opens, so the CUDA arm
    does not race the CPU arm's WAL state. F69 fix: check torch.cuda
    availability before launching the CUDA arm and skip-with-reason
    when absent (vs silently falling back to CPU and reporting both
    "backends" as CPU).
    """
    from memory_client import Mazemaker

    # F69: probe CUDA availability up front.
    cuda_available = False
    cuda_reason = "unchecked"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_reason = f"torch.cuda.is_available()=True, device_count={torch.cuda.device_count()}"
        else:
            cuda_reason = "torch.cuda.is_available()=False"
    except Exception as e:
        cuda_reason = f"torch import failed: {e}"

    results: Dict[str, Any] = {"cuda_available": cuda_available, "cuda_reason": cuda_reason}

    for backend in ["cpu", "cuda"]:
        print(f"\n  [GPU vs CPU] Testing {backend}...")
        if backend == "cuda" and not cuda_available:
            print(f"    Skipping cuda: {cuda_reason}")
            results[backend] = {"skipped": True, "reason": cuda_reason}
            continue

        nm = None
        # Isolate backends with per-backend temp DBs so WAL / index state
        # from one arm cannot leak into the other.
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=f".{backend}.db", prefix="gpu-vs-",
                                    delete=False) as _tf_db:
            tmp_db = _tf_db.name
        try:
            try:
                nm = Mazemaker(db_path=tmp_db, embedding_backend=backend)
            except RuntimeError as e:
                print(f"    Skipping {backend}: {e}")
                results[backend] = {"error": str(e)}
                continue

            # Store memories. Setup for the recall benchmark — auto_connect off.
            t0 = time.perf_counter()
            for m in memories[:1000]:  # Limit for speed
                nm.remember(m["text"], label=m["label"],
                            auto_connect=False, detect_conflicts=False)
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
        finally:
            # F108 fix (audit 2026-05-13): always close + unlink so we don't
            # leak temp DBs across runs.
            if nm is not None:
                try:
                    nm.close()
                except Exception:
                    pass
            for ext in ("", "-wal", "-shm"):
                try:
                    Path(tmp_db + ext).unlink(missing_ok=True)
                except Exception:
                    pass

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
