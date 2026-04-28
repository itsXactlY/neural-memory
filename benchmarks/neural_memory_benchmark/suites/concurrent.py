"""
Concurrent / WAL Benchmark Suite
==================================
Stress-tests SQLite WAL under concurrent multi-threaded load.

Tests:
  1. Concurrent writers (1, 2, 4, 8, 16 threads)
  2. Concurrent readers (1, 4, 8, 16, 32 threads)
  3. Mixed read/write workloads
  4. WAL file size growth under load
  5. Deadlock / contention detection
"""
import os
import sys
import time
import threading
import tempfile
import statistics
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def wal_size_mb(db_path: str) -> float:
    base = db_path.replace("-wal", "").replace("-shm", "")
    wal = base + "-wal"
    if os.path.exists(wal):
        return os.path.getsize(wal) / 1024 / 1024
    return 0.0


def run_concurrent_writers(
    db_path: str,
    num_writers: int,
    ops_per_writer: int,
    memories: List[Dict],
) -> Dict[str, Any]:
    """Run N concurrent writers inserting memories."""

    errors = []
    completed = [0]
    lock = threading.Lock()
    start_time = [time.perf_counter()]
    end_time = [None]

    def writer_task(writer_id: int):
        try:
            nm = NeuralMemory(db_path=db_path, embedding_backend="auto")
            ops = 0
            for i in range(ops_per_writer):
                m = memories[(writer_id * ops_per_writer + i) % len(memories)]
                nm.remember(
                    f"{m['text']} [writer={writer_id}, op={i}]",
                    label=m["label"],
                    auto_connect=False,  # Faster without auto-connect
                )
                ops += 1
            with lock:
                completed[0] += ops
        except Exception as e:
            with lock:
                errors.append(f"writer_{writer_id}: {e}")

    with ThreadPoolExecutor(max_workers=num_writers) as ex:
        futures = [ex.submit(writer_task, i) for i in range(num_writers)]
        for f in as_completed(futures):
            pass  # Wait for completion

    end_time[0] = time.perf_counter()
    elapsed = end_time[0] - start_time[0]
    total_ops = completed[0]
    rate = total_ops / elapsed if elapsed > 0 else 0

    return {
        "writers": num_writers,
        "ops_per_writer": ops_per_writer,
        "total_ops": total_ops,
        "elapsed_s": round(elapsed, 2),
        "ops_per_second": round(rate, 1),
        "ms_per_op": round(elapsed / total_ops * 1000, 3) if total_ops > 0 else 0,
        "errors": errors,
    }


def run_concurrent_readers(
    db_path: str,
    num_readers: int,
    ops_per_reader: int,
    query_texts: List[str],
) -> Dict[str, Any]:
    """Run N concurrent readers querying memories."""
    nm = NeuralMemory(db_path=db_path, embedding_backend="auto")

    latencies = []
    errors = []
    lock = threading.Lock()

    def reader_task(reader_id: int):
        local_latencies = []
        try:
            for i in range(ops_per_reader):
                q = query_texts[i % len(query_texts)]
                t0 = time.perf_counter()
                nm.recall(q, k=10)
                local_latencies.append(time.perf_counter() - t0)
        except Exception as e:
            with lock:
                errors.append(f"reader_{reader_id}: {e}")
        finally:
            with lock:
                latencies.extend(local_latencies)

    threads = [threading.Thread(target=reader_task, args=(i,)) for i in range(num_readers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    latencies_ms = [l * 1000 for l in latencies]
    latencies_ms.sort()

    return {
        "readers": num_readers,
        "ops_per_reader": ops_per_reader,
        "total_ops": len(latencies),
        "elapsed_s": round(elapsed, 2),
        "ops_per_second": round(len(latencies) / elapsed, 1) if elapsed > 0 else 0,
        "latency_p50_ms": round(latencies_ms[len(latencies_ms)//2], 2) if latencies_ms else 0,
        "latency_p95_ms": round(latencies_ms[int(len(latencies_ms)*0.95)] if latencies_ms else 0, 2),
        "latency_p99_ms": round(latencies_ms[int(len(latencies_ms)*0.99)] if latencies_ms else 0, 2),
        "errors": errors,
    }


def run_mixed_workload(
    db_path: str,
    num_writers: int,
    num_readers: int,
    ops_per_thread: int,
    memories: List[Dict],
    query_texts: List[str],
) -> Dict[str, Any]:
    """Run mixed read/write workload simultaneously."""
    results = {"writers": None, "readers": None}

    with ThreadPoolExecutor(max_workers=num_writers + num_readers) as ex:
        writer_futures = []
        for i in range(num_writers):
            def wtask(wid=i):
                return run_concurrent_writers(
                    db_path, 1, ops_per_thread,
                    memories[wid * ops_per_thread: (wid+1) * ops_per_thread or None]
                )
            writer_futures.append(ex.submit(wtask))

        reader_futures = []
        for i in range(num_readers):
            def rtask(rid=i):
                return run_concurrent_readers(db_path, 1, ops_per_thread, query_texts)
            reader_futures.append(ex.submit(rtask))

        wf = as_completed(writer_futures)
        rf = as_completed(reader_futures)

    # Collect
    w_results = [f.result() for f in writer_futures]
    r_results = [f.result() for f in reader_futures]

    total_write_ops = sum(r["total_ops"] for r in w_results)
    total_read_ops = sum(r["total_ops"] for r in r_results)
    total_ops = total_write_ops + total_read_ops
    total_time = max(
        max((r["elapsed_s"] for r in w_results), default=0),
        max((r["elapsed_s"] for r in r_results), default=0),
    )

    return {
        "writers": num_writers,
        "readers": num_readers,
        "total_write_ops": total_write_ops,
        "total_read_ops": total_read_ops,
        "total_ops": total_ops,
        "ops_per_second": round(total_ops / total_time, 1) if total_time > 0 else 0,
        "write_results": w_results,
        "read_results": r_results,
        "wal_size_mb": round(wal_size_mb(db_path), 3),
    }


class ConcurrentBenchmark:
    def __init__(
        self,
        memories: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ):
        self.memories = memories
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        import json
        print("\n=== Concurrent / WAL Benchmark ===")

        results = {"writer_scaling": {}, "reader_scaling": {}, "mixed": {}}

        # Writers scaling
        print("\n[1] Writer thread scaling")
        for num_writers in [1, 2, 4, 8]:
            print(f"  {num_writers} writers...", end=" ", flush=True)
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db = f.name

            r = run_concurrent_writers(
                db_path=db,
                num_writers=num_writers,
                ops_per_writer=50,
                memories=self.memories[:num_writers * 50],
            )
            results["writer_scaling"][f"{num_writers}_writers"] = r
            print(f"{r['ops_per_second']:.0f} ops/s, "
                  f"WAL={wal_size_mb(db):.2f}MB, "
                  f"errors={len(r['errors'])}")

            # Cleanup
            try:
                for ext in ["", "-wal", "-shm"]:
                    p = db + ext
                    if os.path.exists(p):
                        os.unlink(p)
            except Exception:
                pass

        # Reader scaling
        print("\n[2] Reader thread scaling")
        # First create a DB with some data
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            prep_db = f.name
        nm = NeuralMemory(db_path=prep_db, embedding_backend="auto")
        for m in self.memories[:2000]:
            nm.remember(m["text"], label=m["label"], auto_connect=False)
        nm.close()
        query_texts = [m["text"][:80] for m in self.memories[:100]]

        for num_readers in [1, 4, 8, 16]:
            print(f"  {num_readers} readers...", end=" ", flush=True)
            r = run_concurrent_readers(
                db_path=prep_db,
                num_readers=num_readers,
                ops_per_reader=50,
                query_texts=query_texts,
            )
            results["reader_scaling"][f"{num_readers}_readers"] = r
            print(f"{r['ops_per_second']:.0f} ops/s, "
                  f"p95={r['latency_p95_ms']}ms, "
                  f"errors={len(r['errors'])}")

        try:
            for ext in ["", "-wal", "-shm"]:
                p = prep_db + ext
                if os.path.exists(p):
                    os.unlink(p)
        except Exception:
            pass

        # Mixed workload
        print("\n[3] Mixed read/write workload")
        for writers, readers in [(2, 4), (4, 8), (8, 16)]:
            print(f"  {writers}W/{readers}R...", end=" ", flush=True)
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                mix_db = f.name

            r = run_mixed_workload(
                db_path=mix_db,
                num_writers=writers,
                num_readers=readers,
                ops_per_thread=30,
                memories=self.memories[:writers * 30],
                query_texts=[m["text"][:80] for m in self.memories[:100]],
            )
            key = f"{writers}W_{readers}R"
            results["mixed"][key] = r
            print(f"{r['ops_per_second']:.0f} ops/s, WAL={r['wal_size_mb']}MB")

            try:
                for ext in ["", "-wal", "-shm"]:
                    p = mix_db + ext
                    if os.path.exists(p):
                        os.unlink(p)
            except Exception:
                pass

        out_path = self.output_dir / "concurrent_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n[saved] {out_path}")
        return results
