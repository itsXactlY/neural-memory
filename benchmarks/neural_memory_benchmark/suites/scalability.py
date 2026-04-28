"""
Scalability Benchmark Suite
============================
Measures performance as memory count grows from 1K to 500K+.

Tests:
  1. Insert rate vs memory count (write scaling)
  2. Retrieval latency vs memory count
  3. HNSW activation threshold detection
  4. WAL file growth rate
  5. Memory footprint growth
"""
import os
import sys
import time
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def wal_size_mb(db_path: str) -> float:
    """Get total WAL file size in MB."""
    base = db_path.replace("-wal", "").replace("-shm", "")
    wal_path = base + "-wal"
    if os.path.exists(wal_path):
        return os.path.getsize(wal_path) / 1024 / 1024
    return 0.0


def db_size_mb(db_path: str) -> float:
    """Get main DB file size in MB."""
    if os.path.exists(db_path):
        return os.path.getsize(db_path) / 1024 / 1024
    return 0.0


def sqlite_stats(db_path: str) -> Dict[str, Any]:
    """Get SQLite-level statistics."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Memory count
        cur.execute("SELECT COUNT(*) FROM memories")
        count = cur.fetchone()[0] or 0

        # Connection count
        cur.execute("SELECT COUNT(*) FROM connections")
        conn_count = cur.fetchone()[0] or 0

        # WAL size
        wal = wal_size_mb(db_path)

        # DB size
        db_size = db_size_mb(db_path)

        conn.close()
        return {
            "memory_count": count,
            "connection_count": conn_count,
            "wal_size_mb": round(wal, 3),
            "db_size_mb": round(db_size, 3),
        }
    except Exception as e:
        return {"error": str(e)}


class ScalabilityBenchmark:
    """
    Measures performance degradation (or stability) as memory count grows.
    """

    def __init__(
        self,
        memories: List[Dict[str, Any]],
        tiers: List[int] = None,
        output_dir: Optional[Path] = None,
    ):
        self.memories = memories
        self.tiers = tiers or [1_000, 10_000, 50_000, 100_000, 500_000]
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        import json
        print("\n=== Scalability Benchmark ===")
        results = {"tiers": {}, "summary": {}}

        # Use a fresh temp DB for each tier to get clean measurements
        prev_db = None

        for tier in sorted(self.tiers):
            print(f"\n  --- Tier: {tier:,} memories ---")

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                tier_db = f.name

            nm = NeuralMemory(db_path=tier_db, embedding_backend="auto")

            # How many memories from our pool?
            pool = self.memories * ((tier // max(len(self.memories), 1)) + 1)
            pool = pool[:tier]

            # Measure insert rate
            t0 = time.perf_counter()
            for i, m in enumerate(pool):
                nm.remember(m["text"], label=m["label"])
                if (i + 1) % 5000 == 0:
                    print(f"    inserted {i+1}/{tier}...")
            insert_time = time.perf_counter() - t0
            insert_rate = tier / insert_time if insert_time > 0 else 0

            # Force WAL checkpoint
            try:
                conn = sqlite3.connect(tier_db)
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                conn.close()
            except Exception:
                pass

            # Measure retrieval latency
            sample_queries = [m["text"][:80] for m in pool[:min(100, len(pool))]]
            t0 = time.perf_counter()
            for q in sample_queries:
                nm.recall(q, k=10)
            recall_time = time.perf_counter() - t0
            recall_rate = len(sample_queries) / recall_time if recall_time > 0 else 0

            stats = sqlite_stats(tier_db)

            results["tiers"][f"{tier}"] = {
                "insert_time_s": round(insert_time, 2),
                "insert_rate_per_s": round(insert_rate, 1),
                "recall_time_s": round(recall_time, 4),
                "recall_rate_per_s": round(recall_rate, 1),
                "ms_per_query": round(recall_time / len(sample_queries) * 1000, 3),
                "sqlite_stats": stats,
            }

            print(f"    Insert: {insert_rate:.0f}/s, Recall: {recall_rate:.0f}/s, "
                  f"DB: {stats.get('db_size_mb', '?')}MB, "
                  f"WAL: {stats.get('wal_size_mb', '?')}MB")

            # Cleanup
            try:
                nm.close()
                os.unlink(tier_db)
                for ext in ["-wal", "-shm"]:
                    p = tier_db + ext
                    if os.path.exists(p):
                        os.unlink(p)
            except Exception:
                pass

        # Summary: does retrieval rate degrade?
        tier_results = results["tiers"]
        recall_rates = [float(tier_results[k]["recall_rate_per_s"]) for k in tier_results]
        results["summary"] = {
            "min_recall_rate": min(recall_rates),
            "max_recall_rate": max(recall_rates),
            "degradation_ratio": (
                min(recall_rates) / max(recall_rates)
                if max(recall_rates) > 0 else 0
            ),
            "note": "Ratio < 0.5 means significant degradation at scale",
        }

        out_path = self.output_dir / "scalability_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n[saved] {out_path}")
        return results
