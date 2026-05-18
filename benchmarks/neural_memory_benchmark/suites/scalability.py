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

from memory_client import Mazemaker


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

            nm = Mazemaker(db_path=tier_db, embedding_backend="auto")

            # How many memories from our pool?
            # F25 fix (audit 2026-05-13): `self.memories * N` shallow-replicates
            # the same dict references, so the embedding cache hits on EVERY
            # duplicate — measuring cache speed, not real-world ingestion.
            # F26 fix: checkpoint WAL periodically during the loop so disk
            # usage stays bounded on 500k-row tiers.
            import copy as _copy
            base = self.memories or []
            repeats = (tier // max(len(base), 1)) + 1
            pool: List[Dict[str, Any]] = []
            for r in range(repeats):
                if len(pool) >= tier:
                    break
                for src in base:
                    if len(pool) >= tier:
                        break
                    mem = _copy.deepcopy(src)
                    # Suffix the text so identical inputs become distinct
                    # cache keys; embedding pipeline now sees unique work.
                    mem["text"] = f"{mem.get('text','')} [rep{r}]"
                    mem["id"] = f"{mem.get('id','mem')}-r{r}-{len(pool):07d}"
                    pool.append(mem)

            # Measure insert rate. Per PERF_STRATEGY.md, throughput suites
            # use auto_connect=False + detect_conflicts=False so we measure
            # bare-ingest cost. Graph-aware ingest is covered by graph.py.
            t0 = time.perf_counter()
            _checkpoint_every = 50_000
            for i, m in enumerate(pool):
                nm.remember(m["text"], label=m["label"],
                            auto_connect=False, detect_conflicts=False)
                if (i + 1) % 5000 == 0:
                    print(f"    inserted {i+1}/{tier}...")
                if (i + 1) % _checkpoint_every == 0:
                    try:
                        _conn = sqlite3.connect(tier_db)
                        _conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                        _conn.close()
                    except Exception:
                        pass
            insert_time = time.perf_counter() - t0
            insert_rate = tier / insert_time if insert_time > 0 else 0

            # Force WAL checkpoint
            try:
                conn = sqlite3.connect(tier_db)
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                conn.close()
            except Exception:
                pass

            # Measure retrieval latency.
            # F84 fix (audit 2026-05-13): the previous query sample was the
            # FIRST 100 entries of the pool, biasing toward most-recent
            # writes. Use a deterministic random sample across the entire
            # pool so latency reflects steady-state retrieval.
            import random as _random
            _qrng = _random.Random(self.cfg_seed if hasattr(self, "cfg_seed") else 42 + tier)
            _sample_idx = _qrng.sample(range(len(pool)), min(100, len(pool)))
            sample_queries = [pool[i]["text"][:80] for i in _sample_idx]
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

            # Cleanup — F85 fix (audit 2026-05-13): the previous block
            # wrapped close + unlink in one try/except, so a `close()`
            # failure caused `unlink()` to be skipped, leaking the temp
            # DB on disk. Each step now lives in its own except so
            # cleanup is best-effort but won't be short-circuited.
            try:
                nm.close()
            except Exception:
                pass
            for ext in ("", "-wal", "-shm"):
                p = tier_db + ext
                try:
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
