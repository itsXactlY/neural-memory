"""
Conflict Detection Benchmark Suite
====================================
Tests conflict detection and memory supersession.

Tests:
  1. Conflict pair detection accuracy
  2. Supersession chain recording
  3. Recall quality after conflict resolution
  4. Salience decay over time / access patterns
  5. Poison memory resistance
"""
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import Mazemaker


class ConflictBenchmark:
    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        conflict_groups: int = 20,
    ):
        self.db_path = db_path
        self.memories = memories
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.conflict_groups = conflict_groups
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> Mazemaker:
        nm = Mazemaker(db_path=self.db_path, embedding_backend="auto")
        for m in self.memories:
            nm.remember(m["text"], label=m["label"], auto_connect=True)
        return nm

    def run(self) -> Dict[str, Any]:
        import json
        print("\n=== Conflict Detection Benchmark ===")

        results = {}

        # 1. Store conflicting memory pairs
        print("\n[1] Storing conflicting memory pairs...")
        nm = Mazemaker(db_path=self.db_path, embedding_backend="auto")

        conflict_pairs = [
            ("FastEmbed loads in under 1 second.", "FastEmbed loads in 4-5 seconds on average."),
            ("The Dream Engine runs NREM first.", "The Dream Engine runs REM before NREM."),
            ("HNSW activates at 5,000 memories.", "HNSW activates at 10,000 memories."),
            ("Salience decays at 0.95 per day.", "Salience decays at 0.5 per day."),
            ("WAL allows concurrent reads.", "WAL blocks reads during writes."),
            ("Postgres syncs one-way from SQLite.", "Postgres syncs bidirectionally."),
            ("BGE-M3 uses 1024 dimensions.", "BGE-M3 uses 768 dimensions."),
            ("PPR uses teleportation probability 0.15.", "PPR teleportation is 0.3."),
        ]

        for i, (orig, conflict) in enumerate(conflict_pairs):
            # Store original
            mid1 = nm.remember(orig, label="conflict_test", auto_connect=False)
            # Store conflict — should be detected
            mid2 = nm.remember(conflict, label="conflict_test", auto_connect=False)

        print(f"  Stored {len(conflict_pairs) * 2} conflicting memories")

        # 2. Check for supersession markers
        print("\n[2] Checking for supersession markers...")
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE '%[SUPERSEDED]%'"
        )
        superseded_count = cur.fetchone()[0] or 0
        conn.close()

        results["supersession"] = {
            "conflict_pairs_stored": len(conflict_pairs),
            "superseded_detected": superseded_count,
        }
        print(f"  Superseded markers found: {superseded_count}")

        # 3. Recall quality on conflicting vs non-conflicting
        print("\n[3] Recall quality comparison...")
        # Query original facts — should still find them
        nm._retrieval_mode = "semantic"
        test_queries = [
            ("FastEmbed", True),
            ("Dream Engine phases", True),
            ("HNSW threshold", True),
            ("something unrelated", False),
        ]

        quality_results = {}
        for query, should_match in test_queries:
            r = nm.recall(query, k=5)
            hits = len(r)
            quality_results[query] = {
                "hits": hits,
                "should_match": should_match,
                "correct": (hits > 0) == should_match,
            }

        results["recall_quality"] = quality_results

        # 4. Salience decay simulation
        print("\n[4] Salience decay simulation...")
        t0 = time.perf_counter()
        for i in range(100):
            nm.recall(random.choice(conflict_pairs)[0], k=3)
        access_time = time.perf_counter() - t0

        # Check if accessing memories changes their salience
        sample = nm.recall(".", k=20)
        saliences = [r.get("salience", r.get("score", 1.0)) for r in sample]

        results["salience"] = {
            "access_time_100_calls_s": round(access_time, 3),
            "sample_saliences": {
                "min": round(min(saliences), 3) if saliences else 0,
                "max": round(max(saliences), 3) if saliences else 0,
                "mean": round(sum(saliences)/len(saliences), 3) if saliences else 0,
            },
        }

        # 5. Adversarial poison resistance
        print("\n[5] Adversarial poison resistance...")
        poison_texts = [
            "X" * 5000,  # Very long
            "",  # Empty
            "   ",  # Whitespace
            "SELECT * FROM memories; DROP TABLE memories;",  # Injection
            "\x00\x01\x02",  # Binary
        ]
        for pt in poison_texts:
            try:
                nm.remember(pt, label="adversarial:poison", auto_connect=False)
            except Exception as e:
                print(f"  Rejected poison input: {e}")

        stats = nm.stats()
        results["adversarial"] = {
            "poison_attempts": len(poison_texts),
            "post_poison_stats": stats,
        }
        print(f"  Post-poison stats: {stats}")

        out_path = self.output_dir / "conflict_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n[saved] {out_path}")
        return results
