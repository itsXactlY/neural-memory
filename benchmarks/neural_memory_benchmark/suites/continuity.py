"""
Cross-Session Continuity Benchmark
===================================
The actual hermes-agent use case: a fact stored in session 1 must be
recallable in session N, even after the store has accumulated noise
from sessions 2..N-1. This is what *semantic memory* is supposed to
enable, and a vector-DB-with-extra-steps wouldn't necessarily nail it.

Setup:
  1. Store M_target unique paraphrase facts under label "session-1".
  2. For sessions 2..N-1, store K_noise UNRELATED paraphrase facts each
     under "session-i" labels.  These act as distractors — they share
     the corpus and grow the index without the test fact being among
     them.
  3. In "session N", issue the paraphrased question for each target
     fact and check whether the original session-1 memory is in top-k.

A pure recency-biased retriever would perform worst here (the target
is the OLDEST item in the store). A correct semantic-memory system
should still find it.

Output:
  recall_at_k by noise tier so you can see how cleanly the curve
  flattens vs how steeply it falls — flat curve = robust continuity.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory

try:
    from dataset_v2 import generate_continuity_pairs, ParaphraseGenerator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_v2 import generate_continuity_pairs, ParaphraseGenerator


class ContinuityBenchmark:
    def __init__(
        self,
        db_path: str,
        output_dir: Optional[Path] = None,
        target_facts: int = 50,
        noise_tiers: Optional[List[int]] = None,
        seed: int = 42,
        k: int = 5,
    ):
        self.db_path = db_path
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.target_facts = target_facts
        # Noise-per-tier — the cumulative noise after tier i is sum(noise_tiers[:i+1]).
        self.noise_tiers = noise_tiers or [0, 200, 1000, 5000]
        self.seed = seed
        self.k = k
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Cross-Session Continuity Benchmark ===")
        # Build target pairs (fact + paraphrased query) up front; each pair
        # has a globally unique anchor so we can check recall by anchor match.
        targets = generate_continuity_pairs(seed=self.seed, count=self.target_facts)

        nm = NeuralMemory(db_path=self.db_path, embedding_backend="auto")

        # Session 1: store all target facts.
        print(f"  [session 1] storing {len(targets)} target facts...")
        for t in targets:
            nm.remember(t["memory"]["text"], label="session-1", auto_connect=True)

        results: Dict[str, Any] = {"tiers": {}}
        noise_gen = ParaphraseGenerator(seed=self.seed + 1)
        cumulative_noise = 0

        for tier_idx, noise_count in enumerate(self.noise_tiers):
            tier_id = f"tier_{tier_idx}_noise_{cumulative_noise + noise_count}"
            # Add this tier's noise (skipped on tier 0 if noise_count == 0).
            if noise_count > 0:
                print(f"  [session {tier_idx + 2}] adding {noise_count} noise facts...")
                noise_mems, _ = noise_gen.generate(noise_count)
                for nm_dict in noise_mems:
                    nm.remember(nm_dict["text"], label=f"session-{tier_idx + 2}", auto_connect=True)
                cumulative_noise += noise_count

            # Session N: query each target.
            hits = 0
            rrs: List[float] = []
            for t in targets:
                results_list = nm.recall(t["query"], k=self.k)
                anchor = t["memory"]["metadata"]["anchor"]
                rank = 0
                for i, r in enumerate(results_list, 1):
                    if anchor in (r.get("content") or ""):
                        rank = i
                        break
                hits += 1 if rank > 0 else 0
                rrs.append(1.0 / rank if rank > 0 else 0.0)
            n = len(targets)
            tier_result = {
                "cumulative_noise": cumulative_noise,
                "recall_at_k": round(hits / n, 4),
                "mrr": round(statistics.mean(rrs), 4) if rrs else 0.0,
                "n_targets": n,
            }
            results["tiers"][tier_id] = tier_result
            print(f"    {tier_id}: recall@{self.k}={tier_result['recall_at_k']}  "
                  f"MRR={tier_result['mrr']}")

        # Compute the "drop curve" so a reader can see at a glance how
        # robust continuity is.
        recalls = [v["recall_at_k"] for v in results["tiers"].values()]
        results["analysis"] = {
            "recall_at_zero_noise": recalls[0] if recalls else 0.0,
            "recall_at_max_noise": recalls[-1] if recalls else 0.0,
            "absolute_drop": round((recalls[0] - recalls[-1]) if recalls else 0.0, 4),
            "relative_drop_pct": (
                round((recalls[0] - recalls[-1]) / recalls[0] * 100, 2)
                if recalls and recalls[0] > 0 else None
            ),
            "note": (
                "Lower drop = better cross-session continuity. A pure recency "
                "retriever would drop sharply; a semantic system should hold."
            ),
        }

        out = self.output_dir / "continuity_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
