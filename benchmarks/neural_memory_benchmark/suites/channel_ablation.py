"""
Channel Ablation Benchmark
==========================
The skynet retrieval mode fuses semantic + BM25 + entity + temporal +
PPR via Reciprocal Rank Fusion. The current results don't tell us
which channels are pulling weight and which might be hurting. This
suite runs skynet with one channel zero-weighted at a time and
attributes the contribution to each.

It uses NeuralMemory's `channel_weights` parameter (Memory exposes it
as a constructor knob). For each channel C, we set channel_weights[C]
= 0 (effectively disabling C) and rebuild a Memory instance — same
data, same queries, same embedding model. Delta from skynet-with-all
quantifies C's contribution.

Output: per-channel recall@k and MRR delta. Negative deltas mean the
channel HURTS recall on this dataset — a real finding worth surfacing.
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


CHANNELS = ["semantic", "bm25", "entity", "temporal", "ppr", "salience"]


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
    return {
        "recall_at_k": round(hits / n, 4),
        "mrr": round(statistics.mean(rrs), 4) if rrs else 0.0,
        "p50_ms": round(sorted(latencies_ms)[len(latencies_ms) // 2], 3),
        "n": n,
    }


class ChannelAblationBenchmark:
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

    def _build(self, channel_weights: Optional[Dict[str, float]]) -> Memory:
        mem = Memory(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="skynet",
            channel_weights=channel_weights,
        )
        for m in self.memories:
            mem.remember(m["text"], label=m["label"], auto_connect=True)
        return mem

    def run(self) -> Dict[str, Any]:
        print("\n=== Channel Ablation Benchmark ===")
        # Default skynet (all channels at their natural weights).
        baseline_mem = self._build(channel_weights=None)
        baseline = _measure(baseline_mem, self.queries, self.k)
        print(f"  all channels   : R@{self.k}={baseline['recall_at_k']}  MRR={baseline['mrr']}")

        results: Dict[str, Any] = {"all_channels": baseline, "ablation": {}}

        # Default channel weights as documented in memory_client.NeuralMemory
        # — pulled from the public defaults so we know what 0-ing one does.
        default_weights = {
            "semantic":  1.0,
            "bm25":      0.55,
            "entity":    0.45,
            "temporal":  0.20,
            "ppr":       0.55,
            "salience":  0.25,
        }

        for ch in CHANNELS:
            # Each ablation gets its own DB so leftover state from a prior
            # build doesn't bias the result (in particular, HNSW indexes
            # warmed by an earlier run).
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                ablate_db = f.name
            self.db_path_orig = self.db_path
            self.db_path = ablate_db
            weights = dict(default_weights)
            weights[ch] = 0.0
            try:
                mem = self._build(channel_weights=weights)
                stats = _measure(mem, self.queries, self.k)
                stats["delta_recall"] = round(
                    stats["recall_at_k"] - baseline["recall_at_k"], 4
                )
                stats["delta_mrr"] = round(stats["mrr"] - baseline["mrr"], 4)
                results["ablation"][f"no_{ch}"] = stats
                print(f"  no_{ch:9s}: R@{self.k}={stats['recall_at_k']}  "
                      f"MRR={stats['mrr']}  Δrecall={stats['delta_recall']:+.4f}  "
                      f"Δmrr={stats['delta_mrr']:+.4f}")
            finally:
                self.db_path = self.db_path_orig
                try:
                    Path(ablate_db).unlink(missing_ok=True)
                except Exception:
                    pass

        # Identify which channels matter.
        contribs = {
            ch: -results["ablation"][f"no_{ch}"]["delta_recall"]
            for ch in CHANNELS
        }
        results["analysis"] = {
            "per_channel_contribution_to_recall": contribs,
            "most_helpful_channel": max(contribs, key=contribs.get),
            "most_harmful_channel": min(contribs, key=contribs.get),
            "interpretation": (
                "per_channel_contribution = -delta_recall when that channel "
                "is removed. Positive = removing it hurt recall (channel "
                "was helping). Negative = removing it improved recall "
                "(channel was hurting on this dataset)."
            ),
        }

        out = self.output_dir / "channel_ablation_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
