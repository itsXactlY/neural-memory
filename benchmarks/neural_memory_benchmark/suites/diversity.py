"""
Diversity Benchmark Suite — MMR & score_floor sweep
====================================================
Audit-J added MMR (Maximal Marginal Relevance) reranking and a
score_floor cutoff to the public Memory.recall() API. This suite
demonstrates whether either knob actually improves result quality on
the paraphrase ground-truth set.

Two questions:
  1. Does MMR diversification *cost* recall@k, and by how much, in
     exchange for the diversity it produces? (We measure result-set
     entropy via topic distribution.)
  2. Does score_floor cleanly drop low-similarity noise without
     dropping good hits? (Recall@k stable, hit_rate at low confidence
     drops by exactly the cutoff fraction.)

NO ground-truth ids are revealed to the recall pipeline — the only
signal that lets us check correctness is matching the synthetic anchor
token in the result content.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import NeuralMemory


def _topic_of(label: str) -> str:
    # "paraphrase:metric" → "metric"; else label
    return label.split(":", 1)[-1] if ":" in label else label


class DiversityBenchmark:
    """Sweeps mmr_lambda and score_floor on a fixed query set."""

    MMR_LAMBDAS = [0.0, 0.3, 0.5, 0.7]
    SCORE_FLOORS = [0.0, 0.2, 0.4]

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
        self.nm: Optional[NeuralMemory] = None

    def setup(self) -> None:
        self.nm = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )
        for m in self.memories:
            self.nm.remember(m["text"], label=m["label"], auto_connect=True)

    def _evaluate(self, mmr_lambda: float, score_floor: float) -> Dict[str, Any]:
        hits = 0
        rrs: List[float] = []
        all_topics: List[str] = []
        result_counts: List[int] = []

        for q in self.queries:
            results = self.nm.recall(
                q["query"],
                k=self.k,
                mmr_lambda=mmr_lambda,
                score_floor=score_floor,
            )
            result_counts.append(len(results))
            anchor = q.get("anchor", "")
            rank = 0
            for i, r in enumerate(results, 1):
                if anchor and anchor in (r.get("content") or ""):
                    rank = i
                    break
            hits += 1 if rank > 0 else 0
            rrs.append(1.0 / rank if rank > 0 else 0.0)
            for r in results:
                all_topics.append(_topic_of(r.get("label", "") or ""))

        topic_dist = Counter(all_topics)
        n_topics = len(topic_dist)
        # Shannon entropy of the result-set topic distribution. Higher = more
        # diverse; if MMR works as advertised this should rise with lambda.
        total = sum(topic_dist.values()) or 1
        import math
        entropy = -sum((c / total) * math.log2(c / total)
                       for c in topic_dist.values() if c > 0)

        n = len(self.queries)
        return {
            "recall_at_k": round(hits / n, 4) if n else 0.0,
            "mrr": round(statistics.mean(rrs), 4) if rrs else 0.0,
            "mean_results_returned": round(statistics.mean(result_counts), 2),
            "topic_entropy_bits": round(entropy, 4),
            "topic_count": n_topics,
            "topic_distribution_top5": dict(topic_dist.most_common(5)),
        }

    def run(self) -> Dict[str, Any]:
        print("\n=== Diversity Benchmark (MMR / score_floor sweep) ===")
        self.setup()

        results: Dict[str, Any] = {"sweep": {}}
        for lam in self.MMR_LAMBDAS:
            for floor in self.SCORE_FLOORS:
                key = f"mmr={lam}_floor={floor}"
                r = self._evaluate(lam, floor)
                results["sweep"][key] = r
                print(f"  {key}: R@k={r['recall_at_k']}  "
                      f"MRR={r['mrr']}  entropy={r['topic_entropy_bits']}  "
                      f"avg_returned={r['mean_results_returned']}")

        # Summarise the trade-off so a reader can spot the operating point.
        results["analysis"] = self._analyse(results["sweep"])
        out = self.output_dir / "diversity_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results

    @staticmethod
    def _analyse(sweep: Dict[str, Any]) -> Dict[str, Any]:
        # Find the (mmr, floor) point that maximises recall, and the one that
        # maximises entropy without dropping recall by more than 5%.
        best_recall = max(sweep.values(), key=lambda v: v["recall_at_k"])
        best_recall_key = max(sweep, key=lambda k: sweep[k]["recall_at_k"])
        target = best_recall["recall_at_k"] * 0.95
        diversity_candidates = [
            (k, v) for k, v in sweep.items() if v["recall_at_k"] >= target
        ]
        best_div = max(diversity_candidates, key=lambda kv: kv[1]["topic_entropy_bits"]) if diversity_candidates else (best_recall_key, best_recall)
        return {
            "best_recall_setting": best_recall_key,
            "best_recall_value": best_recall["recall_at_k"],
            "best_diversity_within_5pct_recall": best_div[0],
            "best_diversity_entropy": best_div[1]["topic_entropy_bits"],
            "note": (
                "If best_diversity_within_5pct_recall != best_recall_setting, "
                "MMR is producing a real diversity/recall trade-off."
            ),
        }
