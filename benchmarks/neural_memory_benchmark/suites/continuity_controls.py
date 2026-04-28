"""
Continuity Controls Benchmark
=============================
Codex audit 2026-04-28 flagged that continuity.py measures absolute
recall under noise but offers no point of comparison. Without controls
we can't tell whether a 30%/56%/46% recall curve is "good for a
semantic memory system" or "trivially what any vector DB would do".

This suite runs the SAME continuity scenario against three retrievers:

  1. neural-memory-adapter (Memory.recall)
  2. raw cosine (numpy, same embedder) — "would a 50-line vector store
     do this?"
  3. recency baseline — return the K most-recently-stored memories,
     regardless of query. This is the dumb pathological control —
     anything that loses to recency is broken.

For each retriever, sweep the same noise tiers (0 / 200 / 1000 / 5000)
and report recall@k + MRR. The shape of the curve matters more than
absolute numbers — neural-memory should at minimum hold steady or
degrade gracefully where recency-only collapses.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from neural_memory import Memory
from embed_provider import EmbeddingProvider

try:
    from dataset_v2 import generate_continuity_pairs, ParaphraseGenerator, _GLOBAL_ANCHORS
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_v2 import generate_continuity_pairs, ParaphraseGenerator, _GLOBAL_ANCHORS


class _RawCosineStore:
    """In-process cosine-only baseline. Everything is in memory; no DB."""

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self.texts: List[str] = []
        self.anchors: List[str] = []
        self.mat = None  # (N, dim) row-normalised

    def add(self, text: str, anchor: str = "") -> None:
        self.texts.append(text)
        self.anchors.append(anchor)
        v = np.asarray(self.embedder.embed(text), dtype=np.float32)
        v /= np.linalg.norm(v).clip(min=1e-12)
        self.mat = v[None] if self.mat is None else np.vstack([self.mat, v])

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.mat is None or len(self.texts) == 0:
            return []
        q = np.asarray(self.embedder.embed(query), dtype=np.float32)
        q /= np.linalg.norm(q).clip(min=1e-12)
        sims = self.mat @ q
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            {"content": self.texts[int(i)], "anchor": self.anchors[int(i)],
             "similarity": float(sims[int(i)])}
            for i in idx
        ]


class _RecencyStore:
    """Returns the K most-recently-added memories regardless of query.

    The pathological control — anything beating "newest is best" must
    actually be doing semantic work, not just exploiting insertion order.
    """

    def __init__(self):
        self.records: deque = deque()

    def add(self, text: str, anchor: str = "") -> None:
        self.records.append({"content": text, "anchor": anchor})

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Return tail (most recent) — but we want OLDEST to fail this
        # particular task (target was stored earliest). Tail-first naturally
        # demonstrates that.
        if not self.records:
            return []
        return list(self.records)[-k:][::-1]


def _eval(recall_fn, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    hits = 0
    rrs: List[float] = []
    for q in queries:
        results = recall_fn(q["query"], k=k)
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
    }


class ContinuityControlsBenchmark:
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
        self.noise_tiers = noise_tiers or [0, 200, 1000, 5000]
        self.seed = seed
        self.k = k
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Continuity Controls Benchmark ===")
        # Reset the global anchor registry so this run's anchors don't
        # collide with anything a prior suite minted (codex 2026-04-28 fix).
        _GLOBAL_ANCHORS.clear()

        targets = generate_continuity_pairs(seed=self.seed, count=self.target_facts)
        target_queries = [
            {"query": t["query"], "anchor": t["memory"]["metadata"]["anchor"]}
            for t in targets
        ]

        # Three retrievers — same data goes into each, in the same order.
        embedder = EmbeddingProvider(backend="auto")
        nm = Memory(db_path=self.db_path, embedding_backend="auto")
        raw = _RawCosineStore(embedder)
        recency = _RecencyStore()

        # Session 1: store target facts in all three.
        print(f"  [session 1] storing {len(targets)} target facts...")
        for t in targets:
            text = t["memory"]["text"]
            anchor = t["memory"]["metadata"]["anchor"]
            nm.remember(text, label="session-1", auto_connect=True)
            raw.add(text, anchor=anchor)
            recency.add(text, anchor=anchor)

        results: Dict[str, Any] = {"tiers": {}}
        noise_gen = ParaphraseGenerator(seed=self.seed + 1)
        cumulative = 0

        for tier_i, count in enumerate(self.noise_tiers):
            if count > 0:
                print(f"  [session {tier_i+2}] adding {count} noise...")
                noise_mems, _ = noise_gen.generate(count)
                for nm_dict in noise_mems:
                    nm.remember(nm_dict["text"], label=f"session-{tier_i+2}", auto_connect=True)
                    raw.add(nm_dict["text"], anchor=nm_dict["metadata"]["anchor"])
                    recency.add(nm_dict["text"], anchor=nm_dict["metadata"]["anchor"])
                cumulative += count

            tier_id = f"tier_{tier_i}_noise_{cumulative}"
            tier = {
                "cumulative_noise": cumulative,
                "neural_memory": _eval(lambda q, k=5: nm.recall(q, k=k), target_queries, self.k),
                "raw_cosine":   _eval(lambda q, k=5: raw.recall(q, k=k), target_queries, self.k),
                "recency_only": _eval(lambda q, k=5: recency.recall(q, k=k), target_queries, self.k),
            }
            results["tiers"][tier_id] = tier
            print(f"    {tier_id}:  nm={tier['neural_memory']['recall_at_k']}  "
                  f"raw={tier['raw_cosine']['recall_at_k']}  "
                  f"recency={tier['recency_only']['recall_at_k']}")

        # Curve analysis — does NM hold up better than raw cosine, and does
        # recency-only collapse as expected?
        nm_curve = [v["neural_memory"]["recall_at_k"] for v in results["tiers"].values()]
        raw_curve = [v["raw_cosine"]["recall_at_k"] for v in results["tiers"].values()]
        recency_curve = [v["recency_only"]["recall_at_k"] for v in results["tiers"].values()]

        results["analysis"] = {
            "nm_curve": nm_curve,
            "raw_cosine_curve": raw_curve,
            "recency_curve": recency_curve,
            "nm_vs_raw_lift_at_max_noise": round(nm_curve[-1] - raw_curve[-1], 4) if nm_curve else 0,
            "recency_floor_at_max_noise": recency_curve[-1] if recency_curve else 0,
            "interpretation": (
                "If nm_vs_raw_lift_at_max_noise <= 0, neural-memory does NOT "
                "outperform a 50-line vector store on the cross-session task. "
                "If recency_floor_at_max_noise > nm_curve[-1], the system loses "
                "to a pathological 'newest wins' baseline."
            ),
        }

        out = self.output_dir / "continuity_controls_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
