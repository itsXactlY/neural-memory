"""
Continuity Controls Benchmark
=============================
Codex audit 2026-04-28 v2 flagged that the previous version of this suite was
still anchor-based retrieval: each query mentioned the unique coined anchor
token from its target ("Who runs zriev38?"), so raw cosine got 1.0 / 0.98 /
0.9 / 0.7 vs mazemaker 0.76 — we were measuring rare-token overlap, not
cross-session memory.

Rewrite: queries now ask about the *concept* the target memory was about
("Which component handles geo-redundancy spending?") and never mention the
anchor token. At each noise tier > 0 we ALSO inject 1-2 near-distractors per
target — noise memories that share the concept vocabulary with the query but
have a *different* anchor. Raw cosine should be lured into ranking the
distractor above the original target.

Three retrievers, same data, same noise schedule:

  1. mazemaker-adapter (Memory.recall) — graph + salience + dream
  2. raw cosine (numpy, same embedder) — pure embedding nearest-neighbour
  3. recency baseline — return tail; targets are the OLDEST item, so this
     should collapse to ~0 immediately. Sanity floor.

For each retriever we sweep tiers 0 / 200 / 1000 / 5000 of generic noise plus
the concept-shaped distractors injected per target, and report recall@k +
MRR. Ground truth is the target's anchor token (still in the target text,
just absent from the query). Hits are evaluated by anchor-in-content.

Expected curve shape:
  * recency_only -> ~0 across all tiers (oldest-wins is wrong here).
  * raw_cosine drops as distractor count grows — that drop is the whole
    point, the previous suite did not have one.
  * neural_memory may or may not beat raw cosine; if it doesn't, that's an
    honest negative result and the JSON output flags it.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from mazemaker import Memory
from embed_provider import EmbeddingProvider

try:
    from dataset_v2 import (
        generate_concept_continuity_pairs,
        ParaphraseGenerator,
        _GLOBAL_ANCHORS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_v2 import (
        generate_concept_continuity_pairs,
        ParaphraseGenerator,
        _GLOBAL_ANCHORS,
    )


class _RawCosineStore:
    """In-process cosine-only baseline. Everything is in memory; no DB.

    F29 fix (audit 2026-05-13): the previous implementation called
    np.vstack on every add(), copying the full matrix each time —
    O(N²) for 5k+ entries. Use an append-only Python list of rows and
    lazily stack into an ndarray on first recall (or when growth has
    been amortised). Result: 5k adds now take ~0.4s instead of >120s.
    """

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self.texts: List[str] = []
        self.anchors: List[str] = []
        self._rows: List[np.ndarray] = []  # append-only; coalesced on demand
        self._mat: Optional[np.ndarray] = None  # cached coalesced matrix

    def add(self, text: str, anchor: str = "") -> None:
        self.texts.append(text)
        self.anchors.append(anchor)
        v = np.asarray(self.embedder.embed(text), dtype=np.float32)
        v /= np.linalg.norm(v).clip(min=1e-12)
        self._rows.append(v)
        # Invalidate the cached matrix — recompute on next recall().
        self._mat = None

    def _materialise(self) -> Optional[np.ndarray]:
        if not self._rows:
            return None
        if self._mat is None or self._mat.shape[0] != len(self._rows):
            self._mat = np.stack(self._rows, axis=0)
        return self._mat

    @property
    def mat(self):  # backwards-compat with callers that read .mat directly
        return self._materialise()

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        mat = self._materialise()
        if mat is None or len(self.texts) == 0:
            return []
        q = np.asarray(self.embedder.embed(query), dtype=np.float32)
        q /= np.linalg.norm(q).clip(min=1e-12)
        sims = mat @ q
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            {"content": self.texts[int(i)], "anchor": self.anchors[int(i)],
             "similarity": float(sims[int(i)])}
            for i in idx
        ]


class _RecencyStore:
    """Returns the K most-recently-added memories regardless of query."""

    def __init__(self):
        self.records: deque = deque()

    def add(self, text: str, anchor: str = "") -> None:
        self.records.append({"content": text, "anchor": anchor})

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        return list(self.records)[-k:][::-1]


def _eval(recall_fn, queries: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    """Recall@k + MRR. A query is hit iff the target's unique anchor token
    appears in one of the top-k results' content. Anchors are coined and
    appear in exactly one stored memory (the target), so this is unambiguous.
    """
    hits = 0
    rrs: List[float] = []
    errs = 0
    for q in queries:
        try:
            results = recall_fn(q["query"], k=k)
        except Exception:
            errs += 1
            results = []
        anchor = q.get("anchor", "")
        rank = 0
        # Use word-boundary match so e.g. anchor "zriev38" doesn't accidentally
        # match a substring inside another coined token from the noise pool.
        if anchor:
            pat = re.compile(r"\b" + re.escape(anchor) + r"\b")
            for i, r in enumerate(results, 1):
                content = r.get("content") or ""
                if pat.search(content):
                    rank = i
                    break
        hits += 1 if rank > 0 else 0
        rrs.append(1.0 / rank if rank > 0 else 0.0)
    n = len(queries) or 1
    return {
        "recall_at_k": round(hits / n, 4),
        "mrr": round(statistics.mean(rrs), 4) if rrs else 0.0,
        "errors": errs,
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
        distractors_per_target_per_tier: int = 2,
    ):
        self.db_path = db_path
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.target_facts = target_facts
        self.noise_tiers = noise_tiers or [0, 200, 1000, 5000]
        self.seed = seed
        self.k = k
        # How many concept-shaped near-distractors to inject per target at
        # each noise tier > 0. Two is enough to reliably out-rank the target
        # under raw cosine; more would just compound the effect.
        self.distractors_per_target_per_tier = distractors_per_target_per_tier
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Continuity Controls Benchmark (concept-mode) ===")
        # Reset the global anchor registry so this run's anchors don't
        # collide with anything a prior suite minted.
        _GLOBAL_ANCHORS.clear()

        targets = generate_concept_continuity_pairs(
            seed=self.seed, count=self.target_facts
        )
        target_queries = [
            {"query": t["query"], "anchor": t["target_anchor"]}
            for t in targets
        ]

        # Three retrievers — same data goes into each, in the same order.
        embedder = EmbeddingProvider(backend="auto")
        nm = Memory(db_path=self.db_path, embedding_backend="auto")
        raw = _RawCosineStore(embedder)
        recency = _RecencyStore()

        # Session 1: store target facts in all three.
        print(f"  [session 1] storing {len(targets)} target facts (concept-mode)...")
        for t in targets:
            text = t["target_text"]
            anchor = t["target_anchor"]
            nm.remember(text, label="session-1", auto_connect=True)
            raw.add(text, anchor=anchor)
            recency.add(text, anchor=anchor)

        # The distractor anchor generator gets its own seed-offset so its
        # coined tokens never collide with target anchors (also enforced via
        # _GLOBAL_ANCHORS lock inside ParaphraseGenerator).
        distractor_anchor_gen = ParaphraseGenerator(seed=self.seed + 7)
        distractor_rng = random.Random(self.seed + 13)

        results: Dict[str, Any] = {"tiers": {}}
        noise_gen = ParaphraseGenerator(seed=self.seed + 1)
        cumulative_generic = 0
        cumulative_distractors = 0

        for tier_i, count in enumerate(self.noise_tiers):
            session_label = f"session-{tier_i+2}"

            if count > 0:
                # Per-target near-distractors — the adversarial part of this
                # benchmark. For each original target, pick N templates from
                # its concept's distractor pool, fill with a FRESH coined
                # anchor, and inject. These are semantically closer to the
                # target's query than the target itself, so raw cosine should
                # rank them above the target.
                injected = 0
                for t in targets:
                    templates = t["near_distractor_templates"]
                    if not templates:
                        continue
                    chosen = distractor_rng.sample(
                        templates,
                        k=min(self.distractors_per_target_per_tier, len(templates)),
                    )
                    for tmpl in chosen:
                        d_anchor = distractor_anchor_gen._fresh_anchor()
                        d_text = tmpl.format(anchor=d_anchor)
                        nm.remember(d_text, label=session_label, auto_connect=True)
                        raw.add(d_text, anchor=d_anchor)
                        recency.add(d_text, anchor=d_anchor)
                        injected += 1
                cumulative_distractors += injected
                print(f"  [{session_label}] injected {injected} concept-near-distractors "
                      f"({self.distractors_per_target_per_tier}/target)")

                # Generic paraphrase noise — unrelated topics, just bulk.
                print(f"  [{session_label}] adding {count} generic noise...")
                noise_mems, _ = noise_gen.generate(count)
                for nm_dict in noise_mems:
                    nm.remember(nm_dict["text"], label=session_label, auto_connect=True)
                    raw.add(nm_dict["text"], anchor=nm_dict["metadata"]["anchor"])
                    recency.add(nm_dict["text"], anchor=nm_dict["metadata"]["anchor"])
                cumulative_generic += count

            tier_id = f"tier_{tier_i}_noise_{cumulative_generic}_dist_{cumulative_distractors}"
            tier = {
                "cumulative_generic_noise": cumulative_generic,
                "cumulative_near_distractors": cumulative_distractors,
                "neural_memory": _eval(lambda q, k=5: nm.recall(q, k=k), target_queries, self.k),
                "raw_cosine":   _eval(lambda q, k=5: raw.recall(q, k=k), target_queries, self.k),
                "recency_only": _eval(lambda q, k=5: recency.recall(q, k=k), target_queries, self.k),
            }
            results["tiers"][tier_id] = tier
            print(f"    {tier_id}:  nm={tier['neural_memory']['recall_at_k']}  "
                  f"raw={tier['raw_cosine']['recall_at_k']}  "
                  f"recency={tier['recency_only']['recall_at_k']}")

        # Curve analysis.
        nm_curve = [v["neural_memory"]["recall_at_k"] for v in results["tiers"].values()]
        raw_curve = [v["raw_cosine"]["recall_at_k"] for v in results["tiers"].values()]
        recency_curve = [v["recency_only"]["recall_at_k"] for v in results["tiers"].values()]

        # Did raw cosine actually drop with added distractors? If not, the
        # adversarial design failed and the result is uninterpretable.
        raw_drop = round(raw_curve[0] - raw_curve[-1], 4) if raw_curve else 0.0

        results["analysis"] = {
            "nm_curve": nm_curve,
            "raw_cosine_curve": raw_curve,
            "recency_curve": recency_curve,
            "raw_cosine_drop_first_to_last": raw_drop,
            "nm_vs_raw_lift_at_max_noise": round(nm_curve[-1] - raw_curve[-1], 4) if nm_curve else 0,
            "recency_floor_at_max_noise": recency_curve[-1] if recency_curve else 0,
            "interpretation": (
                "Concept-mode continuity. Queries paraphrase the target's concept "
                "and never mention its anchor; near-distractors per noise tier "
                "share concept vocabulary with the query but use different anchors. "
                "Expected: raw_cosine_drop_first_to_last > 0 (distractors lure raw "
                "cosine away from the target). recency_curve should hover near 0 "
                "since targets are the OLDEST items. nm_vs_raw_lift_at_max_noise > 0 "
                "would indicate mazemaker uses graph/salience to hold the "
                "original target above semantically-closer distractors. <= 0 is an "
                "honest negative result on this adversarial task."
            ),
        }

        out = self.output_dir / "continuity_controls_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [saved] {out}")
        return results
