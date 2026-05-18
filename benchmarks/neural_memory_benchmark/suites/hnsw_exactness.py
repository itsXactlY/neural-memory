"""
HNSW Exactness Benchmark
========================
HNSW is an approximate nearest-neighbor index — it trades recall for
speed. The mazemaker-adapter activates HNSW automatically above
some threshold ("auto"). This suite quantifies the trade:

  At memory tiers 1k / 10k / 50k:
    * exact (use_hnsw=False) — semantic recall via full cosine over
      the memory matrix.  Ground truth for ANN recall.
    * HNSW (use_hnsw=True)   — approximate recall via hnswlib.

Isolation flags (BOTH arms)
---------------------------
To make HNSW vs exact the only changing variable, both arms construct
``Mazemaker`` with::

    use_cpp=False    # disable the C++ bridge / kNN engine
    rerank=False     # disable cross-encoder reranker

The exact arm additionally passes ``use_hnsw=False`` (kills the HNSW
codepath at the top of ``_ensure_hnsw``); the HNSW arm passes
``use_hnsw=True`` (NOT "auto") so it cannot silently fall through to the
brute-force path when a tier sits below the auto-activation threshold
(currently 1000 memories, see ``memory_client._ensure_hnsw``).

Activation assertions
---------------------
After ingest, each arm asserts the path it actually took:
  * exact: ``nm._hnsw_index is None`` — bail out the tier loudly if not.
  * HNSW:  triggers ``_ensure_hnsw()`` via a probe recall and checks
    ``nm._hnsw_index is not None``. If the index didn't build (hnswlib
    missing, dim-mismatch, etc.) the tier records
    ``hnsw_did_not_activate: true`` and skips the comparison rather
    than reporting a misleading overlap=1.0.

For each tier we measure:
  * recall_at_k_overlap — for each query, fraction of HNSW's top-k
    that are also in exact's top-k. 1.0 = HNSW is lossless. 0.0 = HNSW
    returned a completely different set.
  * latency speedup.

This is the only suite that tells you whether HNSW is hurting recall
quality at the cost of latency on YOUR data, or earning the latency
honestly.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import Mazemaker


def _build_and_query(
    use_hnsw: bool,
    memories: List[Dict[str, Any]],
    queries: List[str],
    k: int,
) -> Dict[str, Any]:
    """Build a clean Mazemaker with HNSW/exact isolated, ingest, and query.

    Returns a dict with timings, retrieved ids, and the actual retrieval
    path taken (``retrieval_path``). On the HNSW arm, if the index never
    activated, ``hnsw_did_not_activate=True`` is returned and the caller
    must skip the comparison.
    """
    arm = "hnsw" if use_hnsw else "exact"
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Both arms: kill C++ bridge and reranker so the only variable
        # between arms is the HNSW codepath itself. Pass use_hnsw as an
        # explicit bool (NOT "auto") so the HNSW arm can't silently
        # fall back to brute force at sub-threshold tiers.
        nm = Mazemaker(
            db_path=db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
            use_hnsw=use_hnsw,
            use_cpp=False,
            rerank=False,
        )

        # Belt-and-braces: even if the C++ bridge sneakily attached
        # itself, null it out so recall() takes pure-Python paths.
        #
        # F23 fix (audit 2026-05-13): the original blanket `nm._cpp = None`
        # also disabled any C++ helpers used outside kNN (SIMD cosine,
        # graph ops), giving a degraded "exact" baseline that wasn't
        # comparable to the HNSW arm. Switch to use_cpp=False at engine
        # construction (already done above) and only null _cpp if it
        # exposes a search method — the kNN search path is what we want
        # to suppress, not the auxiliary helpers.
        cpp = getattr(nm, "_cpp", None)
        if cpp is not None and hasattr(cpp, "search"):
            # Wrap rather than null so other helpers keep working.
            class _NoKnnCpp:
                def __getattr__(self, name):
                    if name == "search":
                        raise AttributeError("kNN disabled for exactness control")
                    return getattr(cpp, name)
            nm._cpp = _NoKnnCpp()

        # Sanity: confirm config flags landed where we think they did.
        assert nm._rerank is False, "rerank flag did not propagate"
        if not use_hnsw:
            assert nm._hnsw_enabled is False, (
                f"exact arm: expected _hnsw_enabled=False, got {nm._hnsw_enabled!r}"
            )

        t0 = time.perf_counter()
        for m in memories:
            nm.remember(m["text"], label=m["label"], auto_connect=False)
        setup_s = time.perf_counter() - t0

        # Probe recall to materialize the HNSW index (lazy via _ensure_hnsw)
        # before we start timing query latency. We discard the result.
        # F75 fix (audit 2026-05-13): `if queries:` does not protect against
        # an empty string at index 0 — guard explicitly.
        _probe = next((q for q in queries if isinstance(q, str) and q.strip()), None)
        if _probe is not None:
            try:
                nm.recall(_probe, k=k)
            except Exception:
                pass

        hnsw_did_not_activate = False
        if use_hnsw:
            # HNSW arm: index MUST be live now. If hnswlib is missing, or
            # dim filter rejected everything, _hnsw_index stays None.
            if nm._hnsw_index is None:
                hnsw_did_not_activate = True
                retrieval_path = "hnsw_did_not_activate"
            else:
                retrieval_path = (
                    f"hnsw_lib (index_size={len(nm._hnsw_ids)}, "
                    f"capacity={nm._hnsw_capacity})"
                )
        else:
            # Exact arm: index MUST be absent.
            if nm._hnsw_index is not None:
                raise RuntimeError(
                    "exact arm leaked into HNSW path: _hnsw_index is not None "
                    "despite use_hnsw=False — aborting tier to avoid a bogus "
                    "1.0 overlap."
                )
            retrieval_path = (
                f"exact_pythonic (cosine over ({len(nm._graph_nodes)},{nm.dim}) "
                "store matrix)"
            )

        latencies: List[float] = []
        # F24 fix (audit 2026-05-13): recall results can carry non-numeric
        # ids ("episodic-000000") under the synthetic dataset, so the
        # blanket int() coercion crashed. Use whatever id-like key the
        # backend exposes verbatim — overlap comparisons compare strings
        # just as well as ints.
        ids_per_query: List[List[Any]] = []
        if not hnsw_did_not_activate:
            for q in queries:
                t0 = time.perf_counter()
                results = nm.recall(q, k=k)
                latencies.append(time.perf_counter() - t0)
                ids_per_query.append([
                    r.get("id", r.get("memory_id", r.get("label")))
                    for r in results
                ])

        print(f"      [{arm}] retrieval_path: {retrieval_path}")

        return {
            "setup_s": round(setup_s, 2),
            "p50_ms": round(sorted(latencies)[len(latencies) // 2] * 1000, 3) if latencies else None,
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 3) if latencies else None,
            "ids_per_query": ids_per_query,
            "use_hnsw": use_hnsw,
            "retrieval_path": retrieval_path,
            "hnsw_did_not_activate": hnsw_did_not_activate,
        }
    finally:
        for ext in ("", "-wal", "-shm"):
            try:
                os.unlink(db_path + ext)
            except OSError:
                pass


class HNSWExactnessBenchmark:
    def __init__(
        self,
        memories: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        tiers: Optional[List[int]] = None,
        k: int = 10,
    ):
        self.memories = memories
        self.queries = queries
        self.tiers = tiers or [1_000, 10_000]
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.k = k
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== HNSW Exactness Benchmark ===")
        print("    isolation: use_cpp=False, rerank=False on both arms;")
        print("               exact arm use_hnsw=False, HNSW arm use_hnsw=True (forced, not auto)")
        results: Dict[str, Any] = {"tiers": {}, "k": self.k}
        # Pad memories by cycling — the original corpus may not reach 10k.
        pool = self.memories
        if len(pool) < max(self.tiers):
            repeats = (max(self.tiers) // max(len(pool), 1)) + 1
            scaled: List[Dict[str, Any]] = []
            for r in range(repeats):
                for i, m in enumerate(pool):
                    md = dict(m)
                    md["id"] = f"{md.get('id','m')}-rep{r}-{i}"
                    md["text"] = md["text"] + f" [rep{r}-{i}]"
                    scaled.append(md)
            pool = scaled[: max(self.tiers)]

        query_strings = [q["query"] for q in self.queries]

        for tier in sorted(self.tiers):
            print(f"\n  --- tier: {tier:,} memories ---")
            mems_tier = pool[:tier]

            print("    [exact] building...")
            exact = _build_and_query(False, mems_tier, query_strings, self.k)
            print("    [hnsw]  building...")
            ann = _build_and_query(True, mems_tier, query_strings, self.k)

            tier_result: Dict[str, Any] = {
                "exact": {
                    "setup_s": exact["setup_s"],
                    "p50_ms": exact["p50_ms"],
                    "p95_ms": exact["p95_ms"],
                    "retrieval_path": exact["retrieval_path"],
                },
                "hnsw": {
                    "setup_s": ann["setup_s"],
                    "p50_ms": ann["p50_ms"],
                    "p95_ms": ann["p95_ms"],
                    "retrieval_path": ann["retrieval_path"],
                },
            }

            if ann.get("hnsw_did_not_activate"):
                # Loud failure rather than silent overlap=1.0.
                tier_result["hnsw_did_not_activate"] = True
                tier_result["ann_recall_overlap"] = None
                tier_result["ann_recall_floor"] = None
                tier_result["speedup_p50"] = None
                results["tiers"][f"{tier}"] = tier_result
                print(f"    !! HNSW DID NOT ACTIVATE at tier {tier} — comparison skipped.")
                print(f"    !! (likely cause: hnswlib missing, or tier below auto threshold")
                print(f"    !!  even though use_hnsw=True was requested explicitly.)")
                continue

            # Per-query overlap of HNSW's top-k with exact's top-k.
            # F24 + F76 fix (audit 2026-05-13):
            #   * Ids may be strings — filter on truthiness, not `>= 0`.
            #   * Old metric `intersection / |exact|` was asymmetric: if
            #     `exact` returned only 2 results and `hnsw` returned 10
            #     wildly different ones, both could contain those 2 and
            #     score 1.0. Use Jaccard over the union, which penalises
            #     spurious extras AND missed hits symmetrically.
            overlaps: List[float] = []
            for ex_ids, ann_ids in zip(exact["ids_per_query"], ann["ids_per_query"]):
                ex_set = {i for i in ex_ids if i not in (None, "", -1)}
                ann_set = {i for i in ann_ids if i not in (None, "", -1)}
                union = ex_set | ann_set
                if not union:
                    continue
                overlaps.append(len(ex_set & ann_set) / len(union))

            speedup_p50 = (
                exact["p50_ms"] / ann["p50_ms"]
                if ann["p50_ms"] and exact["p50_ms"] is not None and ann["p50_ms"] > 0 else None
            )
            tier_result["ann_recall_overlap"] = round(statistics.mean(overlaps), 4) if overlaps else 0.0
            tier_result["ann_recall_floor"] = round(min(overlaps), 4) if overlaps else 0.0
            tier_result["speedup_p50"] = round(speedup_p50, 2) if speedup_p50 else None
            results["tiers"][f"{tier}"] = tier_result
            print(f"    overlap={tier_result['ann_recall_overlap']}  "
                  f"speedup_p50={tier_result['speedup_p50']}x  "
                  f"exact={exact['p50_ms']}ms hnsw={ann['p50_ms']}ms")

        out = self.output_dir / "hnsw_exactness_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  [saved] {out}")
        return results
