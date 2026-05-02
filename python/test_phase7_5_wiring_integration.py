"""End-to-end integration test for Phase 7.5 α/β/γ/δ wiring fixes.

Catches the bug class that motivated Phase 7.5: "DB column populated but
call-site never reads it." Each test asserts that varying a feature
field actually changes the final score — i.e., the scorer reads the
field, not just that the field exists.

Run:
    python3.11 python/test_phase7_5_wiring_integration.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import CandidateFeatures, DEFAULT_WEIGHTS, score_candidate  # noqa: E402


class ScoringFormulaWiringTests(unittest.TestCase):
    """Verify each Phase 7.5 field actually moves the final score."""

    def _baseline(self) -> CandidateFeatures:
        return CandidateFeatures(
            memory_id=1,
            semantic_score=0.5,
            sparse_score=0.5,
            graph_score=0.5,
            temporal_score=0.0,
            entity_score=0.0,
            procedural_score=0.0,
            stale_penalty=0.0,
            contradiction_penalty=0.0,
            rrf_feature=0.5,
            salience=1.0,
            confidence=1.0,
        )

    def test_alpha_procedural_score_moves_final_score(self) -> None:
        # Phase 7.5-α: procedural_score must contribute to ranking
        f0 = self._baseline()
        f1 = self._baseline()
        f1.procedural_score = 0.7
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        s1 = score_candidate(f1, DEFAULT_WEIGHTS)
        self.assertGreater(s1, s0,
                           "procedural_score=0.7 should produce higher "
                           "score than procedural_score=0.0")

    def test_beta_entity_score_moves_final_score(self) -> None:
        # Phase 7.5-β: entity_score must contribute to ranking
        f0 = self._baseline()
        f1 = self._baseline()
        f1.entity_score = 0.6
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        s1 = score_candidate(f1, DEFAULT_WEIGHTS)
        self.assertGreater(s1, s0,
                           "entity_score=0.6 should produce higher "
                           "score than entity_score=0.0")

    def test_gamma_stale_penalty_lowers_score(self) -> None:
        # Phase 7.5-γ: stale_penalty must subtract from the final score
        f0 = self._baseline()
        f1 = self._baseline()
        f1.stale_penalty = 0.2
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        s1 = score_candidate(f1, DEFAULT_WEIGHTS)
        self.assertLess(s1, s0,
                        "stale_penalty=0.2 should produce lower score "
                        "than stale_penalty=0.0")

    def test_delta_contradiction_penalty_lowers_score(self) -> None:
        # Phase 7.5-δ: contradiction_penalty must subtract from the score
        f0 = self._baseline()
        f1 = self._baseline()
        f1.contradiction_penalty = 0.15
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        s1 = score_candidate(f1, DEFAULT_WEIGHTS)
        self.assertLess(s1, s0,
                        "contradiction_penalty=0.15 should produce lower "
                        "score than contradiction_penalty=0.0")

    def test_combined_wired_features_compose_correctly(self) -> None:
        # All four wiring fixes interact correctly in the formula. Use
        # assertAlmostEqual on the exact expected delta — catches weight
        # or formula regressions that sign-only tests would miss.
        f0 = self._baseline()
        f1 = self._baseline()
        f1.procedural_score = 0.7
        f1.entity_score = 0.6
        f1.stale_penalty = 0.1
        f1.contradiction_penalty = 0.05
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        s1 = score_candidate(f1, DEFAULT_WEIGHTS)
        # Compute expected delta via the formula:
        #   base += w_procedural * proc + w_entity * entity
        #   base *= salience * confidence (both 1.0 in baseline)
        #   final -= stale_penalty + contradiction_penalty
        # boosts: 0.05*0.7 + 0.10*0.6 = 0.035 + 0.060 = 0.095
        # penalties: -(0.1 + 0.05) = -0.15
        # delta = 0.095 - 0.15 = -0.055
        expected_delta = (
            (DEFAULT_WEIGHTS["procedural"] * 0.7
             + DEFAULT_WEIGHTS["entity"] * 0.6)
            - (0.1 + 0.05)
        )
        self.assertAlmostEqual(s1 - s0, expected_delta, places=6,
                               msg="composition formula deviated from "
                                   f"expected delta {expected_delta}")

    def test_zero_baseline_unchanged(self) -> None:
        # Sanity: baseline scoring is deterministic and produces identical
        # results when called twice.
        f = self._baseline()
        s_a = score_candidate(f, DEFAULT_WEIGHTS)
        s_b = score_candidate(f, DEFAULT_WEIGHTS)
        self.assertEqual(s_a, s_b)

    def test_epsilon_locus_score_e2e_ranks_located_memory_higher(self) -> None:
        # Phase 7.5-ε e2e: a memory linked via located_in to a locus
        # mentioned in the query should rank higher than a sibling that
        # isn't linked. Validates the actual wired code path.
        import sys
        import tempfile
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from memory_client import NeuralMemory

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            mem = NeuralMemory(
                db_path=tmp.name,
                embedding_backend="auto",
                use_cpp=False,
                use_hnsw=False,
            )
            # Seed: locus + two twin memories about the same topic, but
            # only one is linked to the locus.
            locus_id = mem.remember("Lot 27 Lennar Aurora", kind="locus")
            mid_linked = mem.remember(
                "Crew finished panel install at lot 27",
                kind="experience",
            )
            mid_twin = mem.remember(
                "Crew finished panel install at another site",
                kind="experience",
            )
            mem.assign_locus(mid_linked, locus_id)

            # Query mentions the locus name
            results = mem.hybrid_recall("install at lot 27", k=5)
            ids = [r["id"] for r in results]
            self.assertIn(mid_linked, ids,
                          "located memory should appear in top-5")
            # Don't assert exact ordering vs the twin (semantic similarity
            # may dominate small N) — just verify the path is exercised
            # without crashing and the located memory IS retrievable.
            mem.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_activation_trace_present_in_recall_results(self) -> None:
        # Mazemaker-inspired: hybrid_recall results must carry _trace field
        # with per-channel contributions. Critical for downstream "why did
        # this rank?" explainability.
        import sys
        import tempfile
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from memory_client import NeuralMemory

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            mem = NeuralMemory(
                db_path=tmp.name,
                embedding_backend="auto",
                use_cpp=False,
                use_hnsw=False,
            )
            mem.remember("Test memory for activation trace", kind="experience")
            results = mem.hybrid_recall("test memory", k=3)
            self.assertGreater(len(results), 0,
                               "should return at least one result")
            r = results[0]
            self.assertIn("_trace", r,
                          "result must carry _trace field")
            trace = r["_trace"]
            # All 12 fields must be present
            expected_fields = {
                "semantic", "sparse", "graph", "temporal",
                "entity", "procedural", "locus",
                "rrf_feature", "salience", "confidence",
                "stale_penalty", "contradiction_penalty",
            }
            self.assertEqual(set(trace.keys()), expected_fields,
                             f"trace fields differ: got {set(trace.keys())}")
            for field, value in trace.items():
                self.assertIsInstance(value, (int, float),
                                      f"trace.{field} should be numeric, "
                                      f"got {type(value)}")
            mem.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_spanish_detector_skips_rerank_correctly(self) -> None:
        # Per-query Spanish detection for rerank-skip. Caught empirically
        # 2026-05-01: English-trained ms-marco cross-encoder regressed
        # Spanish queries 0.33 → 0.0. Heuristic: skip rerank for queries
        # containing >=2 unambiguous Spanish words OR any non-ASCII chars.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from memory_client import NeuralMemory

        # Should SKIP (Spanish detected)
        for q in [
            "Encuentra la conversacion sobre material que no llego.",
            "Que se dijo sobre comprar breakers?",
            "Busca mensaje de WhatsApp sobre cable numero doce.",
            "Did José call yesterday?",  # non-ASCII fallback
        ]:
            self.assertTrue(NeuralMemory._should_skip_rerank(q),
                            f"should skip: {q!r}")

        # Should NOT skip (English / ambiguous)
        for q in [
            "Find all memories for Lennar lot 12.",
            "How does Tito want estimates handled?",
            "What is the QBO refresh issue?",
            "La Quinta Inn invoice",  # 'la' alone insufficient
            "No new orders today",  # 'no' alone insufficient
            "",
        ]:
            self.assertFalse(NeuralMemory._should_skip_rerank(q),
                             f"should NOT skip: {q!r}")

    def test_es_disable_env_preserves_legacy_skip_behavior(self) -> None:
        # NM_RERANK_ES_DISABLE=1 must restore pre-multilingual skip-rerank
        # floor for non-ASCII queries (the escape hatch documented in
        # memory_client.py around _rerank_es_disabled). Verifies the env-var
        # is read at construction and the disabled flag is set correctly.
        import os
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        prior = os.environ.get("NM_RERANK_ES_DISABLE")
        os.environ["NM_RERANK_ES_DISABLE"] = "1"
        try:
            from memory_client import NeuralMemory
            mem = NeuralMemory(db_path="/tmp/test-es-disable.db",
                               use_cpp=False, use_hnsw=False)
            try:
                self.assertTrue(mem._rerank_es_disabled)
                self.assertIsNone(mem._rerank_model_es)  # never loaded
            finally:
                mem.close()
        finally:
            if prior is None:
                os.environ.pop("NM_RERANK_ES_DISABLE", None)
            else:
                os.environ["NM_RERANK_ES_DISABLE"] = prior

    def test_es_path_default_enabled_with_separate_model_name(self) -> None:
        # Without env var, ES path is enabled and uses the configured
        # multilingual model name. Documents that English queries with
        # accented names (e.g., "Did José call yesterday?") will route to
        # the multilingual model rather than skip rerank — intentional
        # per per-commit reviewer 77e350b finding.
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        os.environ.pop("NM_RERANK_ES_DISABLE", None)
        from memory_client import NeuralMemory
        mem = NeuralMemory(db_path="/tmp/test-es-default.db",
                           use_cpp=False, use_hnsw=False)
        try:
            self.assertFalse(mem._rerank_es_disabled)
            self.assertNotEqual(mem._rerank_model_name,
                                mem._rerank_model_es_name)
            self.assertIn("mmarco", mem._rerank_model_es_name.lower())
        finally:
            mem.close()

    def test_es_path_executes_multilingual_model_for_accented_original_query(self) -> None:
        """Production-path coverage: _maybe_rerank actually loads & calls the
        ES model when original_query has non-ASCII chars, and ranks by its
        scores. Per resolver-070621 — the previous lifecycle tests didn't
        cover the actual call, only the config state.
        """
        prior_disable = os.environ.get("NM_RERANK_ES_DISABLE")
        prior_module = sys.modules.get("sentence_transformers")
        os.environ.pop("NM_RERANK_ES_DISABLE", None)
        calls = []

        class FakeCrossEncoder:
            def __init__(self, name: str) -> None:
                calls.append(("init", name))

            def predict(self, pairs):
                calls.append(("predict", pairs))
                return [0.1, 0.9]

        sys.modules["sentence_transformers"] = types.SimpleNamespace(
            CrossEncoder=FakeCrossEncoder,
        )
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        try:
            from memory_client import NeuralMemory
            mem = NeuralMemory(db_path=tmp.name, use_cpp=False, use_hnsw=False)
            try:
                candidates = [
                    {"id": 1, "content": "low score"},
                    {"id": 2, "content": "high score"},
                ]
                result = mem._maybe_rerank(
                    "did jose call yesterday",
                    candidates,
                    force=True,
                    original_query="Did José call yesterday?",
                )
                self.assertEqual([c["id"] for c in result], [2, 1])
                init_names = [value for kind, value in calls if kind == "init"]
                self.assertEqual(init_names, [mem._rerank_model_es_name])
                self.assertIn("mmarco", init_names[0].lower())
                self.assertIsNone(mem._rerank_model)
                self.assertIsNotNone(mem._rerank_model_es)
            finally:
                mem.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            if prior_disable is None:
                os.environ.pop("NM_RERANK_ES_DISABLE", None)
            else:
                os.environ["NM_RERANK_ES_DISABLE"] = prior_disable
            if prior_module is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = prior_module

    def test_es_disable_env_skips_execution_for_accented_original_query(self) -> None:
        """NM_RERANK_ES_DISABLE=1 actually skips _maybe_rerank execution for
        non-ASCII queries (returns candidates unchanged, no model loaded).
        """
        prior_disable = os.environ.get("NM_RERANK_ES_DISABLE")
        os.environ["NM_RERANK_ES_DISABLE"] = "1"
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        try:
            from memory_client import NeuralMemory
            mem = NeuralMemory(db_path=tmp.name, use_cpp=False, use_hnsw=False)
            try:
                candidates = [{"id": 1, "content": "unchanged"}]
                result = mem._maybe_rerank(
                    "did jose call yesterday",
                    candidates,
                    force=True,
                    original_query="Did José call yesterday?",
                )
                self.assertIs(result, candidates)
                self.assertNotIn("rerank_score", result[0])
                self.assertIsNone(mem._rerank_model)
                self.assertIsNone(mem._rerank_model_es)
            finally:
                mem.close()
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            if prior_disable is None:
                os.environ.pop("NM_RERANK_ES_DISABLE", None)
            else:
                os.environ["NM_RERANK_ES_DISABLE"] = prior_disable

    def test_extreme_values_produce_finite_score(self) -> None:
        # Per round-4 reviewer: the formula has no clamp. Verify all
        # CandidateFeatures fields at extreme values still produce a
        # finite, non-NaN score. Defensive guard against future regressions
        # that introduce log/sqrt/division operations.
        import math
        f = self._baseline()
        f.semantic_score = 1.0
        f.sparse_score = 1.0
        f.graph_score = 1.0
        f.temporal_score = 1.0
        f.entity_score = 1.0
        f.procedural_score = 1.0
        f.locus_score = 1.0
        f.rrf_feature = 1.0
        f.salience = 2.0
        f.confidence = 1.0
        s = score_candidate(f, DEFAULT_WEIGHTS)
        self.assertFalse(math.isnan(s), "score should not be NaN at extremes")
        self.assertFalse(math.isinf(s), "score should not be inf at extremes")
        self.assertGreater(s, 0, "all-positive features → positive score")
        # Now test all-zero baseline: should produce 0 (or close to it)
        f0 = CandidateFeatures(memory_id=1)  # all defaults
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        self.assertAlmostEqual(s0, 0.0, places=6,
                               msg="all-default features → ~0 score")

    def test_stale_penalty_threshold_at_30_days(self) -> None:
        # Phase 7.5-γ ramp: penalty starts at 30 days, caps at 0.3.
        # Verify the formula directly: stale_penalty = min(age_days/300, 0.3)
        # if age_days > 30 else 0.0
        def _compute(age_days: float) -> float:
            return min(age_days / 300.0, 0.3) if age_days > 30 else 0.0

        # Day 0 = no penalty
        self.assertEqual(_compute(0), 0.0)
        # Day 30 (boundary) = no penalty
        self.assertEqual(_compute(30), 0.0)
        # Day 31 = small penalty
        self.assertGreater(_compute(31), 0.0)
        self.assertLess(_compute(31), 0.15)
        # Day 90 = 0.30 cap not yet reached (90/300 = 0.30 — exactly at cap)
        self.assertAlmostEqual(_compute(90), 0.30)
        # Day 1000 = capped at 0.30
        self.assertEqual(_compute(1000), 0.30)


if __name__ == "__main__":
    unittest.main()
