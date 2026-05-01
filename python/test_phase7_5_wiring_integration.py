"""End-to-end integration test for Phase 7.5 α/β/γ/δ wiring fixes.

Catches the bug class that motivated Phase 7.5: "DB column populated but
call-site never reads it." Each test asserts that varying a feature
field actually changes the final score — i.e., the scorer reads the
field, not just that the field exists.

Run:
    python3.11 python/test_phase7_5_wiring_integration.py
"""

from __future__ import annotations

import sys
import tempfile
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
        # All four wiring fixes interact correctly in the formula
        f0 = self._baseline()
        f1 = self._baseline()
        # Apply two boosts and two penalties
        f1.procedural_score = 0.7
        f1.entity_score = 0.6
        f1.stale_penalty = 0.1
        f1.contradiction_penalty = 0.05
        s0 = score_candidate(f0, DEFAULT_WEIGHTS)
        s1 = score_candidate(f1, DEFAULT_WEIGHTS)
        # Boosts (procedural 0.05*0.7 + entity 0.10*0.6 = 0.035 + 0.060 = 0.095)
        # outweigh penalties (0.10 + 0.05 = 0.15) — wait, penalties are
        # subtracted directly without weight multiplication. Let me recompute:
        # boosts: 0.05*0.7 + 0.10*0.6 = 0.035 + 0.060 = 0.095
        # boosts × salience × confidence (both 1.0) = 0.095
        # then minus stale 0.1 + contradiction 0.05 = -0.15
        # net delta = 0.095 - 0.15 = -0.055 (negative)
        # So s1 should be LESS than s0.
        self.assertLess(s1, s0,
                        "combined boosts (~0.095) exceeded by combined "
                        "penalties (~0.15) → net negative")

    def test_zero_baseline_unchanged(self) -> None:
        # Sanity: baseline scoring is deterministic and produces identical
        # results when called twice.
        f = self._baseline()
        s_a = score_candidate(f, DEFAULT_WEIGHTS)
        s_b = score_candidate(f, DEFAULT_WEIGHTS)
        self.assertEqual(s_a, s_b)


if __name__ == "__main__":
    unittest.main()
