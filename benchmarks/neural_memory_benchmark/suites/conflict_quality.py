"""
Conflict Resolution Quality Benchmark
======================================
The previous conflict suite stored conflicting pairs and counted
`[SUPERSEDED]` markers. That tells you supersession FIRED but not
whether it was CORRECT — i.e. after the conflict, does recall return
the WINNING fact in rank 1, or is the loser still on top?

This suite uses paraphrase conflict triplets (anchor shared, fact
overwritten) and measures the rank-1 winner rate. It's the actual
quality of the supersession algorithm, not its mere presence.

Two flavours:
  * *strict*  — winner must be rank 1 (recall_top1 = winner_rate)
  * *relaxed* — winner anywhere in top-3 beats loser anywhere
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
    from dataset_v2 import generate_conflict_pairs
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_v2 import generate_conflict_pairs


def _rank_of(token: str, results: List[Dict[str, Any]]) -> int:
    for i, r in enumerate(results, 1):
        if token and token in (r.get("content") or ""):
            return i
    return 0


def _extract_winner_marker(replacement: str) -> str:
    # Use the most distinctive bigram from the replacement that doesn't
    # appear in the original. Best-effort heuristic — the explicit
    # marker is the unique 2-3 word fragment after the verb.
    # Falls back to the whole replacement string.
    return replacement


class ConflictQualityBenchmark:
    def __init__(
        self,
        db_path: str,
        output_dir: Optional[Path] = None,
        n_pairs: int = 30,
        seed: int = 42,
        k: int = 5,
    ):
        self.db_path = db_path
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.n_pairs = n_pairs
        self.seed = seed
        self.k = k
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        print("\n=== Conflict Resolution Quality Benchmark ===")
        pairs = generate_conflict_pairs(seed=self.seed, count=self.n_pairs)

        # Codex audit 2026-04-28: without a detect_conflicts=False control we
        # can't tell whether the winner-rate is supersession working or just
        # recency / vector similarity. Run the same scenario twice, once with
        # supersession enabled (default) and once with it forced off.
        results_per_arm: Dict[str, Dict[str, Any]] = {}

        for arm_label, dc_flag, db_path in (
            ("with_supersession", True, self.db_path),
            ("control_no_supersession", False, self.db_path + ".ctrl"),
        ):
            nm = NeuralMemory(db_path=db_path, embedding_backend="auto")
            # NeuralMemory.remember has a detect_conflicts kwarg; the control
            # arm forces it False so any winner-rate lift comes from
            # mechanisms other than supersession (recency boost, MERGE policy
            # in add_connection, etc.). Step 1: store originals. Step 2:
            # store replacements. Same insertion order both arms.
            for p in pairs:
                nm.remember(
                    p["original"], label=f"conflict:{p['topic']}",
                    auto_connect=True, detect_conflicts=dc_flag,
                )
            for p in pairs:
                nm.remember(
                    p["replacement"], label=f"conflict:{p['topic']}",
                    auto_connect=True, detect_conflicts=dc_flag,
                )
            results_per_arm[arm_label] = self._measure_one_arm(nm, pairs)
            print(f"  [{arm_label}]  winner@1={results_per_arm[arm_label]['winner_rank_1_rate']}  "
                  f"loser>winner={results_per_arm[arm_label]['loser_above_winner_rate']}")

        # Aggregate across arms — the "supersession lift" is the delta in
        # winner_rank_1_rate from control_no_supersession to with_supersession.
        ws = results_per_arm["with_supersession"]
        ctrl = results_per_arm["control_no_supersession"]
        out_results: Dict[str, Any] = {
            "with_supersession": ws,
            "control_no_supersession": ctrl,
            "supersession_lift": {
                "winner_rank_1": round(ws["winner_rank_1_rate"] - ctrl["winner_rank_1_rate"], 4),
                "loser_above_winner_drop": round(
                    ctrl["loser_above_winner_rate"] - ws["loser_above_winner_rate"], 4
                ),
            },
            "interpretation": {
                "supersession_lift.winner_rank_1": (
                    "Positive = supersession is responsible for getting the "
                    "winner to top-1. ~0 = the lift comes from recency / vector "
                    "similarity, not from the supersession algorithm itself."
                ),
                "supersession_lift.loser_above_winner_drop": (
                    "Positive = supersession suppresses the stale fact. ~0 = "
                    "the stale fact is just as likely to outrank the new one "
                    "either way."
                ),
            },
        }
        print(f"  supersession_lift: winner_rank_1={out_results['supersession_lift']['winner_rank_1']:+.4f}  "
              f"loser_drop={out_results['supersession_lift']['loser_above_winner_drop']:+.4f}")

        out = self.output_dir / "conflict_quality_results.json"
        out.write_text(json.dumps(out_results, indent=2))
        print(f"  [saved] {out}")
        return out_results

    def _measure_one_arm(self, nm: NeuralMemory, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        winner_top1 = 0
        winner_top3 = 0
        winner_anywhere = 0
        loser_above_winner = 0
        per_topic: Dict[str, List[int]] = {}

        for p in pairs:
            results = nm.recall(p["query"], k=self.k)
            winner_marker = self._unique_token(p["replacement"], p["original"])
            loser_marker = self._unique_token(p["original"], p["replacement"])

            wrank = _rank_of(winner_marker, results) if winner_marker else 0
            lrank = _rank_of(loser_marker, results) if loser_marker else 0

            if wrank == 1:
                winner_top1 += 1
            if 0 < wrank <= 3:
                winner_top3 += 1
            if wrank > 0:
                winner_anywhere += 1
            if lrank > 0 and (wrank == 0 or lrank < wrank):
                loser_above_winner += 1

            per_topic.setdefault(p["topic"], []).append(wrank)

        n = len(pairs)
        return {
            "n_conflict_pairs": n,
            "winner_rank_1_rate": round(winner_top1 / n, 4) if n else 0.0,
            "winner_in_top_3_rate": round(winner_top3 / n, 4) if n else 0.0,
            "winner_anywhere_rate": round(winner_anywhere / n, 4) if n else 0.0,
            "loser_above_winner_rate": round(loser_above_winner / n, 4) if n else 0.0,
            "per_topic_mean_winner_rank": {
                t: round(statistics.mean([r for r in ranks if r > 0]), 2) if any(r > 0 for r in ranks) else None
                for t, ranks in per_topic.items()
            },
        }

    @staticmethod
    def _unique_token(target: str, other: str) -> str:
        """Return a multi-word substring of `target` that doesn't appear in `other`.

        Used to distinguish the winner from the loser when both share the
        anchor and most of the surface form. Falls back to the longest
        differing word.
        """
        # Try a 3-word sliding window
        t_words = target.split()
        o_lower = other.lower()
        for n in (4, 3, 2, 1):
            for i in range(len(t_words) - n + 1):
                ng = " ".join(t_words[i : i + n]).strip(".,;:")
                if ng and ng.lower() not in o_lower and len(ng) >= 4:
                    return ng
        # Fallback: any single word in target not in other
        for w in t_words:
            wc = w.strip(".,;:").lower()
            if wc and wc not in o_lower and len(wc) >= 4:
                return w.strip(".,;:")
        return ""
