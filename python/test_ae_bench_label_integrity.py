"""Ground-truth integrity for benchmarks/ae_domain_memory_bench/queries.py.

Locks two contracts:

  1. Every ground_truth_ids list contains only memory IDs that ACTUALLY
     exist in the substrate with non-empty content + present embedding.
     If a GT ID is silently deleted or never re-embedded, this test
     surfaces the drift before the next bench run inflates as a false MISS.

  2. The bench scored-query count never silently shrinks. Today's count is
     the floor; future expansions push it up. A drop means a label was
     accidentally cleared or a query removed.

The test SKIPS gracefully when the substrate file is absent (clean CI env).
This is the only way to keep the integrity check in the test suite without
forcing every dev env to materialize a 4GB substrate.
"""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "benchmarks" / "ae_domain_memory_bench"))

from queries import ALL_QUERIES  # noqa: E402

_SUBSTRATE_PATH = Path.home() / ".neural_memory" / "memory.db"

# Floor: queries with non-empty ground_truth_ids as of HEAD when this test
# was added. Any future label expansion increments this; any decrement
# means a label was lost.
_SCORED_QUERY_FLOOR = 57

# Cap: number of pairs of scored queries that may share an identical GT-set.
# These collisions exist by design — the same memory can legitimately be the
# answer to two queries asked from different lenses (e.g. a Lennar lot query
# and a customer-temporal query about that lot's contact). The cap is a
# CEILING: any new collision beyond this count means lazy labeling (someone
# reused an existing GT-set instead of designing a query with disjoint
# evidence). Raise this only with documented justification per new pair.
_DUPLICATE_GT_SET_PAIR_CAP = 18


class BenchLabelIntegrityTests(unittest.TestCase):
    def test_scored_query_count_meets_floor(self) -> None:
        """Total queries with ground_truth_ids never silently shrinks."""
        scored = [q for q in ALL_QUERIES if q["ground_truth_ids"]]
        self.assertGreaterEqual(
            len(scored), _SCORED_QUERY_FLOOR,
            f"Scored query count dropped below floor "
            f"({_SCORED_QUERY_FLOOR}). Found {len(scored)}. "
            f"A label was likely cleared or a query removed; verify before "
            f"lowering the floor.",
        )

    def test_no_query_id_is_duplicated(self) -> None:
        """Every query id is unique — no accidental copy/paste."""
        ids = [q["id"] for q in ALL_QUERIES]
        dupes = {qid for qid in ids if ids.count(qid) > 1}
        self.assertFalse(
            dupes, f"Duplicate query ids detected: {sorted(dupes)}",
        )

    def test_duplicate_ground_truth_set_pair_count_under_cap(self) -> None:
        """No new GT-set collisions beyond the documented design cap.

        Two queries sharing an identical sorted ground_truth_ids tuple is a
        smell: it usually means the second query was lazily labeled with the
        first query's GT instead of being designed against disjoint evidence.

        Some collisions are legitimate (cross-category lenses on the same
        memory). The cap captures the count at the time of the S6 expansion;
        new collisions must either retire an old one or push the cap up with
        a justifying comment.
        """
        scored = [q for q in ALL_QUERIES if q["ground_truth_ids"]]
        seen: dict[tuple[int, ...], str] = {}
        collisions: list[tuple[str, str, tuple[int, ...]]] = []
        for q in scored:
            key = tuple(sorted(q["ground_truth_ids"]))
            if key in seen:
                collisions.append((seen[key], q["id"], key))
            else:
                seen[key] = q["id"]
        self.assertLessEqual(
            len(collisions),
            _DUPLICATE_GT_SET_PAIR_CAP,
            f"Duplicate GT-set pair count {len(collisions)} exceeds cap "
            f"{_DUPLICATE_GT_SET_PAIR_CAP}. New collisions: "
            f"{collisions[_DUPLICATE_GT_SET_PAIR_CAP:]}. Either design "
            f"queries against disjoint memory evidence or raise the cap "
            f"with a documented justification.",
        )

    def test_every_ground_truth_id_exists_in_substrate(self) -> None:
        """Every GT memory id must exist in substrate with non-empty content
        and a present embedding. SKIPS if substrate file is absent."""
        if not _SUBSTRATE_PATH.exists():
            self.skipTest(
                f"Substrate not present at {_SUBSTRATE_PATH}; "
                f"GT integrity check requires live DB."
            )

        all_gt_ids: set[int] = set()
        for q in ALL_QUERIES:
            all_gt_ids.update(q["ground_truth_ids"])
        if not all_gt_ids:
            self.skipTest("No ground_truth_ids in any query — nothing to check.")

        conn = sqlite3.connect(
            f"file:{_SUBSTRATE_PATH}?mode=ro", uri=True, timeout=5,
        )
        try:
            id_list = ",".join(str(i) for i in sorted(all_gt_ids))
            rows = conn.execute(
                f"SELECT id, LENGTH(content), LENGTH(embedding) "
                f"FROM memories WHERE id IN ({id_list})"
            ).fetchall()
        finally:
            conn.close()

        present_ids = {row[0] for row in rows}
        missing = all_gt_ids - present_ids
        self.assertFalse(
            missing,
            f"Ground-truth memory IDs missing from substrate: {sorted(missing)}. "
            f"Every GT ID in queries.py must point to an existing memory.",
        )

        empty_content = [row[0] for row in rows if not row[1]]
        self.assertFalse(
            empty_content,
            f"Ground-truth memories with empty content: {sorted(empty_content)}. "
            f"Cross-encoder rerank cannot score empty content.",
        )

        no_embedding = [row[0] for row in rows if not row[2]]
        self.assertFalse(
            no_embedding,
            f"Ground-truth memories without embedding: {sorted(no_embedding)}. "
            f"Dense retrieval cannot surface these without an embedding.",
        )


if __name__ == "__main__":
    unittest.main()
