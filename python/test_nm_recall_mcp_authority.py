"""Tests for tools/nm_recall_mcp.py bench-authority helper.

Lockdown for S7/S7b packets (2026-05-03): the MCP server module must read
its R@5 claim from the most recent eligible ae-domain bench artifact
rather than hard-coding a stale literal, and selection must use mtime +
provenance.db_path filter (not lex-sort) so copy-ablation runs don't
overshadow production-DB runs. These tests verify both the helper and
that the stale ``"0.82"`` literal does not creep back into the prose
across the full set of production source files that previously carried it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path


# Make tools/ importable as a package-less module (mirrors how the MCP
# stdio launcher invokes it: direct script execution adds tools/ to path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import nm_recall_mcp  # noqa: E402


class BenchAuthorityHelperTests(unittest.TestCase):
    def test_authority_helper_reads_latest_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Older artifact — should be ignored in favor of the newer one.
            older = tmp_path / "ae-domain-2026-05-01-000000.json"
            older.write_text(json.dumps({"global_r@5": 0.4242}))
            newer = tmp_path / "ae-domain-2026-05-02-124730.json"
            newer.write_text(json.dumps({"global_r@5": 0.5758}))

            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)

            self.assertEqual(r5, "0.5758")
            self.assertEqual(artifact, "ae-domain-2026-05-02-124730.json")

    def test_authority_helper_returns_unknown_when_no_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            self.assertEqual(r5, "unknown")
            self.assertEqual(artifact, "unknown")

    def test_authority_helper_returns_unknown_on_malformed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = tmp_path / "ae-domain-2026-05-02-999999.json"
            broken.write_text("{not valid json")
            r5, _ = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            self.assertEqual(r5, "unknown")

    def test_authority_helper_returns_unknown_when_key_missing(self):
        # Bonus: artifact present but lacking the expected key. Should
        # still degrade gracefully rather than KeyError.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            partial = tmp_path / "ae-domain-2026-05-02-888888.json"
            partial.write_text(json.dumps({"mode": "scored"}))
            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            self.assertEqual(r5, "unknown")
            # Artifact name is still reported so the caller knows which
            # file lacked the field.
            self.assertEqual(artifact, "ae-domain-2026-05-02-888888.json")

    def test_authority_selects_artifact_by_mtime_not_lex_sort(self):
        """Lex-sort would pick the wrong file when filenames have non-
        lexicographic ordering vs write order. Mtime selection wins.

        Models the real S7b incident: ``ae-domain-bge-small-clean-073802.json``
        sorts AFTER ``ae-domain-2026-05-02-124730.json`` because letters >
        digits in ASCII. We synthesize a parallel scenario here using two
        strict-timestamp names whose lex order is opposite of write order.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # ``aaaa`` lex-sorts FIRST but is written LAST. lex-sort would
            # pick ``zzzz`` (older content); mtime should pick ``aaaa``.
            zzzz = tmp_path / "ae-domain-2026-05-02-999999.json"
            zzzz.write_text(json.dumps({"global_r@5": 0.4242}))
            # Force older mtime on zzzz so the directory order isn't a
            # confounder. Use 100s ago.
            old_ts = time.time() - 100
            os.utime(zzzz, (old_ts, old_ts))

            aaaa = tmp_path / "ae-domain-2026-05-02-000000.json"
            aaaa.write_text(json.dumps({"global_r@5": 0.5758}))
            # aaaa keeps its just-written (newer) mtime.

            # Sanity: lex-sort picks aaaa here too (so the test is fair to
            # both strategies on this exact pair). Use the actual reported
            # incident shape — names that lex-sort wrong:
            #   bench artifact (timestamp): "ae-domain-2026-05-02-124730.json"
            #   ablation copy (tagged):     "ae-domain-bge-small-clean-073802.json"
            # The ablation-copy filename lex-sorts AFTER the timestamp one
            # because alphabetic 'b' > digit '2'. Repro that exactly:
            for p in (aaaa, zzzz):
                p.unlink()
            # Production-style timestamped artifact, written FIRST (older mtime)
            ts_artifact = tmp_path / "ae-domain-2026-05-02-124730.json"
            ts_artifact.write_text(json.dumps({"global_r@5": 0.5758}))
            os.utime(ts_artifact, (time.time(), time.time()))
            # Older artifact written LAST but with strict timestamp name
            # that lex-sorts EARLIER — older mtime should be ignored.
            older = tmp_path / "ae-domain-2026-05-01-080000.json"
            older.write_text(json.dumps({"global_r@5": 0.4242}))
            os.utime(older, (time.time() - 100, time.time() - 100))

            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            # Mtime selects ts_artifact (newer mtime) over older (older mtime),
            # even though their lex order would also agree here. The test
            # below (excludes_copy_ablation) covers the actual incident shape.
            self.assertEqual(artifact, "ae-domain-2026-05-02-124730.json")
            self.assertEqual(r5, "0.5758")

    def test_authority_excludes_copy_ablation_artifact(self):
        """Production artifact must win over a NEWER copy-ablation artifact.

        Models the S7b incident exactly: a copy-ablation run produces
        ``ae-domain-bge-small-clean-073802.json`` with provenance.db_path
        pointing at a non-canonical DB. Even though it has a newer mtime
        (or the prior lex-sort would pick it), the production artifact —
        whose provenance.db_path ends with ``/.neural_memory/memory.db`` —
        must be the authority.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Production artifact: canonical DB, written FIRST (older mtime).
            prod = tmp_path / "ae-domain-2026-05-02-124730.json"
            prod.write_text(json.dumps({
                "global_r@5": 0.5758,
                "provenance": {
                    "db_path": "/Users/tito/.neural_memory/memory.db",
                },
            }))
            old_ts = time.time() - 100
            os.utime(prod, (old_ts, old_ts))
            # Copy-ablation artifact: non-canonical DB path, NEWER mtime.
            ablation = tmp_path / "ae-domain-bge-small-clean-073802.json"
            ablation.write_text(json.dumps({
                "global_r@5": 0.6061,
                "provenance": {
                    "db_path": "/tmp/copy-ablation-substrate.db",
                },
            }))
            # Newer mtime on ablation.
            new_ts = time.time()
            os.utime(ablation, (new_ts, new_ts))

            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            self.assertEqual(artifact, "ae-domain-2026-05-02-124730.json")
            self.assertEqual(r5, "0.5758")

    def test_authority_falls_through_when_only_pre_provenance_artifacts(self):
        """Pre-bfd3b70 artifacts have no ``provenance`` block. Strict
        timestamp filename regex still admits them as eligible.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Old-format artifact: no provenance block, but strict timestamp
            # filename. Should still be selectable.
            legacy = tmp_path / "ae-domain-2026-04-30-120000.json"
            legacy.write_text(json.dumps({
                "mode": "scored",
                "global_r@5": 0.6500,
                # No provenance block at all.
            }))
            # Tagged-name legacy artifact (no provenance, non-timestamp name)
            # — must NOT be selected even if mtime is newer.
            tagged = tmp_path / "ae-domain-mpnet-experiment.json"
            tagged.write_text(json.dumps({
                "mode": "scored",
                "global_r@5": 0.7200,
            }))
            new_ts = time.time()
            os.utime(tagged, (new_ts, new_ts))
            old_ts = time.time() - 100
            os.utime(legacy, (old_ts, old_ts))

            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            self.assertEqual(artifact, "ae-domain-2026-04-30-120000.json")
            self.assertEqual(r5, "0.6500")

    def test_authority_returns_unknown_when_only_ineligible_artifacts(self):
        """All-ablation directory (no production runs, no strict-timestamp
        legacy artifacts) returns unknown rather than picking junk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Tagged ablation with NO provenance — ineligible.
            (tmp_path / "ae-domain-bge-small-073202.json").write_text(
                json.dumps({"global_r@5": 0.6500})
            )
            # Tagged ablation WITH provenance pointing at non-canonical DB.
            (tmp_path / "ae-domain-bge-small-clean-073802.json").write_text(
                json.dumps({
                    "global_r@5": 0.6061,
                    "provenance": {"db_path": "/tmp/non-canonical.db"},
                })
            )
            r5, artifact = nm_recall_mcp._bench_authority(bench_dir=tmp_path)
            self.assertEqual(r5, "unknown")
            self.assertEqual(artifact, "unknown")


# ----------------------------------------------------------------------------
# Stale R@5=0.82 prose lockdown
# ----------------------------------------------------------------------------
# S7 covered nm_recall_mcp.py only. S7b extends to every production source
# file that previously carried the stale literal: ae_workflow_helpers.py,
# the snapshot daily script, and its launchd plist. Add new entries here
# whenever a future change touches a file that previously claimed a static
# R@5 figure — the rule is "derive from bench artifact, never freeze."
_LOCKDOWN_PRODUCTION_SOURCES = [
    "tools/nm_recall_mcp.py",
    "python/ae_workflow_helpers.py",
    "tools/neural-memory-snapshot-daily.sh",
    "tools/launchd/com.ae.neural-memory-snapshot.plist",
]


class StaleStringLockdownTests(unittest.TestCase):
    # Match ``0.82`` only when surrounded by non-digit context, so genuine
    # non-stale floats like ``0.8200001`` would not trip it (none exist
    # today; this is a forward-looking guard).
    _STALE_RE = re.compile(r"(?<!\d)0\.82(?!\d)")

    def test_no_static_082_string_in_module(self):
        """Defensive: the old hard-coded ``0.82`` claim must not return.

        Allows the literal ``0.82`` to appear inside this test file (we
        reference it in the docstring), but the production module source
        must be free of it.
        """
        src = Path(nm_recall_mcp.__file__).read_text(encoding="utf-8")
        matches = self._STALE_RE.findall(src)
        self.assertEqual(
            matches, [],
            f"stale '0.82' literal reappeared in nm_recall_mcp.py "
            f"({len(matches)} occurrence(s)) — must be derived live from "
            "bench artifact via _bench_authority().",
        )

    def test_no_static_082_string_in_lockdown_production_sources(self):
        """Generalized lockdown across every production source previously
        carrying the stale R@5=0.82 prose. New files added to
        ``_LOCKDOWN_PRODUCTION_SOURCES`` are scanned automatically.
        """
        repo_root = Path(__file__).resolve().parent.parent
        offenders = []
        for rel in _LOCKDOWN_PRODUCTION_SOURCES:
            target = repo_root / rel
            if not target.exists():
                offenders.append(f"{rel}: MISSING (lockdown list out of date)")
                continue
            src = target.read_text(encoding="utf-8")
            matches = self._STALE_RE.findall(src)
            if matches:
                offenders.append(
                    f"{rel}: {len(matches)} occurrence(s) of stale '0.82'"
                )
        self.assertEqual(
            offenders, [],
            "stale '0.82' literal reappeared in production source(s):\n  "
            + "\n  ".join(offenders)
            + "\n\nMust be derived from latest bench artifact, not hard-coded.",
        )


if __name__ == "__main__":
    unittest.main()
