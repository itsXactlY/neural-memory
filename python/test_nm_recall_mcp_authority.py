"""Tests for tools/nm_recall_mcp.py bench-authority helper.

Lockdown for S7 packet (2026-05-03): the MCP server module must read its
R@5 claim from the most recent ae-domain bench artifact rather than
hard-coding a stale literal. These tests verify both the helper and that
the stale ``"0.82"`` literal does not creep back into the prose.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
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


class StaleStringLockdownTests(unittest.TestCase):
    def test_no_static_082_string_in_module(self):
        """Defensive: the old hard-coded ``0.82`` claim must not return.

        Allows the literal ``0.82`` to appear inside this test file (we
        reference it in the docstring), but the production module source
        must be free of it.
        """
        src = Path(nm_recall_mcp.__file__).read_text(encoding="utf-8")
        # Match ``0.82`` only when surrounded by non-digit context, so
        # genuine non-stale floats like ``0.8200001`` would not trip it
        # (none exist today; this is a forward-looking guard).
        matches = re.findall(r"(?<!\d)0\.82(?!\d)", src)
        self.assertEqual(
            matches, [],
            f"stale '0.82' literal reappeared in nm_recall_mcp.py "
            f"({len(matches)} occurrence(s)) — must be derived live from "
            "bench artifact via _bench_authority().",
        )


if __name__ == "__main__":
    unittest.main()
