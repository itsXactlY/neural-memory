"""Tests for embed_provider dim-mismatch guard + reembed_substrate canonical-DB
refusal. Per codex-prescriptive-redesigner Day 2 item: lock the
`bcb480f → 64da6ae` guard against future regressions.

The guard prevents NM_EMBED_MODEL from silently writing wrong-dim vectors
into a substrate whose readers assume DIMENSION (384).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))


class EmbedProviderDimGuardTests(unittest.TestCase):
    """SentenceTransformerBackend must fail-fast on non-DIMENSION models."""

    def setUp(self) -> None:
        # Reset class-level singleton so each test loads fresh
        from embed_provider import SentenceTransformerBackend
        SentenceTransformerBackend._shared_model = None
        SentenceTransformerBackend._shared_dim = 384
        # Snapshot env so we can restore
        self._prior_model = os.environ.get("NM_EMBED_MODEL")

    def tearDown(self) -> None:
        from embed_provider import SentenceTransformerBackend
        SentenceTransformerBackend._shared_model = None
        SentenceTransformerBackend._shared_dim = 384
        if self._prior_model is None:
            os.environ.pop("NM_EMBED_MODEL", None)
        else:
            os.environ["NM_EMBED_MODEL"] = self._prior_model

    def test_default_model_passes_guard(self) -> None:
        """all-MiniLM-L6-v2 (384d default) should pass guard cleanly."""
        os.environ.pop("NM_EMBED_MODEL", None)
        # Force re-evaluation of class-level MODEL_NAME (env was just changed)
        import importlib
        import embed_provider
        importlib.reload(embed_provider)
        b = embed_provider.SentenceTransformerBackend()
        self.assertEqual(b.dim, 384)
        self.assertEqual(b.MODEL_NAME, "all-MiniLM-L6-v2")

    def test_bge_small_passes_guard_384d(self) -> None:
        """bge-small-en-v1.5 (also 384d) should pass guard — drop-in compatible."""
        os.environ["NM_EMBED_MODEL"] = "BAAI/bge-small-en-v1.5"
        import importlib
        import embed_provider
        importlib.reload(embed_provider)
        b = embed_provider.SentenceTransformerBackend()
        self.assertEqual(b.dim, 384)
        self.assertIn("bge-small", b.MODEL_NAME)

    def test_768d_model_raises_value_error(self) -> None:
        """all-mpnet-base-v2 (768d) must trigger guard — substrate is 384d.

        This is the core safety contract: the guard prevents silent
        corruption of the live substrate by a dim-incompatible model.
        """
        os.environ["NM_EMBED_MODEL"] = "sentence-transformers/all-mpnet-base-v2"
        import importlib
        import embed_provider
        importlib.reload(embed_provider)
        with self.assertRaises(ValueError) as ctx:
            embed_provider.SentenceTransformerBackend()
        msg = str(ctx.exception)
        self.assertIn("768d", msg)
        self.assertIn("384", msg)
        self.assertIn("substrate", msg.lower())


class ReembedSubstrateCanonicalDBRefusalTests(unittest.TestCase):
    """tools/reembed_substrate.py must refuse to write to canonical
    production DB unless --i-know-what-im-doing flag is set.
    """

    def test_refuses_canonical_db_path_without_flag(self) -> None:
        """Running re-embed against ~/.neural_memory/memory.db without
        --i-know-what-im-doing should exit code 2."""
        canonical = Path.home() / ".neural_memory" / "memory.db"
        # Skip if canonical doesn't exist (clean test env)
        if not canonical.exists():
            self.skipTest("canonical DB not present in test env")
        py = "/Users/tito/.hermes/hermes-agent/venv/bin/python3"
        if not Path(py).exists():
            py = sys.executable
        result = subprocess.run(
            [py, str(REPO / "tools" / "reembed_substrate.py"), "--db", str(canonical)],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("REFUSING", result.stderr + result.stdout)

    def test_accepts_test_db_copy_path(self) -> None:
        """Re-embed against a test DB path should NOT trigger the canonical refusal.
        (We don't actually run re-embed here — too slow. Just verify the path-check
        passes and the next step is reachable.)
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(b"")  # empty file, will fail later but past the path check
            tmp_path = tmp.name
        try:
            py = "/Users/tito/.hermes/hermes-agent/venv/bin/python3"
            if not Path(py).exists():
                py = sys.executable
            result = subprocess.run(
                [py, str(REPO / "tools" / "reembed_substrate.py"), "--db", tmp_path],
                capture_output=True, text=True, timeout=10,
            )
            # Should NOT be the canonical-refusal exit code
            self.assertNotEqual(result.returncode, 2,
                                f"Test DB rejected as canonical: {result.stderr}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
