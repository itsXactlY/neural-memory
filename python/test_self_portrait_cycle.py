"""Tests for tools/self_portrait_cycle.py — orchestrator STEPS 2-7.

Per packet S-PORTRAIT-2 acceptance criteria:
- scaffold mode writes input packet only (no image-gen, no store)
- complete mode requires --reasoning-text + --prompt-text
- image-gen handles missing OPENAI_API_KEY gracefully
- image-gen handles API failure gracefully
- store writes kind='self_portrait' with attribution metadata (author=agent_name)
- diff_from_prior returns "First portrait cycle for X" when no prior
- orchestrator calls compose_substrate_packet in STEP 2 (scaffold path)
- --dry-run skips the substrate write (store call_count == 0)
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make tools/ importable as a top-level module so we can `import
# self_portrait_cycle` directly. The cycle module also injects python/ into
# sys.path for its own internal imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _REPO_ROOT / "tools"
_PYTHON_DIR = _REPO_ROOT / "python"
for p in (_TOOLS_DIR, _PYTHON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import self_portrait_cycle as cycle  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeStore:
    """Stand-in for SQLiteStore that records every store + execute."""

    def __init__(self, prior_rows: list[tuple] | None = None):
        self.store_calls: list[dict] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self._prior_rows = list(prior_rows or [])
        self._lock = MagicMock()
        self._lock.__enter__ = MagicMock(return_value=None)
        self._lock.__exit__ = MagicMock(return_value=None)
        self.conn = MagicMock()
        self.conn.execute = self._execute

    def _execute(self, sql: str, params: tuple = ()):
        self.execute_calls.append((sql, params))
        cursor = MagicMock()
        cursor.fetchall.return_value = self._prior_rows
        cursor.fetchone.return_value = self._prior_rows[0] if self._prior_rows else None
        return cursor


class _FakeMem:
    """Stand-in for NeuralMemory exposing both .store and .remember."""

    def __init__(self, prior_rows: list[tuple] | None = None):
        self.store = _FakeStore(prior_rows=prior_rows)
        self.remember_calls: list[dict] = []

    def remember(self, text: str, **kwargs) -> int:
        self.remember_calls.append({"text": text, **kwargs})
        return 12345  # fake new memory id


def _make_packet(**overrides) -> dict:
    """Default substrate packet shape (mirrors compose_substrate_packet)."""
    base = {
        "agent": "claude-code",
        "ts": 1000.0,
        "self_memories": [],
        "self_reflections": [],
        "top_entities": [],
        "dream_insights": [],
        "peer_portraits": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# scaffold mode — writes input packet only, no image-gen, no store
# ---------------------------------------------------------------------------


class ScaffoldModeTests(unittest.TestCase):
    def test_scaffold_mode_writes_input_packet_only(self):
        """STEP 2 scaffold: writes input.json, no image-gen, no store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mem = _FakeMem()
            with patch.object(
                cycle, "_PORTRAITS_ROOT", tmp_root
            ), patch(
                "self_portrait_substrate.compose_substrate_packet",
                return_value=_make_packet(self_memories=[{"id": 1}]),
            ):
                result = cycle.run_scaffold_mode(mem, "claude-code", 1700000000.0)

            self.assertEqual(result["mode"], "scaffold")
            self.assertEqual(result["agent_name"], "claude-code")
            self.assertEqual(result["self_memories_count"], 1)

            input_path = Path(result["substrate_packet_path"])
            self.assertTrue(input_path.exists(), "scaffold must write input.json")
            self.assertEqual(input_path.name, "input.json")
            # Exactly the cycle dir we expect.
            self.assertEqual(
                input_path.parent.name, "cycle-1700000000"
            )

            # No store invocation.
            self.assertEqual(mem.remember_calls, [])
            self.assertEqual(mem.store.store_calls, [])

    def test_orchestrator_calls_compose_substrate_packet_in_step_2(self):
        """STEP 2 must invoke compose_substrate_packet from the substrate module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mem = _FakeMem()
            with patch.object(
                cycle, "_PORTRAITS_ROOT", tmp_root
            ), patch(
                "self_portrait_substrate.compose_substrate_packet",
                return_value=_make_packet(),
            ) as compose_mock:
                cycle.run_scaffold_mode(mem, "valiendo", 1700000000.0)
            compose_mock.assert_called_once_with(mem, "valiendo")


# ---------------------------------------------------------------------------
# complete mode CLI guard — requires both reasoning + prompt
# ---------------------------------------------------------------------------


class CompleteModeCLITests(unittest.TestCase):
    def test_complete_mode_requires_reasoning_and_prompt(self):
        """argparse must reject --mode complete missing reasoning+prompt."""
        # argparse.error raises SystemExit. We patch _open_memory so it never
        # gets called even if validation fails to short-circuit.
        with patch.object(cycle, "_open_memory") as open_mem_mock:
            with self.assertRaises(SystemExit):
                cycle.main(["--agent", "claude-code", "--mode", "complete"])
            with self.assertRaises(SystemExit):
                cycle.main([
                    "--agent", "claude-code",
                    "--mode", "complete",
                    "--reasoning-text", "i think therefore i am",
                ])
            with self.assertRaises(SystemExit):
                cycle.main([
                    "--agent", "claude-code",
                    "--mode", "complete",
                    "--prompt-text", "fractured mirrors etc",
                ])
            # Validation fires before _open_memory in all three failures.
            open_mem_mock.assert_not_called()

    def test_validate_agent_prompt_rejects_empty_and_oversize(self):
        """Bounds-only validation per Tito rule #1 (no templating)."""
        with self.assertRaises(ValueError):
            cycle.validate_agent_prompt("")
        with self.assertRaises(ValueError):
            cycle.validate_agent_prompt("   \n\t  ")
        with self.assertRaises(ValueError):
            cycle.validate_agent_prompt("x" * (cycle._MAX_PROMPT_LEN + 1))
        # Exact-bound passes.
        out = cycle.validate_agent_prompt("x" * cycle._MAX_PROMPT_LEN)
        self.assertEqual(len(out), cycle._MAX_PROMPT_LEN)


# ---------------------------------------------------------------------------
# STEP 5 image-gen: graceful failure paths
# ---------------------------------------------------------------------------


class ImageGenTests(unittest.TestCase):
    def test_image_gen_handles_missing_api_key(self):
        """Missing OPENAI_API_KEY → image_path=None, no exception, error captured."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {}, clear=False
        ):
            # Strip the var even if user has it.
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            result = cycle.generate_image(
                "test prompt",
                "claude-code",
                1700000000.0,
                portraits_root=Path(tmpdir),
            )
        self.assertIsNone(result.image_path)
        self.assertIsNone(result.image_url)
        self.assertEqual(result.model_used, cycle._DEFAULT_IMAGE_MODEL)
        self.assertIn("OPENAI_API_KEY", (result.error or ""))

    def test_image_gen_handles_api_failure_gracefully(self):
        """urllib HTTPError 500 → image_path=None, cycle continues."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False
        ), patch.object(
            cycle,
            "_post_openai_images",
            side_effect=urllib.error.HTTPError(
                cycle._OPENAI_IMAGES_URL, 500, "Internal Server Error", {}, None
            ),
        ):
            result = cycle.generate_image(
                "test prompt",
                "claude-code",
                1700000000.0,
                portraits_root=Path(tmpdir),
            )
        self.assertIsNone(result.image_path)
        self.assertIn("HTTP 500", (result.error or ""))

    def test_image_gen_handles_url_error_gracefully(self):
        """Network error → image_path=None, no exception."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False
        ), patch.object(
            cycle,
            "_post_openai_images",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = cycle.generate_image(
                "test prompt",
                "claude-code",
                1700000000.0,
                portraits_root=Path(tmpdir),
            )
        self.assertIsNone(result.image_path)
        self.assertIn("URLError", (result.error or ""))

    def test_image_gen_succeeds_with_b64_response(self):
        """When API returns b64_json: write decoded bytes to disk."""
        import base64

        png_bytes = b"\x89PNG\r\n\x1a\nFAKE"
        b64 = base64.b64encode(png_bytes).decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"OPENAI_API_KEY": "fake-key"}, clear=False
        ), patch.object(
            cycle,
            "_post_openai_images",
            return_value={"data": [{"b64_json": b64}]},
        ):
            result = cycle.generate_image(
                "test prompt",
                "claude-code",
                1700000000.0,
                portraits_root=Path(tmpdir),
            )
            # Assertions must stay INSIDE the tempdir context — once it
            # exits, tmpdir is cleaned and Path.exists() flips to False.
            self.assertIsNotNone(result.image_path)
            self.assertTrue(Path(result.image_path).exists())
            self.assertEqual(Path(result.image_path).read_bytes(), png_bytes)


# ---------------------------------------------------------------------------
# STEP 7: store contract
# ---------------------------------------------------------------------------


class StoreContractTests(unittest.TestCase):
    def test_store_writes_self_portrait_kind_with_attribution_metadata(self):
        """Store must write kind='self_portrait', origin_system=agent_name,
        metadata.author=agent_name (S-PORTRAIT-1 attribution recommendation)."""
        mem = _FakeMem()
        new_id = cycle.store_portrait(
            mem,
            agent_name="claude-code",
            cycle_ts=1700000000.0,
            reasoning_text="i am evolving",
            prompt_text="a luminous lattice",
            image_path="/tmp/fake.png",
            image_url=None,
            model_used="gpt-image-2",
            diff_from_prior_summary="First portrait cycle for claude-code.",
            substrate_packet_path="/tmp/scaffold/input.json",
        )
        self.assertEqual(new_id, 12345)
        self.assertEqual(len(mem.remember_calls), 1)
        call = mem.remember_calls[0]
        self.assertEqual(call["kind"], "self_portrait")
        self.assertEqual(call["origin_system"], "claude-code")
        self.assertEqual(call["source"], "self_portrait_cycle")
        self.assertEqual(call["confidence"], 1.0)
        self.assertEqual(call["salience"], 0.8)
        self.assertEqual(call["valid_from"], 1700000000.0)
        # Reasoning is the searchable content per spec.
        self.assertEqual(call["text"], "i am evolving")
        # Attribution metadata.
        md = call["metadata"]
        self.assertEqual(md["author"], "claude-code")
        self.assertEqual(md["agent_name"], "claude-code")
        self.assertEqual(md["prompt_text"], "a luminous lattice")
        self.assertEqual(md["image_path"], "/tmp/fake.png")
        self.assertIsNone(md["image_url"])
        self.assertEqual(md["model_used"], "gpt-image-2")
        self.assertEqual(
            md["diff_from_prior"], "First portrait cycle for claude-code."
        )
        self.assertEqual(md["substrate_packet_path"], "/tmp/scaffold/input.json")
        # detect_conflicts must be False for self-portrait rows.
        self.assertEqual(call["detect_conflicts"], False)


# ---------------------------------------------------------------------------
# STEP 6: diff-from-prior
# ---------------------------------------------------------------------------


class DiffFromPriorTests(unittest.TestCase):
    def test_diff_from_prior_returns_first_portrait_message_when_no_prior(self):
        """When _read_prior_reasoning returns None → 'First portrait cycle for X.'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = _FakeMem(prior_rows=[])  # empty fetch → no prior
            with patch.dict("os.environ", {}, clear=False):
                import os

                os.environ.pop("OPENAI_API_KEY", None)
                result = cycle.run_complete_mode(
                    mem,
                    agent_name="codex",
                    reasoning_text="first ever reflection",
                    prompt_text="a quiet circuit",
                    cycle_ts=1700000000.0,
                    dry_run=True,  # don't actually write to substrate
                    portraits_root=Path(tmpdir),
                )
        self.assertEqual(result["diff_from_prior"], "First portrait cycle for codex.")

    def test_diff_from_prior_identical_reports_stable(self):
        same = "evolving toward stillness"
        self.assertEqual(
            cycle._diff_summary(same, same), "Reasoning stable; same themes."
        )

    def test_diff_from_prior_low_overlap_reports_major_shift(self):
        prior = "alpha beta gamma delta epsilon"
        current = "lambda mu nu xi omicron"
        out = cycle._diff_summary(prior, current)
        self.assertTrue(out.startswith("Major shift"), out)


# ---------------------------------------------------------------------------
# --dry-run contract: store call_count == 0
# ---------------------------------------------------------------------------


class DryRunTests(unittest.TestCase):
    def test_dry_run_skips_store(self):
        """--dry-run: STEPS 4-6 still run; STEP 7 (store) is skipped entirely."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {}, clear=False
        ):
            # Strip API key so image-gen short-circuits cleanly.
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            mem = _FakeMem(prior_rows=[])
            result = cycle.run_complete_mode(
                mem,
                agent_name="claude-code",
                reasoning_text="dry run reflection",
                prompt_text="a glass figure",
                cycle_ts=1700000000.0,
                dry_run=True,
                portraits_root=Path(tmpdir),
            )
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["stored"])
        self.assertEqual(result["memory_id"], -1)
        # Critical: zero store invocations.
        self.assertEqual(len(mem.remember_calls), 0)
        self.assertEqual(len(mem.store.store_calls), 0)


if __name__ == "__main__":
    unittest.main()
