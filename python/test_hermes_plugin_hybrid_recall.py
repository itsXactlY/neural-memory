"""Hermes plugin queue_prefetch hybrid_recall opt-in contracts.

The plugin file imports Hermes runtime interfaces, so this test stubs only the
minimal module surface needed to instantiate NeuralMemoryProvider. It does not
import or start the real Hermes runtime.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _install_hermes_stubs() -> None:
    agent_module = types.ModuleType("agent")
    memory_provider_module = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider_module.MemoryProvider = MemoryProvider
    agent_module.memory_provider = memory_provider_module
    sys.modules.setdefault("agent", agent_module)
    sys.modules.setdefault("agent.memory_provider", memory_provider_module)

    tools_module = types.ModuleType("tools")
    registry_module = types.ModuleType("tools.registry")
    registry_module.tool_error = lambda message: {"error": message}
    tools_module.registry = registry_module
    sys.modules.setdefault("tools", tools_module)
    sys.modules.setdefault("tools.registry", registry_module)


def _load_plugin_module():
    _install_hermes_stubs()
    plugin_path = Path(__file__).resolve().parent.parent / "hermes-plugin" / "__init__.py"
    spec = importlib.util.spec_from_file_location("nm_hermes_plugin_for_tests", plugin_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plugin module from {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLUGIN = _load_plugin_module()


class HermesPluginHybridRecallTests(unittest.TestCase):
    def _provider(self):
        provider = PLUGIN.NeuralMemoryProvider()
        provider._config = {"prefetch_limit": 3}
        provider._memory = MagicMock()
        provider._memory.recall.return_value = [
            {"id": 1, "content": "dense result", "similarity": 0.7}
        ]
        # Real hybrid_recall returns rows keyed `combined` (per memory_client.py
        # line 2260), not `similarity`. The plugin normalizes — verified below.
        provider._memory.hybrid_recall.return_value = [
            {"id": 2, "content": "hybrid result", "combined": 0.9}
        ]
        return provider

    def _run_prefetch(self, provider) -> str:
        provider.queue_prefetch("panel materials")
        self.assertIsNotNone(provider._prefetch_thread)
        provider._prefetch_thread.join(timeout=2)
        self.assertFalse(provider._prefetch_thread.is_alive())
        return provider._prefetch_result

    def test_default_env_uses_recall_not_hybrid(self) -> None:
        provider = self._provider()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_HERMES_HYBRID_RECALL", None)
            result = self._run_prefetch(provider)

        provider._memory.recall.assert_called_once_with("panel materials", k=3)
        provider._memory.hybrid_recall.assert_not_called()
        self.assertIn("dense result", result)

    def test_env_flag_set_routes_to_hybrid_recall(self) -> None:
        provider = self._provider()
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            result = self._run_prefetch(provider)

        provider._memory.hybrid_recall.assert_called_once_with(
            "panel materials", k=3, rerank=False)
        provider._memory.recall.assert_not_called()
        self.assertIn("hybrid result", result)

    def test_hybrid_recall_exception_falls_back_to_recall(self) -> None:
        provider = self._provider()
        provider._memory.hybrid_recall.side_effect = RuntimeError("hybrid unavailable")
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            result = self._run_prefetch(provider)

        provider._memory.hybrid_recall.assert_called_once_with(
            "panel materials", k=3, rerank=False)
        provider._memory.recall.assert_called_once_with("panel materials", k=3)
        self.assertIn("dense result", result)

    def test_env_flag_other_value_does_not_route(self) -> None:
        for value in ("0", "", "true", "yes"):
            with self.subTest(value=value):
                provider = self._provider()
                with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": value},
                                clear=False):
                    result = self._run_prefetch(provider)

                provider._memory.recall.assert_called_once_with(
                    "panel materials", k=3)
                provider._memory.hybrid_recall.assert_not_called()
                self.assertIn("dense result", result)

    # ------------------------------------------------------------------
    # LIVE_FEED #11 — score key normalization (queue_prefetch)
    # ------------------------------------------------------------------

    def test_queue_prefetch_score_mapping_uses_similarity_key(self) -> None:
        """hybrid_recall returns `combined`; plugin must expose `similarity`
        so downstream prefetch formatter (which reads r.get('similarity'))
        renders the score, not 0."""
        provider = self._provider()
        provider._memory.hybrid_recall.return_value = [
            {"id": 7, "content": "hybrid panel result", "combined": 0.42}
        ]
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            result = self._run_prefetch(provider)

        # Formatter writes "[0.42] hybrid panel result", not "[0.00]"
        self.assertIn("[0.42]", result)
        self.assertIn("hybrid panel result", result)
        # And the underlying list now has `similarity` set (additive — combined preserved)
        normalized = provider._memory.hybrid_recall.return_value
        self.assertEqual(normalized[0]["similarity"], 0.42)
        self.assertEqual(normalized[0]["combined"], 0.42)

    def test_queue_prefetch_score_mapping_handles_underscore_combined(self) -> None:
        """Some hybrid_recall intermediates (rerank top-N) carry `_combined`."""
        provider = self._provider()
        provider._memory.hybrid_recall.return_value = [
            {"id": 8, "content": "underscore variant", "_combined": 0.31}
        ]
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            result = self._run_prefetch(provider)
        self.assertIn("[0.31]", result)
        normalized = provider._memory.hybrid_recall.return_value
        self.assertEqual(normalized[0]["similarity"], 0.31)

    # ------------------------------------------------------------------
    # LIVE_FEED #10 — _handle_recall hybrid routing
    # ------------------------------------------------------------------

    def _handle_recall_provider(self):
        provider = PLUGIN.NeuralMemoryProvider()
        provider._memory = MagicMock()
        provider._memory.recall.return_value = [
            {"id": 11, "label": "plain", "content": "plain recall row", "similarity": 0.55}
        ]
        provider._memory.hybrid_recall.return_value = [
            {"id": 22, "label": "hybrid", "content": "hybrid recall row", "combined": 0.81}
        ]
        return provider

    def test_handle_recall_default_uses_recall_not_hybrid(self) -> None:
        provider = self._handle_recall_provider()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_HERMES_HYBRID_RECALL", None)
            payload = provider._handle_recall({"query": "panel breakers", "limit": 4})

        provider._memory.recall.assert_called_once_with("panel breakers", k=4)
        provider._memory.hybrid_recall.assert_not_called()
        self.assertIn("plain recall row", payload)

    def test_handle_recall_env_flag_routes_to_hybrid_recall_with_rerank_true(self) -> None:
        provider = self._handle_recall_provider()
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            payload = provider._handle_recall({"query": "panel breakers", "limit": 4})

        # Per LIVE_FEED #10: rerank=True (NOT False like queue_prefetch).
        provider._memory.hybrid_recall.assert_called_once_with(
            "panel breakers", k=4, rerank=True)
        provider._memory.recall.assert_not_called()
        self.assertIn("hybrid recall row", payload)

    def test_handle_recall_hybrid_exception_falls_back_to_recall(self) -> None:
        provider = self._handle_recall_provider()
        provider._memory.hybrid_recall.side_effect = RuntimeError("hybrid down")
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            payload = provider._handle_recall({"query": "panel breakers", "limit": 4})

        provider._memory.hybrid_recall.assert_called_once_with(
            "panel breakers", k=4, rerank=True)
        provider._memory.recall.assert_called_once_with("panel breakers", k=4)
        self.assertIn("plain recall row", payload)

    def test_handle_recall_score_mapping_uses_similarity_key(self) -> None:
        """hybrid_recall returns `combined` — _handle_recall payload must show
        the actual score under `similarity`, not 0."""
        import json as _json
        provider = self._handle_recall_provider()
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            payload = provider._handle_recall({"query": "panel breakers", "limit": 4})

        parsed = _json.loads(payload)
        self.assertEqual(parsed["count"], 1)
        self.assertEqual(parsed["results"][0]["similarity"], 0.81)


if __name__ == "__main__":
    unittest.main()
