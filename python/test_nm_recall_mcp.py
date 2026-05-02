"""Smoke tests for nm_recall_mcp.py — cross-agent JSON-RPC server.

Per holistic-reviewer-round-1 finding: nm_recall_mcp.py (365 LOC,
exposed to Codex + Hermes + Claude Code) shipped with zero tests.
This closes the coverage gap with stdin/stdout-driven smoke tests.

Contracts:
- Server boots (initialize succeeds)
- tools/list returns 5 tools (nm_recall, nm_sparse_search, nm_remember,
  nm_status, nm_audit)
- nm_status JSON-RPC tool/call returns valid JSON content
- Unknown tool returns proper JSON-RPC error
- Unknown method returns proper JSON-RPC error
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


_MCP_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "nm_recall_mcp.py"


def _rpc(*messages: dict, timeout: int = 60) -> list[dict]:
    """Send messages via stdin, return parsed JSON-RPC responses."""
    stdin_data = "".join(json.dumps(m) + "\n" for m in messages)
    proc = subprocess.run(
        ["python3.11", str(_MCP_SCRIPT)],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    responses = []
    for line in proc.stdout.strip().split("\n"):
        if line.strip():
            responses.append(json.loads(line))
    return responses


class NMRecallMCPSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Skip suite cleanly if substrate doesn't exist (e.g., fresh CI)
        db_path = Path.home() / ".neural_memory" / "memory.db"
        if not db_path.exists():
            raise unittest.SkipTest(
                f"substrate not present at {db_path} — skipping MCP smoke"
            )
        if not _MCP_SCRIPT.exists():
            raise unittest.SkipTest(f"MCP script not at {_MCP_SCRIPT}")

    def test_initialize_returns_protocol_version(self) -> None:
        responses = _rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["jsonrpc"], "2.0")
        self.assertEqual(responses[0]["id"], 1)
        self.assertIn("result", responses[0])
        self.assertEqual(
            responses[0]["result"]["protocolVersion"], "2024-11-05",
        )
        self.assertEqual(
            responses[0]["result"]["serverInfo"]["name"], "nm-recall",
        )

    def test_tools_list_returns_five_tools(self) -> None:
        responses = _rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        self.assertEqual(len(responses), 2)
        tools = responses[1]["result"]["tools"]
        self.assertEqual(len(tools), 5)
        names = {t["name"] for t in tools}
        self.assertEqual(
            names,
            {"nm_recall", "nm_sparse_search", "nm_remember",
             "nm_status", "nm_audit"},
        )

    def test_nm_status_tool_call_returns_substrate_stats(self) -> None:
        responses = _rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "nm_status", "arguments": {}}},
            timeout=120,  # nm_status hits SQLite — may be slow under contention
        )
        self.assertEqual(len(responses), 2)
        result = responses[1].get("result")
        self.assertIsNotNone(result)
        # Content envelope
        content = result["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        # Inner JSON has substrate stats
        stats = json.loads(content[0]["text"])
        self.assertIn("total_memories", stats)
        self.assertGreater(stats["total_memories"], 0)
        self.assertIn("entities", stats)
        self.assertIn("edges", stats)
        # Per reviewer-round-6 reshape:
        self.assertIn("memories_fts_count", stats)
        self.assertIn("non_entity_memories_count", stats)

    def test_unknown_tool_returns_jsonrpc_error(self) -> None:
        responses = _rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "nonexistent_tool", "arguments": {}}},
        )
        self.assertEqual(len(responses), 2)
        self.assertIn("error", responses[1])
        self.assertEqual(responses[1]["error"]["code"], -32601)
        self.assertIn("nonexistent_tool", responses[1]["error"]["message"])

    def test_unknown_method_returns_jsonrpc_error(self) -> None:
        responses = _rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "bogus/method", "params": {}},
        )
        self.assertEqual(len(responses), 1)
        self.assertIn("error", responses[0])
        self.assertEqual(responses[0]["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
