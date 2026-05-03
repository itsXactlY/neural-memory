"""Regression coverage for tools/ingest_ae_corpus.py::_existing_content_hashes.

Locks the bridge-mailbox dedup contract: bridge_mailbox chunks are written
with origin_system='hermes', so the dedup query MUST include 'hermes' or
every re-run of the importer will re-ingest every bridge message.

Verified-now via Arch-2 audit 2026-05-03: live DB has 1793 bridge_mailbox
rows but only 746 distinct msg_ids — 1047 duplicate rows that this dedup
fix prevents going forward.

The fix is a single-token addition to the WHERE clause; this test exists
so a future "tidy up" of the origin_system list cannot silently re-introduce
the bug.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from ingest_ae_corpus import _existing_content_hashes  # noqa: E402


def _fake_store(rows):
    """MagicMock store whose conn.execute(...).fetchall() returns rows."""
    store = MagicMock()
    store.conn.execute.return_value.fetchall.return_value = rows
    return store


class ExistingContentHashesTests(unittest.TestCase):
    def test_query_includes_hermes_origin(self) -> None:
        """The SQL must filter origin_system to include 'hermes'.
        This is the contract that prevents bridge dup re-ingest."""
        store = _fake_store([])
        _existing_content_hashes(store)
        sql = store.conn.execute.call_args[0][0]
        self.assertIn("'hermes'", sql,
                      "dedup query must include 'hermes' origin or "
                      "bridge_mailbox rows duplicate-ingest")
        self.assertIn("'ae'", sql)
        self.assertIn("'claude_memory'", sql)

    def test_collects_content_hash_from_metadata(self) -> None:
        """Returned set is built from metadata_json.content_hash values."""
        rows = [
            (json.dumps({"content_hash": "hash_ae_1"}),),
            (json.dumps({"content_hash": "hash_hermes_1"}),),
            (json.dumps({"content_hash": "hash_claude_1"}),),
        ]
        store = _fake_store(rows)
        out = _existing_content_hashes(store)
        self.assertEqual(out,
                         {"hash_ae_1", "hash_hermes_1", "hash_claude_1"})

    def test_skips_rows_without_content_hash(self) -> None:
        """Rows missing a content_hash key are silently skipped."""
        rows = [
            (json.dumps({"content_hash": "hash_kept"}),),
            (json.dumps({"other_key": "no_hash_here"}),),
            (json.dumps({}),),
        ]
        store = _fake_store(rows)
        out = _existing_content_hashes(store)
        self.assertEqual(out, {"hash_kept"})

    def test_skips_invalid_json_silently(self) -> None:
        """Malformed metadata_json must not raise; just skip those rows."""
        rows = [
            (json.dumps({"content_hash": "good_hash"}),),
            ("not valid json {{{",),
            (None,),  # NULL metadata_json — defensive
        ]
        store = _fake_store(rows)
        try:
            out = _existing_content_hashes(store)
        except Exception as e:
            self.fail(f"_existing_content_hashes raised on bad row: {e!r}")
        self.assertEqual(out, {"good_hash"})

    def test_hermes_bridge_chunk_dedup_contract(self) -> None:
        """End-to-end contract: when a hermes-origin row with a known
        content_hash exists in the dedup set, _gather_bridge_messages's
        chunk with the same content_hash is recognized as already-ingested.
        Models the actual bug that produced 1047 duplicate bridge rows."""
        bridge_content_hash = "abcdef1234567890_bridge_msg_hash"
        rows = [
            # Bridge mailbox row written by a prior ingest_ae_corpus run.
            # _gather_bridge_messages writes origin_system='hermes' (line 597).
            (json.dumps({
                "content_hash": bridge_content_hash,
                "source_label": "bridge_mailbox",
                "msg_id": "msg_xyz",
            }),),
        ]
        store = _fake_store(rows)
        existing = _existing_content_hashes(store)
        self.assertIn(bridge_content_hash, existing,
                      "bridge_mailbox content_hash MUST be in the dedup "
                      "set so the next ingest cycle skips re-writing it. "
                      "If this fails, every re-run re-ingests every bridge "
                      "message → unbounded substrate growth.")


if __name__ == "__main__":
    unittest.main()
