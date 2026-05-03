"""S2b 2026-05-03 — multi-source namespacing acceptance tests.

Proves that the v2 ledger contract holds end-to-end across all current
producers of typed evidence:

  • record_wa_crew_event       (source_system='hermes_wa_bridge')
  • record_estimate_evidence   (source_system='ae_dashboard')
  • record_material_price_evidence (source_system=f'vendor:{vendor}')
  • tools/ingest_sent_pdf_sidecars (source_system='sent_estimate_pdf_miner')

The producers above are all REGISTERED in test_ae_evidence_ingest.py for their
own behavioural contracts. This file complements those by asserting the
namespacing invariants the v2 schema relies on:

  1. Distinct helpers compute distinct evidence_ids for plausible inputs.
  2. The v2 ledger admits the dual-source 'sent_pdf' case
     (sent_estimate_pdf_miner vs ae_dashboard with same source_record_id),
     validating that S2b unblocked a structurally possible collision.
  3. Within a single source_system, evidence_id stability holds: same
     (evidence_type, source_system, source_record_id) → same evidence_id
     across calls / processes (deterministic key).

Tests use a tmp_path fixture DB (NEVER touches ~/.neural_memory/memory.db).

Stdlib unittest. Run:
    python3 -m unittest python.test_evidence_ledger_namespace
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ae_workflow_helpers import _compute_evidence_id  # noqa: E402
from schema_upgrade import SchemaUpgrade  # noqa: E402


# Mirrors the legacy SCHEMA constant in memory_client.py — same shape used
# elsewhere in the test suite (test_schema_upgrade._LEGACY_SCHEMA).
_LEGACY_SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT (unixepoch()),
    last_accessed REAL DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0
);

CREATE TABLE connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    edge_type TEXT DEFAULT 'similar',
    created_at REAL DEFAULT (unixepoch()),
    event_time REAL DEFAULT NULL,
    ingestion_time REAL DEFAULT NULL,
    valid_from REAL DEFAULT NULL,
    valid_to REAL DEFAULT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);
"""


# --------------------------------------------------------------------------
# Per-producer source_record_id reference table (S2b 2026-05-03).
# Mirrors the helper formulas in ae_workflow_helpers.py + tools/ingest_*.py
# so the test contract is self-documenting and any drift is caught.
# --------------------------------------------------------------------------
PRODUCER_SHAPES = [
    {
        "producer": "record_wa_crew_event",
        "evidence_type": "wa_crew_message",
        "source_system": "hermes_wa_bridge",
        # f"{thread_id}:{int(ts*1_000_000)}:{md5(raw_text)[:8]}"
        "source_record_id": "1234567890@s.whatsapp.net:1714723200000000:0123abcd",
    },
    {
        "producer": "record_estimate_evidence (no pdf)",
        "evidence_type": "estimate_event",
        "source_system": "ae_dashboard",
        # f"{estimate_id}:{event_type}"
        "source_record_id": "EST-2026-001:draft",
    },
    {
        "producer": "record_estimate_evidence (with pdf)",
        "evidence_type": "sent_pdf",
        "source_system": "ae_dashboard",
        # f"{estimate_id}:{event_type}"
        "source_record_id": "EST-2026-001:sent",
    },
    {
        "producer": "record_material_price_evidence",
        "evidence_type": "material_price",
        "source_system": "vendor:amperage",
        # f"{sku}:{vendor}:{int(quoted_at)}"
        "source_record_id": "BREAKER-20A:amperage:1714723200",
    },
    {
        "producer": "tools/ingest_sent_pdf_sidecars (filename)",
        "evidence_type": "sent_pdf",
        "source_system": "sent_estimate_pdf_miner",
        # f"{msg_id}:{filename}"
        "source_record_id": "19dcf38bc5b87cde:Bharath_Sambandam_estimate.pdf",
    },
    {
        "producer": "tools/ingest_sent_pdf_sidecars (filehash fallback)",
        "evidence_type": "sent_pdf",
        "source_system": "sent_estimate_pdf_miner",
        # f"{msg_id}:{filehash[:16]}"
        "source_record_id": "19dcf38bc5b87cde:7b3f2a1e6c9d4e8b",
    },
]


class EvidenceIdNamespaceTests(unittest.TestCase):
    """Pure-function tests on _compute_evidence_id — no DB needed."""

    def test_evidence_id_distinct_per_producer_in_typical_corpus(self) -> None:
        """Real-shape inputs from each producer yield distinct evidence_ids.
        Sanity check that the deterministic hash is doing its job; not a
        proof of cross-source uniqueness in the abstract (which is what the
        v2 index covers structurally)."""
        seen = {}
        for spec in PRODUCER_SHAPES:
            eid = _compute_evidence_id(
                spec["evidence_type"], spec["source_system"],
                spec["source_record_id"],
            )
            self.assertNotIn(
                eid, seen,
                f"evidence_id collision: {spec['producer']!r} produced "
                f"{eid!r}, already claimed by {seen.get(eid)!r}",
            )
            seen[eid] = spec["producer"]

    def test_evidence_id_deterministic(self) -> None:
        """Same triple → same evidence_id, every time."""
        for spec in PRODUCER_SHAPES:
            a = _compute_evidence_id(
                spec["evidence_type"], spec["source_system"],
                spec["source_record_id"],
            )
            b = _compute_evidence_id(
                spec["evidence_type"], spec["source_system"],
                spec["source_record_id"],
            )
            self.assertEqual(a, b,
                             f"{spec['producer']} evidence_id is non-deterministic")

    def test_evidence_id_changes_with_source_system(self) -> None:
        """Same (evidence_type, source_record_id) but different source_system
        → different evidence_id. This is what makes per-source namespacing
        work at the hash layer (matches the wider v2 unique index)."""
        eid_miner = _compute_evidence_id(
            "sent_pdf", "sent_estimate_pdf_miner", "shared-key",
        )
        eid_dashboard = _compute_evidence_id(
            "sent_pdf", "ae_dashboard", "shared-key",
        )
        self.assertNotEqual(
            eid_miner, eid_dashboard,
            "evidence_id must vary with source_system — without this, two "
            "sources could emit the same evidence_id under different "
            "(system, record_id) pairs",
        )


class LedgerNamespacingAcceptanceTests(unittest.TestCase):
    """v2 ledger admits the multi-source 'sent_pdf' case end-to-end.

    Uses tmp_path-style fixture: builds a fresh DB, applies SchemaUpgrade,
    then directly INSERTs the ledger rows the corresponding producers WOULD
    insert. This validates the v2 INDEX shape against the producer contract
    without requiring a full NeuralMemory instance (no model load) — keeps
    the test fast + isolated.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db = self.tmp_path / "memory.db"
        with sqlite3.connect(str(self.db)) as conn:
            conn.executescript(_LEGACY_SCHEMA)
        SchemaUpgrade(str(self.db)).upgrade()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _insert(
        self, *, evidence_type: str, source_system: str, source_record_id: str,
    ) -> str:
        eid = _compute_evidence_id(evidence_type, source_system, source_record_id)
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute(
                "INSERT INTO evidence_ledger "
                "(evidence_id, evidence_type, source_system, source_record_id) "
                "VALUES (?, ?, ?, ?)",
                (eid, evidence_type, source_system, source_record_id),
            )
            conn.commit()
        return eid

    def test_dual_source_sent_pdf_with_shared_record_id(self) -> None:
        """The structurally-possible collision flagged by the synth probe:
        evidence_type='sent_pdf' from 'sent_estimate_pdf_miner' AND from
        'ae_dashboard' with overlapping source_record_id values. v2 admits
        both as distinct rows."""
        miner_eid = self._insert(
            evidence_type="sent_pdf",
            source_system="sent_estimate_pdf_miner",
            source_record_id="overlap-key",
        )
        dashboard_eid = self._insert(
            evidence_type="sent_pdf",
            source_system="ae_dashboard",
            source_record_id="overlap-key",
        )
        self.assertNotEqual(miner_eid, dashboard_eid)
        with sqlite3.connect(str(self.db)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger "
                "WHERE source_record_id = 'overlap-key'"
            ).fetchone()[0]
        self.assertEqual(count, 2)

    def test_same_source_same_record_blocked(self) -> None:
        """Per-source replay safety: same source_system can't insert the
        same source_record_id twice (regardless of evidence_type)."""
        self._insert(
            evidence_type="sent_pdf",
            source_system="ae_dashboard",
            source_record_id="some-rec",
        )
        # Different evidence_type but same (source_system, source_record_id):
        # blocked by idx_evidence_ledger_source.
        with self.assertRaises(sqlite3.IntegrityError):
            self._insert(
                evidence_type="estimate_event",
                source_system="ae_dashboard",
                source_record_id="some-rec",
            )

    def test_full_triple_dedup_blocked(self) -> None:
        """The v2 wider index blocks duplicate
        (evidence_type, source_system, source_record_id) triples — the
        replay-authority guarantee."""
        self._insert(
            evidence_type="material_price",
            source_system="vendor:amperage",
            source_record_id="BREAKER-20A:amperage:1714723200",
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self._insert(
                evidence_type="material_price",
                source_system="vendor:amperage",
                source_record_id="BREAKER-20A:amperage:1714723200",
            )

    def test_all_producer_shapes_coexist(self) -> None:
        """All producer shapes from PRODUCER_SHAPES insert successfully into
        a single fresh ledger — no cross-producer collisions in the
        representative input set."""
        for spec in PRODUCER_SHAPES:
            self._insert(
                evidence_type=spec["evidence_type"],
                source_system=spec["source_system"],
                source_record_id=spec["source_record_id"],
            )
        with sqlite3.connect(str(self.db)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger"
            ).fetchone()[0]
        self.assertEqual(count, len(PRODUCER_SHAPES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
