"""Contract tests for AEEvidenceIngest v0 (Theme 8 typed evidence record).

Per codex-prescriptive-redesigner Day 4 prescription. These contracts
define what evidence ingest MUST do — Tito approval gates LIVE ingestion
but the contract itself can be locked now.

Tests cover:
- record_evidence_artifact: generic typed record with provenance
- record_wa_crew_event: WA crew chat ingest schema (Hermes → NM contract)
- record_estimate_evidence: estimate pipeline events with PDF provenance
- record_material_price_evidence: bi-temporal price quotes
- Validation: privacy_class, evidence_type, confidence, delivery_status
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ae_workflow_helpers import (  # noqa: E402
    EVIDENCE_PRIVACY_CLASSES,
    EVIDENCE_TYPES,
    record_evidence_artifact,
    record_estimate_evidence,
    record_material_price_evidence,
    record_wa_crew_event,
)
from memory_client import NeuralMemory  # noqa: E402


class AEEvidenceIngestContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    # ------------------------------------------------------------ generic
    def test_record_evidence_artifact_persists_provenance(self) -> None:
        mid = record_evidence_artifact(
            self.mem,
            evidence_type="sent_pdf",
            capability_id="ITEM-9",
            source_system="ae_dashboard",
            source_path="/path/to/sent_estimates_pdfs/EST-001.pdf",
            content="Estimate EST-001 sent to customer alice@example.com",
            privacy_class="financial",
            confidence=0.95,
            source_record_id="EST-001:sent",
            consumer_hint="dashboard:/api/estimates/pipeline",
        )
        full = self.mem.get_memory(mid)
        self.assertIsNotNone(full)
        import json
        md = json.loads(full["metadata_json"])
        self.assertEqual(md["evidence_type"], "sent_pdf")
        self.assertEqual(md["capability_id"], "ITEM-9")
        self.assertEqual(md["source_system"], "ae_dashboard")
        self.assertEqual(md["source_path"], "/path/to/sent_estimates_pdfs/EST-001.pdf")
        self.assertEqual(md["privacy_class"], "financial")
        self.assertEqual(md["consumer_hint"], "dashboard:/api/estimates/pipeline")

    def test_evidence_artifact_rejects_invalid_privacy_class(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            record_evidence_artifact(
                self.mem,
                evidence_type="sent_pdf",
                capability_id="ITEM-9",
                source_system="x", source_path="/x", content="x",
                privacy_class="OBVIOUSLY_INVALID",
            )
        self.assertIn("privacy_class", str(ctx.exception))

    def test_evidence_artifact_rejects_invalid_evidence_type(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            record_evidence_artifact(
                self.mem,
                evidence_type="not_a_real_type",
                capability_id="ITEM-9",
                source_system="x", source_path="/x", content="x",
            )
        self.assertIn("evidence_type", str(ctx.exception))

    def test_evidence_artifact_rejects_out_of_range_confidence(self) -> None:
        with self.assertRaises(ValueError):
            record_evidence_artifact(
                self.mem, evidence_type="sent_pdf", capability_id="X",
                source_system="x", source_path="/x", content="x",
                confidence=1.5,
            )

    def test_caller_metadata_cannot_override_contract_keys(self) -> None:
        """Caller-supplied extra_metadata must NOT override the contract
        fields (capability_id, source_system, source_path, evidence_type,
        privacy_class) — provenance is non-negotiable."""
        mid = record_evidence_artifact(
            self.mem,
            evidence_type="sent_pdf",
            capability_id="REAL-CAP",
            source_system="real_system",
            source_path="/real/path",
            content="content",
            privacy_class="internal",
            extra_metadata={
                "capability_id": "FAKE-CAP",
                "privacy_class": "public",
                "extra_field": "ok-to-add",
            },
        )
        import json
        md = json.loads(self.mem.get_memory(mid)["metadata_json"])
        self.assertEqual(md["capability_id"], "REAL-CAP")
        self.assertEqual(md["privacy_class"], "internal")
        self.assertEqual(md["extra_field"], "ok-to-add")

    # ------------------------------------------------------------ WA
    def test_wa_crew_event_persists_full_contract_schema(self) -> None:
        ts = time.time()
        mid = record_wa_crew_event(
            self.mem,
            capability_id="ITEM-WA",
            thread_id="crew-spanish-default",
            sender="miguel",
            raw_text="el cable 12 ya llegó al lote 27",
            ts=ts,
            lang="es",
            normalized_text="the 12 wire arrived at lot 27",
            delivery_status="read",
            auth_proof="receipt:abc123",
        )
        import json
        md = json.loads(self.mem.get_memory(mid)["metadata_json"])
        self.assertEqual(md["evidence_type"], "wa_crew_message")
        self.assertEqual(md["door"], "wa")
        self.assertEqual(md["thread_id"], "crew-spanish-default")
        self.assertEqual(md["sender"], "miguel")
        self.assertEqual(md["lang"], "es")
        self.assertEqual(md["delivery_status"], "read")
        self.assertEqual(md["auth_proof"], "receipt:abc123")
        # Read receipts get higher confidence
        self.assertEqual(self.mem.get_memory(mid)["confidence"], 0.95)

    def test_wa_crew_event_rejects_empty_raw_text(self) -> None:
        with self.assertRaises(ValueError):
            record_wa_crew_event(
                self.mem, capability_id="X", thread_id="t",
                sender="s", raw_text="", ts=time.time(),
            )

    def test_wa_crew_event_pending_status_lowers_confidence(self) -> None:
        mid = record_wa_crew_event(
            self.mem, capability_id="X", thread_id="t",
            sender="s", raw_text="hello", ts=time.time(),
            delivery_status="pending",
        )
        # Pending status → lower confidence (delivery not confirmed)
        self.assertEqual(self.mem.get_memory(mid)["confidence"], 0.85)

    # ------------------------------------------------------------ estimate
    def test_estimate_evidence_with_pdf_path_classifies_as_sent_pdf(self) -> None:
        mid = record_estimate_evidence(
            self.mem,
            capability_id="ITEM-9",
            estimate_id="EST-042",
            customer_id="CUST-7",
            event_type="sent",
            pdf_path="/path/EST-042.pdf",
            amount_cents=125000,
            sent_to="alice@example.com",
            sent_at=time.time(),
        )
        import json
        md = json.loads(self.mem.get_memory(mid)["metadata_json"])
        self.assertEqual(md["evidence_type"], "sent_pdf")
        self.assertEqual(md["estimate_id"], "EST-042")
        self.assertEqual(md["amount_cents"], 125000)

    def test_estimate_evidence_without_pdf_classifies_as_estimate_event(self) -> None:
        mid = record_estimate_evidence(
            self.mem, capability_id="ITEM-9",
            estimate_id="EST-042", customer_id="CUST-7",
            event_type="approved",
        )
        import json
        md = json.loads(self.mem.get_memory(mid)["metadata_json"])
        self.assertEqual(md["evidence_type"], "estimate_event")
        self.assertEqual(md["event_type"], "approved")

    def test_estimate_evidence_rejects_invalid_event_type(self) -> None:
        with self.assertRaises(ValueError):
            record_estimate_evidence(
                self.mem, capability_id="X",
                estimate_id="E1", customer_id="C1",
                event_type="invalid_event",
            )

    # ------------------------------------------------------------ material price
    def test_material_price_evidence_records_bi_temporal(self) -> None:
        quoted_at = time.time() - 86400 * 30  # 30 days ago
        valid_to = time.time()  # superseded today
        mid = record_material_price_evidence(
            self.mem,
            capability_id="ITEM-MAT",
            sku="ROMEX-12-2",
            vendor="hd",
            price_cents=8999,
            unit="100ft",
            quoted_at=quoted_at,
            valid_to=valid_to,
            quote_source_path="hd:catalog:2026-04",
        )
        import json
        full = self.mem.get_memory(mid)
        md = json.loads(full["metadata_json"])
        self.assertEqual(md["sku"], "ROMEX-12-2")
        self.assertEqual(md["vendor"], "hd")
        self.assertEqual(md["price_cents"], 8999)
        # Bi-temporal: quoted_at = valid_from, valid_to set to now
        self.assertEqual(full["valid_from"], quoted_at)
        self.assertEqual(full["valid_to"], valid_to)

    def test_material_price_evidence_rejects_negative_price(self) -> None:
        with self.assertRaises(ValueError):
            record_material_price_evidence(
                self.mem, capability_id="X",
                sku="X", vendor="x", price_cents=-1,
                quoted_at=time.time(),
                quote_source_path="/x",
            )

    # ------------------------------------------------------------ enums
    def test_privacy_class_enum_is_immutable(self) -> None:
        """Defines the canonical 5 privacy classes — additions require a
        contract change + re-review."""
        self.assertEqual(
            EVIDENCE_PRIVACY_CLASSES,
            {"public", "internal", "pii_low", "pii_high", "financial"},
        )

    def test_evidence_type_enum_covers_theme_8_scope(self) -> None:
        """Defines the canonical 8 evidence types — Theme 8 scope."""
        expected = {"wa_crew_message", "sent_pdf", "estimate_event",
                    "material_price", "appscript_row", "blueprint_excerpt",
                    "qbo_transaction", "calendar_event"}
        self.assertEqual(EVIDENCE_TYPES, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
