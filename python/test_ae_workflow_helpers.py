"""Tests for ae_workflow_helpers — verify helpers populate the right Phase 7
typed kwargs and produce expected memory shapes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ae_workflow_helpers import (  # noqa: E402
    initialize_ae_locus_overlay,
    record_customer_interaction,
    record_financial_event,
    record_invoice_status_change,
    record_job_event,
    record_sop,
    record_whatsapp_message,
)
from memory_client import NeuralMemory  # noqa: E402


class AEWorkflowHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = NeuralMemory(
            db_path=str(Path(self._tmp.name) / "memory.db"),
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

    def test_record_customer_interaction(self) -> None:
        mid = record_customer_interaction(
            self.mem,
            customer="Vibha Choudhury",
            topic="permit jurisdiction",
            body="Asked when the rough-in inspection is scheduled.",
            channel="phone",
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "experience")
        self.assertEqual(row["source"], "phone")
        self.assertEqual(row["origin_system"], "ae")
        self.assertIsNotNone(row["valid_from"])
        # Vibha and Choudhury should be auto-extracted as entities
        ents = self.mem.get_entities_for_memory(mid)
        labels = {e["label"] for e in ents}
        self.assertTrue("Vibha Choudhury" in labels or "Vibha" in labels)

    def test_record_job_event(self) -> None:
        mid = record_job_event(
            self.mem,
            job_id="Lennar lot 27",
            event_type="inspection_scheduled",
            body="Friday 10am; need GFCI labels in panel.",
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "experience")
        self.assertEqual(row["source"], "dashboard")
        # Lennar should be in entities
        ents = self.mem.get_entities_for_memory(mid)
        labels = {e["label"] for e in ents}
        self.assertIn("Lennar", labels)

    def test_record_whatsapp_classifies_procedural_when_imperative(self) -> None:
        mid = record_whatsapp_message(
            self.mem,
            crew_member="miguel",
            text="Cuando el inspector llegue, mostrarle las etiquetas del panel.",
            thread_id="t123",
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "procedural")
        self.assertEqual(row["source"], "whatsapp")

    def test_record_whatsapp_classifies_experience_for_event(self) -> None:
        mid = record_whatsapp_message(
            self.mem,
            crew_member="miguel",
            text="Llegamos al sitio. Falta material.",
            thread_id="t124",
        )
        row = self.mem.get_memory(mid)
        # Default fallback for non-procedural Spanish text → 'experience'
        self.assertEqual(row["kind"], "experience")

    def test_record_sop(self) -> None:
        mid = record_sop(
            self.mem,
            label="sop:panel-upgrade",
            content="When estimating panel upgrades, check load calc first.",
            confidence=0.99,
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "procedural")
        self.assertAlmostEqual(row["confidence"], 0.99)
        self.assertEqual(row["source"], "manual")

    def test_record_sop_with_evidence_creates_derived_from_edges(self) -> None:
        # Plant two experience memories the SOP cites
        e1 = self.mem.remember("Job A had permit delay due to jurisdiction confusion.",
                               detect_conflicts=False, kind="experience")
        e2 = self.mem.remember("Job B had similar jurisdiction issue.",
                               detect_conflicts=False, kind="experience")
        sop = record_sop(
            self.mem,
            label="sop:permit-jurisdiction",
            content="Always verify permit jurisdiction before quoting.",
            evidence_ids=[e1, e2],
        )
        # Both edges should exist
        edges = self.mem.get_edges(sop)
        derived = {e["target_id"] for e in edges if e["edge_type"] == "derived_from"}
        self.assertEqual(derived, {e1, e2})

    def test_record_invoice_status_change_creates_pair_with_validity(self) -> None:
        old, new = record_invoice_status_change(
            self.mem,
            invoice_id="INV-042",
            old_status="pending",
            new_status="paid",
            transition_ts=1500.0,
            customer="Vibha",
            amount_cents=125000,
        )
        old_row = self.mem.get_memory(old)
        new_row = self.mem.get_memory(new)
        self.assertEqual(old_row["valid_to"], 1500.0)
        self.assertEqual(new_row["valid_from"], 1500.0)
        self.assertTrue(self.mem.has_edge(old, new, edge_type="contradicts"))

    def test_record_financial_event(self) -> None:
        mid = record_financial_event(
            self.mem,
            event_type="invoice_due",
            due_date_iso="2026-05-15",
            note="Vibha remodel final invoice.",
            amount_cents=500000,
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["source"], "financial_calendar")
        self.assertEqual(row["kind"], "experience")

    def test_initialize_ae_locus_overlay_idempotent(self) -> None:
        loci_a = initialize_ae_locus_overlay(self.mem)
        loci_b = initialize_ae_locus_overlay(self.mem)
        self.assertEqual(loci_a, loci_b, "locus init must be idempotent")
        self.assertEqual(len(loci_a), 6)
        # Each is a kind='locus' node
        for room_id in loci_a.values():
            row = self.mem.get_memory(room_id)
            self.assertEqual(row["kind"], "locus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
