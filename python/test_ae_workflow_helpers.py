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
    recall_customer_by_name,
    recall_estimates_for_customer,
    recall_for_dashboard,
    recall_recent_leads,
    recall_template_for_job,
    record_customer,
    record_customer_interaction,
    record_estimate,
    record_financial_event,
    record_invoice_status_change,
    record_job_event,
    record_lead,
    record_sop,
    record_template,
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


class RecallForDashboardTests(unittest.TestCase):
    """Verify the bench-validated dashboard query helper."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = NeuralMemory(
            db_path=str(Path(self._tmp.name) / "memory.db"),
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )
        self.mid = self.mem.remember(
            "Lennar lot 27 panel install scheduled",
            kind="experience",
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_returns_results_with_trace(self) -> None:
        results = recall_for_dashboard(self.mem, "Lennar lot 27", k=3)
        self.assertGreater(len(results), 0, "should return results")
        # Activation trace must be present (per Phase 7.5 contract)
        self.assertIn("_trace", results[0])
        # Trace has all 12 expected fields
        trace = results[0]["_trace"]
        for field in ("semantic", "sparse", "graph", "temporal",
                      "entity", "procedural", "locus",
                      "stale_penalty", "contradiction_penalty"):
            self.assertIn(field, trace, f"trace missing {field}")

    def test_kind_filter_passed_through(self) -> None:
        # When kind='procedural' is requested, results should be filtered
        # (no procedural memories in this test DB → empty result expected)
        results = recall_for_dashboard(self.mem, "Lennar lot 27", k=3,
                                        kind="procedural")
        # The mid we seeded was kind='experience' so should NOT appear
        ids = [r["id"] for r in results]
        self.assertNotIn(self.mid, ids,
                         "experience-kind seed should be filtered when "
                         "kind=procedural is passed")


class LookupBeforeCreateHelpersTests(unittest.TestCase):
    """Tests for the 8 lookup-before-create helpers (commit c49800e).
    Per aux-builder msg_9cb0c42a — Hermes-via-chat workflow needs these
    to answer 'do I already have this customer/template/estimate?'
    without QBO round-trip."""

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

    def test_record_customer_creates_entity(self) -> None:
        mid = record_customer(
            self.mem,
            name="Sarah Jones",
            qbo_id="QBO-123",
            email="sarah@example.com",
            phone="555-1234",
            addresses=["123 Main St"],
            source="qbo",
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "entity")
        self.assertEqual(row["source"], "qbo")
        self.assertEqual(row["origin_system"], "ae")
        # Metadata roundtrips
        import json
        md = json.loads(row.get("metadata_json") or "{}")
        self.assertEqual(md["qbo_id"], "QBO-123")
        self.assertEqual(md["email"], "sarah@example.com")

    def test_record_template_creates_procedural(self) -> None:
        mid = record_template(
            self.mem,
            template_id="TPL-001",
            name="Kitchen Remodel Standard",
            job_type="remodel",
            line_items=[{"sku": "GFCI-15", "qty": 4, "unit": "ea"}],
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "procedural")
        import json
        md = json.loads(row.get("metadata_json") or "{}")
        self.assertEqual(md["template_id"], "TPL-001")
        self.assertEqual(md["job_type"], "remodel")
        self.assertEqual(md["line_item_count"], 1)

    def test_record_estimate_stores_dual_pricing(self) -> None:
        mid = record_estimate(
            self.mem,
            estimate_id="EST-100",
            customer_id="CUST-7",
            template_id="TPL-001",
            line_items_customer_pricing=[{"sku": "X", "price": 100}],
            line_items_internal_cost=[{"sku": "X", "cost": 60}],
            status="draft",
            total_amount=100.00,
        )
        row = self.mem.get_memory(mid)
        self.assertEqual(row["kind"], "experience")
        import json
        md = json.loads(row.get("metadata_json") or "{}")
        self.assertEqual(md["customer_id"], "CUST-7")
        self.assertEqual(md["status"], "draft")
        # Dual pricing both stored
        self.assertEqual(md["line_items_customer"][0]["price"], 100)
        self.assertEqual(md["line_items_internal"][0]["cost"], 60)

    def test_record_lead_returns_dict_with_duplicate_check(self) -> None:
        # First lead — no duplicate
        result = record_lead(
            self.mem,
            source="angi",
            contact={"name": "Bob Smith", "email": "bob@example.com"},
            intent="kitchen remodel quote",
        )
        self.assertIsInstance(result, dict)
        self.assertIn("id", result)
        self.assertIsNone(result["duplicate_customer_id"])

    def test_record_lead_flags_duplicate_existing_customer(self) -> None:
        # Seed a customer first
        cust_mid = record_customer(self.mem, name="Bob Smith", source="qbo")
        # Lead with same name should auto-flag
        result = record_lead(
            self.mem,
            source="thumbtack",
            contact={"name": "Bob Smith", "phone": "555-9999"},
            intent="bathroom",
        )
        self.assertEqual(result["duplicate_customer_id"], cust_mid)

    def test_recall_customer_by_name_finds_seeded(self) -> None:
        cust_mid = record_customer(self.mem, name="Vibha Choudhury", source="qbo")
        results = recall_customer_by_name(self.mem, name="Vibha", k=5)
        self.assertIn(cust_mid, [r["id"] for r in results])

    def test_recall_customer_by_name_exact_mode_uses_label_equality(self) -> None:
        """fuzzy=False must use exact label match, not BM25 sparse_search.
        Caught by codex-resolver 2026-05-02 — sparse_search returned non-matches."""
        cust_mid = record_customer(self.mem, name="Vibha Choudhury", source="qbo")
        # Add a noise memory that contains the customer label as substring but
        # is NOT the customer entity row
        self.mem.remember(
            "This note mentions customer:Vibha Choudhury but is not the customer entity.",
            label="note:vibha",
            kind="experience",
        )
        results = recall_customer_by_name(
            self.mem, name="Vibha Choudhury", fuzzy=False, k=5)
        # Exact mode must return ONLY the customer entity, not the note
        self.assertEqual([r["id"] for r in results], [cust_mid])

    def test_recall_template_for_job_returns_seeded_template(self) -> None:
        template_mid = record_template(
            self.mem,
            template_id="TPL-002",
            name="Service Call Standard",
            job_type="service",
            line_items=[{"sku": "BREAKER-20A", "qty": 1, "unit": "ea"}],
        )
        results = recall_template_for_job(self.mem, job_type="service")
        self.assertIn(template_mid, [r["id"] for r in results])
        self.assertIn("template:TPL-002", [r["label"] for r in results])

    def test_recall_estimates_for_customer_metadata_filter(self) -> None:
        # Seed two estimates for different customers
        record_estimate(
            self.mem, estimate_id="E1", customer_id="C1",
            template_id=None,
            line_items_customer_pricing=[], line_items_internal_cost=[],
            status="draft",
        )
        record_estimate(
            self.mem, estimate_id="E2", customer_id="C2",
            template_id=None,
            line_items_customer_pricing=[], line_items_internal_cost=[],
            status="sent",
        )
        # Filter for C1 — should not include C2
        results = recall_estimates_for_customer(self.mem, customer_id="C1", k=10)
        self.assertEqual({r["label"] for r in results}, {"estimate:E1"})
        self.assertEqual({r["metadata"]["customer_id"] for r in results}, {"C1"})

    def test_recall_recent_leads_filter_by_source(self) -> None:
        record_lead(self.mem, source="angi",
                    contact={"name": "X"}, intent="quote",
                    auto_check_existing=False)
        record_lead(self.mem, source="thumbtack",
                    contact={"name": "Y"}, intent="quote",
                    auto_check_existing=False)
        results = recall_recent_leads(self.mem, source="angi", days=30)
        self.assertEqual({r["label"] for r in results}, {"lead:angi:X"})
        self.assertEqual({r["metadata"]["lead_source"] for r in results}, {"angi"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
