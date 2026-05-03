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
    _compute_evidence_id,
    record_evidence_artifact,
    record_estimate_evidence,
    record_material_price_evidence,
    record_wa_crew_event,
)
from memory_client import NeuralMemory  # noqa: E402


def _mid(result: Any) -> int:
    """Extract memory_id from the {memory_id, evidence_id, inserted} dict
    returned by record_evidence_artifact and friends. Lets the bulk of
    the existing assertions stay readable."""
    if isinstance(result, dict):
        return result["memory_id"]
    return result  # legacy int (should not happen post-S2)


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
        mid = _mid(record_evidence_artifact(
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
        ))
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
                source_record_id="test-priv-check",  # satisfy S4b contract; test reaches privacy check
            )
        self.assertIn("privacy_class", str(ctx.exception))

    def test_evidence_artifact_rejects_invalid_evidence_type(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            record_evidence_artifact(
                self.mem,
                evidence_type="not_a_real_type",
                capability_id="ITEM-9",
                source_system="x", source_path="/x", content="x",
                source_record_id="test-type-check",  # satisfy S4b contract; test reaches type check
            )
        self.assertIn("evidence_type", str(ctx.exception))

    def test_evidence_artifact_rejects_out_of_range_confidence(self) -> None:
        with self.assertRaises(ValueError):
            record_evidence_artifact(
                self.mem, evidence_type="sent_pdf", capability_id="X",
                source_system="x", source_path="/x", content="x",
                confidence=1.5,
                source_record_id="test-conf-check",  # satisfy S4b contract; test reaches confidence check
            )

    def test_caller_metadata_cannot_override_contract_keys(self) -> None:
        """Caller-supplied extra_metadata must NOT override the contract
        fields (capability_id, source_system, source_path, evidence_type,
        privacy_class) — provenance is non-negotiable."""
        mid = _mid(record_evidence_artifact(
            self.mem,
            evidence_type="sent_pdf",
            capability_id="REAL-CAP",
            source_system="real_system",
            source_path="/real/path",
            content="content",
            privacy_class="internal",
            source_record_id="OVERRIDE-TEST-1",  # S4b: stable key required
            extra_metadata={
                "capability_id": "FAKE-CAP",
                "privacy_class": "public",
                "extra_field": "ok-to-add",
            },
        ))
        import json
        md = json.loads(self.mem.get_memory(mid)["metadata_json"])
        self.assertEqual(md["capability_id"], "REAL-CAP")
        self.assertEqual(md["privacy_class"], "internal")
        self.assertEqual(md["extra_field"], "ok-to-add")

    # ------------------------------------------------------------ WA
    def test_wa_crew_event_persists_full_contract_schema(self) -> None:
        ts = time.time()
        mid = _mid(record_wa_crew_event(
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
        ))
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
        mid = _mid(record_wa_crew_event(
            self.mem, capability_id="X", thread_id="t",
            sender="s", raw_text="hello", ts=time.time(),
            delivery_status="pending",
        ))
        # Pending status → lower confidence (delivery not confirmed)
        self.assertEqual(self.mem.get_memory(mid)["confidence"], 0.85)

    # ------------------------------------------------------------ estimate
    def test_estimate_evidence_with_pdf_path_classifies_as_sent_pdf(self) -> None:
        mid = _mid(record_estimate_evidence(
            self.mem,
            capability_id="ITEM-9",
            estimate_id="EST-042",
            customer_id="CUST-7",
            event_type="sent",
            pdf_path="/path/EST-042.pdf",
            amount_cents=125000,
            sent_to="alice@example.com",
            sent_at=time.time(),
        ))
        import json
        md = json.loads(self.mem.get_memory(mid)["metadata_json"])
        self.assertEqual(md["evidence_type"], "sent_pdf")
        self.assertEqual(md["estimate_id"], "EST-042")
        self.assertEqual(md["amount_cents"], 125000)

    def test_estimate_evidence_without_pdf_classifies_as_estimate_event(self) -> None:
        mid = _mid(record_estimate_evidence(
            self.mem, capability_id="ITEM-9",
            estimate_id="EST-042", customer_id="CUST-7",
            event_type="approved",
        ))
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
        mid = _mid(record_material_price_evidence(
            self.mem,
            capability_id="ITEM-MAT",
            sku="ROMEX-12-2",
            vendor="hd",
            price_cents=8999,
            unit="100ft",
            quoted_at=quoted_at,
            valid_to=valid_to,
            quote_source_path="hd:catalog:2026-04",
        ))
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

    # ------------------------------------------------------------ no-supersession
    def test_evidence_helpers_preserve_multiple_rows_without_supersession(self) -> None:
        """Repeated evidence inserts MUST NOT collapse through conflict
        detection — each evidence record is its own bi-temporal row.
        Caught by per-commit reviewer of d410019.
        """
        mid1 = _mid(record_evidence_artifact(
            self.mem, evidence_type="estimate_event",
            capability_id="ITEM-9", source_system="ae_dashboard",
            source_path="/path/A", content="estimate sent to alice",
            source_record_id="A",
        ))
        # Same capability_id + similar content — would normally trigger
        # supersession via detect_conflicts. Must NOT here.
        mid2 = _mid(record_evidence_artifact(
            self.mem, evidence_type="estimate_event",
            capability_id="ITEM-9", source_system="ae_dashboard",
            source_path="/path/B", content="estimate sent to alice (resend)",
            source_record_id="B",
        ))
        # Both rows should exist independently
        self.assertNotEqual(mid1, mid2)
        m1 = self.mem.get_memory(mid1)
        m2 = self.mem.get_memory(mid2)
        self.assertIsNotNone(m1, "first evidence row was superseded — contract violation")
        self.assertIsNotNone(m2, "second evidence row missing")
        # Neither should have valid_to set (no supersession applied)
        self.assertIsNone(m1.get("valid_to"),
                          "first evidence got valid_to set without explicit caller intent")

    def test_wa_same_second_ids_do_not_collide(self) -> None:
        """Two WA messages in same thread within one second must have
        distinct provenance — content hash + microsecond ts ensures
        uniqueness. Caught by per-commit reviewer of d410019.
        """
        ts = 1717350000.5  # half-second precision
        mid1 = _mid(record_wa_crew_event(
            self.mem, capability_id="X", thread_id="t",
            sender="m", raw_text="message one", ts=ts,
        ))
        mid2 = _mid(record_wa_crew_event(
            self.mem, capability_id="X", thread_id="t",
            sender="m", raw_text="message two", ts=ts,  # SAME ts
        ))
        self.assertNotEqual(mid1, mid2,
                            "WA messages with same ts collided on memory id")
        import json
        md1 = json.loads(self.mem.get_memory(mid1)["metadata_json"])
        md2 = json.loads(self.mem.get_memory(mid2)["metadata_json"])
        # Both rows persisted distinctly via record_evidence_artifact's
        # detect_conflicts=False contract. Provenance via different content
        # is guaranteed by the new content_hash component of source_record_id.
        self.assertEqual(md1["thread_id"], md2["thread_id"])
        # Verify the labels differ (which they do via source_record_id differing)
        self.assertNotEqual(self.mem.get_memory(mid1)["label"],
                            self.mem.get_memory(mid2)["label"],
                            "WA labels collided despite different content")

    # ------------------------------------------------------------ replay-authority (S2 packet 2026-05-03)
    def test_evidence_id_is_deterministic_across_calls(self) -> None:
        """_compute_evidence_id must be deterministic — same triple
        produces same id every time, in this process and across
        processes. Spec: sha256(f"{type}|{system}|{record}").hex[:16]."""
        a = _compute_evidence_id("wa_crew_message", "hermes_wa_bridge", "abc:123")
        b = _compute_evidence_id("wa_crew_message", "hermes_wa_bridge", "abc:123")
        self.assertEqual(a, b, "evidence_id must be deterministic")
        self.assertEqual(len(a), 16, "evidence_id must be 16 hex chars")
        self.assertTrue(all(c in "0123456789abcdef" for c in a),
                        "evidence_id must be lowercase hex")
        # Different triples produce different ids
        c = _compute_evidence_id("wa_crew_message", "hermes_wa_bridge", "abc:124")
        self.assertNotEqual(a, c, "different source_record_id must produce different id")
        d = _compute_evidence_id("sent_pdf", "hermes_wa_bridge", "abc:123")
        self.assertNotEqual(a, d, "different evidence_type must produce different id")
        e = _compute_evidence_id("wa_crew_message", "ae_dashboard", "abc:123")
        self.assertNotEqual(a, e, "different source_system must produce different id")

    def test_record_evidence_artifact_returns_structured_dict(self) -> None:
        """Return shape contract: {memory_id: int, evidence_id: str, inserted: bool}."""
        result = record_evidence_artifact(
            self.mem,
            evidence_type="sent_pdf",
            capability_id="ITEM-RET",
            source_system="ae_dashboard",
            source_path="/p",
            content="c",
            source_record_id="RET-1",
        )
        self.assertIsInstance(result, dict)
        self.assertIn("memory_id", result)
        self.assertIn("evidence_id", result)
        self.assertIn("inserted", result)
        self.assertIsInstance(result["memory_id"], int)
        self.assertIsInstance(result["evidence_id"], str)
        self.assertIsInstance(result["inserted"], bool)
        self.assertTrue(result["inserted"], "first insert must be inserted=True")
        self.assertEqual(
            result["evidence_id"],
            _compute_evidence_id("sent_pdf", "ae_dashboard", "RET-1"),
        )

    def test_record_evidence_artifact_upsert_returns_existing_memory_id(self) -> None:
        """Second call with the same (type, system, record_id) must return
        the existing memory_id with inserted=False — replay safety."""
        first = record_evidence_artifact(
            self.mem,
            evidence_type="wa_crew_message",
            capability_id="ITEM-UPS",
            source_system="hermes_wa_bridge",
            source_path="/p1",
            content="hola lote 27",
            source_record_id="thread-A:1234567890:abcd1234",
        )
        self.assertTrue(first["inserted"])

        # Replay the same record — different content, different source_path,
        # but identical (type, system, record_id) triple.
        second = record_evidence_artifact(
            self.mem,
            evidence_type="wa_crew_message",
            capability_id="ITEM-UPS",
            source_system="hermes_wa_bridge",
            source_path="/p2-different-but-same-key",
            content="completely different content",
            source_record_id="thread-A:1234567890:abcd1234",
        )
        self.assertFalse(second["inserted"], "replay must NOT insert a duplicate")
        self.assertEqual(first["memory_id"], second["memory_id"],
                         "replay must return the original memory_id")
        self.assertEqual(first["evidence_id"], second["evidence_id"])

        # And substrate still has only one row for this evidence_id
        with self.mem.store._lock:
            cnt = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE id IN (?, ?)",
                (first["memory_id"], second["memory_id"]),
            ).fetchone()[0]
        self.assertEqual(cnt, 1, "exactly one persisted row expected for the replayed key")

    def test_record_wa_crew_event_returns_structured_dict(self) -> None:
        """Specialized wrapper must forward the {memory_id, evidence_id,
        inserted} contract — not collapse to a bare int."""
        result = record_wa_crew_event(
            self.mem,
            capability_id="ITEM-WA-RET",
            thread_id="t-ret",
            sender="m",
            raw_text="message",
            ts=1717350000.5,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("memory_id", result)
        self.assertIn("evidence_id", result)
        self.assertIn("inserted", result)
        self.assertTrue(result["inserted"])

        # And replaying the same row dedupes via evidence_id (the WA-specific
        # source_record_id includes microsecond-ts + content hash, so identical
        # raw_text + ts + thread_id produces the same evidence_id).
        replay = record_wa_crew_event(
            self.mem,
            capability_id="ITEM-WA-RET",
            thread_id="t-ret",
            sender="m",
            raw_text="message",
            ts=1717350000.5,
        )
        self.assertFalse(replay["inserted"], "WA replay must dedupe")
        self.assertEqual(result["memory_id"], replay["memory_id"])

    def test_dryrun_evidence_id_matches_live_ingest(self) -> None:
        """Lockdown: tools/ingest_wa_dryrun.py::to_typed_record must produce
        the SAME evidence_id that record_wa_crew_event does for the same
        input row. Both call sites import _compute_evidence_id from
        ae_workflow_helpers — this test guards against a future refactor
        accidentally inlining a divergent computation."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from ingest_wa_dryrun import to_typed_record  # noqa: E402

        ts = 1717350000.5
        row = {
            "thread_id": "crew-spanish",
            "sender": "miguel",
            "raw_text": "el cable 12 ya llegó al lote 27",
            "ts": ts,
            "lang": "es",
            "delivery_status": "delivered",
            "capability_id": "ITEM-WA",
        }
        dryrun_record = to_typed_record(row)
        live_result = record_wa_crew_event(
            self.mem,
            capability_id="ITEM-WA",
            thread_id="crew-spanish",
            sender="miguel",
            raw_text="el cable 12 ya llegó al lote 27",
            ts=ts,
            lang="es",
        )
        self.assertEqual(
            dryrun_record["evidence_id"],
            live_result["evidence_id"],
            "dry-run and live evidence_id MUST match for the same input row",
        )
        self.assertEqual(
            dryrun_record["metadata"]["evidence_id"],
            live_result["evidence_id"],
            "dry-run metadata must carry the same evidence_id",
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

    # ------------------------------------------------------------ ledger (S2)
    def test_ledger_row_created_on_first_insert(self) -> None:
        """First record_evidence_artifact call must populate the ledger
        with evidence_id, memory_id, and the source identity tuple."""
        result = record_evidence_artifact(
            self.mem,
            evidence_type="sent_pdf",
            capability_id="ITEM-LED-1",
            source_system="ae_dashboard",
            source_path="/p1",
            content="estimate sent",
            source_record_id="LED-1",
        )
        self.assertTrue(result["inserted"])
        with self.mem.store._lock:
            row = self.mem.store.conn.execute(
                "SELECT evidence_id, memory_id, evidence_type, source_system, "
                "source_record_id, status FROM evidence_ledger "
                "WHERE evidence_id = ?",
                (result["evidence_id"],),
            ).fetchone()
        self.assertIsNotNone(row, "ledger row was not created on first insert")
        ev_id, mem_id, ev_type, src_sys, src_rec, status = row
        self.assertEqual(ev_id, result["evidence_id"])
        self.assertEqual(mem_id, result["memory_id"])
        self.assertEqual(ev_type, "sent_pdf")
        self.assertEqual(src_sys, "ae_dashboard")
        self.assertEqual(src_rec, "LED-1")
        self.assertEqual(status, "inserted")

    def test_replay_skips_remember_and_returns_existing(self) -> None:
        """Replay (same evidence_id) must NOT call mem.remember again — it
        must short-circuit through the ledger and return inserted=False
        with the existing memory_id."""
        first = record_evidence_artifact(
            self.mem,
            evidence_type="wa_crew_message",
            capability_id="ITEM-LED-2",
            source_system="hermes_wa_bridge",
            source_path="/p1",
            content="first content",
            source_record_id="thread-X:1700000000:abc",
        )
        self.assertTrue(first["inserted"])

        # Spy on mem.remember — replay must not invoke it.
        original_remember = self.mem.remember
        call_count = {"n": 0}

        def spy_remember(*a: Any, **kw: Any) -> int:
            call_count["n"] += 1
            return original_remember(*a, **kw)

        self.mem.remember = spy_remember  # type: ignore[method-assign]
        try:
            replay = record_evidence_artifact(
                self.mem,
                evidence_type="wa_crew_message",
                capability_id="ITEM-LED-2",
                source_system="hermes_wa_bridge",
                source_path="/p2-different",
                content="completely different content",
                source_record_id="thread-X:1700000000:abc",
            )
        finally:
            self.mem.remember = original_remember  # type: ignore[method-assign]

        self.assertEqual(call_count["n"], 0,
                         "replay path must not invoke mem.remember")
        self.assertFalse(replay["inserted"])
        self.assertEqual(replay["memory_id"], first["memory_id"])
        self.assertEqual(replay["evidence_id"], first["evidence_id"])

        # Ledger still has exactly one row for this evidence_id.
        with self.mem.store._lock:
            cnt = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger WHERE evidence_id = ?",
                (first["evidence_id"],),
            ).fetchone()[0]
        self.assertEqual(cnt, 1)

    def test_concurrent_inserts_yield_one_ledger_row(self) -> None:
        """Race test: 8 threads call record_evidence_artifact with the
        SAME (evidence_type, source_system, source_record_id). The ledger
        must end up with exactly one row, all callers return the same
        memory_id, and exactly one caller reports inserted=True."""
        import threading

        results: list[dict[str, Any]] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                barrier.wait()
                r = record_evidence_artifact(
                    self.mem,
                    evidence_type="estimate_event",
                    capability_id="ITEM-RACE",
                    source_system="ae_dashboard",
                    source_path="/race",
                    content="race content",
                    source_record_id="RACE-KEY-1",
                )
                with results_lock:
                    results.append(r)
            except Exception as e:
                with results_lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors, f"race produced exceptions: {errors}")
        self.assertEqual(len(results), 8)

        # Exactly one ledger row.
        evidence_id = results[0]["evidence_id"]
        with self.mem.store._lock:
            cnt = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()[0]
        self.assertEqual(cnt, 1, "race should produce exactly one ledger row")

        # All callers see the same memory_id + evidence_id.
        memory_ids = {r["memory_id"] for r in results}
        evidence_ids = {r["evidence_id"] for r in results}
        self.assertEqual(len(evidence_ids), 1)
        self.assertEqual(len(memory_ids), 1,
                         f"race produced divergent memory_ids: {memory_ids}")

        # At MOST one inserted=True (the race winner). If a caller raced
        # past the ledger reservation before another caller patched in
        # the memory_id, the loser may have made a fresh mem.remember()
        # — that's the documented fallback. The contract guarantees
        # AT LEAST one True; in practice with the lock-serialized
        # connection it should be exactly one.
        inserted_count = sum(1 for r in results if r["inserted"])
        self.assertGreaterEqual(inserted_count, 1,
                                "at least one caller must report inserted=True")
        self.assertLessEqual(inserted_count, 1,
                             "race should produce exactly one inserted=True caller")

    def test_helper_return_shape_preserved_across_all_helpers(self) -> None:
        """All 4 evidence helpers must return the contract dict
        {memory_id: int, evidence_id: str, inserted: bool}."""
        # 1. record_evidence_artifact
        r1 = record_evidence_artifact(
            self.mem, evidence_type="sent_pdf", capability_id="C1",
            source_system="s1", source_path="/p1", content="c1",
            source_record_id="rec-1",
        )
        # 2. record_wa_crew_event
        r2 = record_wa_crew_event(
            self.mem, capability_id="C2", thread_id="t1",
            sender="m", raw_text="hello", ts=time.time(),
        )
        # 3. record_estimate_evidence
        r3 = record_estimate_evidence(
            self.mem, capability_id="C3", estimate_id="E1",
            customer_id="CU1", event_type="sent",
            pdf_path="/p3.pdf", amount_cents=1000,
            sent_at=time.time(),
        )
        # 4. record_material_price_evidence
        r4 = record_material_price_evidence(
            self.mem, capability_id="C4", sku="SKU-1",
            vendor="hd", price_cents=500,
            quoted_at=time.time(),
            quote_source_path="/p4",
        )
        for label, r in [
            ("artifact", r1), ("wa", r2), ("estimate", r3), ("material", r4),
        ]:
            self.assertIsInstance(r, dict, f"{label} must return dict")
            self.assertEqual(set(r.keys()),
                             {"memory_id", "evidence_id", "inserted"},
                             f"{label} return shape mismatch: {set(r.keys())}")
            self.assertIsInstance(r["memory_id"], int,
                                  f"{label} memory_id must be int")
            self.assertIsInstance(r["evidence_id"], str,
                                  f"{label} evidence_id must be str")
            self.assertIsInstance(r["inserted"], bool,
                                  f"{label} inserted must be bool")
            self.assertEqual(len(r["evidence_id"]), 16,
                             f"{label} evidence_id must be 16 hex chars")


    # ------------------------------------------------------------ S4b: source_record_id required (P0 contract)
    def test_record_evidence_artifact_rejects_none_source_record_id_by_default(
        self,
    ) -> None:
        """record_evidence_artifact must raise ValueError when source_record_id
        is None and allow_unkeyed_nonprod is False (the default). This guards
        production callers from accidentally creating replay-unkeyed rows."""
        with self.assertRaises(ValueError) as ctx:
            record_evidence_artifact(
                self.mem,
                evidence_type="sent_pdf",
                capability_id="ITEM-KEY",
                source_system="ae_dashboard",
                source_path="/p",
                content="c",
                # source_record_id omitted (defaults to None)
                # allow_unkeyed_nonprod omitted (defaults to False)
            )
        msg = str(ctx.exception)
        self.assertIn("source_record_id", msg)
        self.assertIn("allow_unkeyed_nonprod", msg)

    def test_record_evidence_artifact_allows_unkeyed_with_opt_in(self) -> None:
        """record_evidence_artifact with source_record_id=None and
        allow_unkeyed_nonprod=True must succeed and return the standard
        result dict (unkeyed ad-hoc path)."""
        result = record_evidence_artifact(
            self.mem,
            evidence_type="sent_pdf",
            capability_id="ITEM-UNKEYED",
            source_system="ae_dashboard",
            source_path="/p",
            content="unkeyed test content",
            allow_unkeyed_nonprod=True,
            # source_record_id intentionally None
        )
        self.assertIsInstance(result, dict)
        self.assertIn("memory_id", result)
        self.assertIn("evidence_id", result)
        self.assertIn("inserted", result)
        self.assertTrue(result["inserted"])
        self.assertIsInstance(result["memory_id"], int)

    def test_ledger_loser_timeout_fails_closed_no_duplicate_insert(self) -> None:
        """When the ledger loser's timeout expires (winner never patches
        memory_id), record_evidence_artifact must return a pending/failure
        result — NOT create a second row.

        Simulated by: inserting a ledger row with NULL memory_id directly
        (mimicking a mid-flight winner that stalled), then calling
        record_evidence_artifact with the same evidence_id. The function
        will lose the INSERT OR IGNORE race, poll, time out, and must
        return status='pending_winner' without calling mem.remember().
        """
        import threading

        evidence_type = "estimate_event"
        source_system = "ae_dashboard"
        source_record_id = "TIMEOUT-TEST-KEY-1"

        from ae_workflow_helpers import _compute_evidence_id  # noqa: F401 (already imported)
        evidence_id = _compute_evidence_id(evidence_type, source_system, source_record_id)

        # Directly plant a ledger row with NULL memory_id — simulates a winner
        # that claimed the row but hasn't called mem.remember() yet.
        with self.mem.store._lock:
            self.mem.store.conn.execute(
                "INSERT OR IGNORE INTO evidence_ledger "
                "(evidence_id, memory_id, evidence_type, source_system, "
                " source_record_id, status) "
                "VALUES (?, NULL, ?, ?, ?, 'inserted')",
                (evidence_id, evidence_type, source_system, source_record_id),
            )
            self.mem.store.conn.commit()

        # Confirm the planted row exists with NULL memory_id.
        with self.mem.store._lock:
            row = self.mem.store.conn.execute(
                "SELECT memory_id FROM evidence_ledger WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row[0], "planted row should have NULL memory_id")

        # Spy: mem.remember must NOT be called by the loser-timeout path.
        original_remember = self.mem.remember
        call_count = {"n": 0}

        def spy_remember(*a, **kw):
            call_count["n"] += 1
            return original_remember(*a, **kw)

        self.mem.remember = spy_remember  # type: ignore[method-assign]
        try:
            result = record_evidence_artifact(
                self.mem,
                evidence_type=evidence_type,
                capability_id="ITEM-TIMEOUT",
                source_system=source_system,
                source_path="/timeout-test",
                content="should not be inserted",
                source_record_id=source_record_id,
            )
        finally:
            self.mem.remember = original_remember  # type: ignore[method-assign]

        # mem.remember must NOT have been invoked.
        self.assertEqual(call_count["n"], 0,
                         "loser-timeout path must not invoke mem.remember")

        # Return shape must be pending/failure.
        self.assertIsInstance(result, dict)
        self.assertFalse(result["inserted"])
        self.assertIsNone(result["memory_id"])
        self.assertEqual(result.get("status"), "pending_winner",
                         "loser-timeout must return status='pending_winner'")
        self.assertEqual(result["evidence_id"], evidence_id)

        # Exactly one ledger row for this evidence_id (no duplicate insert).
        with self.mem.store._lock:
            cnt = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM evidence_ledger WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()[0]
        self.assertEqual(cnt, 1, "loser-timeout must not create a duplicate ledger row")

        # Exactly zero memory rows referencing this evidence_id (no duplicate evidence).
        with self.mem.store._lock:
            mem_cnt = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE "
                "json_extract(metadata_json, '$.evidence_id') = ?",
                (evidence_id,),
            ).fetchone()[0]
        self.assertEqual(mem_cnt, 0,
                         "loser-timeout must not insert a memory row")


if __name__ == "__main__":
    unittest.main(verbosity=2)
