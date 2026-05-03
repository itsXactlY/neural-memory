"""Tests for tools/ingest_wa_dryrun.py — Sonnet packet S5 (2026-05-03).

Hardened-validator coverage:
  - required-field presence + null detection
  - ts ISO8601 vs epoch numeric rejection
  - thread_id pattern WARNING (group/DM)
  - sender type + non-empty
  - delivery_status enum
  - privacy_class enum (tightened to {internal, financial, pii_low})
  - lang ISO 639-1 WARNING
  - media_paths element-level (type, non-empty, no shell metachars)
  - consumer_hint type
  - boundary_violation_suspect type
  - normalized_text type
  - auth_proof type
  - evidence_id pre-image (format + computed-match)
  - per-row report shape (row_index, valid, errors, warnings,
    computed_evidence_id)
  - exit codes (0 all-valid, 2 io failure, 3 any invalid)
  - evidence_id parity with ae_workflow_helpers._compute_evidence_id

Hermetic — no real WA data, no substrate writes.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import ingest_wa_dryrun as wa  # noqa: E402
from ae_workflow_helpers import _compute_evidence_id  # noqa: E402


def _good_row(**overrides) -> dict:
    """A valid baseline WA row; override per-test."""
    base = {
        "thread_id": "120363021234567890@g.us",
        "sender": "miguel",
        "raw_text": "el cable 12 ya llegó al lote 27",
        "ts": "2026-05-03T10:41:47+00:00",
        "lang": "es",
        "delivery_status": "delivered",
        "privacy_class": "internal",
        "capability_id": "ITEM-WA-LENNAR",
    }
    base.update(overrides)
    return base


class RequiredFieldsTests(unittest.TestCase):
    def test_baseline_row_is_valid(self) -> None:
        rep = wa.validate_row(_good_row(), 1)
        self.assertTrue(rep["valid"], rep["errors"])
        self.assertEqual(rep["errors"], [])
        self.assertIsNotNone(rep["computed_evidence_id"])
        self.assertEqual(rep["row_index"], 1)

    def test_missing_thread_id_rejected(self) -> None:
        row = _good_row()
        del row["thread_id"]
        rep = wa.validate_row(row, 7)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("thread_id" in e for e in rep["errors"]))
        self.assertEqual(rep["row_index"], 7)

    def test_null_sender_rejected(self) -> None:
        rep = wa.validate_row(_good_row(sender=None), 2)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("sender" in e and "null" in e for e in rep["errors"]))

    def test_empty_raw_text_rejected(self) -> None:
        rep = wa.validate_row(_good_row(raw_text="   "), 3)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("raw_text" in e for e in rep["errors"]))

    def test_missing_ts_rejected(self) -> None:
        row = _good_row()
        del row["ts"]
        rep = wa.validate_row(row, 4)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("ts" in e for e in rep["errors"]))

    def test_non_dict_row_rejected(self) -> None:
        rep = wa.validate_row(["not", "a", "dict"], 5)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("JSON object" in e for e in rep["errors"]))


class TimestampTests(unittest.TestCase):
    def test_iso8601_with_z_accepted(self) -> None:
        rep = wa.validate_row(_good_row(ts="2026-05-03T10:41:47Z"), 1)
        self.assertTrue(rep["valid"], rep["errors"])

    def test_iso8601_with_microseconds_accepted(self) -> None:
        rep = wa.validate_row(_good_row(ts="2026-05-03T10:41:47.123456+00:00"), 1)
        self.assertTrue(rep["valid"], rep["errors"])

    def test_epoch_int_rejected(self) -> None:
        rep = wa.validate_row(_good_row(ts=1717350000), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("ts" in e and "ISO8601" in e for e in rep["errors"]))

    def test_epoch_float_rejected(self) -> None:
        rep = wa.validate_row(_good_row(ts=1717350000.5), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("ts" in e and "ISO8601" in e for e in rep["errors"]))

    def test_epoch_as_string_rejected(self) -> None:
        rep = wa.validate_row(_good_row(ts="1717350000"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("ts" in e for e in rep["errors"]))

    def test_ambiguous_format_rejected(self) -> None:
        rep = wa.validate_row(_good_row(ts="May 3 2026 10:41am"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("ts" in e for e in rep["errors"]))


class ThreadIdPatternTests(unittest.TestCase):
    def test_group_thread_no_warning(self) -> None:
        rep = wa.validate_row(_good_row(thread_id="120363021234567890@g.us"), 1)
        self.assertEqual(rep["warnings"], [])
        self.assertTrue(rep["valid"])

    def test_dm_thread_no_warning(self) -> None:
        rep = wa.validate_row(_good_row(thread_id="14155551212@s.whatsapp.net"), 1)
        self.assertEqual(rep["warnings"], [])
        self.assertTrue(rep["valid"])

    def test_unrecognized_pattern_warns_but_valid(self) -> None:
        rep = wa.validate_row(_good_row(thread_id="some-legacy-id"), 1)
        self.assertTrue(rep["valid"], rep["errors"])
        self.assertTrue(any("thread_id" in w for w in rep["warnings"]))


class PrivacyEnumTests(unittest.TestCase):
    def test_internal_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(privacy_class="internal"), 1)["valid"])

    def test_financial_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(privacy_class="financial"), 1)["valid"])

    def test_pii_low_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(privacy_class="pii_low"), 1)["valid"])

    def test_public_rejected_per_packet_contract(self) -> None:
        # Helper allows public/pii_high for non-WA evidence; WA contract
        # tightens to 3-value subset per packet S5.
        rep = wa.validate_row(_good_row(privacy_class="public"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("privacy_class" in e for e in rep["errors"]))

    def test_pii_high_rejected_per_packet_contract(self) -> None:
        rep = wa.validate_row(_good_row(privacy_class="pii_high"), 1)
        self.assertFalse(rep["valid"])

    def test_garbage_rejected(self) -> None:
        rep = wa.validate_row(_good_row(privacy_class="WHATEVER"), 1)
        self.assertFalse(rep["valid"])


class LangCodeTests(unittest.TestCase):
    def test_es_accepted_no_warning(self) -> None:
        rep = wa.validate_row(_good_row(lang="es"), 1)
        self.assertTrue(rep["valid"])
        self.assertEqual(rep["warnings"], [])

    def test_en_accepted_no_warning(self) -> None:
        rep = wa.validate_row(_good_row(lang="en"), 1)
        self.assertTrue(rep["valid"])
        self.assertEqual(rep["warnings"], [])

    def test_three_letter_warns(self) -> None:
        rep = wa.validate_row(_good_row(lang="spa"), 1)
        self.assertTrue(rep["valid"])
        self.assertTrue(any("lang" in w for w in rep["warnings"]))

    def test_uppercase_warns(self) -> None:
        rep = wa.validate_row(_good_row(lang="ES"), 1)
        self.assertTrue(rep["valid"])
        self.assertTrue(any("lang" in w for w in rep["warnings"]))

    def test_non_string_lang_rejected(self) -> None:
        rep = wa.validate_row(_good_row(lang=123), 1)
        self.assertFalse(rep["valid"])


class MediaPathsTests(unittest.TestCase):
    def test_none_accepted(self) -> None:
        row = _good_row()
        row["media_paths"] = None
        # Note: dict.get returns None default already; explicit None allowed.
        self.assertTrue(wa.validate_row(row, 1)["valid"])

    def test_empty_list_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(media_paths=[]), 1)["valid"])

    def test_valid_paths_accepted(self) -> None:
        rep = wa.validate_row(_good_row(media_paths=[
            "/var/wa/img1.jpg",
            "relative/path/audio.ogg",
        ]), 1)
        self.assertTrue(rep["valid"], rep["errors"])

    def test_non_list_rejected(self) -> None:
        rep = wa.validate_row(_good_row(media_paths="just-one-path.jpg"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("media_paths" in e and "list" in e for e in rep["errors"]))

    def test_non_string_element_rejected(self) -> None:
        rep = wa.validate_row(_good_row(media_paths=["ok.jpg", 42]), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("media_paths[1]" in e for e in rep["errors"]))

    def test_empty_string_element_rejected(self) -> None:
        rep = wa.validate_row(_good_row(media_paths=["ok.jpg", "  "]), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("media_paths[1]" in e and "non-empty" in e for e in rep["errors"]))

    def test_shell_metachar_rejected(self) -> None:
        for danger in ["a;rm.jpg", "a|b.jpg", "a`whoami`.jpg", "a$x.jpg",
                       "a&b.jpg", "a<x.jpg", "a>x.jpg"]:
            with self.subTest(path=danger):
                rep = wa.validate_row(_good_row(media_paths=[danger]), 1)
                self.assertFalse(rep["valid"], f"expected reject for {danger!r}")
                self.assertTrue(any("shell metacharacter" in e for e in rep["errors"]))


class TypedFieldTests(unittest.TestCase):
    def test_consumer_hint_string_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(consumer_hint="materials"), 1)["valid"])

    def test_consumer_hint_int_rejected(self) -> None:
        rep = wa.validate_row(_good_row(consumer_hint=42), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("consumer_hint" in e for e in rep["errors"]))

    def test_boundary_violation_suspect_bool_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(boundary_violation_suspect=True), 1)["valid"])
        self.assertTrue(wa.validate_row(_good_row(boundary_violation_suspect=False), 1)["valid"])

    def test_boundary_violation_suspect_string_rejected(self) -> None:
        rep = wa.validate_row(_good_row(boundary_violation_suspect="true"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("boundary_violation_suspect" in e for e in rep["errors"]))

    def test_normalized_text_string_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(normalized_text="The wire 12 arrived at lot 27"), 1)["valid"])

    def test_normalized_text_dict_rejected(self) -> None:
        rep = wa.validate_row(_good_row(normalized_text={"en": "x"}), 1)
        self.assertFalse(rep["valid"])

    def test_auth_proof_dict_accepted(self) -> None:
        self.assertTrue(wa.validate_row(_good_row(auth_proof={"receipt": "abc"}), 1)["valid"])

    def test_auth_proof_string_rejected(self) -> None:
        rep = wa.validate_row(_good_row(auth_proof="raw-receipt-string"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("auth_proof" in e for e in rep["errors"]))


class DeliveryStatusTests(unittest.TestCase):
    def test_all_canonical_accepted(self) -> None:
        for d in ["delivered", "pending", "failed", "read"]:
            with self.subTest(d=d):
                self.assertTrue(wa.validate_row(_good_row(delivery_status=d), 1)["valid"])

    def test_unknown_rejected(self) -> None:
        rep = wa.validate_row(_good_row(delivery_status="ghosted"), 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("delivery_status" in e for e in rep["errors"]))


class EvidenceIdTests(unittest.TestCase):
    def test_computed_evidence_id_present_for_valid_row(self) -> None:
        row = _good_row()
        rep = wa.validate_row(row, 1)
        self.assertTrue(rep["valid"])
        self.assertIsNotNone(rep["computed_evidence_id"])
        # Verify it's the 16-hex shape
        self.assertRegex(rep["computed_evidence_id"], r"^[0-9a-f]{16}$")

    def test_supplied_evidence_id_matching_accepted(self) -> None:
        row = _good_row()
        # Pre-compute what the validator will compute, supply it back
        expected = wa._compute_expected_evidence_id(row)
        row["evidence_id"] = expected
        rep = wa.validate_row(row, 1)
        self.assertTrue(rep["valid"], rep["errors"])

    def test_supplied_evidence_id_mismatch_rejected(self) -> None:
        row = _good_row()
        row["evidence_id"] = "deadbeefcafebabe"  # 16 hex but wrong
        rep = wa.validate_row(row, 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("evidence_id" in e and "mismatch" in e for e in rep["errors"]))

    def test_supplied_evidence_id_wrong_format_rejected(self) -> None:
        row = _good_row()
        row["evidence_id"] = "not-hex"
        rep = wa.validate_row(row, 1)
        self.assertFalse(rep["valid"])
        self.assertTrue(any("evidence_id" in e for e in rep["errors"]))

    def test_supplied_evidence_id_uppercase_rejected(self) -> None:
        row = _good_row()
        # Uppercase hex — should be rejected by the lowercase regex
        row["evidence_id"] = "DEADBEEFCAFEBABE"
        rep = wa.validate_row(row, 1)
        self.assertFalse(rep["valid"])

    def test_supplied_evidence_id_non_string_rejected(self) -> None:
        row = _good_row()
        row["evidence_id"] = 12345
        rep = wa.validate_row(row, 1)
        self.assertFalse(rep["valid"])

    def test_evidence_id_parity_with_helper(self) -> None:
        """The validator's computed evidence_id MUST match what
        ae_workflow_helpers._compute_evidence_id produces for the same
        record_id triple. Locks the formula to single source of truth."""
        row = _good_row(
            thread_id="120363021234567890@g.us",
            ts="2026-05-03T10:41:47+00:00",
            raw_text="parity-check",
        )
        validator_eid = wa._compute_expected_evidence_id(row)

        # Recompute by hand using the helper directly
        ts_epoch = wa._ts_to_epoch(row["ts"])
        record_id = wa._compute_record_id(row["thread_id"], ts_epoch, row["raw_text"])
        helper_eid = _compute_evidence_id(
            evidence_type="wa_crew_message",
            source_system="hermes_wa_bridge",
            source_record_id=record_id,
        )

        self.assertEqual(
            validator_eid, helper_eid,
            "validator evidence_id MUST match _compute_evidence_id helper output",
        )


class ToTypedRecordBackCompatTests(unittest.TestCase):
    """to_typed_record() must accept numeric ts for back-compat with the S2
    lockdown test in test_ae_evidence_ingest.py."""

    def test_numeric_ts_accepted(self) -> None:
        row = {
            "thread_id": "crew-spanish",
            "sender": "miguel",
            "raw_text": "el cable 12 ya llegó al lote 27",
            "ts": 1717350000.5,
            "lang": "es",
            "delivery_status": "delivered",
            "capability_id": "ITEM-WA",
        }
        rec = wa.to_typed_record(row)
        self.assertEqual(rec["valid_from"], 1717350000.5)
        self.assertRegex(rec["evidence_id"], r"^[0-9a-f]{16}$")

    def test_iso8601_ts_accepted(self) -> None:
        row = _good_row(ts="2026-05-03T10:41:47+00:00")
        rec = wa.to_typed_record(row)
        self.assertIsInstance(rec["valid_from"], float)
        self.assertRegex(rec["evidence_id"], r"^[0-9a-f]{16}$")


class S5cAuthProofParityTests(unittest.TestCase):
    """S5c: auth_proof key-presence/is-not-None semantics in to_typed_record.

    Explicit empty auth_proof={} must be preserved in metadata (was dropped
    because `if row.get('auth_proof')` evaluates {} as falsy).
    """

    def test_empty_auth_proof_preserved(self) -> None:
        """auth_proof={} must survive into metadata — key present, value {}."""
        row = _good_row()
        row["auth_proof"] = {}
        rec = wa.to_typed_record(row)
        self.assertIn("auth_proof", rec["metadata"],
                      "empty auth_proof={} must be preserved in metadata")
        self.assertEqual(rec["metadata"]["auth_proof"], {})

    def test_none_auth_proof_dropped(self) -> None:
        """auth_proof=null must be excluded from metadata."""
        row = _good_row()
        row["auth_proof"] = None
        rec = wa.to_typed_record(row)
        self.assertNotIn("auth_proof", rec["metadata"],
                         "null auth_proof must be excluded from metadata")

    def test_non_empty_auth_proof_preserved(self) -> None:
        """Non-empty auth_proof={"receipt": "abc"} is preserved as before."""
        row = _good_row()
        row["auth_proof"] = {"receipt": "abc123"}
        rec = wa.to_typed_record(row)
        self.assertIn("auth_proof", rec["metadata"])
        self.assertEqual(rec["metadata"]["auth_proof"], {"receipt": "abc123"})

    def test_absent_auth_proof_dropped(self) -> None:
        """Row with no auth_proof key must produce no auth_proof in metadata."""
        row = {k: v for k, v in _good_row().items() if k != "auth_proof"}
        rec = wa.to_typed_record(row)
        self.assertNotIn("auth_proof", rec["metadata"])


class CLIExitCodeTests(unittest.TestCase):
    """End-to-end CLI invocation in a tempdir to verify exit-code semantics."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.tool = ROOT / "tools" / "ingest_wa_dryrun.py"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, batch_lines: list[str], extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
        batch = self.tmp / "batch.jsonl"
        batch.write_text("\n".join(batch_lines) + "\n")
        out_dir = self.tmp / "dryruns"
        cmd = [sys.executable, str(self.tool), "--in", str(batch), "--out-dir", str(out_dir)]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_exit_0_all_valid(self) -> None:
        line = json.dumps(_good_row())
        r = self._run([line])
        self.assertEqual(r.returncode, 0, f"stdout={r.stdout!r} stderr={r.stderr!r}")
        self.assertIn("valid: 1", r.stdout)
        self.assertIn("invalid: 0", r.stdout)

    def test_exit_3_any_invalid(self) -> None:
        good = json.dumps(_good_row())
        bad = json.dumps(_good_row(privacy_class="WHATEVER"))
        r = self._run([good, bad])
        self.assertEqual(r.returncode, 3)
        self.assertIn("valid: 1", r.stdout)
        self.assertIn("invalid: 1", r.stdout)
        self.assertIn("rejection reasons", r.stdout)

    def test_exit_2_missing_input_file(self) -> None:
        cmd = [
            sys.executable, str(self.tool),
            "--in", str(self.tmp / "nope.jsonl"),
            "--out-dir", str(self.tmp / "dryruns"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("input file not found", r.stderr)

    def test_report_jsonl_mode_emits_per_row_records(self) -> None:
        good = json.dumps(_good_row())
        bad = json.dumps(_good_row(ts=1717350000))
        r = self._run([good, bad], extra_args=["--report-jsonl"])
        self.assertEqual(r.returncode, 3)
        lines = [ln for ln in r.stdout.strip().split("\n") if ln.strip()]
        self.assertEqual(len(lines), 2)
        for ln in lines:
            rep = json.loads(ln)
            self.assertIn("row_index", rep)
            self.assertIn("valid", rep)
            self.assertIn("errors", rep)
            self.assertIn("warnings", rep)
            self.assertIn("computed_evidence_id", rep)
        self.assertTrue(json.loads(lines[0])["valid"])
        self.assertFalse(json.loads(lines[1])["valid"])

    def test_invalid_rows_excluded_from_dryrun_output(self) -> None:
        good = json.dumps(_good_row())
        bad = json.dumps(_good_row(thread_id=None))
        r = self._run([good, bad])
        self.assertEqual(r.returncode, 3)
        out_dir = self.tmp / "dryruns"
        out_files = list(out_dir.glob("wa-*.jsonl"))
        self.assertEqual(len(out_files), 1)
        records = [json.loads(ln) for ln in out_files[0].read_text().strip().split("\n") if ln.strip()]
        self.assertEqual(len(records), 1, "only valid row should be in dryrun output")

    def test_blank_and_comment_lines_skipped(self) -> None:
        good = json.dumps(_good_row())
        r = self._run(["", "# comment", good, "", "# trailing"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("valid: 1", r.stdout)


class LegacyShimTests(unittest.TestCase):
    """validate_row_legacy returns the pre-S5 (ok, err) tuple shape."""

    def test_valid_returns_true_none(self) -> None:
        ok, err = wa.validate_row_legacy(_good_row(), 1)
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_invalid_returns_false_message(self) -> None:
        ok, err = wa.validate_row_legacy(_good_row(privacy_class="bogus"), 1)
        self.assertFalse(ok)
        self.assertIsNotNone(err)
        self.assertIn("privacy_class", err)


if __name__ == "__main__":
    unittest.main()
