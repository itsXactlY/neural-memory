"""Tests for tools/ingest_sent_pdf_sidecars.py — NM-side tail of AE
sent-PDF sidecars per Sonnet packet S-OptE (2026-05-03).

These tests are hermetic — they patch ae_workflow_helpers.record_evidence_artifact
and use synthetic sidecars in a tempdir. No substrate write, no real Gmail data.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the tool importable as a module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import ingest_sent_pdf_sidecars as ingest  # noqa: E402


def make_sidecar(
    sidecar_dir: Path,
    msg_id: str,
    *,
    text: str = "Sample PDF text",
    downloaded_at: str = "2026-05-02T05:13:53.556661+00:00",
    filename: str = "estimate.pdf",
    page_count: int = 1,
    extra: dict | None = None,
) -> Path:
    payload = {
        "msg_id": msg_id,
        "thread_id": msg_id,
        "subject": "Test estimate",
        "from": "\"Angel's Electric\" <angelselectricservice@gmail.com>",
        "to": "customer@example.com",
        "date": "Wed, 15 Apr 2026 20:27:36 -0500",
        "filename": filename,
        "size_bytes": 12345,
        "text": text,
        "extraction": {"page_count": page_count, "method": "pdfplumber"},
        "dollar_total_guess": None,
        "downloaded_at": downloaded_at,
    }
    if extra:
        payload.update(extra)
    p = sidecar_dir / f"{msg_id}_customer_{filename.replace('.pdf', '')}.json"
    p.write_text(json.dumps(payload))
    return p


class IngestSentPdfSidecarsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sidecar_dir = self.tmp / "sidecars"
        self.sidecar_dir.mkdir()
        self.watermark = self.tmp / "watermark.json"
        self.dryrun_dir = self.tmp / "dryrun"
        self.live_dir = self.tmp / "live"
        # Patch the module-level output dirs so tests don't pollute ~/.neural_memory
        self._dr_patcher = patch.object(ingest, "DRYRUN_DIR", self.dryrun_dir)
        self._lv_patcher = patch.object(ingest, "LIVE_DIR", self.live_dir)
        self._dr_patcher.start()
        self._lv_patcher.start()

    def tearDown(self) -> None:
        self._dr_patcher.stop()
        self._lv_patcher.stop()
        self._tmp.cleanup()

    def _run(self, *argv: str, mock_record=None) -> int:
        full_argv = [
            "ingest_sent_pdf_sidecars.py",
            "--sidecar-dir", str(self.sidecar_dir),
            "--watermark", str(self.watermark),
            *argv,
        ]
        with patch.object(sys, "argv", full_argv):
            if mock_record is None:
                # Default: fail if the tool tries to call record_evidence_artifact
                with patch.object(
                    ingest, "record_evidence_artifact",
                    side_effect=AssertionError(
                        "record_evidence_artifact must NOT be called in dry-run"
                    ),
                ):
                    return ingest.main()
            with patch.object(ingest, "record_evidence_artifact", mock_record):
                # --live also imports memory_client — stub it
                with patch.dict(sys.modules, {"memory_client": MagicMock()}):
                    return ingest.main()

    # ---------------------------------------------------- contract tests

    def test_dry_run_default_no_substrate_write(self) -> None:
        make_sidecar(self.sidecar_dir, "msg_a")
        make_sidecar(self.sidecar_dir, "msg_b")
        # default mock_record asserts record_evidence_artifact is NOT called
        rc = self._run("--backfill")
        self.assertEqual(rc, 0)
        # exactly one dry-run output file
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        self.assertEqual(len(outs), 1)
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertTrue(r["dry_run"])
            self.assertIsNone(r["memory_id"])
            self.assertIsNone(r["inserted"])
            self.assertIn("evidence_id", r)

    def test_live_mode_calls_record_evidence_artifact(self) -> None:
        make_sidecar(self.sidecar_dir, "msg_a", text="Estimate body A")
        mock = MagicMock(return_value={
            "memory_id": 42, "evidence_id": "abc123", "inserted": True,
        })
        rc = self._run("--backfill", "--live", mock_record=mock)
        self.assertEqual(rc, 0)
        self.assertEqual(mock.call_count, 1)
        _, kwargs = mock.call_args
        self.assertEqual(kwargs["evidence_type"], "sent_pdf")
        self.assertEqual(kwargs["source_system"], "sent_estimate_pdf_miner")
        self.assertEqual(kwargs["source_record_id"], "msg_a")
        self.assertEqual(kwargs["content"], "Estimate body A")
        self.assertEqual(kwargs["capability_id"], "ITEM-SENT-PDF")
        self.assertEqual(kwargs["privacy_class"], "financial")
        self.assertIn("subject", kwargs["extra_metadata"])
        self.assertIn("filename", kwargs["extra_metadata"])
        self.assertIn("page_count", kwargs["extra_metadata"])
        self.assertEqual(kwargs["extra_metadata"]["extraction_method"], "pdfplumber")
        # valid_from must be parsed into an epoch float
        self.assertIsInstance(kwargs["valid_from"], float)
        self.assertGreater(kwargs["valid_from"], 1_700_000_000)

    def test_idempotent_via_msg_id(self) -> None:
        """Re-running over same sidecar with same evidence_id uses upsert
        path (record_evidence_artifact returns inserted=False the second time).
        We simulate that on the mock side."""
        make_sidecar(self.sidecar_dir, "msg_dup")

        # First call: inserted; second call (replay): existing
        call_results = [
            {"memory_id": 7, "evidence_id": "evid", "inserted": True},
            {"memory_id": 7, "evidence_id": "evid", "inserted": False},
        ]

        mock = MagicMock(side_effect=call_results)
        # Backfill once, then again; both should call (mock simulates dedup)
        rc1 = self._run("--backfill", "--live", mock_record=mock)
        # Reset watermark side-effect by deleting it (--backfill should write
        # the watermark; we want to simulate a true replay over same sidecar)
        if self.watermark.exists():
            self.watermark.unlink()
        rc2 = self._run("--backfill", "--live", mock_record=mock)
        self.assertEqual(rc1, 0)
        self.assertEqual(rc2, 0)
        self.assertEqual(mock.call_count, 2)
        # The source_record_id passed both times must be identical (msg_id)
        first_kwargs = mock.call_args_list[0].kwargs
        second_kwargs = mock.call_args_list[1].kwargs
        self.assertEqual(first_kwargs["source_record_id"], "msg_dup")
        self.assertEqual(second_kwargs["source_record_id"], "msg_dup")

    def test_watermark_skips_already_processed(self) -> None:
        make_sidecar(self.sidecar_dir, "msg_old")
        make_sidecar(self.sidecar_dir, "msg_new")

        # Seed watermark with msg_old as already processed
        self.watermark.parent.mkdir(parents=True, exist_ok=True)
        self.watermark.write_text(json.dumps({
            "processed_msg_ids": ["msg_old"],
            "last_run_ts": 1700000000,
        }))

        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        # Live + NO backfill → must consult watermark
        rc = self._run("--live", mock_record=mock)
        self.assertEqual(rc, 0)
        self.assertEqual(mock.call_count, 1)
        only_call_kwargs = mock.call_args.kwargs
        self.assertEqual(only_call_kwargs["source_record_id"], "msg_new")
        # Watermark must now contain BOTH (merge of pre-existing + new)
        wm = json.loads(self.watermark.read_text())
        self.assertEqual(set(wm["processed_msg_ids"]), {"msg_old", "msg_new"})

    def test_backfill_ignores_watermark(self) -> None:
        make_sidecar(self.sidecar_dir, "msg_a")
        make_sidecar(self.sidecar_dir, "msg_b")
        self.watermark.parent.mkdir(parents=True, exist_ok=True)
        self.watermark.write_text(json.dumps({
            "processed_msg_ids": ["msg_a", "msg_b"],
            "last_run_ts": 1700000000,
        }))
        # Dry-run + backfill: should still process both even though watermark
        # has them; default mock_record asserts NO record_evidence_artifact call
        rc = self._run("--backfill")
        self.assertEqual(rc, 0)
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        # All rows processed (dry_run=True), none skipped_watermark=True
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertTrue(r.get("dry_run"))
            self.assertNotIn("skipped_watermark", r)

    def test_handles_malformed_sidecar(self) -> None:
        # Good sidecar
        make_sidecar(self.sidecar_dir, "msg_good")
        # Bad JSON
        (self.sidecar_dir / "bad_json.json").write_text("{not valid json")
        # Missing required field (no msg_id)
        (self.sidecar_dir / "missing_field.json").write_text(json.dumps({
            "thread_id": "x", "text": "y", "downloaded_at": "2026-05-02T00:00:00+00:00",
        }))

        rc = self._run("--backfill")
        # Errors present → exit 3 (not 0), but tool should NOT crash
        self.assertEqual(rc, 3)
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        # 3 sidecar files → 3 rows (1 ok + 2 errors)
        self.assertEqual(len(rows), 3)
        ok = [r for r in rows if "error" not in r]
        bad = [r for r in rows if "error" in r]
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(bad), 2)
        # Both error rows include the file path
        for r in bad:
            self.assertIn("sidecar_path", r)
            self.assertIsInstance(r["error"], str)

    def test_metadata_shape_matches_spec(self) -> None:
        make_sidecar(
            self.sidecar_dir, "msg_meta",
            text="body",
            extra={"dollar_total_guess": 1234.56},
        )
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run("--backfill", "--live", mock_record=mock)
        self.assertEqual(rc, 0)
        kwargs = mock.call_args.kwargs
        meta = kwargs["extra_metadata"]
        # Per packet spec — these MUST appear in the metadata
        for key in ("thread_id", "subject", "from", "to", "date",
                    "filename", "dollar_total_guess"):
            self.assertIn(key, meta, f"metadata missing required key {key!r}")
        # Bonus fields from extraction
        self.assertIn("page_count", meta)
        self.assertEqual(meta["dollar_total_guess"], 1234.56)
        # Top-level contract fields
        self.assertEqual(kwargs["evidence_type"], "sent_pdf")
        self.assertEqual(kwargs["capability_id"], "ITEM-SENT-PDF")
        # capability_id is on top-level kwargs (record_evidence_artifact
        # writes it to metadata.capability_id internally), per ae_workflow_helpers
        # contract — so don't expect it inside extra_metadata.

    # ---------------------------------------------------- aux contracts

    def test_parse_downloaded_at_iso_string(self) -> None:
        ts = ingest._parse_downloaded_at("2026-05-02T05:13:53.556661+00:00")
        self.assertIsInstance(ts, float)
        self.assertGreater(ts, 1700000000)

    def test_parse_downloaded_at_epoch_float_passthrough(self) -> None:
        ts = ingest._parse_downloaded_at(1714627200.5)
        self.assertEqual(ts, 1714627200.5)

    def test_empty_sidecar_dir_no_op(self) -> None:
        # No sidecars at all — should produce empty report and exit 0
        rc = self._run("--backfill")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
