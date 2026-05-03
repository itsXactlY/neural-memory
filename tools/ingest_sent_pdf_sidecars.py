#!/usr/bin/env python3
"""ingest_sent_pdf_sidecars.py — tail AE sent-PDF sidecar JSONs into NM substrate.

Per Sonnet packet S-OptE (NM-side tail of AE sent-PDF sidecars, 2026-05-03):

  AE-builder lane already runs `sent_estimate_pdf_miner.py` which writes typed
  sidecar JSONs to /Users/tito/.../LangGraph/data/sent-estimates-pdfs/. NM just
  tails them. Zero AE-side patch required.

  Tito approved no-privacy-gating ("idc about privacy. i'm only user dude.")
  — all sent-PDF sidecars are ingested by default.

Sidecar shape (verified from sent_estimate_pdf_miner lines 180-194):
  {msg_id, thread_id, subject, from, to, date, filename, size_bytes, text,
   extraction, dollar_total_guess, downloaded_at}

NM mapping:
  evidence_type    = "sent_pdf"           (canonical EVIDENCE_TYPES entry;
                                           packet specified "sent_estimate_pdf"
                                           but that's not in the enum — using
                                           the canonical "sent_pdf" instead)
  source_system    = "sent_estimate_pdf_miner"
  source_record_id = sidecar.msg_id        (Gmail-stable; deterministic
                                            evidence_id consistent on re-mine)
  source_path      = absolute sidecar path
  content          = sidecar.text
  valid_from       = parsed(sidecar.downloaded_at) → epoch float
                     (packet text said "epoch float" but the field is actually
                     an ISO-8601 string; we parse it)
  metadata         = {thread_id, subject, from, to, date, filename,
                      dollar_total_guess, size_bytes, page_count,
                      extraction_method, capability_id: "ITEM-SENT-PDF"}

DEFAULT MODE = dry-run. NO substrate write. Validates sidecars and emits
typed records to ~/.neural_memory/ingest-dryruns/sent-pdf-{ts}.jsonl for
inspection. --live is explicit opt-in and writes to canonical substrate
via record_evidence_artifact (which is replay-safe via evidence_id upsert).

Watermark format (default ~/.neural_memory/state/sent-pdf-watermark.json):
  {"processed_msg_ids": [...], "last_run_ts": <epoch_float>}
  Tracks the SET of processed msg_ids — sidecars don't have a stable
  creation order so a single cursor would be wrong. --backfill ignores it.

Usage:
    # Dry-run smoke against all 47 historical sidecars (READ-ONLY):
    python3 tools/ingest_sent_pdf_sidecars.py --backfill

    # Live ingest of all sidecars (writes canonical substrate):
    python3 tools/ingest_sent_pdf_sidecars.py --backfill --live

    # Tail-only (incremental, watermarked):
    python3 tools/ingest_sent_pdf_sidecars.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Import canonical helpers from neural-memory python/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ae_workflow_helpers import (  # noqa: E402
    EVIDENCE_TYPES,
    _compute_evidence_id,
    record_evidence_artifact,
)

DEFAULT_SIDECAR_DIR = Path(
    "/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/"
    "data/sent-estimates-pdfs/"
)
DEFAULT_WATERMARK = Path.home() / ".neural_memory" / "state" / "sent-pdf-watermark.json"
DRYRUN_DIR = Path.home() / ".neural_memory" / "ingest-dryruns"
LIVE_DIR = Path.home() / ".neural_memory" / "ingest-live"

# Canonical mapping constants
EVIDENCE_TYPE = "sent_pdf"          # canon, see ae_workflow_helpers.EVIDENCE_TYPES
SOURCE_SYSTEM = "sent_estimate_pdf_miner"
CAPABILITY_ID = "ITEM-SENT-PDF"
PRIVACY_CLASS = "financial"         # estimates carry pricing; matches record_estimate_evidence default
CONFIDENCE = 0.95

REQUIRED_FIELDS = {"msg_id", "text", "downloaded_at"}


def _parse_downloaded_at(value: object) -> float:
    """Sidecar `downloaded_at` is ISO-8601 (verified-now). Accept epoch float
    too in case the upstream miner ever changes — be liberal in what we read.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Python 3.11+ supports `Z` and `+HH:MM` natively
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            # Final fallback: try date.parsefmt for RFC-2822-ish strings
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(value).timestamp()
    raise ValueError(f"downloaded_at has unsupported type {type(value).__name__}: {value!r}")


def load_watermark(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        ids = data.get("processed_msg_ids", [])
        if not isinstance(ids, list):
            return set()
        return set(str(x) for x in ids)
    except Exception as e:
        print(f"WARNING: watermark file unreadable ({e}); treating as empty", file=sys.stderr)
        return set()


def save_watermark(path: Path, processed_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_msg_ids": sorted(processed_ids),
        "last_run_ts": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)  # atomic


def build_record(sidecar: dict, sidecar_path: Path) -> dict:
    """Build the NM call kwargs from a sidecar dict. Caller passes these
    straight through to record_evidence_artifact in --live mode, OR writes
    the dict to the dry-run JSONL.
    """
    msg_id = sidecar["msg_id"]
    text = sidecar["text"]
    valid_from = _parse_downloaded_at(sidecar["downloaded_at"])

    extra_metadata = {
        "thread_id": sidecar.get("thread_id"),
        "subject": sidecar.get("subject"),
        "from": sidecar.get("from"),
        "to": sidecar.get("to"),
        "date": sidecar.get("date"),
        "filename": sidecar.get("filename"),
        "size_bytes": sidecar.get("size_bytes"),
        "dollar_total_guess": sidecar.get("dollar_total_guess"),
    }
    extraction = sidecar.get("extraction") or {}
    if isinstance(extraction, dict):
        if "page_count" in extraction:
            extra_metadata["page_count"] = extraction["page_count"]
        if "method" in extraction:
            extra_metadata["extraction_method"] = extraction["method"]
    # Strip None values — keeps the metadata clean
    extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}

    # Pre-computed evidence_id for dry-run preview parity with live
    evidence_id = _compute_evidence_id(
        evidence_type=EVIDENCE_TYPE,
        source_system=SOURCE_SYSTEM,
        source_record_id=msg_id,
    )

    return {
        "evidence_type": EVIDENCE_TYPE,
        "capability_id": CAPABILITY_ID,
        "source_system": SOURCE_SYSTEM,
        "source_path": str(sidecar_path),
        "content": text,
        "privacy_class": PRIVACY_CLASS,
        "confidence": CONFIDENCE,
        "source_record_id": msg_id,
        "valid_from": valid_from,
        "extra_metadata": extra_metadata,
        # Pre-computed for dry-run parity (live ingest will recompute identically)
        "_preview_evidence_id": evidence_id,
        "_msg_id": msg_id,
    }


def discover_sidecars(sidecar_dir: Path) -> list[Path]:
    if not sidecar_dir.exists():
        return []
    return sorted(p for p in sidecar_dir.iterdir()
                  if p.is_file() and p.suffix == ".json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sidecar-dir", default=str(DEFAULT_SIDECAR_DIR),
        help=f"Directory of sidecar JSONs (default: {DEFAULT_SIDECAR_DIR})",
    )
    parser.add_argument(
        "--watermark", default=str(DEFAULT_WATERMARK),
        help=f"Watermark JSON path (default: {DEFAULT_WATERMARK})",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Ignore watermark; process every sidecar in the dir.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="LIVE mode — call record_evidence_artifact (writes canonical "
             "substrate). Default is dry-run.",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="(--live only) Path to NM substrate DB. Default = NeuralMemory's "
             "default (~/.neural_memory/memory.db).",
    )
    parser.add_argument(
        "--show", type=int, default=0,
        help="Print first N built records to stdout for inspection.",
    )
    args = parser.parse_args()

    sidecar_dir = Path(args.sidecar_dir).expanduser()
    watermark_path = Path(args.watermark).expanduser()

    # Sanity check
    if EVIDENCE_TYPE not in EVIDENCE_TYPES:
        print(
            f"FATAL: hard-coded EVIDENCE_TYPE={EVIDENCE_TYPE!r} not in "
            f"ae_workflow_helpers.EVIDENCE_TYPES — refusing to run.",
            file=sys.stderr,
        )
        return 4

    sidecars = discover_sidecars(sidecar_dir)
    if not sidecars:
        print(
            f"WARNING: no sidecars found at {sidecar_dir} — nothing to do.",
            file=sys.stderr,
        )
        # Still produce an empty report so callers can detect the no-op cleanly
        out_dir = LIVE_DIR if args.live else DRYRUN_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_now = int(time.time())
        out_path = out_dir / f"sent-pdf-{ts_now}.jsonl"
        out_path.write_text("")
        print(f"empty report: {out_path}")
        return 0

    skip_set: set[str] = set() if args.backfill else load_watermark(watermark_path)

    out_dir = LIVE_DIR if args.live else DRYRUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_now = int(time.time())
    suffix = "live" if args.live else "dry"
    out_path = out_dir / f"sent-pdf-{ts_now}.{suffix}.jsonl"

    # Live mode: open NM lazily so dry-run never imports memory_client
    mem = None
    if args.live:
        from memory_client import NeuralMemory  # noqa: WPS433 (deferred import is intentional)
        if args.db_path:
            mem = NeuralMemory(db_path=args.db_path)
        else:
            mem = NeuralMemory()

    skipped_watermark = 0
    processed = 0
    errors = 0
    inserted = 0
    deduped = 0
    rows_for_show: list[dict] = []
    new_processed_ids: set[str] = set()

    with open(out_path, "w") as fh:
        for sidecar_path in sidecars:
            row_report: dict = {"sidecar_path": str(sidecar_path)}
            try:
                with open(sidecar_path) as sf:
                    sidecar = json.load(sf)
                missing = REQUIRED_FIELDS - set(sidecar.keys())
                if missing:
                    raise ValueError(f"missing required fields: {sorted(missing)}")
                msg_id = sidecar["msg_id"]
                row_report["msg_id"] = msg_id

                if msg_id in skip_set:
                    skipped_watermark += 1
                    row_report["skipped_watermark"] = True
                    fh.write(json.dumps(row_report) + "\n")
                    continue

                kwargs = build_record(sidecar, sidecar_path)
                preview_evidence_id = kwargs.pop("_preview_evidence_id")
                kwargs.pop("_msg_id")

                if args.live:
                    result = record_evidence_artifact(mem, **kwargs)
                    row_report.update({
                        "memory_id": result["memory_id"],
                        "evidence_id": result["evidence_id"],
                        "inserted": result["inserted"],
                    })
                    if result["inserted"]:
                        inserted += 1
                    else:
                        deduped += 1
                else:
                    row_report.update({
                        "evidence_id": preview_evidence_id,
                        "memory_id": None,
                        "inserted": None,
                        "dry_run": True,
                        "would_call": {
                            "evidence_type": kwargs["evidence_type"],
                            "capability_id": kwargs["capability_id"],
                            "source_system": kwargs["source_system"],
                            "source_path": kwargs["source_path"],
                            "source_record_id": kwargs["source_record_id"],
                            "privacy_class": kwargs["privacy_class"],
                            "valid_from": kwargs["valid_from"],
                            "extra_metadata": kwargs["extra_metadata"],
                            "content_len": len(kwargs["content"]),
                        },
                    })
                processed += 1
                new_processed_ids.add(msg_id)
                if args.show and len(rows_for_show) < args.show:
                    rows_for_show.append(row_report)
            except Exception as e:
                errors += 1
                row_report["error"] = f"{type(e).__name__}: {e}"
            fh.write(json.dumps(row_report, default=str) + "\n")

    # Update watermark only on --live success — dry-run is purely informational
    # and shouldn't advance state.
    if args.live and not args.backfill and new_processed_ids:
        merged = (skip_set | new_processed_ids)
        save_watermark(watermark_path, merged)
    elif args.live and args.backfill and new_processed_ids:
        # Backfill should still seed the watermark so a subsequent
        # non-backfill run skips what we just ingested.
        save_watermark(watermark_path, new_processed_ids | skip_set)

    print(f"=== sent-PDF sidecar ingest report ===")
    print(f"  mode: {'LIVE (canonical substrate write)' if args.live else 'dry-run'}")
    print(f"  sidecar dir: {sidecar_dir}")
    print(f"  total found: {len(sidecars)}")
    print(f"  skipped (watermark): {skipped_watermark}")
    print(f"  processed: {processed}")
    if args.live:
        print(f"    inserted (new): {inserted}")
        print(f"    deduped (existing evidence_id): {deduped}")
    print(f"  errors: {errors}")
    print(f"  output: {out_path}")
    if rows_for_show:
        print(f"\n=== first {len(rows_for_show)} rows ===")
        for row in rows_for_show:
            print(json.dumps(row, indent=2, default=str))

    return 0 if errors == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
