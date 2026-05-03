#!/usr/bin/env python3
"""ingest_wa_dryrun.py — typed dry-run validator for WA crew chat batches.

Per codex-prescriptive-redesigner Day 5: "wire ingest to produce typed
dry-run JSONL output to ~/.neural_memory/ingest-dryruns/. NO DB write
by default."

This tool accepts a Hermes-produced WA JSONL batch (one message per line,
matching the AEEvidenceIngest v0 WA contract) and:

  1. Validates each row against the locked contract (fields, enums, types)
  2. Computes the AEEvidenceIngest record shape (provenance, content_hash,
     bi-temporal validity) WITHOUT writing to substrate
  3. Emits typed dry-run records to ~/.neural_memory/ingest-dryruns/wa-{ts}.jsonl
  4. Returns a per-batch validation report

Hermes uses this to verify her WA batches BEFORE NM accepts them live.
Tito uses this to inspect what NM would store from a batch BEFORE
approving live ingest (Tito decision #1 in NM-builder Tito-decision queue).

LIVE INGEST is gated on Tito decision — this tool is read/validate only.

Usage:
    # Validate a Hermes-produced batch:
    python3 tools/ingest_wa_dryrun.py --in /path/to/wa_batch.jsonl

    # Validate stdin:
    cat wa_batch.jsonl | python3 tools/ingest_wa_dryrun.py

    # Show first N validated records:
    python3 tools/ingest_wa_dryrun.py --in batch.jsonl --show 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# Import the canonical evidence_id computation from ae_workflow_helpers
# so this dry-run tool produces IDs identical to live ingest. Single
# source of truth — a lockdown test in test_ae_evidence_ingest.py
# asserts both call sites match if a future refactor diverges.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ae_workflow_helpers import _compute_evidence_id  # noqa: E402

DRYRUN_DIR = Path.home() / ".neural_memory" / "ingest-dryruns"

REQUIRED_FIELDS = {"thread_id", "sender", "raw_text", "ts"}
ALLOWED_DELIVERY = {"delivered", "pending", "failed", "read"}
ALLOWED_PRIVACY = {"public", "internal", "pii_low", "pii_high", "financial"}


def validate_row(row: dict, line_no: int) -> tuple[bool, str | None]:
    """Validate a single WA row against the contract. Returns (ok, error_msg)."""
    missing = REQUIRED_FIELDS - set(row.keys())
    if missing:
        return False, f"line {line_no}: missing required fields: {sorted(missing)}"

    if not isinstance(row["raw_text"], str) or not row["raw_text"].strip():
        return False, f"line {line_no}: raw_text must be non-empty string"

    if not isinstance(row["ts"], (int, float)):
        return False, f"line {line_no}: ts must be numeric (epoch seconds)"

    delivery = row.get("delivery_status", "delivered")
    if delivery not in ALLOWED_DELIVERY:
        return False, (
            f"line {line_no}: delivery_status={delivery!r} must be one of "
            f"{sorted(ALLOWED_DELIVERY)}"
        )

    privacy = row.get("privacy_class", "internal")
    if privacy not in ALLOWED_PRIVACY:
        return False, (
            f"line {line_no}: privacy_class={privacy!r} must be one of "
            f"{sorted(ALLOWED_PRIVACY)}"
        )

    media = row.get("media_paths")
    if media is not None and not isinstance(media, list):
        return False, f"line {line_no}: media_paths must be list[str] or null"

    return True, None


def to_typed_record(row: dict) -> dict:
    """Compute the typed AEEvidenceIngest record shape this row would
    produce in substrate, WITHOUT writing it.
    """
    raw_text = row["raw_text"]
    ts = float(row["ts"])
    thread_id = row["thread_id"]
    delivery = row.get("delivery_status", "delivered")
    privacy = row.get("privacy_class", "internal")
    lang = row.get("lang", "es")

    # Mirror record_wa_crew_event's provenance computation
    content_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:8]
    ts_us = int(ts * 1_000_000)
    record_id = f"{thread_id}:{ts_us}:{content_hash}"

    content_parts = [raw_text]
    if row.get("normalized_text") and row["normalized_text"] != raw_text:
        content_parts.append(f"\n[normalized:{lang}→en]\n{row['normalized_text']}")
    content = "".join(content_parts)

    confidence = 0.95 if delivery == "read" else 0.85

    # Replay-authority key (S2 packet 2026-05-03): identical to what
    # record_evidence_artifact will compute when this row is later live-ingested.
    # Imported from ae_workflow_helpers so divergence is impossible by construction.
    evidence_id = _compute_evidence_id(
        evidence_type="wa_crew_message",
        source_system="hermes_wa_bridge",
        source_record_id=record_id,
    )

    metadata = {
        "evidence_type": "wa_crew_message",
        "capability_id": row.get("capability_id", "ITEM-WA-UNTAGGED"),
        "source_system": "hermes_wa_bridge",
        "source_path": f"wa_bridge:{record_id}",
        "privacy_class": privacy,
        "evidence_id": evidence_id,
        "source_record_id": record_id,
        "thread_id": thread_id,
        "sender": row["sender"],
        "lang": lang,
        "delivery_status": delivery,
        "door": "wa",
    }
    if row.get("media_paths"):
        metadata["media_paths"] = row["media_paths"]
    if row.get("auth_proof"):
        metadata["auth_proof"] = row["auth_proof"]
    if row.get("normalized_text"):
        metadata["normalized_text"] = row["normalized_text"]
    if row.get("consumer_hint"):
        metadata["consumer_hint"] = row["consumer_hint"]

    return {
        "label": f"evidence:wa_crew_message:{metadata['capability_id']}:{record_id}",
        "content": content,
        "kind": "experience",
        "confidence": confidence,
        "source": "hermes_wa_bridge",
        "origin_system": "ae",
        "valid_from": ts,
        "valid_to": None,
        "evidence_id": evidence_id,  # surfaced top-level for replay-replay tooling
        "metadata": metadata,
        # Note: detect_conflicts=False is enforced server-side by record_evidence_artifact
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", default=None,
                        help="JSONL batch file (default: read stdin)")
    parser.add_argument("--out-dir", default=str(DRYRUN_DIR),
                        help="Directory for dry-run output (default: ~/.neural_memory/ingest-dryruns/)")
    parser.add_argument("--show", type=int, default=0,
                        help="Print first N validated records to stdout for inspection")
    args = parser.parse_args()

    # Open input source
    if args.infile:
        if not Path(args.infile).exists():
            print(f"ERROR: input file not found: {args.infile}", file=sys.stderr)
            return 1
        infile = open(args.infile)
    else:
        if sys.stdin.isatty():
            print("ERROR: no --in file and stdin is a tty", file=sys.stderr)
            print(__doc__, file=sys.stderr)
            return 2
        infile = sys.stdin

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_now = int(time.time())
    out_path = out_dir / f"wa-{ts_now}.jsonl"

    valid_count = 0
    invalid_count = 0
    errors: list[str] = []
    typed_records: list[dict] = []

    for line_no, raw_line in enumerate(infile, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {line_no}: JSON parse error: {e}")
            invalid_count += 1
            continue

        ok, err = validate_row(row, line_no)
        if not ok:
            errors.append(err or f"line {line_no}: unknown validation error")
            invalid_count += 1
            continue

        typed = to_typed_record(row)
        typed_records.append(typed)
        valid_count += 1

    # Write typed records to dry-run output
    with open(out_path, "w") as fh:
        for rec in typed_records:
            fh.write(json.dumps(rec) + "\n")

    # Report
    print(f"=== WA dry-run validation report ===")
    print(f"  valid: {valid_count}")
    print(f"  invalid: {invalid_count}")
    print(f"  output: {out_path}")
    if errors:
        print(f"\n=== validation errors (first 20) ===")
        for e in errors[:20]:
            print(f"  ✗ {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    if args.show > 0 and typed_records:
        print(f"\n=== first {min(args.show, len(typed_records))} typed records ===")
        for rec in typed_records[:args.show]:
            print(json.dumps(rec, indent=2))

    # Exit code: 0 if all valid, 3 if any invalid (lets CI gate on this)
    return 0 if invalid_count == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
