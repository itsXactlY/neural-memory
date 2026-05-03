#!/usr/bin/env python3
"""ingest_wa_dryrun.py — typed dry-run validator for WA crew chat batches.

Per codex-prescriptive-redesigner Day 5: "wire ingest to produce typed
dry-run JSONL output to ~/.neural_memory/ingest-dryruns/. NO DB write
by default."

Hardened by Sonnet packet S5 (2026-05-03) per NM-builder synth contract:
"keep validating the required handoff shape while real WA source remains
TITO_BLOCKED; no canonical writes." Hardening adds:

  - Strict ISO8601 ts format check (reject epoch ints/floats and ambiguous
    formats from the input JSONL contract; numeric ts still accepted by
    to_typed_record() for backward compat with the S2 lockdown test).
  - thread_id pattern WARNING (not reject) for non-WA shapes.
  - privacy_class enum tightened to packet contract {internal, financial,
    pii_low}.
  - lang code WARNING if not 2-letter ISO 639-1.
  - media_paths element-level checks (non-empty strings, no shell
    metachars).
  - consumer_hint type check.
  - boundary_violation_suspect type check.
  - evidence_id pre-image check (if present, must match the deterministic
    sha256 first-16-hex computed by ae_workflow_helpers._compute_evidence_id).
  - Per-row structured output: {row_index, valid, errors, warnings,
    computed_evidence_id}.
  - Tightened exit codes: 0 all-valid, 2 input/io failure, 3 any invalid.

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
There is NO --live mode by design; the dryrun tool only ever validates.

Usage:
    # Validate a Hermes-produced batch:
    python3 tools/ingest_wa_dryrun.py --in /path/to/wa_batch.jsonl

    # Validate stdin:
    cat wa_batch.jsonl | python3 tools/ingest_wa_dryrun.py

    # Show first N validated records:
    python3 tools/ingest_wa_dryrun.py --in batch.jsonl --show 5

    # Emit per-row JSONL report (instead of human summary):
    python3 tools/ingest_wa_dryrun.py --in batch.jsonl --report-jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
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

REQUIRED_FIELDS = ("thread_id", "sender", "raw_text", "ts")
ALLOWED_DELIVERY = {"delivered", "pending", "failed", "read"}
# Packet S5 contract: tightened to {internal, financial, pii_low}. The
# helper still allows the wider 5-value Theme-8 set for non-WA evidence
# types; WA crew messages are constrained to this 3-value subset because
# WA carries crew-internal ops (internal), pricing/quotes (financial),
# and low-PII names/threads (pii_low). public/pii_high are out-of-scope
# for WA and signal an upstream Hermes labeling bug if seen.
ALLOWED_PRIVACY = {"internal", "financial", "pii_low"}

# WA thread_id shapes: group threads end @g.us, DM threads end @s.whatsapp.net.
# Pattern is a WARNING only — shape may evolve, and Hermes lane may surface
# legacy shapes; we want signal not blockage.
_WA_GROUP_RE = re.compile(r"^\d+@g\.us$")
_WA_DM_RE = re.compile(r"^\d+@s\.whatsapp\.net$")

# 2-letter ISO 639-1 lowercase. Warning-only.
_LANG_RE = re.compile(r"^[a-z]{2}$")

# evidence_id is sha256 first 16 hex chars per _compute_evidence_id contract.
_EVIDENCE_ID_RE = re.compile(r"^[0-9a-f]{16}$")

# Basic shell-metachar safety for media_paths. Reject any path containing
# these — Hermes lane should hand us already-resolved relative or absolute
# filesystem paths, never shell expressions.
_SHELL_META = set(";&|`$<>\n\r\t")


def _parse_iso8601(value: Any) -> tuple[bool, float | None]:
    """Parse ISO8601 string → (ok, epoch_seconds). Reject epoch ints/floats
    and ambiguous formats. Returns (False, None) on any non-conforming input.

    Accepts:
      - 2026-05-03T10:41:47Z
      - 2026-05-03T10:41:47+00:00
      - 2026-05-03T10:41:47.123456+00:00
      - 2026-05-03T10:41:47 (treated as naive; valid ISO8601 but warned upstream)

    Rejects:
      - 1717350000 (epoch int)
      - 1717350000.5 (epoch float)
      - "1717350000" (epoch as string)
      - "May 3 2026" (non-ISO)
      - "" / None
    """
    if not isinstance(value, str) or not value.strip():
        return False, None
    s = value.strip()
    # Reject pure-numeric strings (epoch-as-string is ambiguous).
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return False, None
    try:
        # Python 3.11+: fromisoformat accepts 'Z' suffix. For older 3.x we
        # normalize 'Z' → '+00:00' first to be safe.
        normalized = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = _dt.datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return False, None
    # If naive (no tzinfo), interpret as UTC for epoch computation but signal
    # via tzinfo absence — caller decides whether to warn.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return True, dt.timestamp()


def _ts_to_epoch(ts: Any) -> float:
    """Internal coercion: ISO8601 string → float; numeric → float (back-compat
    with S2 lockdown test in test_ae_evidence_ingest.py which calls
    to_typed_record with ts=1717350000.5)."""
    if isinstance(ts, str):
        ok, epoch = _parse_iso8601(ts)
        if ok and epoch is not None:
            return epoch
        # Fall through — caller is responsible for upstream validation.
        raise ValueError(f"ts string is not ISO8601: {ts!r}")
    if isinstance(ts, (int, float)):
        return float(ts)
    raise ValueError(f"ts must be ISO8601 string or number, got {type(ts).__name__}")


def _compute_record_id(thread_id: str, ts_epoch: float, raw_text: str) -> str:
    """Mirror record_wa_crew_event's record_id formula EXACTLY:
        record_id = f"{thread_id}:{ts_us}:{content_hash}"
    where ts_us = int(ts * 1_000_000) and content_hash = md5(raw_text)[:8].
    """
    content_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:8]
    ts_us = int(ts_epoch * 1_000_000)
    return f"{thread_id}:{ts_us}:{content_hash}"


def _compute_expected_evidence_id(row: dict) -> str | None:
    """Compute the deterministic evidence_id this row WOULD produce if
    live-ingested. Mirrors record_wa_crew_event's call into
    _compute_evidence_id. Returns None if required fields for the formula
    are missing/unparseable (caller surfaces as part of validation errors).
    """
    thread_id = row.get("thread_id")
    raw_text = row.get("raw_text")
    ts = row.get("ts")
    if not isinstance(thread_id, str) or not isinstance(raw_text, str):
        return None
    try:
        ts_epoch = _ts_to_epoch(ts)
    except ValueError:
        return None
    record_id = _compute_record_id(thread_id, ts_epoch, raw_text)
    return _compute_evidence_id(
        evidence_type="wa_crew_message",
        source_system="hermes_wa_bridge",
        source_record_id=record_id,
    )


def validate_row(row: Any, line_no: int) -> dict:
    """Validate a single WA row against the contract.

    Returns structured report:
        {
          "row_index": int,        # 1-based line number
          "valid": bool,
          "errors": list[str],     # empty iff valid
          "warnings": list[str],   # warnings do NOT invalidate
          "computed_evidence_id": str | None,
        }

    Backward-compat shim: tests may still consume the legacy (ok, err)
    tuple via validate_row_legacy(); see below.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(row, dict):
        return {
            "row_index": line_no,
            "valid": False,
            "errors": [f"line {line_no}: row must be a JSON object, got {type(row).__name__}"],
            "warnings": [],
            "computed_evidence_id": None,
        }

    # ---- required-field presence + non-null ---------------------------
    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"line {line_no}: missing required field {field!r}")
        elif row[field] is None:
            errors.append(f"line {line_no}: required field {field!r} is null")

    # If any required field is missing/null, skip type-deep checks for that
    # field but continue collecting other errors so the report is full.

    # ---- raw_text type/non-empty --------------------------------------
    rt = row.get("raw_text")
    if rt is not None:
        if not isinstance(rt, str):
            errors.append(f"line {line_no}: raw_text must be string, got {type(rt).__name__}")
        elif not rt.strip():
            errors.append(f"line {line_no}: raw_text must be non-empty")

    # ---- thread_id pattern (WARNING) ----------------------------------
    tid = row.get("thread_id")
    if isinstance(tid, str) and tid:
        if not (_WA_GROUP_RE.match(tid) or _WA_DM_RE.match(tid)):
            warnings.append(
                f"line {line_no}: thread_id={tid!r} does not match WA group "
                f"(<numeric>@g.us) or DM (<numeric>@s.whatsapp.net) pattern"
            )
    elif tid is not None and not isinstance(tid, str):
        errors.append(f"line {line_no}: thread_id must be string, got {type(tid).__name__}")

    # ---- sender type --------------------------------------------------
    sender = row.get("sender")
    if sender is not None and not isinstance(sender, str):
        errors.append(f"line {line_no}: sender must be string, got {type(sender).__name__}")
    elif isinstance(sender, str) and not sender.strip():
        errors.append(f"line {line_no}: sender must be non-empty string")

    # ---- ts ISO8601 ---------------------------------------------------
    ts_val = row.get("ts")
    if ts_val is not None:
        if isinstance(ts_val, (int, float)) and not isinstance(ts_val, bool):
            errors.append(
                f"line {line_no}: ts must be ISO8601 string (got numeric "
                f"{ts_val!r}); convert epoch with "
                f"datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()"
            )
        elif isinstance(ts_val, str):
            ok, _epoch = _parse_iso8601(ts_val)
            if not ok:
                errors.append(
                    f"line {line_no}: ts={ts_val!r} is not parseable as ISO8601 "
                    f"(reject ambiguous/non-ISO formats)"
                )
        else:
            errors.append(f"line {line_no}: ts must be ISO8601 string, got {type(ts_val).__name__}")

    # ---- delivery_status enum -----------------------------------------
    delivery = row.get("delivery_status", "delivered")
    if delivery not in ALLOWED_DELIVERY:
        errors.append(
            f"line {line_no}: delivery_status={delivery!r} must be one of "
            f"{sorted(ALLOWED_DELIVERY)}"
        )

    # ---- privacy_class enum (tightened S5) ----------------------------
    privacy = row.get("privacy_class", "internal")
    if privacy not in ALLOWED_PRIVACY:
        errors.append(
            f"line {line_no}: privacy_class={privacy!r} must be one of "
            f"{sorted(ALLOWED_PRIVACY)}"
        )

    # ---- lang ISO 639-1 (WARNING) -------------------------------------
    lang = row.get("lang", "es")
    if not isinstance(lang, str):
        errors.append(f"line {line_no}: lang must be string, got {type(lang).__name__}")
    elif not _LANG_RE.match(lang):
        warnings.append(
            f"line {line_no}: lang={lang!r} is not a 2-letter ISO 639-1 code"
        )

    # ---- media_paths element-level ------------------------------------
    media = row.get("media_paths")
    if media is not None:
        if not isinstance(media, list):
            errors.append(
                f"line {line_no}: media_paths must be list[str] or null"
            )
        else:
            for i, item in enumerate(media):
                if not isinstance(item, str):
                    errors.append(
                        f"line {line_no}: media_paths[{i}] must be string, "
                        f"got {type(item).__name__}"
                    )
                elif not item.strip():
                    errors.append(
                        f"line {line_no}: media_paths[{i}] must be non-empty"
                    )
                elif any(c in _SHELL_META for c in item):
                    errors.append(
                        f"line {line_no}: media_paths[{i}]={item!r} contains "
                        f"shell metacharacter; pass resolved filesystem paths only"
                    )

    # ---- consumer_hint type -------------------------------------------
    ch = row.get("consumer_hint")
    if ch is not None and not isinstance(ch, str):
        errors.append(
            f"line {line_no}: consumer_hint must be string or null, "
            f"got {type(ch).__name__}"
        )

    # ---- boundary_violation_suspect type ------------------------------
    bvs = row.get("boundary_violation_suspect")
    if bvs is not None and not isinstance(bvs, bool):
        # int 0/1 would silently pass isinstance(int, bool=True) on older
        # Python, but Python 3 has bool as subclass of int so guard order
        # matters: bool first, then "is not bool".
        errors.append(
            f"line {line_no}: boundary_violation_suspect must be bool or null, "
            f"got {type(bvs).__name__}"
        )

    # ---- normalized_text type -----------------------------------------
    nt = row.get("normalized_text")
    if nt is not None and not isinstance(nt, str):
        errors.append(
            f"line {line_no}: normalized_text must be string or null, "
            f"got {type(nt).__name__}"
        )

    # ---- auth_proof type ----------------------------------------------
    ap = row.get("auth_proof")
    if ap is not None and not isinstance(ap, dict):
        errors.append(
            f"line {line_no}: auth_proof must be object or null, "
            f"got {type(ap).__name__}"
        )

    # ---- evidence_id pre-image ----------------------------------------
    expected_eid = _compute_expected_evidence_id(row) if not errors else None
    supplied_eid = row.get("evidence_id")
    if supplied_eid is not None:
        if not isinstance(supplied_eid, str):
            errors.append(
                f"line {line_no}: evidence_id must be string, "
                f"got {type(supplied_eid).__name__}"
            )
        elif not _EVIDENCE_ID_RE.match(supplied_eid):
            errors.append(
                f"line {line_no}: evidence_id={supplied_eid!r} must be 16 "
                f"lowercase hex chars (sha256 first 16)"
            )
        elif expected_eid is not None and supplied_eid != expected_eid:
            errors.append(
                f"line {line_no}: evidence_id={supplied_eid!r} mismatch — "
                f"expected {expected_eid!r} from sha256(\"wa_crew_message|"
                f"hermes_wa_bridge|<record_id>\")[:16]"
            )

    return {
        "row_index": line_no,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "computed_evidence_id": expected_eid,
    }


def validate_row_legacy(row: dict, line_no: int) -> tuple[bool, str | None]:
    """Legacy (ok, err) tuple wrapper for any pre-S5 caller. Prefer
    validate_row() for new code — it returns a richer report."""
    rep = validate_row(row, line_no)
    if rep["valid"]:
        return True, None
    return False, "; ".join(rep["errors"])


def to_typed_record(row: dict) -> dict:
    """Compute the typed AEEvidenceIngest record shape this row would
    produce in substrate, WITHOUT writing it.

    Note: backward-compat with the S2 lockdown test
    (test_dryrun_evidence_id_matches_live_ingest in
    python/test_ae_evidence_ingest.py) requires this function to accept
    numeric ts. The user-facing JSONL contract enforced by validate_row
    is stricter (ISO8601 only) — that's the input contract; this function
    is the live-ingest-shape mirror that must accept whatever form the
    helper accepts.
    """
    raw_text = row["raw_text"]
    ts_epoch = _ts_to_epoch(row["ts"])
    thread_id = row["thread_id"]
    delivery = row.get("delivery_status", "delivered")
    privacy = row.get("privacy_class", "internal")
    lang = row.get("lang", "es")

    # Mirror record_wa_crew_event's provenance computation
    record_id = _compute_record_id(thread_id, ts_epoch, raw_text)

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
    if "auth_proof" in row and row["auth_proof"] is not None:
        metadata["auth_proof"] = row["auth_proof"]
    if row.get("normalized_text"):
        metadata["normalized_text"] = row["normalized_text"]
    if row.get("consumer_hint"):
        metadata["consumer_hint"] = row["consumer_hint"]
    if row.get("boundary_violation_suspect") is not None:
        metadata["boundary_violation_suspect"] = bool(row["boundary_violation_suspect"])

    return {
        "label": f"evidence:wa_crew_message:{metadata['capability_id']}:{record_id}",
        "content": content,
        "kind": "experience",
        "confidence": confidence,
        "source": "hermes_wa_bridge",
        "origin_system": "ae",
        "valid_from": ts_epoch,
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
    parser.add_argument("--report-jsonl", action="store_true",
                        help="Emit per-row validation report as JSONL on stdout (overrides human summary)")
    args = parser.parse_args()

    # Open input source
    try:
        if args.infile:
            if not Path(args.infile).exists():
                print(f"ERROR: input file not found: {args.infile}", file=sys.stderr)
                return 2
            infile = open(args.infile)
        else:
            if sys.stdin.isatty():
                print("ERROR: no --in file and stdin is a tty", file=sys.stderr)
                print(__doc__, file=sys.stderr)
                return 2
            infile = sys.stdin
    except OSError as e:
        print(f"ERROR: cannot open input: {e}", file=sys.stderr)
        return 2

    try:
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: cannot create out-dir: {e}", file=sys.stderr)
        return 2

    ts_now = int(time.time())
    out_path = out_dir / f"wa-{ts_now}.jsonl"

    valid_count = 0
    invalid_count = 0
    reports: list[dict] = []
    typed_records: list[dict] = []
    rejection_reasons: dict[str, int] = {}  # error-prefix → count

    for line_no, raw_line in enumerate(infile, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            rep = {
                "row_index": line_no,
                "valid": False,
                "errors": [f"line {line_no}: JSON parse error: {e}"],
                "warnings": [],
                "computed_evidence_id": None,
            }
            reports.append(rep)
            invalid_count += 1
            _bucket_reasons(rejection_reasons, rep["errors"])
            continue

        rep = validate_row(row, line_no)
        reports.append(rep)
        if rep["valid"]:
            valid_count += 1
            typed = to_typed_record(row)
            typed_records.append(typed)
        else:
            invalid_count += 1
            _bucket_reasons(rejection_reasons, rep["errors"])

    # Write typed records to dry-run output (only valid rows)
    try:
        with open(out_path, "w") as fh:
            for rec in typed_records:
                fh.write(json.dumps(rec) + "\n")
    except OSError as e:
        print(f"ERROR: cannot write dryrun output: {e}", file=sys.stderr)
        return 2

    if args.report_jsonl:
        # Machine-readable mode: each line is a row report
        for rep in reports:
            print(json.dumps(rep))
    else:
        print("=== WA dry-run validation report ===")
        print(f"  valid: {valid_count}")
        print(f"  invalid: {invalid_count}")
        print(f"  output: {out_path}")
        if rejection_reasons:
            print("\n=== rejection reasons (grouped) ===")
            for reason, count in sorted(
                rejection_reasons.items(), key=lambda kv: -kv[1]
            ):
                print(f"  {count:>4}  {reason}")
        # Show first 20 raw errors for context
        all_errors = [e for r in reports for e in r["errors"]]
        if all_errors:
            print("\n=== validation errors (first 20) ===")
            for e in all_errors[:20]:
                print(f"  ✗ {e}")
            if len(all_errors) > 20:
                print(f"  ... and {len(all_errors) - 20} more")
        all_warnings = [w for r in reports for w in r["warnings"]]
        if all_warnings:
            print(f"\n=== warnings ({len(all_warnings)} total, first 10) ===")
            for w in all_warnings[:10]:
                print(f"  ! {w}")

        if args.show > 0 and typed_records:
            print(f"\n=== first {min(args.show, len(typed_records))} typed records ===")
            for rec in typed_records[:args.show]:
                print(json.dumps(rec, indent=2))

    # Exit code: 0 all-valid, 3 any invalid (CI gate signal). 2 reserved for
    # io/setup failures (returned earlier).
    return 0 if invalid_count == 0 else 3


def _bucket_reasons(buckets: dict[str, int], errors: list[str]) -> None:
    """Group rejection messages by their semantic prefix (strip "line N: ")
    so the summary shows distinct reasons + counts."""
    for err in errors:
        # Strip leading "line N: " so equivalent errors group across rows
        normalized = re.sub(r"^line \d+: ", "", err)
        # Truncate any long quoted values for grouping stability
        normalized = re.sub(r"=('[^']*'|\"[^\"]*\")", "=<…>", normalized)
        buckets[normalized] = buckets.get(normalized, 0) + 1


if __name__ == "__main__":
    sys.exit(main())
