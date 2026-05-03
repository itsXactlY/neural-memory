"""AE workflow helpers — Phase 7 ergonomics for AE-side consumption.

Thin wrappers around `NeuralMemory.remember()` shaped to the events AE
actually generates: customer interactions, job events, Spanish crew
messages, SOPs, invoice transitions, financial-calendar entries.

Each helper builds the right Phase 7 typed-kwarg dict (kind, source,
origin_system, valid_from, metadata, etc.) so AE-side callers don't
re-derive the patterns. AE-side decides where to call these (dashboard
hooks, OTTO integration, hermes plugins, financial-calendar bridge).

Usage:
    from memory_client import NeuralMemory
    from ae_workflow_helpers import record_customer_interaction

    mem = NeuralMemory()
    mid = record_customer_interaction(
        mem,
        customer="Vibha Choudhury",
        topic="permit jurisdiction",
        body="Asked when the rough-in inspection is scheduled.",
        channel="phone",
    )

Cross-references (in claude-memory PRIVATE):
    reference_neural_memory_ae_usage_patterns.md  — recipes this implements
    reference_neural_memory_unified_integration_handoff.md  — Phase 7 spec
"""

from __future__ import annotations

import hashlib as _hashlib
import json as _json
import time as _time
from typing import Any, Optional


def _hydrate_recall_rows(mem: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add typed DB fields missing from store.get()-materialized recall rows.
    Specifically: parses metadata_json → metadata dict so callers can filter
    by md.get('customer_id') etc. Bug caught by codex-resolver 2026-05-02 —
    my recall_* helpers post-filtered on r.get('metadata') but hybrid_recall
    only returns metadata_json (str), never metadata (dict). Filter never fired.
    """
    out = []
    for r in rows:
        row = dict(r)
        mid = row.get("id")
        if mid is not None and hasattr(mem, "get_memory"):
            full = mem.get_memory(mid)
            if full:
                row.update(full)
        if "metadata" not in row:
            raw = row.get("metadata_json")
            try:
                row["metadata"] = _json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                row["metadata"] = {}
        out.append(row)
    return out


def record_customer_interaction(
    mem: Any,
    *,
    customer: str,
    topic: str,
    body: str,
    channel: str,
    valid_from: Optional[float] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Record a customer-facing interaction (phone, email, in-person, etc.).
    Auto-classified as 'experience'; entity extraction picks up the customer
    name. Returns the new memory id."""
    metadata = {"customer": customer, "topic": topic, "channel": channel}
    if extra_metadata:
        metadata.update(extra_metadata)
    return mem.remember(
        f"{customer} ({channel}) — {topic}: {body}",
        label=f"customer:{customer}:{topic}",
        kind="experience",
        source=channel,
        origin_system="ae",
        valid_from=valid_from if valid_from is not None else _time.time(),
        metadata=metadata,
    )


def record_job_event(
    mem: Any,
    *,
    job_id: str,
    event_type: str,
    body: str,
    source: str = "dashboard",
    valid_from: Optional[float] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Record a job-related event. job_id should be human-readable
    ("Lennar lot 27") so entity extraction picks it up."""
    metadata = {"job_id": job_id, "event_type": event_type}
    if extra_metadata:
        metadata.update(extra_metadata)
    return mem.remember(
        f"{event_type} on {job_id}: {body}",
        label=f"job:{job_id}:{event_type}",
        kind="experience",
        source=source,
        origin_system="ae",
        valid_from=valid_from if valid_from is not None else _time.time(),
        metadata=metadata,
    )


def record_whatsapp_message(
    mem: Any,
    *,
    crew_member: str,
    text: str,
    thread_id: str,
    ts: Optional[float] = None,
    lang: str = "es",
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Record a Spanish (or other) WhatsApp crew message verbatim. Text
    preserved as-is so the sparse channel can find exact phrases. Phase 7
    classifier may auto-tag procedural patterns (Cuando..., Si...)."""
    metadata = {"lang": lang, "crew_member": crew_member, "thread": thread_id}
    if extra_metadata:
        metadata.update(extra_metadata)
    return mem.remember(
        text,  # raw, preserved
        label=f"wa:{crew_member}:{int(ts or _time.time())}",
        kind=None,  # let classifier decide; "Cuando..." → procedural, default → experience
        source="whatsapp",
        origin_system="ae",
        valid_from=ts if ts is not None else _time.time(),
        metadata=metadata,
    )


def record_sop(
    mem: Any,
    *,
    label: str,
    content: str,
    evidence_ids: Optional[list[int]] = None,
    confidence: float = 0.95,
    pinned: bool = False,
) -> int:
    """Record a Standard Operating Procedure / workflow rule.

    SOPs are pinned by default (no decay) and high-confidence. Optional
    evidence_ids creates derived_from edges back to the experiences that
    motivated the SOP — graph_search can traverse from a procedural rule
    back to its supporting evidence base.
    """
    return mem.remember(
        content,
        label=label,
        kind="procedural",
        confidence=confidence,
        source="manual",
        origin_system="ae",
        evidence_ids=evidence_ids or [],
        metadata={"pinned": pinned},
    )


def record_invoice_status_change(
    mem: Any,
    *,
    invoice_id: str,
    old_status: str,
    new_status: str,
    transition_ts: Optional[float] = None,
    customer: Optional[str] = None,
    amount_cents: Optional[int] = None,
) -> tuple[int, int]:
    """Record an invoice status transition with bi-temporal validity.

    Closes the old status fact at transition_ts, opens the new fact at
    transition_ts. Returns (old_id, new_id). Subsequent recall(query,
    as_of=ts) returns the right invoice state for any past timestamp.

    Dream Memify's contradiction detector will auto-link the pair via
    a contradicts edge on its next run.
    """
    ts = transition_ts if transition_ts is not None else _time.time()
    metadata: dict[str, Any] = {"invoice_id": invoice_id}
    if customer:
        metadata["customer"] = customer
    if amount_cents is not None:
        metadata["amount_cents"] = amount_cents

    # detect_conflicts=False is critical here: the two facts ARE near-
    # duplicates by design (same invoice id, near-identical text). Letting
    # H19 supersession fire would merge them into one row, defeating the
    # bi-temporal record we're constructing.
    old_id = mem.remember(
        f"Invoice {invoice_id} is {old_status}.",
        label=f"invoice:{invoice_id}:{old_status}",
        detect_conflicts=False,
        kind="claim",
        valid_to=ts,
        origin_system="ae",
        source="qbo",
        metadata=metadata,
    )
    new_id = mem.remember(
        f"Invoice {invoice_id} is {new_status}.",
        label=f"invoice:{invoice_id}:{new_status}",
        detect_conflicts=False,
        kind="claim",
        valid_from=ts,
        origin_system="ae",
        source="qbo",
        metadata=metadata,
    )
    # Optional explicit edge — Memify would add this anyway, but adding
    # eagerly avoids the lag until next dream cycle.
    try:
        mem.store.add_connection(old_id, new_id, weight=1.0,
                                 edge_type="contradicts")
    except Exception:
        pass
    return old_id, new_id


def record_financial_event(
    mem: Any,
    *,
    event_type: str,
    due_date_iso: str,
    note: str,
    amount_cents: Optional[int] = None,
    valid_from: Optional[float] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Record a financial-calendar entry. Indexes well for temporal queries
    like 'invoices due this week' (recall with as_of=now)."""
    metadata = {"due_date_iso": due_date_iso, "event_type": event_type}
    if amount_cents is not None:
        metadata["amount_cents"] = amount_cents
    if extra_metadata:
        metadata.update(extra_metadata)
    return mem.remember(
        f"Financial calendar: {event_type} on {due_date_iso}: {note}",
        label=f"fincal:{due_date_iso}:{event_type}",
        kind="experience",
        source="financial_calendar",
        origin_system="ae",
        valid_from=valid_from if valid_from is not None else _time.time(),
        metadata=metadata,
    )


def recall_for_dashboard(
    mem: Any,
    query: str,
    *,
    k: int = 5,
    kind: Optional[str] = None,
    as_of: Optional[float] = None,
) -> list[dict]:
    """Bench-validated recall config for AE dashboard surfaces (/m2 etc.).

    Encapsulates the multi-channel hybrid_recall configuration tracked by
    the F9 AE-domain bench. Live R@5 figures are reported by each scored
    bench artifact under ``~/.neural_memory/bench-history/ae-domain-*.json``
    (top-level ``global_r@5``) — see ``tools/nm_recall_mcp.py``
    ``_bench_authority`` for the canonical reader. Drop-in replacement for
    raw mem.recall() / mem.hybrid_recall() in dashboard-side code.

    Config:
      - hybrid_recall (multi-channel: dense + sparse + graph + temporal
        + entity boost + procedural boost + locus boost + stale/contradict
        penalties). All Phase 7.5 wirings active.
      - rerank=True. The English-trained cross-encoder will auto-skip for
        Spanish queries (per _should_skip_rerank heuristic) so Spanish
        crew messages fall back to dense+sparse instead of getting
        mis-ranked.

    Args:
      mem: NeuralMemory instance
      query: free-text question (English or Spanish)
      k: top-K to return (default 5; empirically the bench-passing window)
      kind: optional filter by memory kind ('procedural', 'experience',
        'world', 'dream_insight', etc.)
      as_of: bi-temporal cutoff (unix ts); recalls memories valid at that
        moment

    Returns: list of result dicts with `combined`, `channels`, `_trace`
    fields per hybrid_recall's contract.

    Use-case guidance:
      - Per-customer / per-job lookups: use kind='experience' filter
      - SOP / how-to questions: use kind='procedural' filter
      - "What does the system know about X" → no kind filter
      - Time-boxed audit / 'as of last Tuesday': pass as_of=<unix_ts>
      - For pure conversation/free-text retrieval where raw semantic is
        enough → call mem.sparse_search() instead (faster, no rerank load)

    Cross-references:
      reference_ae_domain_bench_first_empirical_2026-05-01.md (validation)
      reference_neural_memory_ae_usage_patterns.md (recipes)
    """
    return mem.hybrid_recall(query, k=k, kind=kind, as_of=as_of, rerank=True)


def initialize_ae_locus_overlay(mem: Any) -> dict[str, int]:
    """Idempotently create AE's standard locus rooms. Call once at AE
    bootstrap; safe to re-run (create_locus dedupes by label)."""
    return {
        "ops_compliance": mem.create_locus("Business Ops", "Compliance Room"),
        "ops_customers":  mem.create_locus("Business Ops", "Customers Room"),
        "ops_finance":    mem.create_locus("Business Ops", "Finance Room"),
        "field_jobs":     mem.create_locus("Field Work", "Active Jobs"),
        "field_permits":  mem.create_locus("Field Work", "Permits and Inspections"),
        "engineering":    mem.create_locus("Engineering", "Systems and Tools"),
    }


# --------------------------------------------------------------------------
# Lookup-before-create helpers (for Hermes-via-chat workflow per aux-builder
# msg_9cb0c42a 2026-05-02). Hermes can answer "do I already have this?"
# without QBO round-trip on every chat turn.
# --------------------------------------------------------------------------

def record_customer(
    mem: Any,
    *,
    name: str,
    qbo_id: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    addresses: Optional[list[str]] = None,
    source: str = "qbo",
    notes: Optional[str] = None,
) -> int:
    """Record a customer entity. Use recall_customer_by_name first to
    avoid duplicates. Returns memory id of the (new or existing) row.

    For lookup-before-create: caller does
        existing = recall_customer_by_name(mem, name=name)
        if existing: return existing[0]['id']
        return record_customer(mem, name=name, ...)
    """
    metadata = {
        "qbo_id": qbo_id,
        "email": email,
        "phone": phone,
        "addresses": addresses or [],
        "source": source,
    }
    if notes:
        metadata["notes"] = notes
    body_parts = [name]
    if email: body_parts.append(f"<{email}>")
    if phone: body_parts.append(f"({phone})")
    if addresses: body_parts.append(", ".join(addresses))
    return mem.remember(
        " ".join(body_parts),
        label=f"customer:{name}",
        kind="entity",
        source=source,
        origin_system="ae",
        valid_from=_time.time(),
        metadata=metadata,
    )


def record_template(
    mem: Any,
    *,
    template_id: str,
    name: str,
    job_type: str,  # "remodel" | "production"
    line_items: list[dict[str, Any]],
    base_pricing: Optional[dict[str, Any]] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Record an estimate template (reusable line-item bundle indexed by
    job_type). Used by recall_template_for_job to seed new estimates.

    line_items: each dict has at minimum {sku, description, qty, unit}.
    base_pricing: customer-side defaults (markup, etc.).
    """
    metadata = {
        "template_id": template_id,
        "job_type": job_type,
        "line_item_count": len(line_items),
        "line_items": line_items,
    }
    if base_pricing:
        metadata["base_pricing"] = base_pricing
    if extra_metadata:
        metadata.update(extra_metadata)
    return mem.remember(
        f"Template '{name}' ({job_type}, {len(line_items)} items)",
        label=f"template:{template_id}",
        kind="procedural",
        source="manual",
        origin_system="ae",
        valid_from=_time.time(),
        metadata=metadata,
    )


def record_estimate(
    mem: Any,
    *,
    estimate_id: str,
    customer_id: str,
    template_id: Optional[str],
    line_items_customer_pricing: list[dict[str, Any]],
    line_items_internal_cost: list[dict[str, Any]],
    status: str = "draft",
    total_amount: Optional[float] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Record an estimate with DUAL pricing (customer-facing + internal cost).
    Both stored as metadata so Hermes can show either to the right audience.
    Status: draft | sent | accepted | declined | invoiced.
    """
    metadata = {
        "estimate_id": estimate_id,
        "customer_id": customer_id,
        "template_id": template_id,
        "line_items_customer": line_items_customer_pricing,
        "line_items_internal": line_items_internal_cost,
        "status": status,
        "total_amount": total_amount,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return mem.remember(
        f"Estimate {estimate_id} for customer {customer_id} "
        f"({status}, ${total_amount or '?'})",
        label=f"estimate:{estimate_id}",
        kind="experience",
        source="dashboard",
        origin_system="ae",
        valid_from=_time.time(),
        metadata=metadata,
    )


def record_lead(
    mem: Any,
    *,
    source: str,  # "angi" | "thumbtack" | "hd_pro_referral" | "google_verified" | "email"
    contact: dict[str, Any],
    intent: str,
    raw_data: Optional[dict[str, Any]] = None,
    auto_check_existing: bool = True,
) -> dict[str, Any]:
    """Record a new lead. Auto-flags if a customer with same name/phone/email
    already exists (returns {"id": <new_lead_id>, "duplicate_customer_id": <existing>}).

    contact: {name?, email?, phone?, address?}.
    """
    name = contact.get("name", "")
    duplicate = None
    if auto_check_existing and name:
        existing = recall_customer_by_name(mem, name=name, k=1)
        if existing:
            duplicate = existing[0].get("id")
    metadata = {
        "lead_source": source,
        "contact": contact,
        "intent": intent,
        "duplicate_customer_id": duplicate,
    }
    if raw_data:
        metadata["raw_data"] = raw_data
    body = f"Lead from {source}: {name or '(no name)'} — {intent}"
    if duplicate:
        body += f" [duplicate of customer:{duplicate}]"
    mid = mem.remember(
        body,
        label=f"lead:{source}:{name or 'unknown'}",
        kind="experience",
        source=source,
        origin_system="ae",
        valid_from=_time.time(),
        metadata=metadata,
    )
    return {"id": mid, "duplicate_customer_id": duplicate}


def recall_customer_by_name(
    mem: Any,
    *,
    name: str,
    fuzzy: bool = True,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Look up customer entity by name. fuzzy=True uses hybrid_recall (semantic
    similarity catches "Sarah J", "Sarah Jones", "Sara Jones"); fuzzy=False
    requires exact label match via direct SQL (NOT sparse_search, which is
    BM25 fuzzy under the hood — bug caught by codex-resolver 2026-05-02).
    """
    if fuzzy:
        return mem.hybrid_recall(name, k=k, kind="entity", rerank=True)
    label = f"customer:{name}"
    if hasattr(mem, "store") and hasattr(mem.store, "conn"):
        with mem.store._lock:
            rows = mem.store.conn.execute(
                "SELECT id FROM memories WHERE label = ? ORDER BY id DESC LIMIT ?",
                (label, k),
            ).fetchall()
        out = []
        for row in rows:
            full = mem.get_memory(row[0]) if hasattr(mem, "get_memory") else mem.store.get(row[0])
            if full:
                out.append(full)
        return out
    raise NotImplementedError(
        "recall_customer_by_name(fuzzy=False) requires SQLite store with .conn; "
        "non-SQLite stores must implement direct label lookup."
    )


def recall_template_for_job(
    mem: Any,
    *,
    job_type: str,
    segment: Optional[str] = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Find templates indexed by job_type ('remodel' | 'production') with
    optional segment filter (e.g., 'kitchen', 'bathroom', 'service-call').
    Returns templates ranked by hybrid retrieval."""
    query_parts = [f"template {job_type}"]
    if segment:
        query_parts.append(segment)
    return mem.hybrid_recall(" ".join(query_parts), k=k, kind="procedural", rerank=True)


def recall_estimates_for_customer(
    mem: Any,
    *,
    customer_id: str,
    status: Optional[str] = None,
    k: int = 10,
    days: int = 365,
) -> list[dict[str, Any]]:
    """Get this customer's estimates (optionally filtered by status). Looks
    back `days` days via temporal search on bi-temporal valid_from.
    """
    query = f"estimate customer {customer_id}"
    if status:
        query += f" {status}"
    cutoff = _time.time() - (days * 86400)
    # Hydrate so 'metadata' (dict) is present, not just 'metadata_json' (str).
    # Bug caught by codex-resolver 2026-05-02.
    results = _hydrate_recall_rows(
        mem, mem.hybrid_recall(query, k=k, kind="experience", as_of=None, rerank=True))
    # Post-filter by metadata customer_id + status (hybrid is fuzzy by content)
    out = []
    for r in results:
        md = r.get("metadata") or {}
        if md.get("customer_id") != customer_id:
            continue
        if status and md.get("status") != status:
            continue
        if r.get("valid_from", 0) < cutoff:
            continue
        out.append(r)
    return out[:k]


def recall_recent_leads(
    mem: Any,
    *,
    source: Optional[str] = None,
    days: int = 7,
    k: int = 20,
) -> list[dict[str, Any]]:
    """Get recent leads (optionally filtered by source). Default last 7 days."""
    query = f"lead {source}" if source else "lead recent"
    cutoff = _time.time() - (days * 86400)
    # Hydrate so 'metadata' (dict) is present (bug caught by codex-resolver 2026-05-02).
    results = _hydrate_recall_rows(
        mem, mem.hybrid_recall(query, k=k * 2, kind="experience", rerank=True))
    out = []
    for r in results:
        md = r.get("metadata") or {}
        if "lead_source" not in md:
            continue
        if source and md.get("lead_source") != source:
            continue
        if r.get("valid_from", 0) < cutoff:
            continue
        out.append(r)
    return out[:k]


# =============================================================================
# AEEvidenceIngest v0 — Theme 8 typed evidence record contract
# Per codex-prescriptive-redesigner Day 4 (2026-05-02). NO LIVE INGESTION
# until Tito approves WA path + sent Gmail/PDF scope. Helpers are designed
# as read-only contract definitions; live ingest is gated separately.
#
# Cross-system contract (per redesigner Section D):
#   {capability_id, source_system, source_path, source_record_id,
#    valid_from, valid_to, confidence, privacy_class, consumer_hint}
#
# Privacy classes (gate live ingest by these):
#   - public        — no restriction
#   - internal      — internal AE ops only
#   - pii_low       — names, public phones, work addresses
#   - pii_high      — SSN, payment cards, home addresses (do NOT ingest without redaction)
#   - financial     — QBO, invoice amounts, P&L (separate retention policy)
# =============================================================================

EVIDENCE_PRIVACY_CLASSES = {"public", "internal", "pii_low", "pii_high", "financial"}
EVIDENCE_TYPES = {
    "wa_crew_message",      # WA crew chat snippet (Theme 8 P0)
    "sent_pdf",             # PDF the dashboard sent to a customer
    "estimate_event",       # estimate creation/approval/scheduling
    "material_price",       # supplier price quote / order
    "appscript_row",        # AppScript sheet row
    "blueprint_excerpt",    # blueprint snippet for BOM extraction
    "qbo_transaction",      # QBO transaction record
    "calendar_event",       # GCal event we want to remember
}


def _validate_privacy_class(privacy_class: str) -> str:
    if privacy_class not in EVIDENCE_PRIVACY_CLASSES:
        raise ValueError(
            f"privacy_class={privacy_class!r} not in "
            f"{sorted(EVIDENCE_PRIVACY_CLASSES)}"
        )
    return privacy_class


def _validate_evidence_type(evidence_type: str) -> str:
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(
            f"evidence_type={evidence_type!r} not in {sorted(EVIDENCE_TYPES)}"
        )
    return evidence_type


def _compute_evidence_id(
    evidence_type: str,
    source_system: str,
    source_record_id: Optional[str],
) -> str:
    """Deterministic evidence_id derived from the (type, system, record_id)
    triple. sha256 → first 16 hex chars (~64 bits). Same triple in any
    process produces the same id; this is the replay-authority key the
    upsert path keys on.

    `source_record_id=None` is allowed (unkeyed evidence, e.g. ad-hoc
    record_evidence_artifact calls). Such records share an id within the
    triple but the upsert path skips the lookup for None — see
    record_evidence_artifact — so multiple unkeyed rows still persist
    independently. Only triples with an explicit source_record_id get
    the dedup guarantee.
    """
    payload = f"{evidence_type}|{source_system}|{source_record_id or ''}"
    return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _lookup_existing_evidence_memory_id(
    mem: Any, evidence_id: str
) -> Optional[int]:
    """Find an existing memory whose metadata.evidence_id matches.
    Returns the memory_id of the most-recent match (highest id), or None.

    Uses SQLite JSON1 json_extract for an indexable scan over metadata_json.
    Falls back to substring match on metadata_json if json_extract isn't
    available (older SQLite). This mirrors the SQL-direct access pattern
    used by recall_customer_by_name(fuzzy=False).

    NOTE (S2 packet 2026-05-03): superseded by the SQLite-backed
    `evidence_ledger` table installed via SchemaUpgrade. record_evidence_artifact
    now consults the ledger first; this JSON-scan path remains as a fallback
    for stores where the ledger table doesn't exist (legacy DBs that haven't
    run SchemaUpgrade yet, or non-SQLite stores).
    """
    if not hasattr(mem, "store") or not hasattr(mem.store, "conn"):
        return None
    try:
        with mem.store._lock:
            try:
                row = mem.store.conn.execute(
                    "SELECT id FROM memories "
                    "WHERE json_extract(metadata_json, '$.evidence_id') = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (evidence_id,),
                ).fetchone()
            except Exception:
                # Fallback: substring match on metadata_json. Slower, but
                # works on stores without JSON1. The token is unique enough
                # (16 hex chars after a colon) to avoid false positives.
                like_pat = f'%"evidence_id": "{evidence_id}"%'
                row = mem.store.conn.execute(
                    "SELECT id FROM memories WHERE metadata_json LIKE ? "
                    "ORDER BY id DESC LIMIT 1",
                    (like_pat,),
                ).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def _ledger_table_exists(conn: Any) -> bool:
    """Cheap probe — does this DB carry the evidence_ledger table yet?
    Used so record_evidence_artifact can transparently fall back to the
    legacy json_extract dedup path on un-upgraded stores."""
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='evidence_ledger'"
        ).fetchone() is not None
    except Exception:
        return False


def _ledger_lookup(
    conn: Any, evidence_id: str
) -> Optional[tuple[int, str]]:
    """Return (memory_id, status) for evidence_id, or None if not present.
    memory_id may be None in the DB (rare race window between ledger insert
    and mem.remember); the caller is responsible for treating that as 'still
    inserting' and skipping the dedup."""
    try:
        row = conn.execute(
            "SELECT memory_id, status FROM evidence_ledger "
            "WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    mem_id, status = row[0], row[1]
    if mem_id is None:
        return None
    return (int(mem_id), str(status))


def _ledger_reserve(
    conn: Any,
    *,
    evidence_id: str,
    evidence_type: str,
    source_system: str,
    source_record_id: str,
) -> bool:
    """Attempt to claim the ledger row for this evidence_id. Returns True
    if THIS call won the race (we should now mem.remember() and update the
    row). Returns False if a concurrent process already owns the row.

    Uses INSERT OR IGNORE so the second writer silently no-ops; the
    second writer then re-reads the ledger to discover the winning
    memory_id."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO evidence_ledger "
        "(evidence_id, memory_id, evidence_type, source_system, "
        " source_record_id, status) "
        "VALUES (?, NULL, ?, ?, ?, 'inserted')",
        (evidence_id, evidence_type, source_system, source_record_id),
    )
    return cur.rowcount == 1


def _ledger_set_memory_id(
    conn: Any, *, evidence_id: str, memory_id: int
) -> None:
    """Patch in the memory_id after mem.remember() succeeded for the
    just-reserved ledger row."""
    conn.execute(
        "UPDATE evidence_ledger "
        "SET memory_id = ?, updated_at = datetime('now') "
        "WHERE evidence_id = ?",
        (memory_id, evidence_id),
    )


def _ledger_release(conn: Any, *, evidence_id: str) -> None:
    """Roll back a ledger reservation when mem.remember() fails. Caller
    has already won the row via _ledger_reserve; this delete frees it so
    the next call can retry. Best-effort: failures are swallowed because
    the row will still be junk (memory_id NULL) and a retry will re-claim.
    """
    try:
        conn.execute(
            "DELETE FROM evidence_ledger WHERE evidence_id = ? "
            "AND memory_id IS NULL",
            (evidence_id,),
        )
    except Exception:
        pass


def record_evidence_artifact(
    mem: Any,
    *,
    evidence_type: str,
    capability_id: str,
    source_system: str,
    source_path: str,
    content: str,
    privacy_class: str = "internal",
    confidence: float = 0.9,
    source_record_id: Optional[str] = None,
    valid_from: Optional[float] = None,
    valid_to: Optional[float] = None,
    consumer_hint: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generic typed evidence record — base of the AEEvidenceIngest contract.

    All specialized record_*_evidence helpers below delegate here. The
    contract preserves provenance (source_system + source_path + source_record_id)
    plus bi-temporal validity plus privacy classification, so any later
    audit / privacy redaction / capability-runtime-proof query can reason
    about it without re-parsing content.

    REPLAY AUTHORITY (P0-2 fix, S2 packet 2026-05-03):
    Each row carries a deterministic `evidence_id` in metadata, computed
    from (evidence_type, source_system, source_record_id). Before insert,
    we look up an existing row with the same evidence_id; if found we
    return the existing memory_id with `inserted=False`. This makes WA /
    PDF / QBO ingest replay-safe — re-running the same batch produces no
    duplicates.

    Skipping condition: when `source_record_id is None`, we still compute
    an evidence_id but DO NOT do the lookup — unkeyed records are treated
    as always-new (they can't be reliably deduped without a stable key).

    Returns: {"memory_id": int, "evidence_id": str, "inserted": bool}.

    Per Theme 8 + Theme 9 of north-star: feeds typed normalized facts
    into substrate without losing source-of-truth pointer.
    """
    _validate_evidence_type(evidence_type)
    _validate_privacy_class(privacy_class)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence={confidence} must be in [0.0, 1.0]")

    evidence_id = _compute_evidence_id(evidence_type, source_system, source_record_id)

    # Replay-authority + race-safety (S2 packet 2026-05-03):
    # When the SQLite-backed evidence_ledger is available AND we have a
    # stable source_record_id, claim the ledger row first via INSERT OR
    # IGNORE. The winner of the race calls mem.remember() and patches the
    # ledger row with the resulting memory_id; losers re-read the ledger to
    # find the winning memory_id. When source_record_id is None we always
    # insert (unkeyed evidence has no stable identity).
    #
    # Fallback: stores without an SQLite conn (or DBs that haven't run
    # SchemaUpgrade yet) drop back to the legacy json_extract scan.
    use_ledger = (
        source_record_id is not None
        and hasattr(mem, "store") and hasattr(mem.store, "conn")
    )
    if use_ledger:
        store = mem.store
        with store._lock:
            if not _ledger_table_exists(store.conn):
                use_ledger = False

    if use_ledger:
        store = mem.store
        with store._lock:
            won = _ledger_reserve(
                store.conn,
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                source_system=source_system,
                source_record_id=source_record_id,  # type: ignore[arg-type]
            )
            store.conn.commit()
        if not won:
            # Lost the race. Winner is mid-flight calling mem.remember();
            # poll the ledger until memory_id is patched in (or budget
            # expires). Lock is released between polls so the winner can
            # complete. Without this wait, mem.remember() is non-thread-safe
            # under concurrent load (HNSW/connection-graph dict iteration).
            existing = None
            for _ in range(40):
                with store._lock:
                    existing = _ledger_lookup(store.conn, evidence_id)
                if existing is not None:
                    break
                _time.sleep(0.025)
            if existing is not None:
                return {
                    "memory_id": existing[0],
                    "evidence_id": evidence_id,
                    "inserted": False,
                }
            # Budget exhausted — winner is wedged or extremely slow. Fall
            # through to a fresh mem.remember(); accept potential duplicate
            # rather than block forever.
    elif source_record_id is not None:
        # Legacy / pre-upgrade path: JSON-scan dedup.
        existing_id = _lookup_existing_evidence_memory_id(mem, evidence_id)
        if existing_id is not None:
            return {
                "memory_id": existing_id,
                "evidence_id": evidence_id,
                "inserted": False,
            }

    metadata = {
        "evidence_type": evidence_type,
        "capability_id": capability_id,
        "source_system": source_system,
        "source_path": source_path,
        "privacy_class": privacy_class,
        "evidence_id": evidence_id,
    }
    if source_record_id is not None:
        metadata["source_record_id"] = source_record_id
    if consumer_hint is not None:
        metadata["consumer_hint"] = consumer_hint
    if extra_metadata:
        # Caller-supplied keys cannot override contract keys above
        for k, v in extra_metadata.items():
            metadata.setdefault(k, v)

    label = f"evidence:{evidence_type}:{capability_id}:{source_record_id or 'unkeyed'}"
    # detect_conflicts=False is the bi-temporal audit contract: every evidence
    # record is preserved as its own row. Supersession (valid_to set) is an
    # explicit caller decision, not an implicit conflict-detection side-effect.
    # Caught by per-commit reviewer of d410019. Matches record_invoice_status_change
    # pattern (also bi-temporal, also detect_conflicts=False).
    try:
        memory_id = mem.remember(
            content,
            label=label,
            kind="experience",
            confidence=confidence,
            source=source_system,
            origin_system="ae",
            valid_from=valid_from if valid_from is not None else _time.time(),
            valid_to=valid_to,
            metadata=metadata,
            detect_conflicts=False,
        )
    except Exception:
        # mem.remember failed AFTER we won the ledger reservation — release
        # the row so a retry can succeed. Then re-raise.
        if use_ledger:
            store = mem.store
            with store._lock:
                _ledger_release(store.conn, evidence_id=evidence_id)
                store.conn.commit()
        raise

    if use_ledger:
        store = mem.store
        with store._lock:
            _ledger_set_memory_id(
                store.conn, evidence_id=evidence_id, memory_id=memory_id
            )
            store.conn.commit()

    return {
        "memory_id": memory_id,
        "evidence_id": evidence_id,
        "inserted": True,
    }


def record_wa_crew_event(
    mem: Any,
    *,
    capability_id: str,
    thread_id: str,
    sender: str,
    raw_text: str,
    ts: float,
    lang: str = "es",
    normalized_text: Optional[str] = None,
    media_paths: Optional[list[str]] = None,
    delivery_status: str = "delivered",
    auth_proof: Optional[str] = None,
    privacy_class: str = "internal",
    consumer_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Record a WA crew chat event with full Theme 8 contract schema.

    Per redesigner Section D: Hermes is canonical owner of WA delivery,
    NM is canonical owner of WA recall + evidence typing. This helper is
    the NM-side ingest target — Hermes batches WA into JSONL with this
    shape and NM calls record_wa_crew_event per row.

    Required: thread_id, sender, raw_text, ts. Optional: normalized_text
    (translation/cleanup), media_paths (image/audio attachments),
    auth_proof (delivery receipt / read receipt / signature).

    Source path defaults to "wa_bridge:<thread_id>:<ts>" if no specific
    file source. Bi-temporal: valid_from = ts; valid_to remains NULL
    until message is corrected/superseded.

    Returns: {"memory_id": int, "evidence_id": str, "inserted": bool}
    (same contract as record_evidence_artifact — replay-safe via the
    deterministic evidence_id key).
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("record_wa_crew_event: raw_text must be non-empty")
    if delivery_status not in {"delivered", "pending", "failed", "read"}:
        raise ValueError(
            f"delivery_status={delivery_status!r} must be one of "
            f"delivered/pending/failed/read"
        )

    content_parts = [raw_text]
    if normalized_text and normalized_text != raw_text:
        content_parts.append(f"\n[normalized:{lang}→en]\n{normalized_text}")
    content = "".join(content_parts)

    extra = {
        "thread_id": thread_id,
        "sender": sender,
        "lang": lang,
        "delivery_status": delivery_status,
        "door": "wa",
    }
    if media_paths:
        extra["media_paths"] = media_paths
    if auth_proof:
        extra["auth_proof"] = auth_proof
    if normalized_text:
        extra["normalized_text"] = normalized_text

    # Provenance keys: thread_id + microsecond ts + content hash. Two
    # WA messages in the same thread within one second would collide on
    # int(ts) alone (caught by per-commit reviewer of d410019). Microsecond
    # resolution + 8-char content hash makes collision practically impossible.
    import hashlib
    content_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:8]
    ts_us = int(ts * 1_000_000)  # microsecond precision
    record_id = f"{thread_id}:{ts_us}:{content_hash}"
    return record_evidence_artifact(
        mem,
        evidence_type="wa_crew_message",
        capability_id=capability_id,
        source_system="hermes_wa_bridge",
        source_path=f"wa_bridge:{record_id}",
        content=content,
        privacy_class=privacy_class,
        confidence=0.95 if delivery_status == "read" else 0.85,
        source_record_id=record_id,
        valid_from=ts,
        consumer_hint=consumer_hint,
        extra_metadata=extra,
    )


def record_estimate_evidence(
    mem: Any,
    *,
    capability_id: str,
    estimate_id: str,
    customer_id: str,
    event_type: str,        # draft / sent / approved / scheduled / lost
    pdf_path: Optional[str] = None,
    amount_cents: Optional[int] = None,
    sent_to: Optional[str] = None,
    sent_at: Optional[float] = None,
    privacy_class: str = "financial",
    consumer_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Record an estimate-pipeline event with provenance to the PDF artifact.

    For sent events: pdf_path points at /Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/data/sent-estimates-pdfs/<file>.
    For draft/approved/scheduled/lost: pdf_path optional; the event itself is the evidence.

    The financial privacy class is default since estimates carry pricing
    that's not internal-public. Override only with explicit Tito approval.

    Returns: {"memory_id": int, "evidence_id": str, "inserted": bool}
    (forwarded from record_evidence_artifact for replay safety).
    """
    if event_type not in {"draft", "sent", "approved", "scheduled", "lost"}:
        raise ValueError(
            f"event_type={event_type!r} must be one of "
            f"draft/sent/approved/scheduled/lost"
        )

    content_parts = [f"Estimate {estimate_id} for customer {customer_id}: {event_type}"]
    if amount_cents is not None:
        content_parts.append(f"\nAmount: ${amount_cents/100:.2f}")
    if sent_to:
        content_parts.append(f"\nSent to: {sent_to}")
    if pdf_path:
        content_parts.append(f"\nPDF: {pdf_path}")
    content = "".join(content_parts)

    extra = {
        "estimate_id": estimate_id,
        "customer_id": customer_id,
        "event_type": event_type,
    }
    if amount_cents is not None:
        extra["amount_cents"] = amount_cents
    if sent_to:
        extra["sent_to"] = sent_to
    if sent_at is not None:
        extra["sent_at"] = sent_at
    if pdf_path:
        extra["pdf_path"] = pdf_path

    return record_evidence_artifact(
        mem,
        evidence_type="estimate_event" if not pdf_path else "sent_pdf",
        capability_id=capability_id,
        source_system="ae_dashboard",
        source_path=pdf_path or f"estimate_pipeline:{estimate_id}:{event_type}",
        content=content,
        privacy_class=privacy_class,
        confidence=0.95,
        source_record_id=f"{estimate_id}:{event_type}",
        valid_from=sent_at if sent_at is not None else _time.time(),
        consumer_hint=consumer_hint,
        extra_metadata=extra,
    )


def record_material_price_evidence(
    mem: Any,
    *,
    capability_id: str,
    sku: str,
    vendor: str,
    price_cents: int,
    quoted_at: float,
    quote_source_path: str,
    unit: str = "ea",
    valid_to: Optional[float] = None,
    consumer_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Record a material/SKU price evidence with bi-temporal validity.

    Use valid_to to mark price superseded (don't delete the prior — temporal
    queries need to know what we believed before the supersede).

    Vendor: amperage / gve / amazon / hd / supplyhouse / etc.
    Privacy: 'internal' is default — prices aren't public, but not PII.

    Returns: {"memory_id": int, "evidence_id": str, "inserted": bool}
    (forwarded from record_evidence_artifact for replay safety).
    """
    if price_cents < 0:
        raise ValueError(f"price_cents={price_cents} must be non-negative")

    content = (
        f"{sku} from {vendor}: ${price_cents/100:.2f} per {unit} "
        f"(quoted {_time.strftime('%Y-%m-%d', _time.localtime(quoted_at))})"
    )

    extra = {
        "sku": sku,
        "vendor": vendor,
        "price_cents": price_cents,
        "unit": unit,
        "quoted_at": quoted_at,
    }

    return record_evidence_artifact(
        mem,
        evidence_type="material_price",
        capability_id=capability_id,
        source_system=f"vendor:{vendor}",
        source_path=quote_source_path,
        content=content,
        privacy_class="internal",
        confidence=0.95,
        source_record_id=f"{sku}:{vendor}:{int(quoted_at)}",
        valid_from=quoted_at,
        valid_to=valid_to,
        consumer_hint=consumer_hint,
        extra_metadata=extra,
    )


__all__ = [
    "record_customer_interaction",
    "record_job_event",
    "record_whatsapp_message",
    "record_sop",
    "record_invoice_status_change",
    "record_financial_event",
    "initialize_ae_locus_overlay",
    "recall_for_dashboard",
    # Lookup-before-create helpers (aux-builder msg_9cb0c42a 2026-05-02)
    "record_customer",
    "record_template",
    "record_estimate",
    "record_lead",
    "recall_customer_by_name",
    "recall_template_for_job",
    "recall_estimates_for_customer",
    "recall_recent_leads",
    # AEEvidenceIngest v0 (Theme 8 + 9, codex-redesigner Day 4 2026-05-02)
    "record_evidence_artifact",
    "record_wa_crew_event",
    "record_estimate_evidence",
    "record_material_price_evidence",
    "EVIDENCE_PRIVACY_CLASSES",
    "EVIDENCE_TYPES",
    # Replay-authority key (S2 packet 2026-05-03)
    "_compute_evidence_id",
]
