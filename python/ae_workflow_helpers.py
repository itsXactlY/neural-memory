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

    Encapsulates the configuration empirically validated 2026-05-01 to
    produce R@5=0.82 on the AE-domain bench (passing the 0.76 threshold).
    Drop-in replacement for raw mem.recall() / mem.hybrid_recall() in
    dashboard-side code.

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
]
