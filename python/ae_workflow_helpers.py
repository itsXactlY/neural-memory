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

import time as _time
from typing import Any, Optional


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


__all__ = [
    "record_customer_interaction",
    "record_job_event",
    "record_whatsapp_message",
    "record_sop",
    "record_invoice_status_change",
    "record_financial_event",
    "initialize_ae_locus_overlay",
]
