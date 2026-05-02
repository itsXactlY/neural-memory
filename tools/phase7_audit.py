#!/usr/bin/env python3
"""Phase 7 health audit — read-only inspection of how a live neural-memory
DB exercises the typed/entity/scoring features added in Sprint 2 Phase 7.

Reports:
  - Memory counts by kind (catches "everything classified as 'unknown'" drift)
  - Top entities by mention frequency
  - mentions_entity / derived_from / contradicts edge counts
  - Validity-window coverage (how many memories have valid_from/valid_to set)
  - Memify duplicate candidates (does NOT apply downweight)
  - Contradiction candidates (does NOT add edges)
  - Locus overlay coverage
  - FTS5 index sync sanity check
  - Salience distribution
  - Schema column presence sanity

Default DB: ~/.neural_memory/memory.db. Override with --db.

Usage:
    python3 tools/phase7_audit.py
    python3 tools/phase7_audit.py --db /path/to/memory.db
    python3 tools/phase7_audit.py --json /tmp/phase7_audit.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def _section(title: str) -> str:
    return f"\n=== {title} ===\n"


def audit(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    out: dict = {"db_path": db_path}

    # ---- counts by kind --------------------------------------------------
    kind_counts = dict(conn.execute(
        "SELECT COALESCE(kind, '<NULL>'), COUNT(*) FROM memories GROUP BY kind"
    ).fetchall())
    out["memories_by_kind"] = kind_counts
    out["memories_total"] = sum(kind_counts.values())

    # ---- entity stats -----------------------------------------------------
    entities = conn.execute(
        "SELECT id, label, metadata_json FROM memories "
        "WHERE kind = 'entity' ORDER BY id"
    ).fetchall()
    entity_freqs = []
    for e in entities:
        meta = json.loads(e["metadata_json"]) if e["metadata_json"] else {}
        entity_freqs.append({
            "id": e["id"],
            "label": e["label"],
            "frequency": meta.get("frequency", 0),
            "last_seen": meta.get("last_seen"),
        })
    entity_freqs.sort(key=lambda x: -x["frequency"])
    out["entity_count"] = len(entity_freqs)
    out["top_entities"] = entity_freqs[:10]

    # ---- edge type breakdown ---------------------------------------------
    edge_breakdown = dict(conn.execute(
        "SELECT COALESCE(edge_type, '<NULL>'), COUNT(*) FROM connections "
        "GROUP BY edge_type ORDER BY COUNT(*) DESC"
    ).fetchall())
    out["edges_by_type"] = edge_breakdown
    out["edges_total"] = sum(edge_breakdown.values())

    # ---- validity coverage -----------------------------------------------
    vfm = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE valid_from IS NOT NULL"
    ).fetchone()[0]
    vtm = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE valid_to IS NOT NULL"
    ).fetchone()[0]
    out["validity_coverage"] = {
        "memories_with_valid_from": vfm,
        "memories_with_valid_to": vtm,
        "memories_total": out["memories_total"],
    }

    # ---- memify duplicate candidates -------------------------------------
    dupe_groups = conn.execute(
        "SELECT content, COUNT(*) AS n FROM memories "
        "WHERE (kind IS NULL OR kind != 'entity') AND content IS NOT NULL "
        "GROUP BY content HAVING n > 1 ORDER BY n DESC"
    ).fetchall()
    out["memify_duplicate_groups"] = len(dupe_groups)
    out["memify_top_duplicates"] = [
        {"content_excerpt": (row["content"] or "")[:120], "count": row["n"]}
        for row in dupe_groups[:5]
    ]
    out["memify_extra_rows_if_applied"] = sum(row["n"] - 1 for row in dupe_groups)

    # ---- contradiction candidates ----------------------------------------
    pairs = conn.execute(
        "SELECT a.id AS a_id, a.content AS a_content, a.valid_to AS a_valid_to, "
        "       b.id AS b_id, b.content AS b_content, b.valid_from AS b_valid_from "
        "FROM memories a, memories b "
        "WHERE a.valid_to IS NOT NULL AND b.valid_from IS NOT NULL "
        "  AND a.valid_to < b.valid_from "
        "  AND a.id != b.id "
        "  AND (a.kind IS NULL OR a.kind != 'entity') "
        "  AND (b.kind IS NULL OR b.kind != 'entity')"
    ).fetchall()
    out["contradiction_candidates_by_validity"] = len(pairs)

    # ---- locus overlay ---------------------------------------------------
    locus_count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE kind = 'locus'"
    ).fetchone()[0]
    located_in_count = conn.execute(
        "SELECT COUNT(*) FROM connections WHERE edge_type = 'located_in'"
    ).fetchone()[0]
    out["locus_overlay"] = {
        "locus_nodes": locus_count,
        "located_in_edges": located_in_count,
    }

    # ---- FTS5 sync check -------------------------------------------------
    fts_present = bool(conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'memories_fts'"
    ).fetchone())
    out["fts5"] = {"index_present": fts_present}
    if fts_present:
        fts_rows = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        non_entity_memories = conn.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE content IS NOT NULL "
            "  AND (kind IS NULL OR kind != 'entity')"
        ).fetchone()[0]
        out["fts5"]["fts_row_count"] = fts_rows
        out["fts5"]["expected_indexed_count"] = non_entity_memories
        out["fts5"]["sync_delta"] = fts_rows - non_entity_memories

    # ---- salience distribution -------------------------------------------
    sal_buckets = conn.execute(
        "SELECT "
        "  SUM(CASE WHEN salience IS NULL THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN salience < 0.5 THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN salience >= 0.5 AND salience < 1.0 THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN salience >= 1.0 AND salience < 1.5 THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN salience >= 1.5 THEN 1 ELSE 0 END) "
        "FROM memories"
    ).fetchone()
    out["salience_distribution"] = {
        "null":       sal_buckets[0],
        "below_0.5":  sal_buckets[1],
        "0.5_to_1.0": sal_buckets[2],
        "1.0_to_1.5": sal_buckets[3],
        "above_1.5":  sal_buckets[4],
    }

    # ---- Phase 7 feature usage (shipped but unwired detection) ---------
    # Caught 2026-05-01: Phase 7 commits 1-10 shipped schema + APIs for
    # node kinds (dream_insight, locus, profile_trait), edge types
    # (summarizes, contradicts, located_in, rem_bridge), and metadata
    # fields (valid_to, locus_id, procedural_score). But several features
    # had ZERO production rows in live DB, indicating no consumer wires
    # the API to actually populate them. Surface this so wiring gaps are
    # visible in routine audits.
    feature_usage = {}
    feature_checks = [
        ("dream_insight_nodes",
         "SELECT COUNT(*) FROM memories WHERE kind='dream_insight'"),
        ("locus_nodes",
         "SELECT COUNT(*) FROM memories WHERE kind='locus'"),
        ("profile_trait_nodes",
         "SELECT COUNT(*) FROM memories WHERE kind='profile_trait'"),
        ("summarizes_edges",
         "SELECT COUNT(*) FROM connections WHERE edge_type='summarizes'"),
        ("contradicts_edges",
         "SELECT COUNT(*) FROM connections WHERE edge_type='contradicts'"),
        ("located_in_edges",
         "SELECT COUNT(*) FROM connections WHERE edge_type='located_in'"),
        ("memories_with_valid_to",
         "SELECT COUNT(*) FROM memories WHERE valid_to IS NOT NULL"),
        ("memories_with_locus_id",
         "SELECT COUNT(*) FROM memories WHERE locus_id IS NOT NULL"),
        ("memories_with_procedural_score",
         "SELECT COUNT(*) FROM memories WHERE procedural_score IS NOT NULL"),
    ]
    for name, q in feature_checks:
        try:
            feature_usage[name] = conn.execute(q).fetchone()[0]
        except sqlite3.OperationalError:
            feature_usage[name] = None
    out["phase7_feature_usage"] = feature_usage
    out["phase7_unwired_features"] = sorted(
        k for k, v in feature_usage.items() if v == 0
    )

    # ---- dream_insights bloat detection ---------------------------------
    # Caught 2026-05-01: add_insight() does unconditional INSERT, so every
    # dream cycle re-emits the same bridge/cluster insights. Without a
    # uniqueness guard the table grows linearly with cycle count × node count.
    # Surface the duplication ratio + disk impact so future audits catch
    # regressions before they hit GB-scale.
    dream_present = bool(conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'dream_insights'"
    ).fetchone())
    if dream_present:
        total_insights = conn.execute(
            "SELECT COUNT(*) FROM dream_insights"
        ).fetchone()[0]
        unique_insights = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT 1 FROM dream_insights "
            "  GROUP BY insight_type, source_memory_id, COALESCE(content,'')"
            ")"
        ).fetchone()[0]
        dup_ratio = (
            round(100.0 * (total_insights - unique_insights) / total_insights, 1)
            if total_insights else 0.0
        )
        sessions = conn.execute(
            "SELECT COUNT(*) FROM dream_sessions"
        ).fetchone()[0] if conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'dream_sessions'"
        ).fetchone() else 0
        out["dream_insights"] = {
            "total_rows": total_insights,
            "unique_combos": unique_insights,
            "duplicate_ratio_pct": dup_ratio,
            "dream_sessions": sessions,
            "bloat_flag": dup_ratio > 50.0,
        }

    # ---- schema column sanity -------------------------------------------
    mem_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    conn_cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)")}
    expected_mem = {"kind", "confidence", "transaction_time", "valid_from",
                    "valid_to", "metadata_json", "memory_visibility",
                    "pin_state", "decay_rate", "reuse_count",
                    "last_reinforced_at", "extracted_entities_json",
                    "locus_id", "procedural_score", "origin_system", "source"}
    expected_conn = {"edge_type", "confidence", "valid_from", "valid_to",
                     "transaction_time", "salience", "evidence_count",
                     "metadata_json", "origin_system", "last_strengthened_at"}
    out["schema_phase7_completeness"] = {
        "memories_phase7_cols_present": len(expected_mem & mem_cols),
        "memories_phase7_cols_expected": len(expected_mem),
        "memories_missing": sorted(expected_mem - mem_cols),
        "connections_phase7_cols_present": len(expected_conn & conn_cols),
        "connections_phase7_cols_expected": len(expected_conn),
        "connections_missing": sorted(expected_conn - conn_cols),
    }

    conn.close()
    return out


def render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"Phase 7 Audit Report")
    lines.append(f"DB: {report['db_path']}")
    lines.append(f"Total memories: {report['memories_total']}")
    lines.append(f"Total edges: {report['edges_total']}")

    lines.append(_section("Memories by kind"))
    for kind, n in sorted(report["memories_by_kind"].items(),
                          key=lambda x: -x[1]):
        lines.append(f"  {kind:25s} {n:6d}")

    lines.append(_section("Top entities by frequency"))
    for e in report["top_entities"]:
        lines.append(f"  freq={e['frequency']:4d}  id={e['id']:5d}  "
                     f"{e['label'][:60]}")
    if not report["top_entities"]:
        lines.append("  (none)")

    lines.append(_section("Edges by type"))
    for et, n in sorted(report["edges_by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"  {et:30s} {n:7d}")

    lines.append(_section("Validity coverage"))
    vc = report["validity_coverage"]
    lines.append(f"  memories with valid_from: "
                 f"{vc['memories_with_valid_from']:5d} / {vc['memories_total']}")
    lines.append(f"  memories with valid_to:   "
                 f"{vc['memories_with_valid_to']:5d} / {vc['memories_total']}")

    lines.append(_section("Memify duplicate candidates"))
    lines.append(f"  duplicate groups: {report['memify_duplicate_groups']}")
    lines.append(f"  rows downweighted if applied: "
                 f"{report['memify_extra_rows_if_applied']}")
    for d in report["memify_top_duplicates"]:
        lines.append(f"    x{d['count']}  {d['content_excerpt'][:100]}")

    lines.append(_section("Contradiction candidates by validity sequence"))
    lines.append(f"  pairs (validity-only signal): "
                 f"{report['contradiction_candidates_by_validity']}")

    lines.append(_section("Locus overlay"))
    lo = report["locus_overlay"]
    lines.append(f"  locus nodes:       {lo['locus_nodes']}")
    lines.append(f"  located_in edges:  {lo['located_in_edges']}")

    lines.append(_section("FTS5 index"))
    fts = report["fts5"]
    if fts["index_present"]:
        lines.append(f"  index present:           yes")
        lines.append(f"  fts row count:           {fts['fts_row_count']}")
        lines.append(f"  expected indexed count:  {fts['expected_indexed_count']}")
        lines.append(f"  sync delta:              {fts['sync_delta']}  "
                     f"(0 = perfect sync; nonzero = drift)")
    else:
        lines.append(f"  NOT PRESENT — FTS5 unavailable on this build")

    lines.append(_section("Salience distribution"))
    for bucket, n in report["salience_distribution"].items():
        lines.append(f"  {bucket:12s} {n:6d}")

    if "phase7_feature_usage" in report:
        lines.append(_section("Phase 7 feature usage (live DB row counts)"))
        for name, count in sorted(report["phase7_feature_usage"].items()):
            tag = " <- UNWIRED" if count == 0 else ""
            tag = " <- COL MISSING" if count is None else tag
            lines.append(f"  {name:35s} {count if count is not None else '?':>8}{tag}")
        if report["phase7_unwired_features"]:
            lines.append(f"")
            lines.append(f"  >> {len(report['phase7_unwired_features'])} features with 0 DB rows:")
            for name in report["phase7_unwired_features"]:
                lines.append(f"     - {name}")

    # Phase 7.5 call-site wiring status (manually tracked; bumps when
    # subphases ship). DB-row counts above measure data presence; this
    # section measures whether the scorer's call-site actually reads each
    # field. Both must be true for a feature to influence rankings.
    lines.append(_section("Phase 7.5 call-site wiring status"))
    lines.append(f"  α  procedural_score read in scorer        SHIPPED 2026-05-01")
    lines.append(f"  β  entity_score from mentions_entity edges  SHIPPED 2026-05-01")
    lines.append(f"  γ  stale_penalty from age                   SHIPPED 2026-05-01")
    lines.append(f"  δ  contradiction_penalty from edges         SHIPPED 2026-05-01")
    lines.append(f"  ε  locus_score                              SHIPPED 2026-05-01")
    lines.append(f"     valid_to / contradicts edge writers       DEFERRED (data-side wiring)")
    lines.append(f"  Coverage: integration test suite")
    lines.append(f"  python/test_phase7_5_wiring_integration.py  6 contracts")

    if "dream_insights" in report:
        di = report["dream_insights"]
        lines.append(_section("Dream insights bloat"))
        lines.append(f"  total rows:        {di['total_rows']}")
        lines.append(f"  unique combos:     {di['unique_combos']}")
        lines.append(f"  duplicate ratio:   {di['duplicate_ratio_pct']}%")
        lines.append(f"  dream sessions:    {di['dream_sessions']}")
        if di["bloat_flag"]:
            lines.append(f"  >> BLOAT FLAG: duplicate ratio > 50%; "
                         f"add_insight() likely lacks idempotency guard")

    lines.append(_section("Phase 7 schema completeness"))
    sc = report["schema_phase7_completeness"]
    lines.append(f"  memories cols:    "
                 f"{sc['memories_phase7_cols_present']}/{sc['memories_phase7_cols_expected']}")
    if sc["memories_missing"]:
        lines.append(f"  memories missing: {sc['memories_missing']}")
    lines.append(f"  connections cols: "
                 f"{sc['connections_phase7_cols_present']}/{sc['connections_phase7_cols_expected']}")
    if sc["connections_missing"]:
        lines.append(f"  connections missing: {sc['connections_missing']}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(Path.home() / ".neural_memory" / "memory.db"))
    parser.add_argument("--json", default=None,
                        help="JSON output path (in addition to human-readable)")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    report = audit(args.db)
    print(render_text(report))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"JSON report written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
