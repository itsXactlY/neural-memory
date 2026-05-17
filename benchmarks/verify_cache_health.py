#!/usr/bin/env python3
"""verify_cache_health.py — assert a bake schema has every artifact.

Run as the FINAL gate before snapshot/bench. Prints a green/red dashboard
of the cache schema's row counts vs. minimum-expected thresholds, exits
non-zero if any artifact is missing.

USAGE
    python benchmarks/verify_cache_health.py                  # --variant s
    python benchmarks/verify_cache_health.py --variant oracle
    python benchmarks/verify_cache_health.py --schema custom_name

EXPECTED TABLES (in cache schema)
    memories                    — sessions + AFE facts + chunks
    connections                 — graph edges (REM bridges, AFE supports)
    memory_revisions            — supersession audit
    memory_dae_embeddings       — DAE rerank vectors
    dream_sessions              — per-phase audit rows
    dream_insights              — cluster + bridge summaries
    connection_history          — append-only weight-change log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"


def _dsn() -> str:
    from postgres_store import _build_dsn
    import urllib.parse
    base = _build_dsn()
    p = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(p._replace(path=f"/{CACHE_DB}"))


# (table, column-list-for-SELECT-count-star, min_expected, label, fatal_if_missing)
CHECKS = [
    ("memories",              "*",           5_000, "Memories (sessions+AFE+chunks)",     True),
    ("connections",           "*",          10_000, "Connections (REM bridges + AFE supports)", True),
    ("memory_dae_embeddings", "*",             100, "DAE embeddings",                     False),
    ("memory_revisions",      "*",               0, "Memory revisions (Supersedes)",       False),
    ("dream_sessions",        "*",               1, "Dream sessions",                      True),
    ("dream_insights",        "*",               1, "Dream insights (cluster+bridge)",     True),
    ("connection_history",    "*",               1, "Connection history",                 False),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    p.add_argument("--schema", default=None,
                   help="Override cache schema name (default: longmemeval_<variant>_bgem3_1024)")
    args = p.parse_args()
    schema = args.schema or f"longmemeval_{args.variant}_bgem3_1024"

    import psycopg
    failures = 0
    print(f"\n[verify] DB={CACHE_DB}  schema={schema}\n")
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            # Schema exists?
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name=%s",
                (schema,),
            )
            if not cur.fetchone():
                print(f"  ❌ schema {schema!r} does not exist")
                return 2

            for table, _cols, min_expected, label, fatal in CHECKS:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                if not cur.fetchone():
                    msg = "MISSING TABLE"
                    icon = "❌" if fatal else "⚠"
                    if fatal:
                        failures += 1
                    print(f"  {icon} {label:<45} {msg}")
                    continue
                cur.execute(f'SELECT count(*) FROM "{schema}".{table}')
                n = int(cur.fetchone()[0])
                ok = n >= min_expected
                if not ok and fatal:
                    failures += 1
                icon = "✓" if ok else ("⚠" if not fatal else "❌")
                print(f"  {icon} {label:<45} {n:>10,} rows (min {min_expected:,})")

            # Specific health spot-checks
            print()
            cur.execute(
                f"SELECT count(*) FROM \"{schema}\".memories "
                f"WHERE label LIKE 'session:%' AND label NOT LIKE '%::%'"
            )
            n_sessions = int(cur.fetchone()[0])
            cur.execute(
                f"SELECT count(*) FROM \"{schema}\".memories "
                f"WHERE label LIKE '%::afe::%'"
            )
            n_afe = int(cur.fetchone()[0])
            cur.execute(
                f"SELECT count(*) FROM \"{schema}\".memories "
                f"WHERE label LIKE '%::chunk::%'"
            )
            n_chunks = int(cur.fetchone()[0])
            cur.execute(
                f"SELECT count(*) FROM \"{schema}\".memories "
                f"WHERE colbert_tokens IS NOT NULL"
            )
            n_colbert = int(cur.fetchone()[0])
            cur.execute(
                f"SELECT count(*) FROM \"{schema}\".memories "
                f"WHERE colbert_tokens IS NULL AND length(content) >= 50"
            )
            n_colbert_missing = int(cur.fetchone()[0])

            print(f"  ▸ session rows:        {n_sessions:,}")
            print(f"  ▸ AFE fact rows:       {n_afe:,}")
            print(f"  ▸ chunk rows:          {n_chunks:,}")
            print(f"  ▸ ColBERT-baked rows:  {n_colbert:,}")
            print(f"  ▸ ColBERT MISSING (≥50 chars): {n_colbert_missing:,}")
            if n_sessions < 100:
                failures += 1
                print("  ❌ session count below 100 — bake_longmemeval failed")
            if n_afe < 100:
                failures += 1
                print("  ❌ AFE count below 100 — bake_afe_facts failed")
            if n_colbert_missing > 10:
                failures += 1
                print("  ❌ too many memories lack ColBERT tokens")

            # Insight breakdown
            cur.execute(
                f"SELECT insight_type, count(*) FROM \"{schema}\".dream_insights "
                f"GROUP BY insight_type"
            )
            for itype, n in cur.fetchall():
                print(f"  ▸ insight[{itype}]: {n:,}")

    print()
    if failures:
        print(f"  ❌ VERIFY FAILED: {failures} fatal issue(s)")
        return 1
    print(f"  ✅ VERIFY GREEN: schema is bench-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
