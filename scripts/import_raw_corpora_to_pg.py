#!/usr/bin/env python3
"""One-shot raw bench corpora → mm_bench_raw PG.

Pure data pipeline — no Mazemaker, no embeddings, no engine state.
After this runs, the canonical bench source-of-truth lives in PG and a
single pg_dump captures everything.

Imports:
  * BEAM 10M  — 10 conversations  (~208 K turns; rich plan/batch/group structure)
  * BEAM 1M   — 35 conversations  (~75 K turns;  batch/group only)
  * BEAM 500K — 35 conversations  (~38 K turns;  batch/group only)
  * BEAM 100K — 20 conversations  (~6 K turns;   batch/group only)
  * LongMemEval-S — 500 questions + 247 K session messages
  * Per-conv: topic.json, main_spec, relationships, labels
  * Per-conv: probing_questions.json (10 ability buckets, ~2 q each)
  * Per-corpus: <scale>_topics.json catalog
  * bench_meta.sources — file→sha256+rows provenance

Idempotent: every schema is DROP/CREATE'd.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks/neural_memory_benchmark/mm_10m_eval/corpus"))

import psycopg

from corpus_helpers import load_flat_turns as load_flat_turns_10m


BEAM_ROOT = Path("/tmp/BEAM")

# Canonical LongMemEval source: huggingface xiaowu0162/LongMemEval. Files are
# stored without a `.json` suffix — content is JSON regardless. We use the HF
# snapshot directly (downloaded once into longmemeval_full/) so all three
# variants share the same canonical bytes.
LME_DIR = REPO / "benchmarks/external/data/longmemeval_full"
LME_VARIANTS = {
    "longmemeval_s":      LME_DIR / "longmemeval_s",
    "longmemeval_m":      LME_DIR / "longmemeval_m",
    "longmemeval_oracle": LME_DIR / "longmemeval_oracle",
}

BEAM_SCALES = ["10M", "1M", "500K", "100K"]


def _beam_schema_ddl(scale: str) -> str:
    s = f"beam_{scale.lower()}"
    return f"""
DROP SCHEMA IF EXISTS {s} CASCADE;
CREATE SCHEMA {s};

CREATE TABLE {s}.conversations (
    conv_id        INTEGER PRIMARY KEY,
    topic          JSONB,
    main_spec      TEXT,
    relationships  TEXT,
    labels         TEXT
);

CREATE TABLE {s}.turns (
    conv_id       INTEGER NOT NULL,
    seq           INTEGER NOT NULL,
    plan_id       INTEGER,
    batch_id      INTEGER,
    group_id      INTEGER,
    tidx          INTEGER,
    role          TEXT,
    content       TEXT,
    time_anchor   TEXT,
    question_type TEXT,
    PRIMARY KEY (conv_id, seq)
);
CREATE INDEX {s}_turns_conv ON {s}.turns (conv_id);

CREATE TABLE {s}.topics (payload JSONB NOT NULL);

CREATE TABLE {s}.probing_questions (
    conv_id  INTEGER NOT NULL,
    ability  TEXT    NOT NULL,
    qidx     INTEGER NOT NULL,
    payload  JSONB   NOT NULL,
    PRIMARY KEY (conv_id, ability, qidx)
);
CREATE INDEX {s}_probing_conv ON {s}.probing_questions (conv_id);
"""


def _lme_schema_ddl(schema: str) -> str:
    return f"""
DROP SCHEMA IF EXISTS {schema} CASCADE;
CREATE SCHEMA {schema};

CREATE TABLE {schema}.questions (
    question_id           TEXT PRIMARY KEY,
    question_type         TEXT,
    question              TEXT,
    question_date         TEXT,
    answer                TEXT,
    answer_session_ids    JSONB,
    haystack_dates        JSONB,
    haystack_session_ids  JSONB
);

CREATE TABLE {schema}.sessions (
    question_id  TEXT    NOT NULL,
    session_idx  INTEGER NOT NULL,
    msg_idx      INTEGER NOT NULL,
    role         TEXT,
    content      TEXT,
    PRIMARY KEY (question_id, session_idx, msg_idx)
);
CREATE INDEX {schema}_sessions_qid ON {schema}.sessions (question_id);
"""


_META_DDL = """
DROP SCHEMA IF EXISTS bench_meta CASCADE;
CREATE SCHEMA bench_meta;

CREATE TABLE bench_meta.sources (
    source_name  TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    n_rows       BIGINT NOT NULL,
    imported_at  TIMESTAMPTZ DEFAULT NOW()
);
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(cur, name: str, path: Path, n: int) -> None:
    cur.execute(
        "INSERT INTO bench_meta.sources (source_name, file_path, sha256, n_rows) "
        "VALUES (%s, %s, %s, %s)",
        (name, str(path), _sha256(path), n),
    )


def _read_text_if(p: Path) -> str | None:
    return p.read_text(errors="replace") if p.exists() else None


def _read_json_if(p: Path):
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _parse_beam_smaller(path: Path) -> Iterator[dict]:
    """1M / 500K / 100K: list of batches → groups → turns. plan_id = NULL."""
    d = json.load(path.open())
    seq = 0
    for batch in d:
        batch_id = batch.get("batch_number")
        batch_anchor = batch.get("time_anchor")
        for group_id, group in enumerate(batch.get("turns", [])):
            for tidx, t in enumerate(group):
                if not isinstance(t, dict):
                    continue
                seq += 1
                yield {
                    "seq": seq,
                    "plan_id": None,
                    "batch_id": batch_id,
                    "group_id": group_id,
                    "tidx": tidx,
                    "role": t.get("role"),
                    "content": t.get("content"),
                    "time_anchor": t.get("time_anchor") or batch_anchor,
                    "question_type": t.get("question_type"),
                }


def _parse_beam_10m(conv: int) -> Iterator[dict]:
    """10M: corpus_helpers.load_flat_turns handles the plan-N nested-string format."""
    for t in load_flat_turns_10m(conv):
        yield {
            "seq": int(t["seq"]),
            "plan_id": int(t["plan"]),
            "batch_id": int(t["batch"]),
            "group_id": int(t["group"]),
            "tidx": int(t["tidx"]),
            "role": t["role"],
            "content": t["content"],
            "time_anchor": t.get("time_anchor"),
            "question_type": None,
        }


def import_beam_scale(scale: str, conn: psycopg.Connection) -> None:
    schema = f"beam_{scale.lower()}"
    chats_root = BEAM_ROOT / "chats" / scale
    if not chats_root.exists():
        print(f"  {scale}: NOT FOUND at {chats_root} — skipping", flush=True)
        return

    print(f"\n=== BEAM {scale} ({time.strftime('%H:%M:%S')}) ===", flush=True)

    # Topics-catalog paths in /tmp/BEAM/topics use mixed case across scales
    # (10M/1M uppercase, 100k/500k lowercase). Try both.
    topics_file = None
    for variant in (scale, scale.lower(), scale.upper()):
        candidate = BEAM_ROOT / "topics" / variant
        if candidate.exists():
            for stem in (scale, scale.lower(), scale.upper()):
                hit = candidate / f"{stem}_topics.json"
                if hit.exists():
                    topics_file = hit
                    break
        if topics_file:
            break
    if topics_file:
        with topics_file.open() as f:
            raw = f.read()
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.topics (payload) VALUES (%s::jsonb)", (raw,))
            _record(cur, f"{schema}.topics", topics_file, 1)
        conn.commit()
        print(f"  topics catalog → 1", flush=True)

    conv_dirs = sorted(
        [d for d in chats_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    total_turns = 0
    total_probing = 0

    for conv_dir in conv_dirs:
        conv_id = int(conv_dir.name)
        chat_path = conv_dir / "chat.json"
        if not chat_path.exists():
            continue

        topic_json = _read_json_if(conv_dir / "topic.json")
        main_spec = _read_text_if(conv_dir / "main_spec.txt")
        relationships = _read_text_if(conv_dir / "core_relationships.txt") \
            or _read_text_if(conv_dir / "relationships.txt")
        labels = _read_text_if(conv_dir / "labels.txt")

        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {schema}.conversations "
                "(conv_id, topic, main_spec, relationships, labels) "
                "VALUES (%s, %s::jsonb, %s, %s, %s)",
                (conv_id,
                 json.dumps(topic_json) if topic_json is not None else None,
                 main_spec, relationships, labels),
            )

            n_turns = 0
            parser = _parse_beam_10m(conv_id) if scale == "10M" else _parse_beam_smaller(chat_path)
            with cur.copy(
                f"COPY {schema}.turns "
                "(conv_id, seq, plan_id, batch_id, group_id, tidx, role, content, "
                " time_anchor, question_type) FROM STDIN"
            ) as cp:
                for t in parser:
                    cp.write_row((
                        conv_id, t["seq"], t["plan_id"], t["batch_id"],
                        t["group_id"], t["tidx"], t["role"], t["content"],
                        t["time_anchor"], t["question_type"],
                    ))
                    n_turns += 1
            _record(cur, f"{schema}.conv_{conv_id}", chat_path, n_turns)
            total_turns += n_turns

            probing_file = conv_dir / "probing_questions" / "probing_questions.json"
            if not probing_file.exists():
                probing_file = conv_dir / "probing_questions.json"
            if probing_file.exists():
                pq = _read_json_if(probing_file)
                n_q = 0
                if isinstance(pq, dict):
                    for ability, items in pq.items():
                        if isinstance(items, list):
                            for q_idx, item in enumerate(items):
                                cur.execute(
                                    f"INSERT INTO {schema}.probing_questions "
                                    "(conv_id, ability, qidx, payload) "
                                    "VALUES (%s, %s, %s, %s::jsonb)",
                                    (conv_id, ability, q_idx, json.dumps(item)),
                                )
                                n_q += 1
                        elif isinstance(items, dict):
                            cur.execute(
                                f"INSERT INTO {schema}.probing_questions "
                                "(conv_id, ability, qidx, payload) "
                                "VALUES (%s, %s, %s, %s::jsonb)",
                                (conv_id, ability, 0, json.dumps(items)),
                            )
                            n_q += 1
                if n_q:
                    _record(cur, f"{schema}.probing_{conv_id}", probing_file, n_q)
                total_probing += n_q
        conn.commit()

    print(f"  {len(conv_dirs)} convs, {total_turns} turns, "
          f"{total_probing} probing-questions", flush=True)


def import_lme_variant(schema: str, path: Path, conn: psycopg.Connection) -> None:
    if not path.exists():
        print(f"\n=== {schema} NOT FOUND at {path} — skipping ===", flush=True)
        return
    print(f"\n=== {schema} ({time.strftime('%H:%M:%S')}) ===", flush=True)
    with path.open() as f:
        questions = json.load(f)
    n_q, n_sessions, n_msgs = len(questions), 0, 0

    with conn.cursor() as cur:
        for q in questions:
            cur.execute(
                f"INSERT INTO {schema}.questions "
                "(question_id, question_type, question, question_date, answer, "
                " answer_session_ids, haystack_dates, haystack_session_ids) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)",
                (q["question_id"], q.get("question_type"), q.get("question"),
                 q.get("question_date"), q.get("answer"),
                 json.dumps(q.get("answer_session_ids")) if q.get("answer_session_ids") is not None else None,
                 json.dumps(q.get("haystack_dates")) if q.get("haystack_dates") is not None else None,
                 json.dumps(q.get("haystack_session_ids")) if q.get("haystack_session_ids") is not None else None),
            )

        with cur.copy(
            f"COPY {schema}.sessions (question_id, session_idx, msg_idx, role, content) FROM STDIN"
        ) as cp:
            for q in questions:
                qid = q["question_id"]
                for s_idx, session in enumerate(q.get("haystack_sessions") or []):
                    n_sessions += 1
                    for m_idx, msg in enumerate(session):
                        if not isinstance(msg, dict):
                            continue
                        cp.write_row((qid, s_idx, m_idx, msg.get("role"), msg.get("content")))
                        n_msgs += 1
        _record(cur, f"{schema}.questions", path, n_q)
    conn.commit()
    print(f"  questions → {n_q}, sessions → {n_sessions}, msgs → {n_msgs}", flush=True)


def main() -> int:
    pw = os.environ.get("MM_POSTGRES_PASSWORD", "")
    dsn = f"host=127.0.0.1 port=5432 dbname=mm_bench_raw user=mazemaker password={pw}"
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            for s in BEAM_SCALES:
                cur.execute(_beam_schema_ddl(s))
            for schema in LME_VARIANTS:
                cur.execute(_lme_schema_ddl(schema))
            cur.execute(_META_DDL)
        conn.commit()
        print("Schemas (re)created.", flush=True)

        t0 = time.time()
        for s in BEAM_SCALES:
            import_beam_scale(s, conn)
        for schema, path in LME_VARIANTS.items():
            import_lme_variant(schema, path, conn)
        print(f"\nALL DONE in {(time.time()-t0):.1f}s", flush=True)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
