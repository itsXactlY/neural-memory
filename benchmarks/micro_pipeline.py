#!/usr/bin/env python3
"""micro_pipeline.py — end-to-end YES/NO functional test on a tiny corpus.

Copies N sessions from the full LongMemEval-S cache into a fresh micro
schema, then runs the WHOLE dream pipeline (AFE → chunks → ColBERT
migrate → dream cycle → verify) against it. Total runtime ~3-5 min for
N=30. If anything emits zero where it should be non-zero, exit non-zero.

Use this BEFORE re-running the 333k corpus to catch silent failures
fast.

    python benchmarks/micro_pipeline.py            # N=30 default
    python benchmarks/micro_pipeline.py --n 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"
SOURCE_SCHEMA = "longmemeval_s_bgem3_1024"
MICRO_SCHEMA = "longmemeval_micro_bgem3_1024"  # named so dream_on_cache --variant micro works


def _dsn() -> str:
    from postgres_store import _build_dsn
    base = _build_dsn()
    p = urllib.parse.urlparse(base)
    return urllib.parse.urlunparse(p._replace(path=f"/{CACHE_DB}"))


def step(name: str):
    print(f"\n══ {name} ══", flush=True)


def fail(msg: str, code: int = 1):
    print(f"\n❌ FAIL: {msg}", flush=True)
    sys.exit(code)


def ok(msg: str):
    print(f"✓ {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30,
                   help="Number of source sessions to copy (default 30)")
    p.add_argument("--keep", action="store_true",
                   help="Keep the micro schema after the run (default: drop on success)")
    args = p.parse_args()

    import psycopg
    t0 = time.perf_counter()

    # ── 1. Drop + create micro schema; copy N sessions from S cache ────
    step(f"Step 1: micro schema copy {args.n} sessions from {SOURCE_SCHEMA}")
    with psycopg.connect(_dsn(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS "{MICRO_SCHEMA}" CASCADE')
        cur.execute(f'CREATE SCHEMA "{MICRO_SCHEMA}"')
        cur.execute(
            f'CREATE TABLE "{MICRO_SCHEMA}".memories ('
            f' id BIGSERIAL PRIMARY KEY,'
            f' label TEXT NOT NULL,'
            f' content TEXT NOT NULL,'
            f' embedding vector(1024) NOT NULL,'
            f' vector_dim INTEGER NOT NULL DEFAULT 1024,'
            f' salience DOUBLE PRECISION DEFAULT 1.0,'
            f' colbert_tokens BYTEA,'
            f' question_id TEXT,'
            f' session_idx INTEGER,'
            f' created_at TIMESTAMPTZ DEFAULT NOW(),'
            f' last_accessed TIMESTAMPTZ DEFAULT NOW(),'
            f' access_count INTEGER DEFAULT 0)'
        )
        cur.execute(
            f'INSERT INTO "{MICRO_SCHEMA}".memories '
            f'(label, content, embedding, vector_dim, salience, colbert_tokens) '
            f'SELECT label, content, embedding, vector_dim, salience, colbert_tokens '
            f'FROM "{SOURCE_SCHEMA}".memories '
            f"WHERE label LIKE 'session:%%' AND label NOT LIKE '%%::%%' "
            f"ORDER BY id LIMIT %s",
            (args.n,),
        )
        cur.execute(f'SELECT count(*) FROM "{MICRO_SCHEMA}".memories')
        n_sessions = int(cur.fetchone()[0])
    if n_sessions != args.n:
        fail(f"expected {args.n} sessions, got {n_sessions}")
    ok(f"{n_sessions} sessions copied")

    env = os.environ.copy()
    env["MM_DB_BACKEND"] = "postgres"
    env["MM_POSTGRES_DB"] = CACHE_DB
    env["MM_POSTGRES_SCHEMA"] = MICRO_SCHEMA

    # ── 2. AFE bake ────────────────────────────────────────────────────
    step("Step 2: AFE bake")
    # bake_afe_facts uses --variant; we patch via direct schema env override
    # Use Mazemaker DreamEngine.afe phase directly through a tiny script
    code = """
import os, sys
sys.path.insert(0, 'python')
os.environ['MM_DB_BACKEND']='postgres'
os.environ['MM_POSTGRES_DB']='mm10m_bench'
os.environ['MM_POSTGRES_SCHEMA']='longmemeval_micro_bgem3_1024'
os.environ['MAZEMAKER_AFE_MIN_LEN']='500'
os.environ['MM_DEFER_HNSW']='1'
from memory_client import Mazemaker
from dream_engine import DreamEngine
from dream_postgres_store import DreamPostgresStore
nm = Mazemaker(db_path='/dev/null', embedding_backend='auto', lazy_graph=True, retrieval_mode='semantic', rerank=False)
nm.store._ensure_embedding_column(1024)
de = DreamEngine(DreamPostgresStore(), neural_memory=nm)
for i in range(5):
    s = de._phase_afe()
    print(f'AFE cycle {i+1}: processed={s.get(\"processed\",0)} written={s.get(\"written\",0)} facts={s.get(\"facts_extracted\",0)}', flush=True)
    if s.get('processed', 0) == 0:
        break
"""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        fail(f"AFE crashed: {r.stderr[-800:]}")

    with psycopg.connect(_dsn(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM \"{MICRO_SCHEMA}\".memories WHERE label LIKE '%%::afe::%%'")
        n_afe = int(cur.fetchone()[0])
    if n_afe == 0:
        fail("AFE produced 0 facts")
    ok(f"AFE: {n_afe} facts")

    # ── 3. Chunked sessions ────────────────────────────────────────────
    step("Step 3: chunked sessions")
    code = """
import os, sys
sys.path.insert(0, 'python')
import importlib.util
spec = importlib.util.spec_from_file_location('b', 'benchmarks/bake_chunked_sessions.py')
m = importlib.util.module_from_spec(spec)
m.CACHE_SCHEMA = 'longmemeval_micro_bgem3_1024'
spec.loader.exec_module(m)
# Drive a single bake by calling main() with patched argv
sys.argv = ['bake_chunked_sessions.py', '--variant', 's']  # variant ignored — we already overrode CACHE_SCHEMA above
# Override the module's CACHE_SCHEMA again after main() reads it
m.main()
"""
    # Simpler: invoke the script directly using --variant trick — but it'll target longmemeval_s_bgem3_1024.
    # Instead use a one-shot env-driven mini-bake.
    code = f"""
import sys, time, os
sys.path.insert(0, 'python')
os.environ['MM_DB_BACKEND']='postgres'
os.environ['MM_POSTGRES_DB']='{CACHE_DB}'
os.environ['MM_POSTGRES_SCHEMA']='{MICRO_SCHEMA}'
import psycopg, urllib.parse
from postgres_store import _build_dsn
from embed_provider import EmbeddingProvider
from colbert_helper import colbert_available, encode_tokens_batch, pack_tokens
p = urllib.parse.urlparse(_build_dsn())
dsn = urllib.parse.urlunparse(p._replace(path='/{CACHE_DB}'))
embedder = EmbeddingProvider(backend='auto')
assert colbert_available()
CHUNK = 2000; OVERLAP = 200
def chunk_text(t):
    if len(t) <= CHUNK: return [t]
    out=[]; s=0
    while s<len(t):
        e=min(s+CHUNK,len(t)); out.append(t[s:e])
        if e==len(t): break
        s=e-OVERLAP
    return out
written=0
with psycopg.connect(dsn, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT question_id, session_idx, label, content FROM \\"{MICRO_SCHEMA}\\".memories WHERE label LIKE 'session:%' AND label NOT LIKE '%::%' AND length(content) > 2000")
        sources = cur.fetchall()
    print(f'long sources: {{len(sources)}}', flush=True)
    rows=[]
    for qid, sidx, lbl, content in sources:
        chunks = chunk_text(content)
        for i, ch in enumerate(chunks):
            rows.append((qid, sidx, f'{{lbl}}::chunk::{{i}}', ch))
    print(f'total chunks: {{len(rows)}}', flush=True)
    if not rows:
        print('no chunks needed'); sys.exit(0)
    texts = [r[3] for r in rows]
    t=time.time()
    embs = embedder.embed_batch(texts)
    print(f'embed: {{time.time()-t:.1f}}s', flush=True)
    t=time.time()
    cb_arrs = encode_tokens_batch(texts, top_k=32, batch_size=32)
    print(f'colbert: {{time.time()-t:.1f}}s', flush=True)
    with conn.cursor() as cur:
        with cur.copy('COPY \\"{MICRO_SCHEMA}\\".memories (question_id, session_idx, label, content, embedding, vector_dim, salience, colbert_tokens) FROM STDIN') as cp:
            for (qid, sidx, lbl, ch), emb, cb in zip(rows, embs, cb_arrs):
                emb_text = '[' + ','.join(repr(float(v)) for v in emb) + ']'
                cb_bytes = pack_tokens(cb) if cb is not None else None
                cp.write_row((qid, sidx, lbl, ch, emb_text, 1024, 1.0, cb_bytes))
                written += 1
print(f'chunks written: {{written}}')
"""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        fail(f"chunks crashed: {r.stderr[-800:]}")

    with psycopg.connect(_dsn(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM \"{MICRO_SCHEMA}\".memories WHERE label LIKE '%%::chunk::%%'")
        n_chunks = int(cur.fetchone()[0])
    ok(f"chunks: {n_chunks}")

    # ── 4. ColBERT migrate ────────────────────────────────────────────
    step("Step 4: ColBERT migrate AFE facts")
    r = subprocess.run([sys.executable, "benchmarks/migrate_colbert.py",
                        "--schema", MICRO_SCHEMA, "--label-like", "%::afe::%",
                        "--batch", "64"],
                       env=env, capture_output=True, text=True, timeout=300)
    print(r.stdout)
    if r.returncode != 0:
        fail(f"colbert-migrate crashed: {r.stderr[-800:]}")

    with psycopg.connect(_dsn(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{MICRO_SCHEMA}".memories WHERE colbert_tokens IS NOT NULL')
        n_cb = int(cur.fetchone()[0])
    ok(f"ColBERT-baked: {n_cb}")

    # ── 5. Dream cycle (1 cycle is enough on micro) ────────────────────
    step("Step 5: dream cycle (1 cycle, max_memories=500, max_isolated=300)")
    code = f"""
import os, sys
sys.path.insert(0, 'python')
os.environ['MM_DB_BACKEND']='postgres'
os.environ['MM_POSTGRES_DB']='{CACHE_DB}'
os.environ['MM_POSTGRES_SCHEMA']='{MICRO_SCHEMA}'
os.environ['MM_INSIGHT_MIN_CLUSTER']='3'
os.environ['MM_REM_SIM_HIGH']='0.97'
os.environ.setdefault('MAZEMAKER_AFE_MIN_LEN', '10000000')  # disable AFE re-run
from memory_client import Mazemaker
from dream_engine import DreamEngine
from dream_postgres_store import DreamPostgresStore
nm = Mazemaker(db_path='/dev/null', embedding_backend='auto', lazy_graph=True, retrieval_mode='semantic', rerank=False)
nm.store._ensure_embedding_column(1024)
print(f'GPU recall engine: {{nm._gpu}}', flush=True)
de = DreamEngine(DreamPostgresStore(), neural_memory=nm, max_memories_per_cycle=500, max_isolated_per_cycle=300)
import time; t=time.time()
s = de._run_dream_cycle()
print(f'cycle: {{time.time()-t:.1f}}s', flush=True)
print(f'NREM strengthened={{s.get(\"nrem\",{{}}).get(\"strengthened\",0)}}', flush=True)
print(f'REM bridges={{s.get(\"rem\",{{}}).get(\"bridges\",0)}}', flush=True)
print(f'INS communities={{s.get(\"insights\",{{}}).get(\"communities\",0)}} insights={{s.get(\"insights\",{{}}).get(\"insights\",0)}}', flush=True)
print(f'DAE written={{s.get(\"dae\",{{}}).get(\"written\",0) if isinstance(s.get(\"dae\"),dict) else 0}}', flush=True)
"""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        fail(f"dream crashed: {r.stderr[-1200:]}")

    # ── 6. Verify ──────────────────────────────────────────────────────
    step("Step 6: verify all aux tables populated")
    failures = []
    with psycopg.connect(_dsn(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{MICRO_SCHEMA}".dream_sessions')
        n_sess = int(cur.fetchone()[0])
        cur.execute(f'SELECT count(*) FROM "{MICRO_SCHEMA}".dream_insights')
        n_ins = int(cur.fetchone()[0])
        cur.execute(f'SELECT count(*) FROM "{MICRO_SCHEMA}".memory_dae_embeddings')
        n_dae = int(cur.fetchone()[0])
        cur.execute(f'SELECT edge_type, count(*) FROM "{MICRO_SCHEMA}".connections GROUP BY 1')
        edges = dict(cur.fetchall())
        cur.execute(f'SELECT count(*) FROM "{MICRO_SCHEMA}".memory_revisions')
        n_rev = int(cur.fetchone()[0])
    print(f"dream_sessions={n_sess} dream_insights={n_ins} memory_dae_embeddings={n_dae} memory_revisions={n_rev}")
    print(f"connections by edge_type: {edges}")

    if n_sess < 1: failures.append("dream_sessions=0")
    # dream_insights >0 is the proof that _phase_insights actually emitted
    if n_ins < 1: failures.append("dream_insights=0 — Insight phase emit broken")
    if n_dae < 1: failures.append("memory_dae_embeddings=0 — DAE phase broken")
    # REM should produce SOMETHING on a micro corpus
    if edges.get("bridge", 0) < 1: failures.append("connections.bridge=0 — REM produced no bridges")

    elapsed = time.perf_counter() - t0
    if failures:
        print(f"\n❌ {len(failures)} FAILURES in {elapsed/60:.1f}m:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(f"\n✅ ALL GREEN in {elapsed/60:.1f}m  (n_sessions={n_sessions} n_afe={n_afe} n_chunks={n_chunks} n_sess={n_sess} n_ins={n_ins} n_dae={n_dae})")
    if not args.keep:
        with psycopg.connect(_dsn(), autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{MICRO_SCHEMA}" CASCADE')
        print("(micro schema dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
