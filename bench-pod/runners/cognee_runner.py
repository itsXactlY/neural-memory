"""Cognee runner — topoteretes/cognee 1.x via fully-offline ollama+lancedb.

Cognee's defining operation is `cognify()`: an LLM-driven entity/relation
extraction pass that builds a knowledge graph over ingested text. Vector
retrieval then walks that graph at query time. The graph-construction
step is order-minutes per session, where Mazemaker's mechanical embed
ingest is order-milliseconds. The bench reports both wall-clock surfaces
separately so the tradeoff stays legible.

Methodology mirrors /destruction/cognee/ exactly:
  - LongMemEval-S 500q, session granularity (one cognee `add()` per
    haystack session), top-k=10, substring_match-style judge (rank-of-
    gold by session id).
  - LLM provider forced to local ollama (qwen2.5:3b) and OPENAI_API_KEY
    blanked so any accidental network egress fails closed.
  - Vector store: LanceDB local (cognee default). Embedder: cognee's
    fastembed bridge with BAAI/bge-small-en-v1.5 (384-d) — BGE-M3 (1024-d)
    is not on fastembed's supported list inside cognee 1.0.9, and the
    bench is about cognee's graph contribution, not the embedder.

Isolation: cognee's data is process-global on disk (~/.cognee + the
package-local .cognee_system/). Each question runs `prune.prune_data()`
+ `prune.prune_system()` before ingest so prior questions never leak
into the graph.

CLI:
    python -m runners.cognee_runner \
        --dataset /path/to/longmemeval_s.json \
        --k 10 \
        --output /path/to/result.json \
        [--limit N] [--skip-cognify]

`--skip-cognify` is a debug flag: ingest only, then attempt retrieval
without the LLM-graph step. In cognee 1.0.9 this is a no-op for
recall — the DocumentChunk_text vector collection is populated INSIDE
cognify(), so a `search()` call without it raises NoDataError. The
flag is kept anyway because future cognee versions may decouple raw
chunk indexing from the cognify pipeline; today it serves as a
pipeline-integrity check (verifies prune/add/search wiring isolated
from the slow LLM path).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from runners.common import (
    DatasetMeta,
    Metrics,
    POD_VERSION,
    ResultRecord,
    file_sha256,
    host_info,
    now_iso,
    percentile,
    write_result,
)

SYSTEM = "cognee"

# Local-only defaults — overridable via env, but we ship hardened values
# so a missed env var doesn't silently fall through to OpenAI.
OLLAMA_ENDPOINT = os.environ.get("COGNEE_OLLAMA_ENDPOINT", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("COGNEE_OLLAMA_MODEL", "gemma3:12b")
EMBED_PROVIDER = "fastembed"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
VECTOR_DB_PROVIDER = "lancedb"


# ---------------------------------------------------------------------------
# Cognee offline configuration
# ---------------------------------------------------------------------------

def _force_offline_env() -> None:
    """Hard-blank cloud credentials so any accidental network call fails."""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    # Cognee fires a 30s ping at the LLM endpoint on first add(). Local
    # ollama can be loading the model into VRAM at that exact moment and
    # blow the test budget; we already validated ollama is reachable
    # upstream of this runner, so skip the redundant ping.
    os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")


def _configure_cognee():
    import cognee
    cognee.config.set_llm_provider("ollama")
    cognee.config.set_llm_model(OLLAMA_MODEL)
    cognee.config.set_llm_endpoint(OLLAMA_ENDPOINT)
    cognee.config.set_llm_api_key("ollama")  # sentinel; ollama ignores it
    # Ollama default context is 4096 tokens; cognee's extraction prompts
    # routinely overflow it on session-sized chunks. Push to 16K and
    # cap the response so a runaway model can't waste minutes.
    cognee.config.set_llm_config({
        "llm_args": {"num_ctx": int(os.environ.get("COGNEE_OLLAMA_NUM_CTX", "16384"))},
        "llm_max_completion_tokens": int(os.environ.get("COGNEE_OLLAMA_MAX_TOKENS", "2048")),
        "llm_temperature": 0.0,
    })
    cognee.config.set_embedding_provider(EMBED_PROVIDER)
    cognee.config.set_embedding_model(EMBED_MODEL)
    cognee.config.set_embedding_dimensions(EMBED_DIM)
    cognee.config.set_vector_db_provider(VECTOR_DB_PROVIDER)
    return cognee


def _cognee_config_snapshot() -> dict[str, Any]:
    return {
        "llm_provider": "ollama",
        "llm_model": OLLAMA_MODEL,
        "llm_endpoint": OLLAMA_ENDPOINT,
        "embedding_provider": EMBED_PROVIDER,
        "embedding_model": EMBED_MODEL,
        "embedding_dimensions": EMBED_DIM,
        "vector_db_provider": VECTOR_DB_PROVIDER,
        "isolation": "prune.prune_data + prune.prune_system between questions",
        "openai_api_key_blanked": True,
    }


# ---------------------------------------------------------------------------
# Per-question ingest + retrieval
# ---------------------------------------------------------------------------

def _format_session(sess: list[dict[str, Any]], date: str | None) -> str:
    lines: list[str] = []
    if date:
        lines.append(f"[{date}]")
    for turn in sess:
        lines.append(f"{turn.get('role','?')}: {turn.get('content','')}")
    return "\n".join(lines)


async def _reset_state(cognee) -> None:
    # prune_data wipes ingested data; prune_system wipes graph + vector
    # indexes so the next question starts from a clean slate.
    try:
        await cognee.prune.prune_data()
    except Exception:
        pass
    try:
        await cognee.prune.prune_system(graph=True, vector=True, metadata=True)
    except Exception:
        pass


async def _run_question(cognee, record: dict[str, Any], k: int, skip_cognify: bool) -> dict[str, Any]:
    from cognee.modules.search.types import SearchType

    qid = record["question_id"]
    question = record["question"]
    gold_ids: set[str] = set(record.get("answer_session_ids") or [])
    is_abs = qid.endswith("_abs")

    await _reset_state(cognee)

    # Ingest — tag every session with a node_set carrying its session id
    # so the retrieved chunks can be back-mapped to gold session ids.
    sids = record.get("haystack_session_ids", [])
    sessions = record.get("haystack_sessions", [])
    dates = record.get("haystack_dates", []) or [None] * len(sessions)

    t0 = time.perf_counter()
    n_ingested = 0
    for sid, sess, date in zip(sids, sessions, dates):
        text = _format_session(sess, date)
        if not text.strip():
            continue
        # Prefix the text with an explicit marker so chunk text still
        # carries the session id even if node_set propagation differs
        # across cognee versions.
        await cognee.add(f"SESSION_ID: {sid}\n{text}", node_set=[f"session:{sid}"])
        n_ingested += 1
    ingest_s = time.perf_counter() - t0

    cognify_s = 0.0
    if not skip_cognify:
        t1 = time.perf_counter()
        # chunks_per_batch=1 — cognee's default fans out all chunks
        # concurrently against the LLM and against its own SQLite
        # metadata store. With ollama serialising requests anyway, the
        # only effect of the fanout is "database is locked" retries on
        # the SQLAlchemy metadata writes. Serial mode is honest about
        # the underlying throughput and cleaner to time.
        await cognee.cognify(chunks_per_batch=1, data_per_batch=1)
        cognify_s = time.perf_counter() - t1

    # Retrieval — CHUNKS surfaces the raw chunk text (so we can grep the
    # SESSION_ID marker) while GRAPH_COMPLETION returns an LLM answer
    # that hides the rank signal we need. CHUNKS is the right shape for
    # a session-granularity recall@k judge.
    t2 = time.perf_counter()
    search_type = SearchType.CHUNKS if skip_cognify else SearchType.CHUNKS
    results = await cognee.search(query_text=question, query_type=search_type, top_k=k)
    query_s = time.perf_counter() - t2

    # Rank-of-gold by session id, in the order cognee returned chunks.
    rank: int | None = None
    seen_sids: set[str] = set()
    surfaced: list[str] = []
    for i, r in enumerate(results, start=1):
        sid = _extract_sid(r)
        if sid is None:
            continue
        if sid not in seen_sids:
            seen_sids.add(sid)
            surfaced.append(sid)
        if rank is None and sid in gold_ids:
            rank = i

    return {
        "qid": qid,
        "question_type": record.get("question_type"),
        "is_abstention": is_abs,
        "n_ingested": n_ingested,
        "n_results": len(results),
        "gold_session_ids": sorted(gold_ids),
        "rank_of_gold": rank,
        "surfaced_sids": surfaced[:k],
        "ingest_s": round(ingest_s, 3),
        "cognify_s": round(cognify_s, 3),
        "query_s": round(query_s, 3),
    }


def _extract_sid(result: Any) -> str | None:
    """Pull session id out of a cognee SearchResult.

    Cognee's result schema has drifted between versions; rather than
    hard-couple, we search the stringified payload for the SESSION_ID
    marker we injected at ingest time. Hardfact: deterministic, no
    dependency on internal field names.
    """
    text = ""
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        text = json.dumps(result, default=str)
    else:
        text = getattr(result, "text", None) or getattr(result, "content", None) or str(result)
    marker = "SESSION_ID:"
    idx = text.find(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker):].lstrip()
    # session id ends at the first whitespace or newline
    end = 0
    while end < len(tail) and not tail[end].isspace():
        end += 1
    sid = tail[:end].strip()
    return sid or None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def _drive(args) -> tuple[list[dict[str, Any]], Metrics]:
    _force_offline_env()
    cognee = _configure_cognee()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    records = json.loads(dataset_path.read_text())
    if args.limit and args.limit < len(records):
        records = records[: args.limit]

    per_question: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not args.quiet:
            print(f"[cognee] {i+1}/{len(records)} qid={rec.get('question_id')}", flush=True)
        try:
            row = await _run_question(cognee, rec, args.k, args.skip_cognify)
        except Exception as e:
            row = {
                "qid": rec.get("question_id"),
                "question_type": rec.get("question_type"),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "is_abstention": (rec.get("question_id", "").endswith("_abs")),
                "gold_session_ids": sorted(set(rec.get("answer_session_ids") or [])),
            }
        per_question.append(row)
        if not args.quiet:
            print(
                f"[cognee]   ingest={row.get('ingest_s')}s cognify={row.get('cognify_s')}s "
                f"query={row.get('query_s')}s rank={row.get('rank_of_gold')}",
                flush=True,
            )

    metrics = _aggregate(per_question)
    return per_question, metrics


def _aggregate(per_question: list[dict[str, Any]]) -> Metrics:
    gradeable = [q for q in per_question
                 if q.get("gold_session_ids") and not q.get("is_abstention") and "error" not in q]
    n = max(1, len(gradeable))

    def _hits(thresh: int) -> int:
        return sum(1 for q in gradeable
                   if q.get("rank_of_gold") is not None and q["rank_of_gold"] <= thresh)

    rrs = [1.0 / q["rank_of_gold"] if q.get("rank_of_gold") else 0.0 for q in gradeable]
    query_ms = [q["query_s"] * 1000.0 for q in per_question if "query_s" in q]
    ingest_s = [q["ingest_s"] + q.get("cognify_s", 0.0) for q in per_question if "ingest_s" in q]
    query_s = [q["query_s"] for q in per_question if "query_s" in q]

    errors = [i for i, q in enumerate(per_question) if "error" in q]

    return Metrics(
        r_at_1=round(_hits(1) / n, 4) if gradeable else None,
        r_at_5=round(_hits(5) / n, 4) if gradeable else None,
        r_at_10=round(_hits(10) / n, 4) if gradeable else None,
        mrr=round(sum(rrs) / len(rrs), 4) if rrs else None,
        p50_recall_ms=round(percentile(query_ms, 0.50), 3) if query_ms else None,
        p95_recall_ms=round(percentile(query_ms, 0.95), 3) if query_ms else None,
        wall_seconds_ingest=round(sum(ingest_s), 2) if ingest_s else None,
        wall_seconds_query=round(sum(query_s), 2) if query_s else None,
        llm_tokens_extraction=None,  # cognee 1.0.9 does not expose a token counter on cognify()
        errors=len(errors),
        failed_questions=errors,
    )


def _cognee_version() -> str:
    try:
        import cognee
        return getattr(cognee, "__version__", None) or cognee.get_cognee_version()
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runners.cognee_runner",
                                description="Cognee runner — KG+vector via local ollama + lancedb")
    p.add_argument("--dataset", required=True, help="Path to longmemeval_s.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--output", required=True, help="Path to write the ResultRecord JSON")
    p.add_argument("--limit", type=int, default=0, help="0 = all questions")
    p.add_argument("--skip-cognify", action="store_true",
                   help="Debug: vector retrieval only, no LLM graph step")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    t0 = time.perf_counter()
    per_question, metrics = asyncio.run(_drive(args))
    wall = time.perf_counter() - t0

    ds_size = len(per_question)
    ds_hash = file_sha256(dataset_path)[:16] if dataset_path.exists() else ""

    rec = ResultRecord(
        timestamp=now_iso(),
        bench_pod_version=POD_VERSION,
        system=SYSTEM,
        system_version=f"cognee=={_cognee_version()}",
        system_config={
            **_cognee_config_snapshot(),
            "k": args.k,
            "limit": args.limit,
            "skip_cognify": bool(args.skip_cognify),
            "granularity": "session",
            "dataset_path": str(dataset_path),
            "wall_seconds_total": round(wall, 2),
        },
        dataset=DatasetMeta(name=dataset_path.stem, size=ds_size, hash=ds_hash),
        metrics=metrics,
        host=host_info(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = rec.to_json()
    payload["per_question"] = per_question
    output_path.write_text(json.dumps(payload, indent=2, default=str))

    if not args.quiet:
        print(f"[cognee] wrote {output_path}", flush=True)
        print(f"[cognee] R@1={metrics.r_at_1} R@5={metrics.r_at_5} R@10={metrics.r_at_10} "
              f"MRR={metrics.mrr}  ingest={metrics.wall_seconds_ingest}s "
              f"query={metrics.wall_seconds_query}s  errors={metrics.errors}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
