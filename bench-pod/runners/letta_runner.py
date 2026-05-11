"""Letta (formerly MemGPT) runner for the Comparison Pod.

We measure Letta's *archival memory* retrieval channel directly — that
is, we bypass the tool-call LLM loop and exercise the same code path
that `archival_memory_search` calls under the hood
(`agent_manager.list_passages(query_text=..., embed_query=True)`).
This matches the locked methodology at
https://mazemaker.online/destruction/letta/: we are scoring memory
retrieval, not agentic reasoning, so the LLM round-trip cost
(`llm_tokens_extraction`) is reported as 0 because no LLM is called.

Per-question isolation: each question gets a fresh agent in Letta's
SQLite state DB. We delete the agent at the end of the question so
passages don't leak between questions.

Embedding: monkey-patched to a local BGE-M3 via llama-index +
sentence-transformers, so no OpenAI / memgpt.ai network call is made.
`OPENAI_API_KEY` is forced to empty per the methodology.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from runners.common import (
    DatasetMeta,
    Metrics,
    POD_VERSION,
    ResultRecord,
    base_argparser,
    host_info,
    load_dataset,
    now_iso,
    percentile,
    pod_root,
    write_result,
)

SYSTEM = "letta"

# Forbid network calls from the embedding stack and from Letta's
# OpenAI default. Must be set BEFORE importing letta/llama_index.
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("OPENAI_BASE_URL", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Local-embedding monkey patch
# ---------------------------------------------------------------------------

_LOCAL_EMBEDDER = None


def _install_local_embedder(model_name: str, device: str, batch_size: int) -> None:
    """Replace Letta's embedding_model factory with a local
    sentence-transformers (BGE-M3 by default) wrapped in the
    llama-index interface that Letta expects."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    def _factory(config, user_id=None):
        global _LOCAL_EMBEDDER
        if _LOCAL_EMBEDDER is None:
            _LOCAL_EMBEDDER = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                embed_batch_size=batch_size,
            )
        return _LOCAL_EMBEDDER

    import letta.embeddings
    import letta.services.agent_manager
    import letta.services.passage_manager
    letta.embeddings.embedding_model = _factory
    letta.services.agent_manager.embedding_model = _factory
    letta.services.passage_manager.embedding_model = _factory


# ---------------------------------------------------------------------------
# Ingest + scoring helpers — copied verbatim from longmemeval_s.py so the
# substring-match judge is bit-identical to the Mazemaker reference run.
# ---------------------------------------------------------------------------

def _format_session(sess: list[dict[str, Any]], date: Optional[str]) -> str:
    lines: list[str] = []
    if date:
        lines.append(f"[{date}]")
    for turn in sess:
        lines.append(f"{turn.get('role', '?')}: {turn.get('content', '')}")
    return "\n".join(lines)


def _rank_of_gold_by_sid(result_sids: list[Optional[str]],
                         gold_session_ids: set[str]) -> Optional[int]:
    seen: set[str] = set()
    for i, sid in enumerate(result_sids, start=1):
        if sid is None:
            continue
        if sid in gold_session_ids and sid not in seen:
            return i
        seen.add(sid)
    return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _build_emb_cfg(model_name: str, dim: int):
    from letta.schemas.embedding_config import EmbeddingConfig
    return EmbeddingConfig(
        embedding_endpoint_type="hugging-face",
        embedding_endpoint="local",
        embedding_model=model_name,
        embedding_dim=dim,
        embedding_chunk_size=300,
    )


def _build_llm_cfg():
    from letta.schemas.llm_config import LLMConfig
    # The LLM is never invoked — we only touch archival retrieval —
    # but Letta requires a config to construct the agent.
    return LLMConfig(
        model="not-used",
        model_endpoint_type="openai",
        model_endpoint="http://localhost",
        context_window=8192,
    )


def _run_question(client, emb_cfg, llm_cfg, ChatMemory, record: dict[str, Any],
                  qidx: int, k: int) -> dict[str, Any]:
    qid = record["question_id"]
    question = record["question"]
    gold_ids = set(record.get("answer_session_ids") or [])
    is_abstention = qid.endswith("_abs")

    sids = record.get("haystack_session_ids", [])
    sessions = record.get("haystack_sessions", [])
    dates = record.get("haystack_dates", []) or [None] * len(sessions)

    agent = client.create_agent(
        name=f"lme_{qidx}_{qid[:24].replace('-', '')}",
        embedding_config=emb_cfg,
        llm_config=llm_cfg,
        memory=ChatMemory(human="bench", persona="bench"),
        include_base_tools=False,
    )
    # sid_by_text maps passage text -> session id, since Letta's passage
    # API does not surface our custom label. Letta also CHUNKS each
    # insertion, so we keep a (chunk-prefix -> sid) lookup by checking
    # whether a returned passage's text was contained in the session
    # text we inserted.
    sid_index: list[tuple[str, str]] = []  # (session_text, sid)

    try:
        t0 = time.perf_counter()
        n_ingested = 0
        for sid, sess, date in zip(sids, sessions, dates):
            text = _format_session(sess, date)
            if not text.strip():
                continue
            client.insert_archival_memory(agent.id, text)
            sid_index.append((text, sid))
            n_ingested += 1
        ingest_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        passages = client.server.agent_manager.list_passages(
            actor=client.user,
            agent_id=agent.id,
            query_text=question,
            embed_query=True,
            limit=k,
            embedding_config=emb_cfg,
        )
        recall_ms = (time.perf_counter() - t1) * 1000.0

        # Map each returned chunk back to its source session id.
        result_sids: list[Optional[str]] = []
        for p in passages:
            ptxt = p.text or ""
            matched: Optional[str] = None
            for sess_text, sid in sid_index:
                if ptxt and (ptxt in sess_text):
                    matched = sid
                    break
            result_sids.append(matched)

        rank = _rank_of_gold_by_sid(result_sids, gold_ids) if gold_ids else None
        return {
            "qid": qid,
            "question_type": record.get("question_type"),
            "is_abstention": is_abstention,
            "n_ingested": n_ingested,
            "n_results": len(passages),
            "gold_session_ids": sorted(gold_ids),
            "rank_of_gold": rank,
            "ingest_ms": round(ingest_ms, 2),
            "latency_ms": round(recall_ms, 2),
            "result_sids": result_sids,
        }
    finally:
        try:
            client.delete_agent(agent.id)
        except Exception:
            pass


def run(args) -> int:
    work = Path(args.work).expanduser().resolve()

    raw_records, dataset_meta = load_dataset(work, pod_root(), args.dataset)
    if args.limit:
        raw_records = raw_records[: args.limit]

    if not args.quiet:
        print(f"[letta] dataset={dataset_meta.name} size={dataset_meta.size} "
              f"running on {len(raw_records)} questions, k={args.k}", flush=True)
        print(f"[letta] embedding_model={args.embedding_model} (offline)", flush=True)

    _install_local_embedder(args.embedding_model, args.device, args.embed_batch_size)
    from letta import create_client, ChatMemory  # type: ignore
    import letta as _letta_pkg
    letta_version = getattr(_letta_pkg, "__version__", "unknown")

    client = create_client()
    emb_cfg = _build_emb_cfg(args.embedding_model, args.embedding_dim)
    llm_cfg = _build_llm_cfg()

    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []
    ingest_times: list[float] = []
    failed: list[int] = []

    t_run = time.perf_counter()
    for i, rec in enumerate(raw_records):
        try:
            row = _run_question(client, emb_cfg, llm_cfg, ChatMemory, rec, i, args.k)
            latencies.append(row["latency_ms"])
            ingest_times.append(row["ingest_ms"])
        except Exception as e:
            row = {
                "qid": rec.get("question_id"),
                "question_type": rec.get("question_type"),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "is_abstention": (rec.get("question_id", "").endswith("_abs")),
                "gold_session_ids": sorted(set(rec.get("answer_session_ids") or [])),
            }
            failed.append(i)
        per_question.append(row)
        if not args.quiet:
            print(f"[letta] {i+1}/{len(raw_records)}  rank={row.get('rank_of_gold')} "
                  f"ingest_ms={row.get('ingest_ms')} latency_ms={row.get('latency_ms')}",
                  flush=True)
    wall = time.perf_counter() - t_run

    # Aggregate (same logic as longmemeval_s.aggregate but restricted to
    # gradeable rows — those with gold sessions and not abstention).
    gradeable = [q for q in per_question
                 if q.get("gold_session_ids") and not q.get("is_abstention")
                 and "error" not in q]
    n = max(1, len(gradeable))

    def _hits(thresh: int) -> int:
        return sum(1 for q in gradeable
                   if q.get("rank_of_gold") is not None and q["rank_of_gold"] <= thresh)

    rrs = [(1.0 / q["rank_of_gold"]) if q.get("rank_of_gold") else 0.0
           for q in gradeable]
    mrr = sum(rrs) / len(rrs) if rrs else 0.0

    metrics = Metrics(
        r_at_1=round(_hits(1) / n, 4),
        r_at_5=round(_hits(5) / n, 4),
        r_at_10=round(_hits(10) / n, 4),
        mrr=round(mrr, 4),
        p50_recall_ms=round(percentile(latencies, 0.50), 3),
        p95_recall_ms=round(percentile(latencies, 0.95), 3),
        wall_seconds_ingest=round(sum(ingest_times) / 1000.0, 2) if ingest_times else None,
        wall_seconds_query=round(wall, 2),
        llm_tokens_extraction=0,  # archival path does no extraction LLM call
        errors=len(failed),
        failed_questions=failed,
    )

    rec = ResultRecord(
        timestamp=now_iso(),
        bench_pod_version=POD_VERSION,
        system=SYSTEM,
        system_version=f"pip:{letta_version}",
        system_config={
            "k": args.k,
            "embedding_model": args.embedding_model,
            "embedding_dim": args.embedding_dim,
            "embedding_chunk_size": 300,
            "embedding_device": args.device,
            "embed_batch_size": args.embed_batch_size,
            "granularity": "session",
            "retrieval_path": "agent_manager.list_passages(embed_query=True)",
            "bypassed_llm_tool_loop": True,
            "limit": args.limit,
        },
        dataset=dataset_meta,
        metrics=metrics,
        host=host_info(),
    )
    out_path = write_result(work, SYSTEM, rec)
    if not args.quiet:
        print(f"[letta] wrote {out_path}", flush=True)
        print(f"[letta] R@1={metrics.r_at_1} R@5={metrics.r_at_5} "
              f"R@10={metrics.r_at_10} MRR={metrics.mrr} "
              f"p50={metrics.p50_recall_ms}ms p95={metrics.p95_recall_ms}ms "
              f"errors={metrics.errors}", flush=True)

    # Also write per-question rows next to the result file so debugging
    # is possible without re-running.
    pq_path = out_path.parent / f"{SYSTEM}.per_question.json"
    pq_path.write_text(json.dumps(per_question, indent=2, default=str))
    return 0


def build_parser():
    p = base_argparser(SYSTEM, "Letta runner — measures Letta's archival memory retrieval channel.")
    p.add_argument("-k", "--k", type=int, default=10)
    p.add_argument("--embedding-model", default="BAAI/bge-m3",
                   help="HuggingFace model name for local embedding (default: BGE-M3).")
    p.add_argument("--embedding-dim", type=int, default=1024)
    p.add_argument("--device", default="cpu",
                   help="Device for the local embedder: cpu | cuda (default: cpu — "
                        "set cuda to GPU-accelerate ingest if VRAM is free).")
    p.add_argument("--embed-batch-size", type=int, default=32,
                   help="Embedding batch size for the local embedder (default: 32).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
