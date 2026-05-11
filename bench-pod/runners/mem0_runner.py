"""Mem0 runner for the Comparison Pod (LongMemEval-S, offline).

Mem0's defining property is its LLM-driven extraction pass: each
session goes through a local ollama model that distils "memories",
which are then embedded and indexed. Mazemaker has no such extraction;
this runner is the apples-to-apples comparison.

Methodology — locked, mirrors /destruction/mem0/:
  - Granularity: session-level. Each haystack_session is one
    m.add(messages=[...], user_id=qid, metadata={"session_id": sid})
    so extracted memories carry the gold-id key the judge needs.
  - Isolation: fresh qdrant sub-dir + history_db per question.
  - Retrieval: m.search(query, filters={"user_id": qid}, top_k=k).
  - Judge: identical to mazemaker_runner — first surfaced memory whose
    metadata.session_id is in answer_session_ids is the hit.
  - Offline: OPENAI_API_KEY="" forced; LLM via ollama, embedder local
    HuggingFace BAAI/bge-m3 (1024-d), vector store embedded qdrant.

Hardfact: when extraction returns no memories the question records
rank_of_gold=None (0 contribution to R@k). We do not paper over
empty extractions. llm_tokens_extraction is None because Mem0's
add() response does not expose ollama prompt/completion counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from runners.common import (
    DatasetMeta,
    Metrics,
    POD_VERSION,
    ResultRecord,
    host_info,
    now_iso,
)

SYSTEM = "mem0"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIMS = 1024


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank_of_gold(results: list[dict[str, Any]], gold: set[str]) -> Optional[int]:
    """First-rank hit whose metadata.session_id is in the gold set."""
    seen: set[str] = set()
    for i, r in enumerate(results, start=1):
        sid = (r.get("metadata") or {}).get("session_id")
        if not sid:
            continue
        if sid in gold and sid not in seen:
            return i
        seen.add(sid)
    return None


def _check_ollama(url: str, model: str) -> tuple[bool, str]:
    """Return (ok, message). We do not pull — operator-side concern."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return False, f"ollama not reachable at {url}: {e}"
    names = {m.get("name", "") for m in tags.get("models", [])}
    bases = {n.split(":")[0] for n in names}
    if model in names or model.split(":")[0] in bases:
        return True, "ok"
    return False, f"model '{model}' not loaded in ollama (have: {sorted(names)[:8]}...)"


def _build_memory(qdrant_path: str, history_db_path: str, ollama_url: str,
                  ollama_model: str, embed_device: str):
    from mem0 import Memory
    cfg = {
        "llm": {"provider": "ollama",
                "config": {"model": ollama_model, "ollama_base_url": ollama_url}},
        "embedder": {"provider": "huggingface",
                     "config": {"model": DEFAULT_EMBED_MODEL,
                                "embedding_dims": EMBED_DIMS,
                                # Pin device explicitly: an unspecified device picks CUDA
                                # when available, which contends with ollama on shared GPUs.
                                "model_kwargs": {"device": embed_device}}},
        "vector_store": {"provider": "qdrant",
                         "config": {"path": qdrant_path,
                                    "collection_name": "lme_s",
                                    "embedding_model_dims": EMBED_DIMS}},
        "history_db_path": history_db_path,
    }
    return Memory.from_config(cfg), cfg


def _session_messages(sess: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project a session into the OpenAI-style messages list mem0 expects."""
    out: list[dict[str, str]] = []
    for turn in sess:
        role = turn.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = (turn.get("content") or "").strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def _run_question(m, record: dict[str, Any], k: int) -> dict[str, Any]:
    qid = record["question_id"]
    question = record["question"]
    gold = set(record.get("answer_session_ids") or [])
    is_abs = qid.endswith("_abs")

    sids = record.get("haystack_session_ids", [])
    sessions = record.get("haystack_sessions", [])
    dates = record.get("haystack_dates", []) or [None] * len(sessions)

    n_ingested = 0
    t0 = time.perf_counter()
    for sid, sess, date in zip(sids, sessions, dates):
        msgs = _session_messages(sess)
        if not msgs:
            continue
        try:
            m.add(messages=msgs, user_id=qid,
                  metadata={"session_id": sid, "date": date or ""})
            n_ingested += 1
        except Exception as e:
            # Per-session ingest failure is logged but does not abort
            # the question — gold may still surface from another session.
            print(f"[mem0] WARN: add() failed for qid={qid} sid={sid}: {e}",
                  file=sys.stderr, flush=True)
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    try:
        raw = m.search(question, filters={"user_id": qid}, top_k=k)
    except Exception as e:
        return {"qid": qid, "question_type": record.get("question_type"),
                "is_abstention": is_abs,
                "error": f"search failed: {type(e).__name__}: {e}",
                "gold_session_ids": sorted(gold),
                "ingest_ms": round(ingest_ms, 2)}
    recall_ms = (time.perf_counter() - t1) * 1000.0

    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
    rank = _rank_of_gold(results, gold) if gold else None
    return {
        "qid": qid,
        "question_type": record.get("question_type"),
        "is_abstention": is_abs,
        "n_ingested": n_ingested,
        "n_results": len(results),
        "gold_session_ids": sorted(gold),
        "rank_of_gold": rank,
        "ingest_ms": round(ingest_ms, 2),
        "latency_ms": round(recall_ms, 2),
        "top_memories": [{"memory": (r.get("memory") or "")[:240],
                          "session_id": (r.get("metadata") or {}).get("session_id"),
                          "score": r.get("score")} for r in results[:k]],
    }


def _aggregate(per_question: list[dict[str, Any]]) -> Metrics:
    """Same recipe as longmemeval_s.aggregate(): only gradeable questions
    contribute to R@k. Abstentions and errored questions are excluded from
    the denominator but errors are surfaced in metrics.errors."""
    gradeable = [q for q in per_question
                 if q.get("gold_session_ids") and not q.get("is_abstention")
                 and "error" not in q]
    n = max(1, len(gradeable))

    def hits(thresh: int) -> int:
        return sum(1 for q in gradeable
                   if q.get("rank_of_gold") is not None
                   and q["rank_of_gold"] <= thresh)

    rrs = [1.0 / q["rank_of_gold"] if q.get("rank_of_gold") else 0.0
           for q in gradeable]
    latencies = [q["latency_ms"] for q in per_question if "latency_ms" in q]
    ingests = [q["ingest_ms"] for q in per_question if "ingest_ms" in q]
    errs = [q for q in per_question if "error" in q]

    return Metrics(
        r_at_1=round(hits(1) / n, 4),
        r_at_5=round(hits(5) / n, 4),
        r_at_10=round(hits(10) / n, 4),
        mrr=round(statistics.mean(rrs) if rrs else 0.0, 4),
        p50_recall_ms=round(percentile(latencies, 0.50), 3),
        p95_recall_ms=round(percentile(latencies, 0.95), 3),
        wall_seconds_ingest=round(sum(ingests) / 1000.0, 2) if ingests else None,
        wall_seconds_query=round(sum(latencies) / 1000.0, 2) if latencies else None,
        llm_tokens_extraction=None,   # mem0 add() does not expose ollama counts
        errors=len(errs),
        failed_questions=[i for i, q in enumerate(per_question) if "error" in q],
    )


def _mem0_version() -> str:
    try:
        from importlib.metadata import version
        return f"mem0ai=={version('mem0ai')}"
    except Exception:
        return "mem0ai==unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runners.mem0_runner",
        description="Mem0 runner — LongMemEval-S, offline (ollama + BGE-M3 + qdrant).",
    )
    p.add_argument("--dataset", required=True, help="Path to longmemeval_s.json.")
    p.add_argument("--k", type=int, default=10, help="Retrieval top-k (default 10).")
    p.add_argument("--output", required=True,
                   help="Output JSON path for the canonical ResultRecord.")
    p.add_argument("--limit", type=int, default=0, help="Cap on questions (0 = all).")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--embed-device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for BGE-M3. CPU is portable; CUDA only with >2 GiB free.")
    p.add_argument("--per-question-out", default=None,
                   help="Optional path to write per-question rows (debug).")
    p.add_argument("--quiet", action="store_true")
    return p


def _write_error_record(out_path: Path, dataset_path: Path, args, msg: str) -> None:
    rec = ResultRecord(
        timestamp=now_iso(), bench_pod_version=POD_VERSION, system=SYSTEM,
        system_version=_mem0_version(),
        system_config={"ollama_url": args.ollama_url,
                       "ollama_model": args.ollama_model,
                       "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
                       "error": msg},
        dataset=DatasetMeta(name="longmemeval_s", size=0,
                            hash=_file_sha256(dataset_path)[:16]),
        metrics=Metrics(errors=1), host=host_info(),
    )
    out_path.write_text(json.dumps(rec.to_json(), indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"[mem0] dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    # CRITICAL — force offline mode before any mem0 import touches OpenAI.
    os.environ["OPENAI_API_KEY"] = ""
    os.environ.setdefault("MEM0_TELEMETRY", "False")

    ok, msg = _check_ollama(args.ollama_url, args.ollama_model)
    if not ok:
        print(f"[mem0] {msg}", file=sys.stderr)
        _write_error_record(out_path, dataset_path, args, msg)
        return 3

    records = json.loads(dataset_path.read_text())
    selected = records[: args.limit] if args.limit else records
    if not args.quiet:
        print(f"[mem0] dataset={dataset_path.name} total={len(records)} "
              f"selected={len(selected)} k={args.k}", flush=True)
        print(f"[mem0] ollama={args.ollama_url} model={args.ollama_model} "
              f"embedder=hf:{DEFAULT_EMBED_MODEL} ({EMBED_DIMS}d) "
              f"device={args.embed_device}", flush=True)

    captured_cfg: dict[str, Any] = {}
    per_question: list[dict[str, Any]] = []
    t_run = time.perf_counter()

    for i, rec in enumerate(selected):
        qid = rec.get("question_id", f"q{i}")
        td = tempfile.mkdtemp(prefix=f"mem0-{qid[:10]}-")
        try:
            m, cfg = _build_memory(os.path.join(td, "qdrant"),
                                   os.path.join(td, "history.db"),
                                   args.ollama_url, args.ollama_model,
                                   args.embed_device)
            if not captured_cfg:
                # Per-question paths redacted so the record is comparable
                # across runs.
                captured_cfg = {
                    "llm": cfg["llm"], "embedder": cfg["embedder"],
                    "vector_store": {**cfg["vector_store"],
                                     "config": {**cfg["vector_store"]["config"],
                                                "path": "<per-question-tmpdir>"}},
                    "history_db_path": "<per-question-tmpdir>",
                }
            row = _run_question(m, rec, args.k)
        except Exception as e:
            row = {"qid": qid, "question_type": rec.get("question_type"),
                   "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(),
                   "is_abstention": qid.endswith("_abs"),
                   "gold_session_ids": sorted(set(rec.get("answer_session_ids") or []))}
        finally:
            shutil.rmtree(td, ignore_errors=True)

        per_question.append(row)
        if not args.quiet and (i + 1) % max(1, len(selected) // 20) == 0:
            done = i + 1
            elapsed = time.perf_counter() - t_run
            mid = _aggregate(per_question)
            print(f"[mem0] {done}/{len(selected)}  R@1={mid.r_at_1} "
                  f"R@5={mid.r_at_5} R@10={mid.r_at_10} MRR={mid.mrr}  "
                  f"p50={mid.p50_recall_ms}ms  elapsed={elapsed:.0f}s "
                  f"avg={elapsed/done:.1f}s/q", flush=True)

    metrics = _aggregate(per_question)
    captured_cfg["openai_api_key_set"] = bool(os.environ.get("OPENAI_API_KEY"))

    record = ResultRecord(
        timestamp=now_iso(), bench_pod_version=POD_VERSION, system=SYSTEM,
        system_version=_mem0_version(), system_config=captured_cfg,
        dataset=DatasetMeta(name="longmemeval_s", size=len(selected),
                            hash=_file_sha256(dataset_path)[:16]),
        metrics=metrics, host=host_info(),
    )
    out_path.write_text(json.dumps(record.to_json(), indent=2, default=str))
    if args.per_question_out:
        Path(args.per_question_out).write_text(json.dumps(per_question, indent=2))
    if not args.quiet:
        print(f"[mem0] wrote {out_path}", flush=True)
        print(f"[mem0] R@1={metrics.r_at_1} R@5={metrics.r_at_5} "
              f"R@10={metrics.r_at_10} MRR={metrics.mrr}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
