"""A-MEM runner — drives agiresearch/A-mem against LongMemEval-S with
the mandatory evolution-on / evolution-off ablation.

A-MEM puts an LLM in the write path: each add_note triggers an
evolution pass that links/rewrites neighbours. The /destruction/a-mem/
methodology requires both modes; the delta isolates retrieval from
LLM-driven rewriting.

  --evolution on   default A-MEM; LLM evolution fires per add_note
  --evolution off  monkey-patch process_memory to (False, note);
                   pure vector retrieval over raw notes

LLM is local ollama (qwen2.5:3b at 127.0.0.1:11434). OPENAI_API_KEY
is forced to "" so no upstream call is possible.

Setup (one-time):
    git clone https://github.com/agiresearch/A-mem.git /tmp/amem-source/A-mem
    python3.11 -m venv ~/.venvs/bench-amem
    ~/.venvs/bench-amem/bin/pip install -r /tmp/amem-source/A-mem/requirements.txt
    ~/.venvs/bench-amem/bin/pip install -e /tmp/amem-source/A-mem
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# Force ollama-only LLM, never OpenAI. Must be set BEFORE A-MEM import
# chain pulls litellm.
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("LITELLM_LOG", "ERROR")

from runners.common import (  # noqa: E402
    DatasetMeta, Metrics, POD_VERSION, ResultRecord,
    host_info, now_iso, percentile,
)

SYSTEM = "amem"
AMEM_SOURCE = Path("/tmp/amem-source/A-mem")
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Total LLM tokens consumed across a run. Populated by the wrapper we
# install around OllamaController.get_completion.
_LLM_TOKENS = {"prompt": 0, "completion": 0, "calls": 0}


def _install_token_meter() -> None:
    """Replace OllamaController.get_completion with a metered version that
    captures prompt/completion tokens from litellm's response.usage."""
    from agentic_memory import llm_controller as _lc
    import litellm

    def metered(self, prompt, response_format, temperature=0.7):
        r = litellm.completion(
            model=f"ollama_chat/{self.model}",
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
        )
        try:
            u = r.usage
            _LLM_TOKENS["prompt"] += int(getattr(u, "prompt_tokens", 0) or 0)
            _LLM_TOKENS["completion"] += int(getattr(u, "completion_tokens", 0) or 0)
            _LLM_TOKENS["calls"] += 1
        except Exception:
            pass
        return r.choices[0].message.content

    _lc.OllamaController.get_completion = metered


def _disable_evolution() -> None:
    """Bypass A-MEM's LLM evolution stage. add_note still embeds + writes
    to ChromaDB; search_agentic still walks the vector store; no LLM call."""
    from agentic_memory.memory_system import AgenticMemorySystem
    AgenticMemorySystem.process_memory = lambda self, note: (False, note)
    AgenticMemorySystem.analyze_content = lambda self, content: {
        "keywords": [], "context": "General", "tags": []
    }


def _format_session(sess: list[dict[str, Any]], date: Optional[str]) -> str:
    lines: list[str] = []
    if date:
        lines.append(f"[{date}]")
    for turn in sess:
        lines.append(f"{turn.get('role','?')}: {turn.get('content','')}")
    return "\n".join(lines)


def _gold_rank(results: list[dict[str, Any]], gold: set[str]) -> Optional[int]:
    seen: set[str] = set()
    for i, r in enumerate(results, start=1):
        sid = r.get("category") or ""
        if sid.startswith("session:"):
            sid = sid.split(":", 1)[1]
        else:
            # Fallback: parse from content prefix
            c = r.get("content") or ""
            if c.startswith("[session:"):
                sid = c.split("]", 1)[0][len("[session:"):]
        if sid and sid in gold and sid not in seen:
            return i
        if sid:
            seen.add(sid)
    return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _amem_version() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "-C", str(AMEM_SOURCE), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return f"git:{out.decode().strip()}"
    except Exception:
        return "unknown"


def _run_question(record: dict[str, Any], k: int, evolution: bool,
                  embed_model: str, llm_model: str) -> dict[str, Any]:
    from agentic_memory.memory_system import AgenticMemorySystem

    qid = record.get("question_id", "?")
    question = record["question"]
    gold = set(record.get("answer_session_ids") or [])
    is_abs = qid.endswith("_abs")

    sids = record.get("haystack_session_ids", [])
    sessions = record.get("haystack_sessions", [])
    dates = record.get("haystack_dates", []) or [None] * len(sessions)

    # Fresh AMS per question = fresh ChromaDB collection (constructor
    # resets the in-process client).
    ams = AgenticMemorySystem(
        model_name=embed_model,
        llm_backend="ollama",
        llm_model=llm_model,
    )

    t0 = time.perf_counter()
    n_ingested = 0
    for sid, sess, date in zip(sids, sessions, dates):
        text = _format_session(sess, date)
        if not text.strip():
            continue
        # category carries the session id so we can score by gold session.
        try:
            ams.add_note(content=text, category=f"session:{sid}")
            n_ingested += 1
        except Exception as e:
            print(f"[amem] add_note failed for sid={sid}: {e}", file=sys.stderr)
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    try:
        results = ams.search_agentic(question, k=k)
    except Exception as e:
        results = []
        print(f"[amem] search_agentic failed for {qid}: {e}", file=sys.stderr)
    recall_ms = (time.perf_counter() - t1) * 1000.0

    rank = _gold_rank(results, gold) if gold else None

    # Tear down to free the ChromaDB collection / SentenceTransformer
    try:
        ams.retriever.client.reset()
    except Exception:
        pass
    del ams
    gc.collect()

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
    }


def _aggregate(per_q: list[dict[str, Any]]) -> dict[str, float]:
    gradeable = [q for q in per_q if q.get("gold_session_ids") and not q.get("is_abstention")
                 and "error" not in q]
    n = max(1, len(gradeable))

    def hits(t: int) -> int:
        return sum(1 for q in gradeable
                   if q.get("rank_of_gold") is not None and q["rank_of_gold"] <= t)

    rrs = [1.0 / q["rank_of_gold"] if q.get("rank_of_gold") else 0.0 for q in gradeable]
    lat = [q["latency_ms"] for q in per_q if "latency_ms" in q]

    return {
        "n_gradeable": len(gradeable),
        "recall@1": round(hits(1) / n, 4),
        "recall@5": round(hits(5) / n, 4),
        "recall@10": round(hits(10) / n, 4),
        "MRR": round(statistics.mean(rrs) if rrs else 0.0, 4),
        "p50_ms": round(percentile(lat, 0.50), 3),
        "p95_ms": round(percentile(lat, 0.95), 3),
    }


def _run_one_mode(args, evolution: bool, output_path: Path) -> int:
    # Reset token meter for this mode
    _LLM_TOKENS["prompt"] = 0
    _LLM_TOKENS["completion"] = 0
    _LLM_TOKENS["calls"] = 0

    # Wire up monkey-patches BEFORE any AgenticMemorySystem instantiation
    _install_token_meter()
    if not evolution:
        _disable_evolution()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"[amem] dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    raw = json.loads(dataset_path.read_text())
    selected = raw[: args.limit] if args.limit else raw

    print(f"[amem] mode=evolution-{'on' if evolution else 'off'} "
          f"questions={len(selected)} k={args.k} llm={args.llm_model}", flush=True)

    per_q: list[dict[str, Any]] = []
    failed: list[int] = []
    t_run = time.perf_counter()

    for i, rec in enumerate(selected):
        try:
            row = _run_question(rec, args.k, evolution,
                                args.embed_model, args.llm_model)
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
        per_q.append(row)
        if (i + 1) % max(1, len(selected) // 20 or 1) == 0 or i == 0:
            agg = _aggregate(per_q)
            elapsed = time.perf_counter() - t_run
            print(f"[amem] {i+1}/{len(selected)} "
                  f"R@1={agg['recall@1']:.3f} R@5={agg['recall@5']:.3f} "
                  f"R@10={agg['recall@10']:.3f} MRR={agg['MRR']:.3f} "
                  f"elapsed={elapsed:.0f}s tokens={_LLM_TOKENS['prompt']+_LLM_TOKENS['completion']}",
                  flush=True)

    wall = time.perf_counter() - t_run
    agg = _aggregate(per_q)

    metrics = Metrics(
        r_at_1=agg["recall@1"],
        r_at_5=agg["recall@5"],
        r_at_10=agg["recall@10"],
        mrr=agg["MRR"],
        p50_recall_ms=agg["p50_ms"],
        p95_recall_ms=agg["p95_ms"],
        wall_seconds_ingest=round(sum(q.get("ingest_ms", 0) for q in per_q) / 1000.0, 2),
        wall_seconds_query=round(wall, 2),
        llm_tokens_extraction=(_LLM_TOKENS["prompt"] + _LLM_TOKENS["completion"]) if evolution else 0,
        errors=len(failed),
        failed_questions=failed,
    )

    rec = ResultRecord(
        timestamp=now_iso(),
        bench_pod_version=POD_VERSION,
        system=SYSTEM,
        system_version=_amem_version(),
        system_config={
            "evolution": "on" if evolution else "off",
            "llm_backend": "ollama",
            "llm_endpoint": "http://127.0.0.1:11434",
            "llm_model": args.llm_model,
            "embed_model": args.embed_model,
            "k": args.k,
            "limit": args.limit,
            "llm_calls": _LLM_TOKENS["calls"] if evolution else 0,
            "llm_prompt_tokens": _LLM_TOKENS["prompt"] if evolution else 0,
            "llm_completion_tokens": _LLM_TOKENS["completion"] if evolution else 0,
            "amem_source": str(AMEM_SOURCE),
        },
        dataset=DatasetMeta(
            name="longmemeval_s",
            size=len(raw),
            hash=_file_sha256(dataset_path)[:16],
        ),
        metrics=metrics,
        host=host_info(),
    )

    payload = rec.to_json()
    payload["per_question"] = per_q  # extra, schema allows additionalProperties
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))

    print(f"[amem] wrote {output_path}", flush=True)
    print(f"[amem] R@1={metrics.r_at_1} R@5={metrics.r_at_5} "
          f"R@10={metrics.r_at_10} MRR={metrics.mrr} "
          f"p50={metrics.p50_recall_ms}ms tokens={metrics.llm_tokens_extraction}",
          flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runners.amem_runner",
        description="A-MEM (agiresearch/A-mem) runner with mandatory evolution-on/off ablation.",
    )
    p.add_argument("--dataset", required=True,
                   help="Path to longmemeval_s.json")
    p.add_argument("--output", required=True,
                   help="Path to write the result JSON. With --both, this is used "
                        "as a base path; '_evo-on' / '_evo-off' are inserted before .json.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap on number of questions (0 = all).")
    p.add_argument("--evolution", choices=["on", "off"],
                   help="Required unless --both: 'on' = A-MEM default, "
                        "'off' = bypass LLM evolution stage.")
    p.add_argument("--both", action="store_true",
                   help="Run both evolution modes sequentially; emits TWO output "
                        "JSONs suffixed _evo-on and _evo-off.")
    p.add_argument("--llm-model", default=DEFAULT_OLLAMA_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help=f"SentenceTransformer model (default: {DEFAULT_EMBED_MODEL})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.both and args.evolution is None:
        print("[amem] --evolution {on,off} is required (or pass --both)", file=sys.stderr)
        return 2

    out = Path(args.output).expanduser().resolve()
    if args.both:
        stem, suf = out.with_suffix(""), out.suffix or ".json"
        # When --both, each mode runs in a separate subprocess so the
        # monkey-patches from mode A don't leak into mode B.
        import subprocess
        rc_total = 0
        for mode in ("off", "on"):
            target = Path(f"{stem}_evo-{mode}{suf}")
            cmd = [sys.executable, "-m", "runners.amem_runner",
                   "--dataset", args.dataset, "--k", str(args.k),
                   "--output", str(target), "--evolution", mode,
                   "--llm-model", args.llm_model,
                   "--embed-model", args.embed_model]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            print(f"[amem] --both: launching evolution={mode}", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[amem] mode {mode} exited rc={rc}", file=sys.stderr)
                rc_total = rc
        return rc_total

    return _run_one_mode(args, evolution=(args.evolution == "on"), output_path=out)


if __name__ == "__main__":
    raise SystemExit(main())
