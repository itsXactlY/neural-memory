#!/usr/bin/env python3
"""Demolition Bench — head-to-head Mazemaker QA bench against models that
Hindsight scored as 0/N because they couldn't follow a JSON output schema.

The point we're making:
  Hindsight measures whether the LLM can output JSON.
  Mazemaker measures memory recall — and uses plain-text answers, which
  every one of these "failing" models can produce.

Per question we:

  1. Spin up an isolated Mazemaker on a temp DB.
  2. Ingest the dataset's haystack sessions as memories.
  3. Run Mazemaker's recall to retrieve top-k memories.
  4. Build a plain-text context block from the top-N results.
  5. Ask the model in plain English (NO json schema) for a one-sentence
     answer.
  6. Substring-match the gold answer against the model's response.

That's it. No JSON gating, no schema validation, no "must follow this
format or you get 0". The model only has to put the right words in its
answer somewhere.

Usage
-----
    # Smoke (1 model, 3 questions)
    python -u -m benchmarks.external.demolition_bench \\
        --models gemma3:270m --n 3 --tag smoke

    # Full grid (10 models × 20 questions)
    python -u -m benchmarks.external.demolition_bench --n 20

    # ColBERT-on second pass
    python -u -m benchmarks.external.demolition_bench --n 20 --enable-colbert

    # Custom subset
    python -u -m benchmarks.external.demolition_bench \\
        --models qwen2.5:3b llama3.2:latest --n 50

Output
------
JSON + Markdown table under benchmarks/external/results/
"""
from __future__ import annotations

import os

# Reproducibility rule: ollama is the only process that should touch the
# GPU during this bench. The bench python itself does NOT need CUDA —
# Mazemaker recall over a tempdir with 1–2 ingested memories is trivially
# CPU-bound, and FastEmbed defaults to CPU. By hiding the device from
# torch / onnxruntime / fastembed up front, we guarantee no contention
# with ollama no matter what the engine pulls in transitively. Set this
# BEFORE any torch/transformers/onnxruntime import — it's a runtime read.
# Operator can still force GPU with MM_BENCH_ALLOW_CUDA=1.
if not os.environ.get("MM_BENCH_ALLOW_CUDA"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MM_COLBERT_DEVICE", "cpu")
    os.environ.setdefault("MM_FORCE_CPU", "1")

import argparse
import datetime as _dt
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from memory_client import Mazemaker  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "longmemeval_s"
DATASET_PATH = DATA_DIR / "longmemeval_s.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# The 10 models Hindsight publicly listed as scoring 0/N because they
# couldn't follow the required JSON schema. We're going to score them
# on the actual task: memory recall + plain-text answer.
DEFAULT_MODELS = [
    "gemma3:1b",
    "gemma3:12b",
    "gemma3:270m",
    "qwen2.5:0.5b",
    "qwen2.5:3b",
    "smollm2:1.7b",
    "deepseek-r1:1.5b",
    "granite3.1-dense:2b",
    "llama3.2:latest",
    "ministral-3:3b",
]


# ---------------------------------------------------------------------------
# Ollama HTTP shim (no extra dependency — stdlib urllib.request only)
# ---------------------------------------------------------------------------

def _http_post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _http_get_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def ollama_list() -> list[str]:
    try:
        data = _http_get_json(f"{OLLAMA_URL}/api/tags")
    except Exception:
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def ollama_has_model(name: str, cached: Optional[set[str]] = None) -> bool:
    pool = cached if cached is not None else set(ollama_list())
    if name in pool:
        return True
    # `latest` tag tolerance: "llama3.2" == "llama3.2:latest"
    if ":" not in name and f"{name}:latest" in pool:
        return True
    if name.endswith(":latest") and name[: -len(":latest")] in pool:
        return True
    return False


def ollama_pull(name: str, log_prefix: str = "[demo-bench]") -> bool:
    """Pull a model via the CLI (handles progress UI streamingly).

    Returns True on success, False otherwise.
    """
    print(f"{log_prefix} pulling {name} ...", flush=True)
    try:
        proc = subprocess.run(
            ["ollama", "pull", name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60 * 30,
        )
    except Exception as e:
        print(f"{log_prefix} pull failed: {type(e).__name__}: {e}", flush=True)
        return False
    ok = proc.returncode == 0
    tail = "\n".join((proc.stdout or "").splitlines()[-3:])
    print(f"{log_prefix} pull {'ok' if ok else 'FAIL'}: {tail}", flush=True)
    return ok


def ollama_evict(model: str, timeout: float = 30.0) -> bool:
    """Tell ollama to unload `model` from VRAM (best-effort).

    Sequential model runs leave the previous model resident in VRAM
    until ollama's keep-alive (default 5 min) elapses. With ColBERT also
    holding GPU memory we hit OOM and ollama returns HTTP 500. Forcing
    `keep_alive: 0` tears the model down deterministically.
    """
    try:
        _http_post_json(f"{OLLAMA_URL}/api/generate", {
            "model": model,
            "prompt": "",
            "keep_alive": 0,
        }, timeout=timeout)
        return True
    except Exception:
        return False


def ollama_chat(model: str, prompt: str, timeout: float = 120.0) -> tuple[str, dict]:
    """Plain-text chat call. Returns (answer_text, raw_response)."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer concisely in plain "
                    "English. Do not use JSON, markdown, or any structured "
                    "output. Just answer the question in one sentence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            # Keep it predictable for benchmark reproducibility.
            "temperature": 0.0,
            "num_predict": 256,
        },
    }
    raw = _http_post_json(f"{OLLAMA_URL}/api/chat", payload, timeout=timeout)
    msg = raw.get("message") or {}
    text = msg.get("content") or raw.get("response") or ""
    return text.strip(), raw


# ---------------------------------------------------------------------------
# Dataset shaping — adapt LongMemEval-S to a generic QA shape
# ---------------------------------------------------------------------------

def _format_session(sess: list[dict[str, Any]], date: str | None) -> str:
    lines: list[str] = []
    if date:
        lines.append(f"[{date}]")
    for turn in sess:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def lme_to_qa_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project LongMemEval-S records to {question, gold_substring, sessions}.

    `sessions` is a list of {label, text} dicts so the harness can ingest
    them directly. `gold_substring` is taken from `answer`.
    """
    out = []
    for rec in records:
        if rec.get("question_id", "").endswith("_abs"):
            # Skip abstention questions — they have no gold substring,
            # so the substring-matcher can't grade them. Keep the bench
            # focused on positive recall.
            continue
        ans = (rec.get("answer") or "").strip()
        if not ans:
            continue
        sids = rec.get("haystack_session_ids", [])
        sessions = rec.get("haystack_sessions", [])
        dates = rec.get("haystack_dates", []) or [None] * len(sessions)
        sess_blobs = []
        for sid, sess, date in zip(sids, sessions, dates):
            text = _format_session(sess, date)
            if not text.strip():
                continue
            sess_blobs.append({"label": f"session:{sid}", "text": text})
        if not sess_blobs:
            continue
        out.append({
            "qid": rec.get("question_id", ""),
            "question": rec.get("question", ""),
            "question_type": rec.get("question_type", "unknown"),
            "gold_substring": ans,
            "sessions": sess_blobs,
            "answer_session_ids": list(rec.get("answer_session_ids") or []),
        })
    return out


# 20-question synthetic fallback covering the same surfaces Hindsight
# claims to test, for when LongMemEval-S is unavailable.
SYNTHETIC_RECORDS: list[dict[str, Any]] = [
    # factual recall (4)
    {"qid": "syn-fr-1", "question_type": "factual",
     "question": "What is my dog's name?",
     "gold_substring": "Pepper",
     "sessions": [
         {"label": "session:syn-fr-1-a", "text": "user: I just adopted a beagle named Pepper.\nassistant: Congrats!"},
         {"label": "session:syn-fr-1-b", "text": "user: I love hiking on weekends.\nassistant: Nice."},
     ]},
    {"qid": "syn-fr-2", "question_type": "factual",
     "question": "What city did I move to?",
     "gold_substring": "Lisbon",
     "sessions": [
         {"label": "session:syn-fr-2-a", "text": "user: I'm relocating to Lisbon next month for work.\nassistant: Exciting move."},
         {"label": "session:syn-fr-2-b", "text": "user: My favorite color is teal.\nassistant: Noted."},
     ]},
    {"qid": "syn-fr-3", "question_type": "factual",
     "question": "What programming language do I prefer?",
     "gold_substring": "Rust",
     "sessions": [
         {"label": "session:syn-fr-3-a", "text": "user: I prefer Rust for systems work; the borrow checker keeps me sane.\nassistant: Got it."},
         {"label": "session:syn-fr-3-b", "text": "user: I drink black coffee.\nassistant: OK."},
     ]},
    {"qid": "syn-fr-4", "question_type": "factual",
     "question": "What is my partner's profession?",
     "gold_substring": "architect",
     "sessions": [
         {"label": "session:syn-fr-4-a", "text": "user: My partner is an architect — works on commercial spaces.\nassistant: Cool."},
         {"label": "session:syn-fr-4-b", "text": "user: We're vegetarian.\nassistant: Got it."},
     ]},
    # multi-session (4)
    {"qid": "syn-ms-1", "question_type": "multi-session",
     "question": "Which country am I planning my next vacation in?",
     "gold_substring": "Japan",
     "sessions": [
         {"label": "session:syn-ms-1-a", "text": "user: I've been thinking about a trip to Asia later this year.\nassistant: Anywhere specific?"},
         {"label": "session:syn-ms-1-b", "text": "user: Decided — Japan in October. Booked flights to Tokyo.\nassistant: Have a great trip."},
     ]},
    {"qid": "syn-ms-2", "question_type": "multi-session",
     "question": "What instrument did I take up?",
     "gold_substring": "violin",
     "sessions": [
         {"label": "session:syn-ms-2-a", "text": "user: I want to learn an instrument this year.\nassistant: Any in mind?"},
         {"label": "session:syn-ms-2-b", "text": "user: Started violin lessons last week, it's brutal but I love it.\nassistant: Stick with it."},
     ]},
    {"qid": "syn-ms-3", "question_type": "multi-session",
     "question": "What did I name my new car?",
     "gold_substring": "Bumblebee",
     "sessions": [
         {"label": "session:syn-ms-3-a", "text": "user: Bought a yellow hatchback yesterday.\nassistant: Nice."},
         {"label": "session:syn-ms-3-b", "text": "user: Named the car Bumblebee, obvious choice.\nassistant: Ha!"},
     ]},
    {"qid": "syn-ms-4", "question_type": "multi-session",
     "question": "What's the topic of the book I'm writing?",
     "gold_substring": "octopus",
     "sessions": [
         {"label": "session:syn-ms-4-a", "text": "user: I want to write a popular-science book.\nassistant: On what?"},
         {"label": "session:syn-ms-4-b", "text": "user: Settled on octopus cognition. Three chapters drafted.\nassistant: Sounds great."},
     ]},
    # temporal (4)
    {"qid": "syn-tm-1", "question_type": "temporal",
     "question": "In what month did I start my new job?",
     "gold_substring": "March",
     "sessions": [
         {"label": "session:syn-tm-1-a", "text": "user: Got the offer, signed yesterday.\nassistant: Congrats."},
         {"label": "session:syn-tm-1-b", "text": "user: First day was March 6th. Long but good.\nassistant: How'd it go?"},
     ]},
    {"qid": "syn-tm-2", "question_type": "temporal",
     "question": "How many years have I lived in Berlin?",
     "gold_substring": "seven",
     "sessions": [
         {"label": "session:syn-tm-2-a", "text": "user: I moved to Berlin in 2019.\nassistant: Nice."},
         {"label": "session:syn-tm-2-b", "text": "user: It's now 2026 — seven years in Berlin, hard to believe.\nassistant: Time flies."},
     ]},
    {"qid": "syn-tm-3", "question_type": "temporal",
     "question": "Which season do I prefer for hiking?",
     "gold_substring": "autumn",
     "sessions": [
         {"label": "session:syn-tm-3-a", "text": "user: Spring rains keep me indoors.\nassistant: Mm."},
         {"label": "session:syn-tm-3-b", "text": "user: Autumn is peak hiking — crisp air, no bugs.\nassistant: Agreed."},
     ]},
    {"qid": "syn-tm-4", "question_type": "temporal",
     "question": "When does my lease end?",
     "gold_substring": "August",
     "sessions": [
         {"label": "session:syn-tm-4-a", "text": "user: Signed a one-year lease last August.\nassistant: OK."},
         {"label": "session:syn-tm-4-b", "text": "user: Lease is up at the end of August. Need to decide whether to renew.\nassistant: Mm."},
     ]},
    # entity-tracking (4)
    {"qid": "syn-et-1", "question_type": "entity-tracking",
     "question": "Who is my dentist?",
     "gold_substring": "Dr. Mendes",
     "sessions": [
         {"label": "session:syn-et-1-a", "text": "user: New dentist. Dr. Mendes — recommended by my coworker.\nassistant: Hope it goes well."},
         {"label": "session:syn-et-1-b", "text": "user: Filling went fine. Dr. Mendes was gentle.\nassistant: Glad."},
     ]},
    {"qid": "syn-et-2", "question_type": "entity-tracking",
     "question": "What's my company called?",
     "gold_substring": "Lumen",
     "sessions": [
         {"label": "session:syn-et-2-a", "text": "user: Founded a company last year — Lumen Analytics.\nassistant: Cool."},
         {"label": "session:syn-et-2-b", "text": "user: Lumen just hired its 5th engineer.\nassistant: Growing fast."},
     ]},
    {"qid": "syn-et-3", "question_type": "entity-tracking",
     "question": "Which gym do I go to?",
     "gold_substring": "IronWorks",
     "sessions": [
         {"label": "session:syn-et-3-a", "text": "user: Joined IronWorks last weekend. Sweaty but quiet.\nassistant: Good."},
         {"label": "session:syn-et-3-b", "text": "user: Coach at IronWorks said my squat form is solid.\nassistant: Nice."},
     ]},
    {"qid": "syn-et-4", "question_type": "entity-tracking",
     "question": "What's my best friend's name?",
     "gold_substring": "Marisol",
     "sessions": [
         {"label": "session:syn-et-4-a", "text": "user: My best friend Marisol is visiting next week.\nassistant: Fun."},
         {"label": "session:syn-et-4-b", "text": "user: Marisol and I have known each other since college.\nassistant: Sweet."},
     ]},
    # multi-step / inference (4) — substring still in source
    {"qid": "syn-mh-1", "question_type": "multi-step",
     "question": "What allergy do I have to be careful about at restaurants?",
     "gold_substring": "shellfish",
     "sessions": [
         {"label": "session:syn-mh-1-a", "text": "user: Had a scary ER visit last year — turned out to be a shellfish allergy.\nassistant: Yikes."},
         {"label": "session:syn-mh-1-b", "text": "user: I always read the menu carefully now.\nassistant: Smart."},
     ]},
    {"qid": "syn-mh-2", "question_type": "multi-step",
     "question": "What size shoe do I wear?",
     "gold_substring": "11",
     "sessions": [
         {"label": "session:syn-mh-2-a", "text": "user: Ordered new running shoes in size 11.\nassistant: Fast delivery hopefully."},
         {"label": "session:syn-mh-2-b", "text": "user: Shoes fit perfectly.\nassistant: Good."},
     ]},
    {"qid": "syn-mh-3", "question_type": "multi-step",
     "question": "What's my child's favorite snack?",
     "gold_substring": "blueberries",
     "sessions": [
         {"label": "session:syn-mh-3-a", "text": "user: Toddler will only eat blueberries at snacktime.\nassistant: Could be worse."},
         {"label": "session:syn-mh-3-b", "text": "user: Bought another carton of blueberries today. She's obsessed.\nassistant: Ha."},
     ]},
    {"qid": "syn-mh-4", "question_type": "multi-step",
     "question": "What hobby did I pick up during the pandemic?",
     "gold_substring": "sourdough",
     "sessions": [
         {"label": "session:syn-mh-4-a", "text": "user: Started baking sourdough during lockdown — still going strong years later.\nassistant: Nice."},
         {"label": "session:syn-mh-4-b", "text": "user: My starter is named Doughy McBreadface.\nassistant: Lol."},
     ]},
]


def load_qa_records(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.dataset == "synthetic":
        recs = list(SYNTHETIC_RECORDS)
        if args.n and args.n < len(recs):
            recs = recs[: args.n]
        h = hashlib.sha256(json.dumps(SYNTHETIC_RECORDS, sort_keys=True).encode()).hexdigest()
        return recs, {"name": "synthetic_v1", "size": len(recs), "hash": h[:16]}

    # default: longmemeval_s subset
    if not DATASET_PATH.exists():
        print(f"[demo-bench] LongMemEval-S not found at {DATASET_PATH}, "
              f"falling back to synthetic dataset.", flush=True)
        recs = list(SYNTHETIC_RECORDS)
        if args.n and args.n < len(recs):
            recs = recs[: args.n]
        h = hashlib.sha256(json.dumps(SYNTHETIC_RECORDS, sort_keys=True).encode()).hexdigest()
        return recs, {"name": "synthetic_v1_fallback", "size": len(recs), "hash": h[:16]}

    raw = json.loads(DATASET_PATH.read_text())
    qa = lme_to_qa_records(raw)
    # Stratify by question_type so a small subset still spans the surface.
    if args.n and args.n < len(qa):
        bucket: dict[str, list[dict[str, Any]]] = {}
        for r in qa:
            bucket.setdefault(r["question_type"], []).append(r)
        import random
        rng = random.Random(args.seed)
        out: list[dict[str, Any]] = []
        per_type = max(1, args.n // max(1, len(bucket)))
        for qt, rows in bucket.items():
            rng.shuffle(rows)
            out.extend(rows[:per_type])
        # top up if rounding lost us a few
        if len(out) < args.n:
            extras = [r for r in qa if r not in out]
            rng.shuffle(extras)
            out.extend(extras[: args.n - len(out)])
        rng.shuffle(out)
        qa = out[: args.n]

    h = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    return qa, {
        "name": f"longmemeval_s_subset{len(qa)}",
        "size": len(qa),
        "hash": h[:16],
        "source_sha256": h,
    }


# ---------------------------------------------------------------------------
# Engine builder + per-question run
# ---------------------------------------------------------------------------

def _build_engine(args, db_path: str) -> Mazemaker:
    if args.enable_colbert:
        os.environ["MM_COLBERT_ENABLED"] = "1"
        # ColBERT loads its own copy of BGE-M3 (~1.4 GB VRAM). In this
        # harness ollama also wants the GPU for the LLM serving path —
        # the contention manifests as HTTP 500s on the model side. Force
        # the helper to CPU unless the operator has explicitly overridden.
        os.environ.setdefault("MM_COLBERT_DEVICE", "cpu")
    return Mazemaker(
        db_path=db_path,
        embedding_backend=args.backend,
        use_cpp=False,
        retrieval_mode=args.recall_mode,
        use_hnsw=False,
        lazy_graph=True,
        think_engine="bfs",
        rerank=args.rerank,
    )


def _ingest(nm: Mazemaker, sessions: list[dict[str, str]]) -> int:
    n = 0
    for s in sessions:
        text = s.get("text", "")
        if not text.strip():
            continue
        nm.remember(
            text,
            label=s.get("label", ""),
            auto_connect=False,
            detect_conflicts=False,
        )
        n += 1
    return n


def _retrieve_context(nm: Mazemaker, args, question: str) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {
        "k": args.k_retrieval,
        "hybrid": (args.recall_mode in {"hybrid", "advanced", "skynet", "lean", "trim"}),
        "rerank": args.rerank,
    }
    if args.enable_colbert:
        kwargs["enable_colbert"] = True
        if args.colbert_weight is not None:
            kwargs["colbert_weight"] = float(args.colbert_weight)
    results = nm.recall(question, **kwargs)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return results, dt_ms


def _build_prompt(question: str, results: list[dict[str, Any]], top_n: int = 5) -> str:
    snippets = []
    for i, r in enumerate(results[:top_n], 1):
        body = r.get("content") or r.get("text") or ""
        if not body:
            continue
        # cap each snippet so a fat session doesn't blow the context
        snippets.append(f"[memory {i}]\n{body[:1200]}")
    context = "\n\n".join(snippets) if snippets else "(no memories retrieved)"
    return (
        f"Use the following memories from a chat history to answer the "
        f"question.\n\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer in one sentence."
    )


def run_question_for_model(args, model: str, qa_record: dict[str, Any]) -> dict[str, Any]:
    qid = qa_record["qid"]
    question = qa_record["question"]
    gold = (qa_record.get("gold_substring") or "").strip()
    qtype = qa_record.get("question_type", "unknown")

    with tempfile.TemporaryDirectory(prefix=f"demo-{qid[:8]}-") as td:
        db = str(Path(td) / "bench.db")
        nm = _build_engine(args, db)
        try:
            t0 = time.perf_counter()
            n_ingested = _ingest(nm, qa_record["sessions"])
            ingest_ms = (time.perf_counter() - t0) * 1000.0

            results, recall_ms = _retrieve_context(nm, args, question)

            prompt = _build_prompt(question, results, top_n=args.context_top_n)
            t1 = time.perf_counter()
            try:
                answer, raw = ollama_chat(
                    model, prompt, timeout=args.model_timeout)
                err = None
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                answer, raw, err = "", {}, f"{type(e).__name__}: {e}"
            except Exception as e:
                answer, raw, err = "", {}, f"{type(e).__name__}: {e}"
            model_ms = (time.perf_counter() - t1) * 1000.0

            is_correct = bool(gold) and (gold.lower() in (answer or "").lower())
            looks_like_json = answer.strip().startswith(("{", "["))

            return {
                "qid": qid,
                "question_type": qtype,
                "question": question,
                "gold_substring": gold,
                "n_ingested": n_ingested,
                "n_retrieved": len(results),
                "answer": answer,
                "is_correct": is_correct,
                "looks_like_json": looks_like_json,
                "ingest_ms": round(ingest_ms, 2),
                "recall_ms": round(recall_ms, 2),
                "model_ms": round(model_ms, 2),
                "error": err,
            }
        finally:
            try:
                nm.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in rows if not r.get("error")]
    correct = sum(1 for r in answered if r.get("is_correct"))
    n = len(rows)
    errors = sum(1 for r in rows if r.get("error"))
    json_leaks = sum(1 for r in answered if r.get("looks_like_json"))
    latencies = [r["model_ms"] for r in answered if "model_ms" in r]
    recall_lat = [r["recall_ms"] for r in answered if "recall_ms" in r]
    return {
        "correct": correct,
        "total": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "errors": errors,
        "json_leaks": json_leaks,
        "avg_model_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "avg_recall_latency_ms": round(statistics.mean(recall_lat), 1) if recall_lat else 0.0,
    }


def write_outputs(args, dataset_meta: dict[str, Any],
                  per_model: dict[str, dict[str, Any]],
                  per_model_rows: dict[str, list[dict[str, Any]]]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"demolition_{ts}"
    if args.tag:
        base = f"demolition_{args.tag}_{ts}"

    payload = {
        "timestamp": ts,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "system_config": {
            "embedding_backend": args.backend,
            "recall_mode": args.recall_mode,
            "rerank": bool(args.rerank),
            "enable_colbert": bool(args.enable_colbert),
            "colbert_weight": (float(args.colbert_weight) if args.colbert_weight is not None else None),
            "k_retrieval": args.k_retrieval,
            "context_top_n": args.context_top_n,
            "judge_method": "substring_match",
            "ollama_url": OLLAMA_URL,
            "model_timeout_s": args.model_timeout,
        },
        "dataset": dataset_meta,
        "models": per_model,
        "per_question": per_model_rows,
    }
    json_path = RESULTS_DIR / f"{base}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md_path = RESULTS_DIR / f"{base}.md"
    md_path.write_text(_render_markdown(payload, args))
    return json_path, md_path


def _render_markdown(payload: dict[str, Any], args) -> str:
    ds = payload["dataset"]
    cfg = payload["system_config"]
    n = ds.get("size", "?")
    lines: list[str] = []
    lines.append(f"# Demolition Bench — {payload['timestamp']}")
    lines.append("")
    lines.append(
        "**Headline:** Mazemaker measures memory. Hindsight measures "
        "whether the LLM can output JSON."
    )
    lines.append("")
    lines.append(
        f"Engine `git_sha`: `{payload['git_sha']}`"
        + (" (dirty)" if payload.get("git_dirty") else "")
    )
    lines.append(f"Dataset: `{ds.get('name')}` (n={n}, hash={ds.get('hash')})")
    lines.append(
        f"Config: backend=`{cfg['embedding_backend']}`, recall_mode=`{cfg['recall_mode']}`, "
        f"rerank={cfg['rerank']}, enable_colbert={cfg['enable_colbert']}, "
        f"k_retrieval={cfg['k_retrieval']}, judge=`{cfg['judge_method']}`"
    )
    lines.append("")
    lines.append("| Model | Hindsight Result | Mazemaker Accuracy | Avg model latency | Avg recall latency | JSON leaks | Errors |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, m in payload["models"].items():
        acc = f"{m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%"
        ml = f"{m['avg_model_latency_ms']/1000:.2f}s"
        rl = f"{m['avg_recall_latency_ms']/1000:.2f}s"
        lines.append(
            f"| {model} | 0/N (couldn't follow JSON schema) | {acc} | {ml} | {rl} | "
            f"{m['json_leaks']} | {m['errors']} |"
        )
    lines.append("")
    lines.append("**Methodology:** for every question we spin up an isolated Mazemaker, ingest the haystack sessions, "
                 f"retrieve top-{cfg['k_retrieval']} memories, build a plain-English prompt from the top "
                 f"{cfg['context_top_n']}, and ask the model in natural language for a one-sentence answer. "
                 "Scoring = the dataset's gold substring appears in the model's response (case-insensitive).")
    lines.append("")
    lines.append("`JSON leaks` counts answers that began with `{` or `[` despite the plain-text prompt — "
                 "indicating the model insisted on a structured shape we never asked for. "
                 "These were still scored on substring match.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demolition Bench (Mazemaker vs Hindsight's failed-models list)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Ollama model names. Default: the 10 Hindsight-fail models.")
    p.add_argument("--n", type=int, default=20, help="Number of questions per model")
    p.add_argument("--dataset", default="longmemeval_s",
                   choices=["longmemeval_s", "synthetic"],
                   help="Question source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="Optional tag prefix on the output filename")

    # Engine knobs
    p.add_argument("--backend", default="auto",
                   help="Embedding backend (auto = BGE-M3 via shared server)")
    p.add_argument("--recall-mode", default="advanced",
                   choices=["semantic", "hybrid", "advanced", "skynet", "lean", "trim"])
    p.add_argument("--rerank", action="store_true", default=True,
                   help="Cross-encoder reranker (default: on for advanced/skynet)")
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--enable-colbert", action="store_true",
                   help="Enable ColBERT late-interaction rerank channel")
    p.add_argument("--colbert-weight", type=float, default=None,
                   help="Override ColBERT channel weight (default: per recall-mode)")
    p.add_argument("--k-retrieval", type=int, default=10)
    p.add_argument("--context-top-n", type=int, default=5,
                   help="How many retrieved memories to put in the model prompt")

    # Ollama knobs
    p.add_argument("--auto-pull", action="store_true", default=True,
                   help="Pull missing models with `ollama pull` before running")
    p.add_argument("--no-auto-pull", dest="auto_pull", action="store_false")
    p.add_argument("--model-timeout", type=float, default=180.0,
                   help="Per-call HTTP timeout in seconds")
    p.add_argument("--skip-missing", action="store_true",
                   help="If a model isn't pulled and --no-auto-pull is set, skip it")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    qa_records, dataset_meta = load_qa_records(args)
    if not qa_records:
        print("[demo-bench] no questions loaded — aborting", flush=True)
        return 2

    models = args.models or DEFAULT_MODELS

    print(f"[demo-bench] dataset={dataset_meta['name']} n={dataset_meta['size']}", flush=True)
    print(f"[demo-bench] models={models}", flush=True)
    print(f"[demo-bench] config: recall_mode={args.recall_mode} rerank={args.rerank} "
          f"colbert={args.enable_colbert} k={args.k_retrieval} top_n={args.context_top_n}",
          flush=True)

    # Verify ollama is reachable up front
    available_now = ollama_list()
    if not available_now and not args.auto_pull:
        print("[demo-bench] ollama list returned nothing and --no-auto-pull set; "
              "make sure ollama is running.", flush=True)

    per_model: dict[str, dict[str, Any]] = {}
    per_model_rows: dict[str, list[dict[str, Any]]] = {}

    cached = set(available_now)
    prev_model: Optional[str] = None
    for model in models:
        # Evict the previous model from VRAM before loading the next one.
        # ollama's default 5-min keep-alive otherwise stacks the previous
        # model alongside the new one, and (with ColBERT on or any other
        # GPU resident) the second model hits OOM and returns HTTP 500.
        if prev_model and prev_model != model:
            evicted = ollama_evict(prev_model)
            print(f"[demo-bench] evict {prev_model}: {'ok' if evicted else 'best-effort'}",
                  flush=True)
        if not ollama_has_model(model, cached):
            if args.auto_pull:
                if ollama_pull(model):
                    cached = set(ollama_list())
                else:
                    per_model[model] = {
                        "correct": 0, "total": 0, "accuracy": 0.0,
                        "errors": len(qa_records), "json_leaks": 0,
                        "avg_model_latency_ms": 0.0, "avg_recall_latency_ms": 0.0,
                        "skipped_reason": "pull_failed",
                    }
                    per_model_rows[model] = []
                    print(f"[demo-bench] {model}: pull failed, skipping", flush=True)
                    continue
            elif args.skip_missing:
                per_model[model] = {
                    "correct": 0, "total": 0, "accuracy": 0.0,
                    "errors": len(qa_records), "json_leaks": 0,
                    "avg_model_latency_ms": 0.0, "avg_recall_latency_ms": 0.0,
                    "skipped_reason": "not_pulled",
                }
                per_model_rows[model] = []
                continue

        rows: list[dict[str, Any]] = []
        t_model = time.perf_counter()
        for i, qa in enumerate(qa_records):
            try:
                row = run_question_for_model(args, model, qa)
            except Exception as e:
                row = {
                    "qid": qa.get("qid"),
                    "question_type": qa.get("question_type"),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            rows.append(row)
            stamp = "OK " if row.get("is_correct") else ("ERR" if row.get("error") else "X  ")
            ans = (row.get("answer") or "")[:80].replace("\n", " ")
            print(
                f"[demo-bench] {model} {i+1}/{len(qa_records)} {stamp} "
                f"qid={row.get('qid')} ans={ans!r}",
                flush=True,
            )
        per_model_rows[model] = rows
        agg = aggregate_model(rows)
        agg["wall_seconds"] = round(time.perf_counter() - t_model, 1)
        per_model[model] = agg
        print(
            f"[demo-bench] {model}: {agg['correct']}/{agg['total']} = "
            f"{agg['accuracy']*100:.1f}%  (errors={agg['errors']}, "
            f"json_leaks={agg['json_leaks']}, wall={agg['wall_seconds']}s)",
            flush=True,
        )
        prev_model = model

    # Evict the final model so we don't leave a dangling resident copy.
    if prev_model:
        ollama_evict(prev_model)

    json_path, md_path = write_outputs(args, dataset_meta, per_model, per_model_rows)

    print("\n=== Demolition Bench Summary ===")
    for m, agg in per_model.items():
        print(f"  {m:<24}  {agg['correct']:>3}/{agg['total']:<3}  "
              f"acc={agg['accuracy']*100:5.1f}%  errors={agg['errors']:<2} "
              f"json_leaks={agg['json_leaks']}  ({agg.get('avg_model_latency_ms', 0)/1000:.2f}s/call)")
    print(f"\njson:     {json_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
