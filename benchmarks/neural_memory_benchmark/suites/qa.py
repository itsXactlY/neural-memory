"""
QA Benchmark Suite — End-to-end answer correctness via Ollama gpt-oss-120b
==========================================================================
LongMemEval-style: store memories, recall top-k for each question, hand the
context to gpt-oss:120b-cloud, score whether the model's answer contains the
gold needle (case-insensitive substring).

Reports:
  - retrieval_accuracy: did the gold context appear in top-k? (upper bound)
  - qa_accuracy:        did the LLM answer correctly given the retrieved ctx?
  - compounding_loss:   retrieval_accuracy - qa_accuracy
  - llm_latency_*:      ollama call timing percentiles
"""
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project's python/ to sys.path so we can import memory_client
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from memory_client import Mazemaker  # noqa: E402


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("NEURAL_BENCH_QA_MODEL", "gpt-oss:120b-cloud")
QA_TIMEOUT_S = float(os.environ.get("NEURAL_BENCH_QA_TIMEOUT", "60"))


# ── Dataset ──────────────────────────────────────────────────────────────────
# Mirrors lme_eval.synthetic_records but with extra distractor facts so the
# corpus isn't trivially small. Each row: (context, question, gold_substring).

QA_FACTS: List[tuple] = [
    ("Ada stores the launch key under the blue ceramic owl.",
     "Where does Ada store the launch key?", "blue ceramic owl"),
    ("Bruno's backup server is called pinecone-seven.",
     "What is Bruno's backup server called?", "pinecone-seven"),
    ("The migration window for Atlas is 03:40 UTC on Sunday.",
     "When is the Atlas migration window?", "03:40"),
    ("Mira prefers FastEmbed over sentence-transformers for cold start speed.",
     "What does Mira prefer for cold start speed?", "fastembed"),
    ("The red notebook says project Zephyr uses port 7443.",
     "Which port does project Zephyr use?", "7443"),
    ("Kai's dog Lou reacts badly to chicken treats.",
     "Which treats are bad for Kai's dog Lou?", "chicken"),
    ("The demo API key was rotated after incident ORCHID-19.",
     "What incident caused the demo API key rotation?", "orchid-19"),
    ("The telemetry dashboard lives behind the cloudflared tunnel.",
     "Where does the telemetry dashboard live?", "cloudflared"),
    ("Session summaries should be stored, but raw turn dumps should stay opt-in.",
     "What should stay opt-in?", "raw turn dumps"),
    ("The dream engine insight phase now uses Louvain community detection.",
     "What does the dream engine insight phase use?", "louvain"),
    ("PULSE findings need content-hash dedup before neural ingestion.",
     "What dedup is needed before neural ingestion?", "content-hash"),
    ("The C++ bridge is optional; Python fallback must remain production-safe.",
     "What must remain production-safe if the C++ bridge is absent?", "python fallback"),
    ("HNSW should activate automatically only when the corpus is large enough.",
     "When should HNSW activate automatically?", "corpus is large"),
    ("The weekly rollup should write a WEEKLY.md brief after Insight.",
     "What file does the weekly rollup write?", "weekly.md"),
    ("The primary mazemaker database file is named memory.db.",
     "What is the primary mazemaker database file named?", "memory.db"),
    ("Ola's on-call shift starts every Tuesday at 09:00 CET.",
     "When does Ola's on-call shift start?", "tuesday"),
    ("The fallback embedding backend is sentence-transformers when FastEmbed is missing.",
     "What is the fallback embedding backend?", "sentence-transformers"),
    ("Salience decays with a factor of 0.95 per day in the default config.",
     "What is the default daily salience decay factor?", "0.95"),
    ("The Postgres bridge syncs cold archives one-way from SQLite to Postgres.",
     "Which direction does the Postgres bridge sync?", "sqlite to postgres"),
    ("The shared embedding server uses a UNIX domain socket at ~/.neural_memory/embed.sock.",
     "Where is the shared embedding server's UNIX socket?", "embed.sock"),
]


# ── Ollama client ────────────────────────────────────────────────────────────

def ollama_chat(question: str, context_text: str,
                model: str = OLLAMA_MODEL,
                timeout: float = QA_TIMEOUT_S) -> Dict[str, Any]:
    """Call ollama /api/chat with the given context+question. Returns
    {answer, latency_s, eval_count, error}."""
    system = (
        "You answer the user's question using ONLY the provided memory context. "
        "Reply with the shortest possible answer, ideally a single phrase. "
        "If the answer is not in the context, reply 'unknown'."
    )
    user = f"Context (top retrieved memories):\n{context_text}\n\nQuestion: {question}\nAnswer:"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"num_predict": 400, "temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        elapsed = time.perf_counter() - t0
        data = json.loads(raw)
        msg = data.get("message", {})
        return {
            "answer": (msg.get("content") or "").strip(),
            "thinking": (msg.get("thinking") or "").strip(),
            "latency_s": elapsed,
            "eval_count": data.get("eval_count", 0),
            "done_reason": data.get("done_reason"),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - t0}
    except json.JSONDecodeError as e:
        return {"error": f"json decode: {e}", "latency_s": time.perf_counter() - t0}


def ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ── Scoring ──────────────────────────────────────────────────────────────────

def needle_in(text: str, needle: str) -> bool:
    return (needle or "").lower() in (text or "").lower()


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int((len(s) - 1) * p / 100)
    return float(s[min(idx, len(s) - 1)])


# ── Benchmark ────────────────────────────────────────────────────────────────

class QABenchmark:
    """End-to-end QA accuracy with Ollama gpt-oss-120b as the reader model."""

    def __init__(
        self,
        db_path: str,
        memories: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        modes: Optional[List[str]] = None,
        top_k: int = 5,
        model: str = OLLAMA_MODEL,
    ):
        self.db_path = db_path
        self.distractor_memories = memories or []
        self.output_dir = output_dir or Path.home() / ".neural_memory_benchmark" / "results"
        self.modes = modes or ["semantic", "hybrid"]
        self.top_k = top_k
        self.model = model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nm: Optional[Mazemaker] = None
        self._needle_to_db_id: Dict[str, int] = {}

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self) -> Dict[str, Any]:
        print(f"  [setup] Storing {len(QA_FACTS)} QA facts + "
              f"{len(self.distractor_memories)} distractors in {self.db_path}")
        t0 = time.perf_counter()
        self.nm = Mazemaker(
            db_path=self.db_path,
            embedding_backend="auto",
            retrieval_mode="semantic",
        )
        # 1) Distractors first so QA facts get the higher rowids (and stay
        #    findable; ordering doesn't affect retrieval but it's cleaner).
        for m in self.distractor_memories:
            self.nm.remember(m["text"], label=m.get("label", "distractor"),
                             auto_connect=False)
        # 2) QA facts. Capture the rowid so we can verify ground-truth recall.
        for ctx, _q, needle in QA_FACTS:
            db_id = self.nm.remember(ctx, label="qa_fact", auto_connect=False)
            if db_id is not None:
                self._needle_to_db_id[needle.lower()] = int(db_id)

        elapsed = time.perf_counter() - t0
        return {
            "qa_facts_stored": len(QA_FACTS),
            "distractors_stored": len(self.distractor_memories),
            "setup_elapsed_s": round(elapsed, 2),
            "stats": self.nm.stats(),
        }

    # ── Per-question evaluation ──────────────────────────────────────────────

    def _eval_one(self, question: str, gold_needle: str, mode: str) -> Dict[str, Any]:
        self.nm._retrieval_mode = mode
        retrieved = self.nm.recall(question, k=self.top_k)
        # Retrieval correctness: needle present in any retrieved memory's content
        retrieved_texts = [r.get("content", "") for r in retrieved]
        retrieval_hit = any(needle_in(t, gold_needle) for t in retrieved_texts)
        # Build context block for the LLM
        context_text = "\n".join(
            f"- {t}" for t in retrieved_texts if t
        ) or "(no memories retrieved)"
        # LLM call
        llm = ollama_chat(question, context_text, model=self.model)
        if "error" in llm:
            return {
                "question": question, "gold": gold_needle, "mode": mode,
                "retrieval_hit": retrieval_hit,
                "qa_correct": False,
                "llm_error": llm["error"],
                "llm_latency_s": llm.get("latency_s", 0.0),
                "answer": None,
            }
        qa_correct = needle_in(llm["answer"], gold_needle)
        return {
            "question": question, "gold": gold_needle, "mode": mode,
            "retrieval_hit": retrieval_hit,
            "qa_correct": qa_correct,
            "answer": llm["answer"][:200],
            "llm_latency_s": round(llm["latency_s"], 3),
            "eval_count": llm.get("eval_count", 0),
        }

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        print("\n=== QA Benchmark (Ollama LLM-judged) ===")
        if not ollama_available():
            return {
                "error": f"Ollama not reachable at {OLLAMA_URL}. "
                         f"Skipping QA suite.",
            }

        results: Dict[str, Any] = {
            "model": self.model,
            "top_k": self.top_k,
            "setup": self.setup(),
            "modes": {},
            "per_question": {mode: [] for mode in self.modes},
        }

        for mode in self.modes:
            print(f"\n  === Mode: {mode} ===")
            mode_records: List[Dict[str, Any]] = []
            for ctx, question, gold in QA_FACTS:
                rec = self._eval_one(question, gold, mode)
                mode_records.append(rec)
                marker_r = "R" if rec["retrieval_hit"] else "."
                marker_q = "Q" if rec["qa_correct"] else "."
                err = f" ERR={rec.get('llm_error','')[:40]}" if "llm_error" in rec else ""
                print(f"    [{marker_r}{marker_q}] {question[:55]:<55} → "
                      f"{(rec.get('answer') or '<err>')[:40]:<40} "
                      f"({rec.get('llm_latency_s', 0):.1f}s){err}")
            results["per_question"][mode] = mode_records

            ret_hits = sum(r["retrieval_hit"] for r in mode_records)
            qa_hits = sum(r["qa_correct"] for r in mode_records)
            n = len(mode_records)
            latencies = [r["llm_latency_s"] for r in mode_records
                         if "llm_error" not in r]
            results["modes"][mode] = {
                f"retrieval_accuracy@{self.top_k}":
                    round(ret_hits / n, 4) if n else 0.0,
                "qa_accuracy":
                    round(qa_hits / n, 4) if n else 0.0,
                "compounding_loss":
                    round((ret_hits - qa_hits) / n, 4) if n else 0.0,
                "llm_latency_p50_s": round(percentile(latencies, 50), 2),
                "llm_latency_p95_s": round(percentile(latencies, 95), 2),
                "llm_latency_mean_s": round(
                    statistics.mean(latencies) if latencies else 0.0, 2
                ),
                "llm_errors": sum("llm_error" in r for r in mode_records),
                "n": n,
            }
            m = results["modes"][mode]
            print(f"\n    {mode}: retrieval@{self.top_k}={m[f'retrieval_accuracy@{self.top_k}']:.3f}, "
                  f"qa={m['qa_accuracy']:.3f}, "
                  f"loss={m['compounding_loss']:.3f}, "
                  f"p50={m['llm_latency_p50_s']}s")

        # Save
        out_path = self.output_dir / "qa_results.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[saved] {out_path}")
        return results
