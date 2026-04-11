#!/usr/bin/env python3
"""
Neural Memory Benchmark — EvoMem datasets with local llama-server.

Standalone script that benchmarks our Neural Memory retriever against
EvoMem datasets using a local llama-server as LLM judge.

Usage:
    # Start llama-server first:
    llama-server --model gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf \
        -c 393000 --alias "GPT" -np 3 -fa on --cache-prompt \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        -ngl 999 -t 8 -tb 8 -b 8192 -ub 4096 \
        --no-mmap --host 0.0.0.0 --port 8080 \
        --temp 0.5 --top-p 1.0 --min-p 0.02 \
        --repeat-penalty 1.05 --keep -1

    # Run benchmark:
    python bench_neural.py --dataset mmlu_pro --tasks 50
    python bench_neural.py --dataset gpqa_diamond --tasks 50
    python bench_neural.py --dataset aime_2024
    python bench_neural.py --all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080/v1")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "GPT")  # matches --alias
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = Path(os.environ.get("EVO_MEM_DATA", Path.home() / "projects/evo_mem/data"))
RESULTS_DIR = Path(__file__).parent / "results" / "neural_memory"

# ---------------------------------------------------------------------------
# Embedding Retriever
# ---------------------------------------------------------------------------

class NeuralRetriever:
    """Sentence-transformers retriever with temporal scoring."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[retriever] {model_name} on {device}, dim={self.dim}")

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False, batch_size=64)
        return np.array(vecs)

    def retrieve(self, query: str, memories: List[Dict],
                 top_k: int = 4) -> List[Dict]:
        """Retrieve top-k memories by cosine similarity + temporal boost."""
        if not memories:
            return []

        q_vec = self.encode(query)
        mem_vecs = np.array([m["embedding"] for m in memories])

        # Cosine similarity (embeddings already normalized)
        sims = mem_vecs @ q_vec

        # Temporal boost: more recent = slight advantage
        n = len(memories)
        temporal = np.array([0.97 + 0.03 * (i / max(n, 1))
                             for i in range(n)])

        combined = 0.7 * sims + 0.3 * temporal
        top_idx = np.argsort(combined)[-top_k:][::-1]

        return [{"memory": memories[i], "score": float(combined[i]),
                 "similarity": float(sims[i])}
                for i in top_idx]

    def retrieve_with_spreading(self, query: str, memories: List[Dict],
                                 top_k: int = 4) -> List[Dict]:
        """Retrieve with spreading activation from top results."""
        initial = self.retrieve(query, memories, top_k=top_k * 2)

        # Spread: boost memories similar to top results
        if len(initial) >= 2:
            top_vecs = np.array([memories[m["memory"]["index"]]["embedding"]
                                 for m in initial[:3]])
            all_vecs = np.array([m["embedding"] for m in memories])
            spread_scores = (all_vecs @ top_vecs.T).max(axis=1)

            for i, r in enumerate(initial):
                idx = r["memory"]["index"]
                r["score"] = 0.6 * r["score"] + 0.4 * float(spread_scores[idx])

            initial.sort(key=lambda x: -x["score"])

        return initial[:top_k]


# ---------------------------------------------------------------------------
# LLM Client (llama-server)
# ---------------------------------------------------------------------------

class LlamaClient:
    """OpenAI-compatible client for llama-server."""

    def __init__(self, base_url: str = LLAMA_SERVER_URL,
                 model: str = LLAMA_MODEL):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 300,
                 temperature: float = 0.0) -> str:
        """Generate completion. Returns answer text.

        Handles reasoning models that return content in reasoning_content
        instead of content (e.g., Gemma-4 turbo quant via llama-server).
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,        # deterministic — override server default
                seed=42,                # client-side seed for reproducibility
                extra_body={
                    "reasoning_budget": -1,  # cap CoT, force content output
                    "top_p": 1.0,
                    "min_p": 0.02,
                    "repeat_penalty": 1.05,
                },
            )
            msg = resp.choices[0].message
            # llama-server reasoning models: content="" but reasoning_content has the answer
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            # If content is empty but reasoning has text, use reasoning
            if not content.strip() and reasoning.strip():
                return reasoning.strip()
            return content.strip()
        except Exception as e:
            print(f"[llm] Error: {e}")
            return ""


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_dataset(name: str, max_tasks: int = 0) -> List[Dict]:
    """Load EvoMem dataset from JSON."""
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        print(f"[data] Not found: {path}")
        return []

    with open(path) as f:
        data = json.load(f)

    tasks = data if isinstance(data, list) else data.get("tasks", data.get("data", []))
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    print(f"[data] Loaded {len(tasks)} tasks from {name}")
    return tasks


# ---------------------------------------------------------------------------
# Answer Extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str, dataset: str) -> str:
    """Extract the final answer from LLM response.

    Handles reasoning model output where answer is at the END of
    a long reasoning chain. Strips markdown that kills exact_match.
    """
    import re
    response = strip_markdown(response.strip())
    if not response:
        return ""

    # For AIME: look for integer answer
    if dataset == "aime_2024":
        # Look for "answer is 42" or "= 42" at the end
        patterns = [
            r"(?:answer|Answer|ANSWER)[\s:=]*(\d{1,3})\b",
            r"(?:=|equals?)\s*(\d{1,3})\s*$",
            r"\\boxed\{(\d{1,3})\}",
            r"\*\*(\d{1,3})\*\*",
        ]
        for pat in patterns:
            m = re.search(pat, response)
            if m:
                return m.group(1)
        # Fallback: last number
        nums = re.findall(r"\b(\d{1,3})\b", response)
        return nums[-1] if nums else ""

    # For MCQ (MMLU-Pro, GPQA): look for letter answer
    # Search from END of response — reasoning models bury the answer deep
    patterns = [
        r"(?:answer|Answer|ANSWER)[\s:]*([A-Da-d])\b",
        r"(?:the answer is|The answer is)\s*([A-Da-d])\b",
        r"\*\*([A-Da-d])\*\*",
        r"\\boxed\{([A-Da-d])\}",
        r"(?:option|Option)\s*([A-Da-d])\b",
        r"^([A-Da-d])[\.\)\:]",
        r"^([A-Da-d])\s*$",
    ]

    # Try from end of text first (last 500 chars)
    tail = response[-500:]
    for pat in patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).upper()

    # Try full text
    for pat in patterns:
        m = re.search(pat, response)
        if m:
            return m.group(1).upper()

    # Last resort: last single letter A-D in the text
    letters = re.findall(r"\b([A-Da-d])\b", response)
    return letters[-1].upper() if letters else response[:1].upper()


def strip_markdown(text: str) -> str:
    """Strip markdown bold/italic that kills exact_match scoring."""
    import re
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_prompt_no_memory(task: Dict, dataset: str) -> str:
    """Build prompt WITHOUT memory context."""
    if dataset in ("mmlu_pro", "gpqa_diamond"):
        q = task.get("question", "")
        choices = task.get("choices", [])
        if choices:
            letters = "ABCD" if len(choices) == 4 else "ABCDEFGHIJ"[:len(choices)]
            opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            return f"""Answer the following multiple choice question. Reply with ONLY the letter (A, B, C, or D).

Question: {q}

{opts}

Answer:"""
        return f"Answer the following question with a single letter (A-D):\n\n{q}\n\nAnswer:"

    elif dataset == "aime_2024":
        q = task.get("problem", task.get("question", ""))
        return f"""Solve the following AIME math problem. Give your final answer as an integer from 0-999.

Problem: {q}

Solution:"""

    return str(task)


def build_prompt_with_memory(task: Dict, memories: List[Dict],
                              retriever: NeuralRetriever,
                              dataset: str) -> str:
    """Build prompt WITH retrieved memory context."""
    question = task.get("question", task.get("problem", str(task)))

    # Retrieve relevant memories
    if memories:
        results = retriever.retrieve_with_spreading(question, memories, top_k=4)
        mem_text = "\n".join(
            f"- [{r['score']:.2f}] {r['memory']['text'][:200]}"
            for r in results if r['score'] > 0.2
        )
    else:
        mem_text = ""

    base = build_prompt_no_memory(task, dataset)

    if mem_text:
        return f"""Relevant context from previous tasks:
{mem_text}

{base}"""
    return base


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(dataset_name: str, tasks: List[Dict],
                  llm: LlamaClient, retriever: NeuralRetriever,
                  use_memory: bool = True,
                  use_mssql: bool = False) -> Dict[str, Any]:
    """Run benchmark on a dataset.

    If use_mssql=True, loads memories from MSSQL (persistent, shared with dream engine).
    Otherwise uses in-memory list (ephemeral, correct answers only).
    """
    from tqdm import tqdm

    correct = 0
    total = 0
    memory: List[Dict] = []
    results = []
    total_retrieve_ms = 0

    # Load existing memories from MSSQL if requested
    mssql_store = None
    if use_mssql and use_memory:
        try:
            from mssql_store import MSSQLStore
            mssql_store = MSSQLStore()
            mssql_mems = mssql_store.get_all()
            for m in mssql_mems:
                if m.get("embedding"):
                    memory.append({
                        "text": m.get("content", ""),
                        "embedding": m["embedding"],
                        "index": len(memory),
                    })
            print(f"[mssql] Loaded {len(memory)} memories from MSSQL")
        except Exception as e:
            print(f"[mssql] Failed to load: {e}, falling back to in-memory")

    for i, task in enumerate(tqdm(tasks, desc=dataset_name)):
        # Build prompt
        if use_memory and memory:
            prompt = build_prompt_with_memory(task, memory, retriever, dataset_name)
            t0 = time.time()
            _ = retriever.retrieve(
                task.get("question", task.get("problem", "")), memory, top_k=4
            )
            total_retrieve_ms += (time.time() - t0) * 1000
        else:
            prompt = build_prompt_no_memory(task, dataset_name)

        # Get answer
        response = llm.complete(prompt, max_tokens=1024, temperature=0.0)
        predicted = extract_answer(response, dataset_name)

        # Ground truth
        gt = task.get("answer", task.get("correct_answer", ""))
        if isinstance(gt, int):
            gt = str(gt)
        elif isinstance(gt, list):
            gt = gt[0] if gt else ""

        is_correct = predicted.upper() == str(gt).upper()[:1].upper()
        if is_correct:
            correct += 1
        total += 1

        # Store in memory (only correct answers to avoid poisoning)
        if use_memory and not use_mssql:
            q_text = task.get("question", task.get("problem", ""))
            embedding = retriever.encode(f"Q: {q_text} A: {gt}")
            memory.append({
                "text": f"Q: {q_text}\nCorrect answer: {gt}",
                "embedding": embedding,
                "index": len(memory),
            })

        results.append({
            "task_id": i,
            "question": task.get("question", task.get("problem", ""))[:100],
            "predicted": predicted,
            "actual": str(gt),
            "correct": is_correct,
        })

    if mssql_store:
        mssql_store.close()

    accuracy = correct / max(total, 1)
    avg_retrieve = total_retrieve_ms / max(total, 1)

    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_retrieve_ms": round(avg_retrieve, 1),
        "memory_size": len(memory),
        "use_memory": use_memory,
        "use_mssql": use_mssql,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = ["mmlu_pro", "gpqa_diamond", "aime_2024"]

def main():
    parser = argparse.ArgumentParser(description="Neural Memory Benchmark")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="mmlu_pro")
    parser.add_argument("--tasks", type=int, default=50, help="Max tasks (0=all)")
    parser.add_argument("--url", default=LLAMA_SERVER_URL, help="llama-server URL")
    parser.add_argument("--model", default=LLAMA_MODEL, help="Model alias")
    parser.add_argument("--no-memory", action="store_true", help="Run without memory (baseline)")
    parser.add_argument("--mssql", action="store_true", help="Load memories from MSSQL (with dream consolidation)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mem_mode = "OFF (baseline)" if args.no_memory else "MSSQL (dream consolidated)" if args.mssql else "ON (in-memory)"
    print("=" * 60)
    print("  Neural Memory Benchmark (EvoMem)")
    print("=" * 60)
    print(f"  LLM:       {args.model} @ {args.url}")
    print(f"  Retriever: {EMBED_MODEL} (CUDA)")
    print(f"  Memory:    {mem_mode}")
    print(f"  Temp:      0.0 (seed=42)")
    print("=" * 60)

    # Init
    retriever = NeuralRetriever()
    llm = LlamaClient(args.url, args.model)

    # Test connection
    test = llm.complete("Say OK", max_tokens=5)
    if not test:
        print("[ERROR] Cannot connect to llama-server. Is it running?")
        print(f"  Expected: {args.url}")
        sys.exit(1)
    print(f"[llm] Connected: '{test.strip()}'")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    all_results = {}
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        tasks = load_dataset(ds_name, args.tasks)
        if not tasks:
            continue

        result = run_benchmark(ds_name, tasks, llm, retriever,
                               use_memory=not args.no_memory,
                               use_mssql=args.mssql)
        all_results[ds_name] = result

        print(f"\n  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        print(f"  Avg retrieve: {result['avg_retrieve_ms']}ms")
        print(f"  Memory size: {result['memory_size']}")

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for ds, r in all_results.items():
        print(f"  {ds:20s}  {r['accuracy']:6.1%}  ({r['correct']}/{r['total']})  retrieve={r['avg_retrieve_ms']}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
