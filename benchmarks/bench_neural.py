#!/usr/bin/env python3
"""
Neural Memory Benchmark — EvoMem datasets with local llama-server.

Uses the FULL Neural Memory stack:
  - NeuralMemory (memory_client.py) with C++ SIMD bridge + Cython fast_ops
  - MSSQLStore for persistence (shared with Dream Engine)
  - sentence-transformers CUDA embeddings
  - llama-server (Gemma-4) as LLM judge

Usage:
    # Start llama-server first, then:
    python bench_neural.py --dataset mmlu_pro --tasks 20
    python bench_neural.py --dataset mmlu_pro --tasks 20 --mssql
    python bench_neural.py --all --mssql
    python bench_neural.py --dataset mmlu_pro --no-memory  # baseline

Environment:
    LLAMA_SERVER_URL  — llama-server endpoint (default: http://localhost:8080/v1)
    LLAMA_MODEL       — model alias (default: GPT)
    EVO_MEM_DATA      — path to EvoMem datasets (default: ~/projects/evo_mem/data)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure python/ is on path for memory_client, fast_ops, mssql_store etc.
_py_dir = str(Path(__file__).resolve().parent.parent / "python")
if _py_dir not in sys.path:
    sys.path.insert(0, _py_dir)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080/v1")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "GPT")
DATA_DIR = Path(os.environ.get("EVO_MEM_DATA", Path.home() / "projects/evo_mem/data"))
RESULTS_DIR = Path(__file__).parent / "results" / "neural_memory"


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """Strip markdown bold/italic that kills exact_match scoring."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# LLM Client (llama-server with reasoning budget cap)
# ---------------------------------------------------------------------------

class LlamaClient:
    """OpenAI-compatible client for llama-server."""

    def __init__(self, base_url: str = LLAMA_SERVER_URL, model: str = LLAMA_MODEL):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate completion. Handles reasoning models."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,        # deterministic
                seed=42,                # reproducibility
                extra_body={
                    "reasoning_budget": -1,  # cap CoT, force content
                    "top_p": 1.0,
                    "min_p": 0.02,
                    "repeat_penalty": 1.05,
                },
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
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
# Answer Extraction (with markdown stripping)
# ---------------------------------------------------------------------------

def extract_answer(response: str, dataset: str) -> str:
    """Extract the final answer from LLM response."""
    response = strip_markdown(response)
    if not response:
        return ""

    if dataset == "aime_2024":
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
        nums = re.findall(r"\b(\d{1,3})\b", response)
        return nums[-1] if nums else ""

    patterns = [
        r"(?:answer|Answer|ANSWER)[\s:]*([A-Da-d])\b",
        r"(?:the answer is|The answer is)\s*([A-Da-d])\b",
        r"\\boxed\{([A-Da-d])\}",
        r"(?:option|Option)\s*([A-Da-d])\b",
        r"^([A-Da-d])[\.\)\:]",
    ]
    tail = response[-500:]
    for pat in patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).upper()
    for pat in patterns:
        m = re.search(pat, response)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"\b([A-Da-d])\b", response)
    return letters[-1].upper() if letters else response[:1].upper()


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_prompt_no_memory(task: Dict, dataset: str) -> str:
    """Build prompt WITHOUT memory context."""
    if dataset in ("mmlu_pro", "gpqa_diamond"):
        q = task.get("question", "")
        choices = task.get("choices", [])
        if choices:
            letters = "ABCDEFGHIJ"[:len(choices)]
            opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            return f"""Answer the following multiple choice question. Reply with ONLY the letter.

Question: {q}

{opts}

Answer:"""
        return f"Answer with a single letter (A-D):\n{q}\n\nAnswer:"
    elif dataset == "aime_2024":
        q = task.get("problem", task.get("question", ""))
        return f"""Solve this AIME problem. Give your final answer as an integer from 0-999.

Problem: {q}

Solution:"""
    return str(task)


def build_prompt_with_memory(task: Dict, memory_results: List[Dict], dataset: str) -> str:
    """Build prompt WITH memory context from NeuralMemory.recall()."""
    question = task.get("question", task.get("problem", str(task)))

    if memory_results:
        mem_lines = []
        for r in memory_results[:4]:
            sim = r.get("similarity", r.get("combined", 0))
            content = r.get("content", "")[:200]
            if sim > 0.15:
                mem_lines.append(f"- [{sim:.2f}] {content}")
        mem_text = "\n".join(mem_lines)
    else:
        mem_text = ""

    base = build_prompt_no_memory(task, dataset)

    if mem_text:
        return f"""Relevant context from previous tasks:
{mem_text}

{base}"""
    return base


# ---------------------------------------------------------------------------
# Benchmark Runner (uses FULL NeuralMemory stack)
# ---------------------------------------------------------------------------

def run_benchmark(dataset_name: str, tasks: List[Dict],
                  llm: LlamaClient,
                  memory = None,  # NeuralMemory instance or None
                  use_memory: bool = True) -> Dict[str, Any]:
    """Run benchmark using NeuralMemory.recall() for retrieval."""
    from tqdm import tqdm

    correct = 0
    total = 0
    results = []
    total_retrieve_ms = 0

    for i, task in enumerate(tqdm(tasks, desc=dataset_name)):
        question = task.get("question", task.get("problem", ""))

        # Retrieve from NeuralMemory (uses C++ SIMD + Cython)
        if use_memory and memory:
            t0 = time.time()
            memory_results = memory.recall(question, k=5)
            total_retrieve_ms += (time.time() - t0) * 1000
            prompt = build_prompt_with_memory(task, memory_results, dataset_name)
        else:
            memory_results = []
            prompt = build_prompt_no_memory(task, dataset_name)

        # Get answer from LLM
        response = llm.complete(prompt, max_tokens=1024)
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

        # Store correct answer in memory (test-time learning)
        if use_memory and memory and is_correct:
            memory.remember(
                f"Q: {question}\nCorrect answer: {gt}",
                label=f"bench-{dataset_name}-{i}",
                detect_conflicts=False
            )

        results.append({
            "task_id": i,
            "question": question[:100],
            "predicted": predicted,
            "actual": str(gt),
            "correct": is_correct,
        })

    accuracy = correct / max(total, 1)
    avg_retrieve = total_retrieve_ms / max(total, 1)

    stats = memory.stats() if memory else {}

    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_retrieve_ms": round(avg_retrieve, 1),
        "memory_size": stats.get("memories", 0),
        "connections": stats.get("connections", 0),
        "cpp_enabled": memory._cpp is not None if memory else False,
        "use_memory": use_memory,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = ["mmlu_pro", "gpqa_diamond", "aime_2024"]

def main():
    parser = argparse.ArgumentParser(description="Neural Memory Benchmark (Full Stack)")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="mmlu_pro")
    parser.add_argument("--tasks", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument("--url", default=LLAMA_SERVER_URL, help="llama-server URL")
    parser.add_argument("--model", default=LLAMA_MODEL, help="Model alias")
    parser.add_argument("--no-memory", action="store_true", help="Baseline (no memory)")
    parser.add_argument("--mssql", action="store_true", help="Use MSSQL backend")
    parser.add_argument("--no-cpp", action="store_true", help="Disable C++ SIMD bridge")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Memory config
    mem_label = "OFF (baseline)" if args.no_memory else \
                "MSSQL + C++ + Cython" if args.mssql else \
                "SQLite + C++ + Cython"

    print("=" * 60)
    print("  Neural Memory Benchmark (EvoMem)")
    print("=" * 60)
    print(f"  LLM:       {args.model} @ {args.url}")
    print(f"  Memory:    {mem_label}")
    print(f"  Temp:      0.0 (seed=42)")
    print(f"  Cython:    ", end="")
    try:
        from fast_ops import cosine_similarity
        print("ON (66x cosine_similarity)")
    except ImportError:
        print("OFF (Python fallback)")
    print("=" * 60)

    # Init LLM
    llm = LlamaClient(args.url, args.model)
    test = llm.complete("Say OK", max_tokens=10)
    if not test:
        print("[ERROR] Cannot connect to llama-server")
        sys.exit(1)
    print(f"[llm] Connected: '{test.strip()}'")

    # Init NeuralMemory (full stack)
    memory = None
    if not args.no_memory:
        from memory_client import NeuralMemory
        memory = NeuralMemory(
            use_mssql=args.mssql,
            use_cpp=not args.no_cpp,
            embedding_backend="sentence-transformers",
        )
        mem_stats = memory.stats()
        print(f"[memory] {mem_stats['memories']} memories, {mem_stats['connections']} connections, "
              f"C++={memory._cpp is not None}")

    # Run benchmarks
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        tasks = load_dataset(ds_name, args.tasks)
        if not tasks:
            continue

        result = run_benchmark(ds_name, tasks, llm, memory,
                               use_memory=not args.no_memory)
        all_results[ds_name] = result

        print(f"\n  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        print(f"  Retrieve: {result['avg_retrieve_ms']}ms avg")
        print(f"  Memory:   {result['memory_size']} memories, {result['connections']} connections")
        print(f"  C++:      {result['cpp_enabled']}")

    # Cleanup
    if memory:
        memory.close()

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Stack: {mem_label}")
    for ds, r in all_results.items():
        print(f"  {ds:20s}  {r['accuracy']:6.1%}  ({r['correct']:2d}/{r['total']:2d})  "
              f"retrieve={r['avg_retrieve_ms']:5.1f}ms  "
              f"cpp={r['cpp_enabled']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
