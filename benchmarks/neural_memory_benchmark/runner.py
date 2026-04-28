#!/usr/bin/env python3
"""
Neural Memory Benchmark — CLI Runner
====================================
Entry point for running the full benchmark suite.

Usage (run as module from project root):
    cd ~/projects/neural-memory-adapter
    python3 -m benchmarks.neural_memory_benchmark.runner --dry-run
    python3 -m benchmarks.neural_memory_benchmark.runner --suite retrieval
    python3 -m benchmarks.neural_memory_benchmark.runner --list

    # Full benchmark
    python3 -m benchmarks.neural_memory_benchmark.runner

    # Generate report from last results
    python3 -m benchmarks.neural_memory_benchmark.report
"""
import importlib.util
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
BENCH_ROOT = THIS_FILE.parent          # neural_memory_benchmark/
BENCHMARKS_DIR = BENCH_ROOT.parent     # benchmarks/
SRC_ROOT = BENCHMARKS_DIR.parent / "python"  # python/ (neural-memory-adapter root)
PROJECT_ROOT = BENCHMARKS_DIR.parent   # project root

# Build sys.path: BENCH_ROOT must end up FIRST so its own modules (config.py,
# dataset.py) win over the project's python/config.py. Insert in REVERSE order
# because each insert(0, …) pushes earlier entries down.
for _p in [str(SRC_ROOT), str(PROJECT_ROOT), str(BENCH_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Load benchmark.py as a module without importing (avoids __package__ issues)
benchmark_path = BENCH_ROOT / "benchmark.py"
spec = importlib.util.spec_from_file_location("_benchmark_main", benchmark_path)
benchmark_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_mod)

if __name__ == "__main__":
    # Patch sys.argv before calling main()
    sys.argv[0] = str(THIS_FILE)
    benchmark_mod.main()

