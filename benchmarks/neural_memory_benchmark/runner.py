#!/usr/bin/env python3
"""
Mazemaker Benchmark — CLI Runner
====================================
Entry point for running the full benchmark suite.

Usage (run as module from project root):
    cd ~/projects/mazemaker-adapter
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
SRC_ROOT = BENCHMARKS_DIR.parent / "python"  # python/ (mazemaker-adapter root)
PROJECT_ROOT = BENCHMARKS_DIR.parent   # project root

# Build sys.path: BENCH_ROOT must end up FIRST so its own modules (config.py,
# dataset.py) win over the project's python/config.py. The project's python/
# directory is intentionally APPENDED (not prepended) so it can't shadow
# benchmark-local modules — engine imports work because suites do their own
# `sys.path.insert` for `memory_client` when they need it.
for _p in [str(BENCH_ROOT), str(PROJECT_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Load benchmark.py as a module without importing.
# F20 fix (audit 2026-05-13): also register the loaded module under the
# canonical name `benchmark` in sys.modules so a downstream `import
# benchmark` finds the same instance — previously the importlib name
# `_benchmark_main` was the only registered key, and a second
# `import benchmark` created a parallel module with its own globals.
benchmark_path = BENCH_ROOT / "benchmark.py"
spec = importlib.util.spec_from_file_location("_benchmark_main", benchmark_path)
benchmark_mod = importlib.util.module_from_spec(spec)
sys.modules["_benchmark_main"] = benchmark_mod
sys.modules.setdefault("benchmark", benchmark_mod)
spec.loader.exec_module(benchmark_mod)

if __name__ == "__main__":
    # F55 fix (audit 2026-05-13): preserve the original argv[0] in
    # NEURAL_BENCH_ARGV0 so suites that do `Path(sys.argv[0])` get the
    # actually-invoked entry point if they need it, while argparse keeps
    # the runner path as the program name.
    os.environ.setdefault("NEURAL_BENCH_ARGV0", sys.argv[0])
    sys.argv[0] = str(THIS_FILE)
    benchmark_mod.main()

