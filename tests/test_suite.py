#!/usr/bin/env python3
"""
Mazemaker Adapter — Comprehensive Test Suite
==================================================
Tests every component, integration point, and edge case.
Run: python3 tests/test_suite.py [--quick] [--verbose]

Components tested:
  1. Embed Provider (bge-m3, 1024d, CUDA)
  2. Mazemaker Client (remember, recall, think)
  3. Dream Engine (NREM, REM, Insight phases)
  4. Database (schema, integrity, indexes)
  5. C++ Bridge / LSTM-KNN Bridge
  6. Hermes Plugin (init, hooks)
  7. Installation Sync (python/ ↔ hermes-plugin/)
  8. Edge Cases (empty queries, large batches, concurrent)
"""

import sys
import os
import time
import json
import sqlite3
import hashlib
import argparse
from pathlib import Path
from typing import Optional

# Add project to path
PROJECT_DIR = Path(__file__).parent.parent
PYTHON_DIR = PROJECT_DIR / "python"
PLUGIN_DIR = PROJECT_DIR / "hermes-plugin"
sys.path.insert(0, str(PYTHON_DIR))

# Test infrastructure
class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []

    def ok(self, name: str, msg: str = ""):
        self.passed += 1
        print(f"  ✓ {name}" + (f" — {msg}" if msg else ""))

    def fail(self, name: str, msg: str):
        self.failed += 1
        self.errors.append(f"{name}: {msg}")
        print(f"  ✗ {name} — {msg}")

    def warn(self, name: str, msg: str):
        self.warnings.append(f"{name}: {msg}")
        print(f"  ⚠ {name} — {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed", end="")
        if self.failed:
            print(f", {self.failed} FAILED", end="")
        if self.warnings:
            print(f", {self.warnings.length if hasattr(self.warnings, 'length') else len(self.warnings)} warnings", end="")
        print()
        if self.errors:
            print(f"\n  FAILURES:")
            for e in self.errors:
                print(f"    ✗ {e}")
        print(f"{'='*60}")
        return self.failed == 0


R = TestResult()

# ========================================================================
# 1. INSTALLATION SYNC
# ========================================================================

def test_install_sync():
    """Verify python/ source files exist and hermes-plugin/ holds metadata only.

    Architecture (since refactor e4f33d9):
      python/         — single source of truth for all .py files
      hermes-plugin/  — Hermes-specific metadata (plugin.yaml, etc.)
      Deploy targets  — SYMLINKS back to python/, created by install.sh

    Pre-refactor this test verified .py copies in hermes-plugin/ were in
    sync; that path no longer exists. Same fix as tests/test_upside_down.py
    (iter 1 commit 52bdc24).
    """
    print("\n[1] INSTALLATION SYNC")
    sources = [
        '__init__.py', 'access_logger.py', 'config.py', 'cpp_bridge.py',
        'dream_engine.py', 'dream_postgres_store.py',
        'embed_provider.py', 'lstm_knn_bridge.py',
        'memory_client.py', 'postgres_store.py',
    ]
    missing = [f for f in sources if not (PYTHON_DIR / f).exists()]
    if missing:
        R.fail("sync/sources", f"missing from python/: {missing}")
    else:
        R.ok("sync/sources", f"all {len(sources)} source files present in python/")

    plugin_only = ['plugin.yaml', 'neural_skin.yaml', 'skills']
    for f in plugin_only:
        if (PLUGIN_DIR / f).exists():
            R.ok(f"sync/plugin-only/{f}", "present")
        else:
            R.fail(f"sync/plugin-only/{f}", "missing from hermes-plugin")

    # Regression guard: hermes-plugin/ should NOT contain .py copies
    # (symlinks to python/ are fine for in-tree testing).
    stray = []
    if PLUGIN_DIR.exists():
        for p in PLUGIN_DIR.glob("*.py"):
            if not p.is_symlink():
                stray.append(p.name)
    if stray:
        R.fail("sync/no-stray-copies",
               f"hermes-plugin/ has non-symlink .py files (regression): {stray}")
    else:
        R.ok("sync/no-stray-copies", "hermes-plugin/ has no stray .py copies")


# ========================================================================
# 2. DATABASE
# ========================================================================

def test_database():
    print("\n[2] DATABASE")
    db_path = os.path.expanduser('~/.neural_memory/memory.db')
    if not os.path.exists(db_path):
        R.fail("db/exists", f"not found at {db_path}")
        return

    size_mb = os.path.getsize(db_path) / 1024 / 1024
    R.ok("db/exists", f"{size_mb:.1f} MB")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Schema check
    expected_tables = {'memories', 'connections', 'connection_history',
                       'dream_sessions', 'dream_insights'}
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    actual = {r[0] for r in cur.fetchall()}
    missing = expected_tables - actual
    if missing:
        R.fail("db/schema", f"missing tables: {missing}")
    else:
        R.ok("db/schema", f"{len(actual)} tables")

    # Row counts
    for table in expected_tables & actual:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        if table == 'memories' and count < 100:
            R.warn(f"db/{table}", f"only {count} memories — seems low")
        else:
            R.ok(f"db/{table}", f"{count} rows")

    # Index check
    cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
    indexes = [r[0] for r in cur.fetchall()]
    if len(indexes) >= 5:
        R.ok("db/indexes", f"{len(indexes)} indexes")
    else:
        R.warn("db/indexes", f"only {len(indexes)} indexes")

    # WAL mode
    cur.execute("PRAGMA journal_mode")
    mode = cur.fetchone()[0]
    if mode.upper() == 'WAL':
        R.ok("db/wal", mode)
    else:
        R.warn("db/wal", f"mode={mode} (expected WAL)")

    # Integrity check
    cur.execute("PRAGMA integrity_check")
    integrity = cur.fetchone()[0]
    if integrity == 'ok':
        R.ok("db/integrity")
    else:
        R.fail("db/integrity", integrity)

    conn.close()


# ========================================================================
# 3. EMBED PROVIDER
# ========================================================================

def test_embed_provider():
    print("\n[3] EMBED PROVIDER")
    try:
        from embed_provider import EmbeddingProvider, DIMENSION
    except Exception as e:
        R.fail("embed/import", str(e))
        return

    # DIMENSION constant
    if DIMENSION == 1024:
        R.ok("embed/DIMENSION", f"{DIMENSION}")
    else:
        R.fail("embed/DIMENSION", f"{DIMENSION} (expected 1024)")

    # Initialize
    try:
        ep = EmbeddingProvider()
        R.ok("embed/init", f"backend={type(ep.backend).__name__}")
    except Exception as e:
        R.fail("embed/init", str(e))
        return

    # Dimension check
    if ep.dim == 1024:
        R.ok("embed/dim", f"{ep.dim}d")
    else:
        R.fail("embed/dim", f"{ep.dim}d (expected 1024)")

    # Single embed
    try:
        vec = ep.embed("Test sentence for mazemaker embedding")
        if len(vec) == 1024:
            R.ok("embed/single", f"len={len(vec)}")
        else:
            R.fail("embed/single", f"len={len(vec)} (expected 1024)")
    except Exception as e:
        R.fail("embed/single", str(e))

    # Batch embed
    try:
        vecs = ep.embed_batch(["Hello", "World", "Neural", "Memory"])
        if len(vecs) == 4 and all(len(v) == 1024 for v in vecs):
            R.ok("embed/batch", f"{len(vecs)} vectors, all 1024d")
        else:
            R.fail("embed/batch", f"got {len(vecs)} vectors")
    except Exception as e:
        R.fail("embed/batch", str(e))

    # Embed server socket
    sock = os.path.expanduser('~/.neural_memory/embed.sock')
    if os.path.exists(sock):
        R.ok("embed/server", "socket exists")
    else:
        R.warn("embed/server", "socket not found (server may not be running)")


# ========================================================================
# 4. NEURAL MEMORY CLIENT
# ========================================================================

def test_memory_client():
    print("\n[4] NEURAL MEMORY CLIENT")
    try:
        from memory_client import Mazemaker
    except Exception as e:
        R.fail("nm/import", str(e))
        return

    # Initialize
    try:
        nm = Mazemaker()
        R.ok("nm/init", f"{len(nm._graph_nodes)} nodes loaded")
    except Exception as e:
        R.fail("nm/init", str(e))
        return

    # Dimension
    if hasattr(nm, 'dim'):
        if nm.dim == 1024:
            R.ok("nm/dim", f"{nm.dim}d")
        else:
            R.fail("nm/dim", f"{nm.dim}d (expected 1024)")

    # REMEMBER
    test_content = f"Test memory {time.time()}: Integration test from test_suite.py"
    try:
        mem_id = nm.remember(test_content, label="test-suite")
        if isinstance(mem_id, int) and mem_id > 0:
            R.ok("nm/remember", f"id={mem_id}")
        else:
            R.fail("nm/remember", f"unexpected id: {mem_id}")
    except Exception as e:
        R.fail("nm/remember", str(e))
        mem_id = None

    # RECALL
    try:
        results = nm.recall("integration test", k=3)
        if isinstance(results, list) and len(results) > 0:
            top = results[0]
            has_content = 'content' in top or 'label' in top
            R.ok("nm/recall", f"{len(results)} results, has_content={has_content}")
        else:
            R.fail("nm/recall", f"unexpected: {results}")
    except Exception as e:
        R.fail("nm/recall", str(e))

    # THINK
    if mem_id:
        try:
            thoughts = nm.think(mem_id, depth=2)
            if isinstance(thoughts, list):
                R.ok("nm/think", f"{len(thoughts)} connections from {mem_id}")
            else:
                R.fail("nm/think", f"unexpected type: {type(thoughts)}")
        except Exception as e:
            R.fail("nm/think", str(e))

    # GRAPH
    try:
        graph = nm.graph()
        nodes = graph.get('nodes', 0)
        edges = graph.get('edges', 0)
        if nodes > 0 and edges > 0:
            R.ok("nm/graph", f"{nodes} nodes, {edges} edges")
        else:
            R.fail("nm/graph", f"nodes={nodes}, edges={edges}")
    except Exception as e:
        R.fail("nm/graph", str(e))

    # Cleanup test memory
    if mem_id:
        try:
            db_path = os.path.expanduser('~/.neural_memory/memory.db')
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            cur.execute("DELETE FROM connections WHERE source_id = ? OR target_id = ?",
                       (mem_id, mem_id))
            conn.commit()
            conn.close()
            R.ok("nm/cleanup", f"removed test memory {mem_id}")
        except Exception as e:
            R.warn("nm/cleanup", str(e))


# ========================================================================
# 5. DREAM ENGINE
# ========================================================================

def test_dream_engine():
    print("\n[5] DREAM ENGINE")
    try:
        from dream_engine import DreamEngine, SQLiteDreamBackend
    except Exception as e:
        R.fail("dream/import", str(e))
        return

    db_path = os.path.expanduser('~/.neural_memory/memory.db')
    try:
        backend = SQLiteDreamBackend(db_path)
        R.ok("dream/backend", "SQLiteDreamBackend")
    except Exception as e:
        R.fail("dream/backend", str(e))
        return

    try:
        engine = DreamEngine(backend)
        R.ok("dream/engine", "DreamEngine initialized")
    except Exception as e:
        R.fail("dream/engine", str(e))
        return

    # Run a dream cycle
    try:
        result = engine.dream_now()
        if 'error' not in result:
            nrem = result.get('nrem', {})
            rem = result.get('rem', {})
            insights = result.get('insights', {})
            R.ok("dream/now",
                 f"NREM: {nrem.get('processed', 0)} processed, "
                 f"REM: {rem.get('bridges', 0)} bridges, "
                 f"Insights: {insights.get('insights', 0)}")
        else:
            R.fail("dream/now", result['error'])
    except Exception as e:
        R.fail("dream/now", str(e))

    # Stats
    try:
        stats = backend.get_dream_stats()
        R.ok("dream/stats",
             f"{stats['sessions']} sessions, {stats['total_insights']} insights")
    except Exception as e:
        R.fail("dream/stats", str(e))


# ========================================================================
# 6. C++ BRIDGE & LSTM-KNN
# ========================================================================

def test_bridges():
    print("\n[6] BRIDGES")

    # C++ Bridge
    try:
        from cpp_bridge import MazemakerCpp
        R.ok("cpp/import", "MazemakerCpp available")
    except Exception as e:
        R.warn("cpp/import", f"not available: {e}")

    # fast_ops (Cython)
    try:
        import fast_ops
        R.ok("cpp/fast_ops", "compiled Cython module")
    except ImportError:
        R.warn("cpp/fast_ops", "not compiled (Python fallback)")

    # LSTM Bridge
    try:
        from lstm_knn_bridge import LSTMBridge, KNNBridge
        R.ok("lstm/import", "LSTMBridge + KNNBridge available")
    except Exception as e:
        R.warn("lstm/import", f"not available: {e}")

    # KNN Bridge test
    try:
        from lstm_knn_bridge import KNNBridge
        knn = KNNBridge()
        R.ok("knn/init", type(knn).__name__)
    except Exception as e:
        R.warn("knn/init", str(e))


# ========================================================================
# 7. EDGE CASES
# ========================================================================

def test_edge_cases():
    print("\n[7] EDGE CASES")
    try:
        from memory_client import Mazemaker
        nm = Mazemaker()
    except Exception as e:
        R.fail("edge/import", str(e))
        return

    # Empty recall
    try:
        results = nm.recall("", k=5)
        R.ok("edge/empty-recall", f"returned {len(results)} results (no crash)")
    except Exception as e:
        R.fail("edge/empty-recall", str(e))

    # Very short text
    try:
        mid = nm.remember("a", label="edge-test")
        R.ok("edge/short-text", f"remembered 1-char, id={mid}")
        # Cleanup
        db = sqlite3.connect(os.path.expanduser('~/.neural_memory/memory.db'))
        db.execute("DELETE FROM memories WHERE id = ?", (mid,))
        db.commit()
        db.close()
    except Exception as e:
        R.fail("edge/short-text", str(e))

    # Unicode text
    try:
        mid = nm.remember("日本語テスト 🚀 Тест на русском", label="edge-unicode")
        R.ok("edge/unicode", f"id={mid}")
        db = sqlite3.connect(os.path.expanduser('~/.neural_memory/memory.db'))
        db.execute("DELETE FROM memories WHERE id = ?", (mid,))
        db.commit()
        db.close()
    except Exception as e:
        R.fail("edge/unicode", str(e))

    # Large k recall
    try:
        results = nm.recall("test", k=100)
        R.ok("edge/large-k", f"k=100, got {len(results)}")
    except Exception as e:
        R.fail("edge/large-k", str(e))

    # Think on non-existent ID
    try:
        thoughts = nm.think(99999999, depth=1)
        R.ok("edge/think-missing", f"returned {len(thoughts)} (no crash)")
    except Exception as e:
        R.fail("edge/think-missing", str(e))

    # Batch embed large
    try:
        from embed_provider import EmbeddingProvider
        ep = EmbeddingProvider()
        big_batch = [f"Sentence number {i} for batch testing" for i in range(50)]
        vecs = ep.embed_batch(big_batch)
        if len(vecs) == 50:
            R.ok("edge/large-batch", f"50 vectors, all {len(vecs[0])}d")
        else:
            R.fail("edge/large-batch", f"got {len(vecs)}")
    except Exception as e:
        R.fail("edge/large-batch", str(e))


# ========================================================================
# 8. NO-MINILM VERIFICATION
# ========================================================================

def test_no_minilm():
    print("\n[8] NO-MINILM VERIFICATION")

    # Check disk
    minilm_paths = [
        os.path.expanduser('~/.neural_memory/models/models--sentence-transformers--all-MiniLM-L6-v2'),
        os.path.expanduser('~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'),
        os.path.expanduser('~/.cache/chroma/onnx_models/all-MiniLM-L6-v2'),
    ]
    for p in minilm_paths:
        if os.path.exists(p):
            R.fail("minilm/disk", f"found at {p}")
        else:
            R.ok("minilm/disk", f"clean: {os.path.basename(p)}")

    # Check code
    import subprocess
    result = subprocess.run(
        ['grep', '-rn', '-i', 'all-minilm', str(PROJECT_DIR),
         '--include=*.py', '--include=*.yaml',
         '--exclude-dir=tests', '--exclude=test_suite.py'],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        R.fail("minilm/code", f"references found:\n{result.stdout[:200]}")
    else:
        R.ok("minilm/code", "no references in code")

    # Verify embed output is 1024d
    try:
        from embed_provider import EmbeddingProvider
        ep = EmbeddingProvider()
        vec = ep.embed("dimension check")
        if len(vec) == 1024:
            R.ok("minilm/dimension", f"confirmed 1024d (not 384d)")
        else:
            R.fail("minilm/dimension", f"{len(vec)}d — wrong model loaded!")
    except Exception as e:
        R.fail("minilm/dimension", str(e))


# ========================================================================
# MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='Mazemaker Test Suite')
    parser.add_argument('--quick', action='store_true', help='Skip slow tests')
    parser.add_argument('--verbose', action='store_true', help='Extra output')
    args = parser.parse_args()

    print("=" * 60)
    print("  NEURAL MEMORY ADAPTER — TEST SUITE")
    print("=" * 60)

    test_install_sync()
    test_database()
    test_embed_provider()
    test_memory_client()
    test_dream_engine()
    test_bridges()
    test_edge_cases()
    test_no_minilm()

    success = R.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
